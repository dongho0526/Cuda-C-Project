#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cmath>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "Timer.h"
#include "cuda_profiler_api.h"
#include "device_launch_parameters.h"

const int P1 = 10;
const int P2 = 120;

__device__ int penalty(int d1, int d2) {
    return (d1 == d2) ? 0 : ((abs(d1 - d2) == 1) ? P1 : P2);
}

__global__ void compute_cost_kernel(const uint8_t* left_img, const uint8_t* right_img, int width, int height, int channels, int* cost, int disparity_range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int d = 0; d < disparity_range; ++d) {
            int sad = 0;
            for (int c = 0; c < channels; ++c) {
                int xl = (x * channels + y * width * channels + c);
                int xr = ((x - d) * channels + y * width * channels + c);

                if (x - d >= 0) {
                    sad += abs(left_img[xl] - right_img[xr]);
                }
                else {
                    sad += 255;
                }
            }
            cost[(y * width + x) * disparity_range + d] = sad;
        }
    }
}



__global__ void aggregate_costs_horizontal_kernel(const int* cost, int* aggr_cost, int width, int height, int disparity_range) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < height) {
        for (int x = 1; x < width; ++x) {
            for (int d = 0; d < disparity_range; ++d) {
                int min_cost = cost[((y * width + (x - 1)) * disparity_range) + d];
                for (int pd = 0; pd < disparity_range; ++pd) {
                    int p = cost[((y * width + (x - 1)) * disparity_range) + pd] + penalty(d, pd);
                    if (p < min_cost) {
                        min_cost = p;
                    }
                }
                atomicAdd(&aggr_cost[(y * width + x) * disparity_range + d], min_cost);
            }
        }
    }
}



__global__ void aggregate_costs_vertical_kernel(const int* cost, int* aggr_cost, int width, int height, int disparity_range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < width) {
        for (int y = 1; y < height; ++y) {
            for (int d = 0; d < disparity_range; ++d) {
                int min_cost = cost[((y - 1) * width + x) * disparity_range + d];
                for (int pd = 0; pd < disparity_range; ++pd) {
                    int p = cost[((y - 1) * width + x) * disparity_range + pd] + penalty(d, pd);
                    if (p < min_cost) {
                        min_cost = p;
                    }
                }
                atomicAdd(&aggr_cost[(y * width + x) * disparity_range + d], min_cost);
            }
        }
    }
}
//

__global__ void compute_disparity_map_kernel(const int* aggr_cost, uint8_t* disparity_map, int width, int height, int disparity_range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int best_disparity = 0;
        int min_cost = aggr_cost[(y * width + x) * disparity_range + 0];
        for (int d = 1; d < disparity_range; ++d) {
            if (aggr_cost[(y * width + x) * disparity_range + d] < min_cost) {
                min_cost = aggr_cost[(y * width + x) * disparity_range + d];
                best_disparity = d;
            }
        }
        disparity_map[y * width + x] = static_cast<uint8_t>(best_disparity * 255 / disparity_range);
    }
}

void apply_color_map(const std::vector<uint8_t>& disparity_map, std::vector<uint8_t>& color_map, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        uint8_t value = disparity_map[i];
        float ratio = value / 255.0f;
        int r = static_cast<int>(std::max(0.0f, 1.0f - std::abs(ratio * 4.0f - 3.0f)) * 255.0f);
        int g = static_cast<int>(std::max(0.0f, 1.0f - std::abs(ratio * 4.0f - 2.0f)) * 255.0f);
        int b = static_cast<int>(std::max(0.0f, 1.0f - std::abs(ratio * 4.0f - 1.0f)) * 255.0f);

        color_map[i * 3 + 0] = r;
        color_map[i * 3 + 1] = g;
        color_map[i * 3 + 2] = b;
    }
}

int main() {

    char* out_file_name = "output_img.png";

    int width, height, channels;

    uint8_t* left_img = stbi_load("im0.png", &width, &height, &channels, 0);
    if (!left_img) {
        std::cerr << "Failed to load left image" << std::endl;
        return -1;
    }

    uint8_t* right_img = stbi_load("im1.png", &width, &height, &channels, 0);
    if (!right_img) {
        std::cerr << "Failed to load right image" << std::endl;
        stbi_image_free(left_img);
        return -1;
    }

    int disparity_range = 25;
    int img_size = width * height;
    int cost_size = img_size * disparity_range;

    std::vector<uint8_t> disparity_map(img_size, 0);
    std::vector<uint8_t> color_map(img_size * 3, 0);

    // Allocate device memory
    uint8_t* d_left_img, * d_right_img;
    int* d_cost, * d_aggr_cost;
    uint8_t* d_disparity_map;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t cudaStatus;

    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaEventRecord(start, 0);

    cudaStatus = cudaMalloc(&d_left_img, width * height * channels * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }
    cudaStatus = cudaMalloc(&d_right_img, width * height * channels * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }
    cudaStatus = cudaMalloc(&d_cost, cost_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }
    cudaStatus = cudaMalloc(&d_aggr_cost, cost_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }
    cudaStatus = cudaMalloc(&d_disparity_map, img_size * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    // Copy images to device
    cudaStatus = cudaMemcpy(d_left_img, left_img, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }
    cudaStatus = cudaMemcpy(d_right_img, right_img, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);


    // Compute cost
    compute_cost_kernel << <numBlocks, threadsPerBlock >> > (d_left_img, d_right_img, width, height, channels, d_cost, disparity_range);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching compute_cost_kernel!\n", cudaStatus);
        //goto Error;
    }

    // Initialize aggregated cost to zero
    cudaMemset(d_aggr_cost, 0, cost_size * sizeof(int));

    // Aggregate costs horizontally
    aggregate_costs_horizontal_kernel << <dim3(1, (height + threadsPerBlock.y - 1) / threadsPerBlock.y), threadsPerBlock >> > (d_cost, d_aggr_cost, width, height, disparity_range);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching aggregate costs horizontal kernel!\n", cudaStatus);
        //goto Error;
    }

    // Aggregate costs vertically
    aggregate_costs_vertical_kernel << <dim3((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 1), threadsPerBlock >> > (d_cost, d_aggr_cost, width, height, disparity_range);
    cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching aggregate costs vertical kernel!\n", cudaStatus);
        //goto Error;
    }

    // Compute disparity map
    compute_disparity_map_kernel << <numBlocks, threadsPerBlock >> > (d_aggr_cost, d_disparity_map, width, height, disparity_range);
    cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching compute disparity map kernel!\n", cudaStatus);
        //goto Error;
    }

    // Copy disparity map to host
    cudaMemcpy(disparity_map.data(), d_disparity_map, img_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Apply color map
    apply_color_map(disparity_map, color_map, width, height);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cudaElapsedTime;
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    printf("StereoVision Depth Map(Cuda C) - Time for execution = %3.1f ms \n", cudaElapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Save output image
    if (stbi_write_png(out_file_name, width, height, 3, color_map.data(), width * 3) == 0) {
        std::cerr << "Failed to save depth map" << std::endl;
    }

    // Free device memory
    cudaFree(d_left_img);
    cudaFree(d_right_img);
    cudaFree(d_cost);
    cudaFree(d_aggr_cost);
    cudaFree(d_disparity_map);

    // Free host memory
    stbi_image_free(left_img);
    stbi_image_free(right_img);

    return 0;
}