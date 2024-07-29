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

__constant__ int d_P1;
__constant__ int d_P2;

__device__ int penalty(int d1, int d2) {
    return (d1 == d2) ? 0 : ((abs(d1 - d2) == 1) ? d_P1 : d_P2);
}


__device__ int compute_sad(const uint8_t* left_img, const uint8_t* right_img, int x, int y, int d, int width, int channels, int height) {
    int sad = 0;
    for (int c = 0; c < channels; ++c) {
        int xl = (x * channels + y * width * channels + c);
        int xr = ((x - d) * channels + y * width * channels + c);

        if (x - d >= 0 && x - d < width) {
            sad += abs(left_img[xl] - right_img[xr]);
        }
        else {
            sad += 255;
        }
    }
    return sad;
}

__device__ void aggregate_costs_vertically(int* sh_cost, int thread_id, int thx, int disparity_range) {
    for (int d = 0; d < disparity_range; ++d) {
        int min_cost = sh_cost[thread_id * disparity_range + d];
        if (thx > 0) {
            for (int pd = 0; pd < disparity_range; ++pd) {
                int p = sh_cost[thread_id * disparity_range + pd] + penalty(d, pd);
                if (p < min_cost) {
                    min_cost = p;
                }
            }
        }
        sh_cost[thread_id * disparity_range + d] = min_cost;
    }
}

__device__ void aggregate_costs_horizontally(int* sh_cost, int thread_id, int thy, int disparity_range) {
    for (int d = 0; d < disparity_range; ++d) {
        int min_cost = sh_cost[thread_id * disparity_range + d];
        if (thy > 0) {
            for (int pd = 0; pd < disparity_range; ++pd) {
                int p = sh_cost[thread_id * disparity_range + pd] + penalty(d, pd);
                if (p < min_cost) {
                    min_cost = p;
                }
            }
        }
        sh_cost[thread_id * disparity_range + d] = min_cost;
    }
}

__device__ int find_best_disparity(int* sh_cost, int thread_id, int disparity_range) {
    int best_disparity = 0;
    int min_cost = sh_cost[thread_id * disparity_range + 0];
    for (int d = 1; d < disparity_range; ++d) {
        if (sh_cost[thread_id * disparity_range + d] < min_cost) {
            min_cost = sh_cost[thread_id * disparity_range + d];
            best_disparity = d;
        }
    }
    return best_disparity;
}


__global__ void stereo_vision_kernel(const uint8_t* left_img, const uint8_t* right_img, uint8_t* disparity_map, int width, int height, int channels, int disparity_range) {
    extern __shared__ int sharedM[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thx = threadIdx.x;
    int thy = threadIdx.y;
    int thread_id = thy * blockDim.x + thx;

    if (x < width && y < height) {
        // Load data into shared memory
        for (int d = 0; d < disparity_range; ++d) {
            int sad = compute_sad(left_img, right_img, x, y, d, width, channels, height);
            sharedM[thread_id * disparity_range + d] = sad;
        }
        __syncthreads();

        // Aggregate costs horizontally
        aggregate_costs_horizontally(sharedM, thread_id, thx, disparity_range);
        __syncthreads();

        // Aggregate costs vertically
        aggregate_costs_vertically(sharedM, thread_id, thy, disparity_range);
        __syncthreads();

        // Compute disparity map
        int best_disparity = find_best_disparity(sharedM, thread_id, disparity_range);
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

    std::vector<uint8_t> disparity_map(img_size, 0);
    std::vector<uint8_t> color_map(img_size * 3, 0);

    // Allocate host pinned memory
    uint8_t* h_left_img_pinned;
    uint8_t* h_right_img_pinned;
    cudaMallocHost(&h_left_img_pinned, width * height * channels * sizeof(uint8_t));
    cudaMallocHost(&h_right_img_pinned, width * height * channels * sizeof(uint8_t));

    memcpy(h_left_img_pinned, left_img, width * height * channels * sizeof(uint8_t));
    memcpy(h_right_img_pinned, right_img, width * height * channels * sizeof(uint8_t));

    // Allocate device memory
    uint8_t* d_left_img, * d_right_img, * d_disparity_map;

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
    }
    cudaStatus = cudaMalloc(&d_right_img, width * height * channels * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(&d_disparity_map, img_size * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy images to device
    cudaStatus = cudaMemcpy(d_left_img, h_left_img_pinned, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMemcpy(d_right_img, h_right_img_pinned, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Copy constants to constant memory
    cudaMemcpyToSymbol(d_P1, &P1, sizeof(int));
    cudaMemcpyToSymbol(d_P2, &P2, sizeof(int));

    // Compute cost, aggregate costs, and compute disparity map in one kernel
    size_t shared_memory_size = threadsPerBlock.x * threadsPerBlock.y * disparity_range * sizeof(int);
    stereo_vision_kernel << <numBlocks, threadsPerBlock, shared_memory_size >> > (d_left_img, d_right_img, d_disparity_map, width, height, channels, disparity_range);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching stereo_vision_kernel!\n", cudaStatus);
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
    cudaFree(d_disparity_map);

    // Free host pinned memory
    cudaFreeHost(h_left_img_pinned);
    cudaFreeHost(h_right_img_pinned);

    // Free host memory
    stbi_image_free(left_img);
    stbi_image_free(right_img);

    return 0;
}
