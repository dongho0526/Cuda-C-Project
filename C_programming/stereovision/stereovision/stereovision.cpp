
//C++

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cmath>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include "Timer.h"


const int P1 = 10;
const int P2 = 120;

int penalty(int d1, int d2) {
    return (d1 == d2) ? 0 : ((abs(d1 - d2) == 1) ? P1 : P2);
}

void compute_cost(const uint8_t* left_img, const uint8_t* right_img, int width, int height, int channels, std::vector<int>& cost, int disparity_range) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
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
}

void aggregate_costs_horizontal(const std::vector<int>& cost, std::vector<int>& aggr_cost, int width, int height, int disparity_range) {
    for (int y = 0; y < height; ++y) {
        for (int x = 1; x < width; ++x) {
            for (int d = 0; d < disparity_range; ++d) {
                int min_cost = cost[((y * width + (x - 1)) * disparity_range) + d];
                for (int pd = 0; pd < disparity_range; ++pd) {
                    int p = cost[((y * width + (x - 1)) * disparity_range) + pd] + penalty(d, pd);
                    if (p < min_cost) {
                        min_cost = p;
                    }
                }
                aggr_cost[(y * width + x) * disparity_range + d] += min_cost;
            }
        }
    }
}

void aggregate_costs_vertical(const std::vector<int>& cost, std::vector<int>& aggr_cost, int width, int height, int disparity_range) {
    for (int x = 0; x < width; ++x) {
        for (int y = 1; y < height; ++y) {
            for (int d = 0; d < disparity_range; ++d) {
                int min_cost = cost[((y - 1) * width + x) * disparity_range + d];
                for (int pd = 0; pd < disparity_range; ++pd) {
                    int p = cost[((y - 1) * width + x) * disparity_range + pd] + penalty(d, pd);
                    if (p < min_cost) {
                        min_cost = p;
                    }
                }
                aggr_cost[(y * width + x) * disparity_range + d] += min_cost;
            }
        }
    }
}

void compute_disparity_map(const std::vector<int>& aggr_cost, std::vector<uint8_t>& disparity_map, int width, int height, int disparity_range) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
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
    //char* out_file_name = "output_img.png";

    int width, height, channels;
    float dCpuTime;
    CPerfCounter counter;

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

    std::vector<int> cost(cost_size, 0);
    std::vector<int> aggr_cost(cost_size, 0);
    std::vector<uint8_t> disparity_map(img_size, 0);
    std::vector<uint8_t> color_map(img_size * 3, 0);

    dCpuTime = 0.0f;
    counter.Reset();
    counter.Start();

    // Compute cost
    compute_cost(left_img, right_img, width, height, channels, cost, disparity_range);

    // Initialize aggregated cost to zero
    std::fill(aggr_cost.begin(), aggr_cost.end(), 0);

    // Aggregate costs horizontally
    aggregate_costs_horizontal(cost, aggr_cost, width, height, disparity_range);

    // Aggregate costs vertically
    aggregate_costs_vertical(cost, aggr_cost, width, height, disparity_range);

    // Compute disparity map
    compute_disparity_map(aggr_cost, disparity_map, width, height, disparity_range);

    // Apply color map
    apply_color_map(disparity_map, color_map, width, height);

    counter.Stop();
    dCpuTime += counter.GetElapsedTime();

    printf("stereovision Depth Map C Performance(ms) = %f \n", dCpuTime * 1000);

    // Save output image
    if (stbi_write_png("output_img.png", width, height, 3, color_map.data(), width * 3) == 0) {
        std::cerr << "Failed to save depth map" << std::endl;
    }

    // Free host memory
    stbi_image_free(left_img);
    stbi_image_free(right_img);

    return 0;
}