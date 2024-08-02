# Stereo Vision

### Why Cuda needs?

1. Efficiency in Parallel Processing
Depth map generation involves complex calculations for numerous pixels.
CUDA leverages the parallel processing capabilities of GPUs, allowing these calculations 
to be performed simultaneously.

2. High Performance Computing
Depth map generation involves complex calculations for numerous pixels.
CUDA leverages the parallel processing capabilities of GPUs, allowing these calculations 
to be performed simultaneously.


------------------------------


### Algorithms

1. Compute Cost

![image](https://github.com/user-attachments/assets/f4b4773a-ef89-4f97-b4cc-4375de7d23b6)

2. Aggregate Costs

![image](https://github.com/user-attachments/assets/a4b122e6-ff4c-4de3-8204-be5c4a094539)

3. Compute Disparity Map

![image](https://github.com/user-attachments/assets/f4c1ff8e-4875-416e-87f9-8d828a761e88)

4. Apply Color Map

![image](https://github.com/user-attachments/assets/94896e24-92ea-4c15-8efd-ee49f4a9d1ca)


----------------------------------


### Cuda Architecture

- Compute Cost Kernel
  - Blocks((image_width + 15) / 15, (image_height + 5) / 15)
  - Threads(16, 16)
- Aggregate Cost Horizontal Kernel
  - Blocks(1, image_height + 15 / 15)
  - Threads(16, 16)
- Aggregate Cost Vertical Kernel
  - Blocks(image_width + 15 / 15, 1)
  - Threads(16, 16)
- Compute Disparity Map Kernel
  - Blocks(image_width + 15) / 15, (image_height + 15), 15)
  - Threads(16, 16)


### Results

| Left Image | Rigth Image |
|:--:|:--:|
| ![left img](https://github.com/user-attachments/assets/16663ed6-fa86-48db-a0bc-52cf9e749e9f) | ![right img](https://github.com/user-attachments/assets/a01018bc-426c-4b34-a8e3-bb2a6351c4fe) |

| C programming, Cuda C, Cuda C with Shared Memory | CV2 StereoSGBM |
|:--:|:--:|
| ![normal](https://github.com/user-attachments/assets/bf5d80d9-e1c5-471f-8d02-8fb0165c5820) | ![cv2](https://github.com/user-attachments/assets/6ca17e10-db04-4d1f-b53e-fc4bb4843801) |

### Performance Comparison

| C programming | Cuda without Shared Memory | Cuda with Shared Memory | CV2 Library |
| :--: | :--: | :--: | :--: |
| 3352.9 ms | 511.8 ms | 25.0 ms | 857.5 ms |
