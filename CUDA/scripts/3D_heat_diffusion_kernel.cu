#include <iostream>
#include <cuda_runtime.h>

// 3D heat diffusion kernel using a grid of thread block clusters (3D grid)
// Each cluster along the z-axis processes one z-slice of the volume.
__global__ void heatDiffusion3D(const float* input, float* output,
                                int width, int height, int depth) {
    // Compute 3D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // cluster index (z slice)
    int z = blockIdx.z;

    int idx = (z * height + y) * width + x;

    if (x >= 1 && x < width-1 && y >= 1 && y < height-1 && z >= 1 && z < depth-1) {
        // 6-point stencil average
        float center = input[idx];
        float xp = input[idx + 1];
        float xm = input[idx - 1];
        float yp = input[idx + width];
        float ym = input[idx - width];
        float zp = input[idx + width*height];
        float zm = input[idx - width*height];
        output[idx] = (center + xp + xm + yp + ym + zp + zm) / 7.0f;
    }
}

int main() {
    // Volume dimensions
    const int width = 128;
    const int height = 128;
    const int depth = 64;
    const size_t size = width * height * depth * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input volume
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (z * height + y) * width + x;
                h_input[idx] = static_cast<float>(idx % 100) / 100.0f;
            }
        }
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define a 3D grid and block dimensions
    dim3 blockDim(16, 16, 1);                   // 256 threads per block
    dim3 gridDim((width+blockDim.x-1)/blockDim.x,
                 (height+blockDim.y-1)/blockDim.y,
                 depth);                        // depth clusters along z-axis

    // Launch kernel
    heatDiffusion3D<<<gridDim, blockDim>>>(d_input, d_output, width, height, depth);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Simple verification: print center voxel
    int cx = width/2, cy = height/2, cz = depth/2;
    int cidx = (cz * height + cy) * width + cx;
    std::cout << "Center voxel after diffusion: " << h_output[cidx] << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
