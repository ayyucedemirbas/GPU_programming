#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise 3D tensor multiplication
__global__ void tensorMultiply3D(const float* A, const float* B, float* C, 
                                  int depth, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = z * (height * width) + y * width + x;
        C[idx] = A[idx] * B[idx];
    }
}

int main() {
    // Tensor dimensions
    int depth = 64;
    int height = 128;
    int width = 128;
    int size = depth * height * width;
    int bytes = size * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize tensors with sample data
    for (int i = 0; i < size; i++) {
        h_A[i] = (float)i / size;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Configure grid and block dimensions
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  (depth + blockSize.z - 1) / blockSize.z);
    
    // Launch kernel
    tensorMultiply3D<<<gridSize, blockSize>>>(d_A, d_B, d_C, depth, height, width);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results (print first few elements)
    printf("First 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
