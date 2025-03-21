#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vec_add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 10;
    float a[n], b[n], c[n];
    float *d_a, *d_b, *d_c;

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate GPU memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel (1 block, 256 threads per block)
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vec_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    // Copy result back to CPU
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("A: ");
    for (int i = 0; i < n; i++) printf("%.2f ", a[i]);
    printf("\nB: ");
    for (int i = 0; i < n; i++) printf("%.2f ", b[i]);
    printf("\nC: ");
    for (int i = 0; i < n; i++) printf("%.2f ", c[i]);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}