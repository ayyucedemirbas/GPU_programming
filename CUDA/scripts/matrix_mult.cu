#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 5 

#define CHECK_CUDA_ERROR(call) {                                         \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                                         \
    }                                                                    \
}

// CUDA kernel for matrix multiplication
__global__ void matrix_mult(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

int main() {
    float *A, *B, *C;      // Host matrices
    float *d_A, *d_B, *d_C; // Device matrices

    // Allocate host memory
    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    // Initialize input matrices (A = identity matrix, B = identity matrix)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 1.0f : 0.0f;
            B[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Print input matrices
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", B[i * N + j]);
        }
        printf("\n");
    }

    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, N * N * sizeof(float)));

    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 block_size(16, 16);    // Threads per block
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                  (N + block_size.y - 1) / block_size.y);

    // Launch kernel
    matrix_mult<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result matrix
    printf("\nMatrix C (Result):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            if (fabs(C[i * N + j] - expected) > 1e-5) {
                correct = false;
                break;
            }
        }
    }

    if (correct) {
        printf("\nMatrix multiplication succeeded!\n");
    } else {
        printf("\nMatrix multiplication failed!\n");
    }

    // Cleanup
    free(A);
    free(B);
    free(C);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return 0;
}
