import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDA kernel: each thread adds one element
kernel_code = """
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)
vector_add = mod.get_function("vectorAdd")

# Problem size
define = 1 << 20  # 1M elements
N = 1 << 20

# Host arrays
h_A = np.arange(N, dtype=np.float32)
h_B = 2 * np.arange(N, dtype=np.float32)
h_C = np.empty_like(h_A)

# Allocate device memory
d_A = cuda.mem_alloc(h_A.nbytes)
d_B = cuda.mem_alloc(h_B.nbytes)
d_C = cuda.mem_alloc(h_C.nbytes)

# Copy inputs to device
cuda.memcpy_htod(d_A, h_A)
cuda.memcpy_htod(d_B, h_B)

# Launch parameters
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# Launch kernel
vector_add(
    d_A, d_B, d_C, np.int32(N),
    block=(threads_per_block, 1, 1),
    grid=(blocks_per_grid, 1, 1)
)

# Copy result back to host
cuda.memcpy_dtoh(h_C, d_C)

# Verify results
for i in range(5):
    expected = h_A[i] + h_B[i]
    if abs(h_C[i] - expected) > 1e-5:
        print("FAIL at index", i)
        break
else:
    print("PASS")
