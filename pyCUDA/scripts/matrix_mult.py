import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray, compiler

# Define the CUDA kernel (vector addition)
kernel_code = """
__global__ void vec_add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# Compile the kernel
mod = compiler.SourceModule(kernel_code)
vec_add = mod.get_function("vec_add")

# Input data
n = 10
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

# Allocate GPU memory and copy data
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.empty_like(a_gpu)

# Launch kernel (1 block, 256 threads per block)
block_size = 256
grid_size = (n + block_size - 1) // block_size
vec_add(a_gpu, b_gpu, c_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

# Copy result back to CPU and verify
c = c_gpu.get()
print("Input A:", a)
print("Input B:", b)
print("Output C:", c)
