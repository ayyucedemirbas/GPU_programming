import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

# CUDA kernel code
kernel_code = """
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
"""

# Compile the kernel
mod = SourceModule(kernel_code)
tensor_multiply = mod.get_function("tensorMultiply3D")

# Tensor dimensions
depth, height, width = 64, 128, 128

# Create input tensors
A = np.random.rand(depth, height, width).astype(np.float32)
B = np.random.rand(depth, height, width).astype(np.float32)
C = np.zeros((depth, height, width), dtype=np.float32)

# Define block and grid sizes
block_size = (8, 8, 8)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1],
    (depth + block_size[2] - 1) // block_size[2]
)

# Launch kernel
tensor_multiply(
    drv.In(A),
    drv.In(B),
    drv.Out(C),
    np.int32(depth),
    np.int32(height),
    np.int32(width),
    block=block_size,
    grid=grid_size
)

# Verify results
expected = A * B
print(f"Results match: {np.allclose(C, expected)}")
print(f"Max error: {np.max(np.abs(C - expected))}")
print(f"\nFirst 5 results:")
print(f"C[0,0,:5] = {C[0,0,:5]}")
