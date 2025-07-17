import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDA kernel as a string
kernel_code = """
__global__ void heatDiffusion3D(const float* input, float* output,
                                int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    int idx = (z * height + y) * width + x;

    if (x >= 1 && x < width-1 && y >= 1 && y < height-1 && z >= 1 && z < depth-1) {
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
"""

# Compile kernel
mod = SourceModule(kernel_code)
heat_diffusion = mod.get_function("heatDiffusion3D")

width, height, depth = 128, 128, 64

# Create host arrays
h_input = np.zeros((depth, height, width), dtype=np.float32)
h_output = np.zeros_like(h_input)

# Initialize input
for z in range(depth):
    h_input[z] = (np.arange(height*width) % 100).reshape(height, width) / 100.0

# Allocate device memory
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

# Copy to device
cuda.memcpy_htod(d_input, h_input)

# Define block and grid dimensions
block = (16, 16, 1)
grid = ( (width + block[0] - 1) // block[0],
         (height + block[1] - 1) // block[1],
         depth )

# Launch kernel
heat_diffusion(d_input, d_output,
               np.int32(width), np.int32(height), np.int32(depth),
               block=block, grid=grid)

# Copy back to host
cuda.memcpy_dtoh(h_output, d_output)

# Simple verification: print center voxel
cx, cy, cz = width//2, height//2, depth//2
print(f"Center voxel after diffusion: {h_output[cz, cy, cx]}")
