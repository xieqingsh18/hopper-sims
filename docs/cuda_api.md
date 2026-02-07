# CUDA-Style API - Complete Implementation

## Overview

The Hopper GPU Simulator now supports a complete CUDA-style programming API with memory management and kernel argument passing, making it work just like real CUDA programming.

## API Functions

### 1. Device Memory Management

#### cudaMalloc
```python
ptr = cuda.cudaMalloc(size_in_bytes)
```
Allocates device memory and returns a DevicePointer.

**CUDA Equivalent:**
```cuda
cudaMalloc(&ptr, size);
```

**Example:**
```python
# Allocate 256 bytes for 64 integers
A_dev = cuda.cudaMalloc(256)
```

#### cudaMemcpyHtoD
```python
cuda.cudaMemcpyHtoD(ptr, host_data_bytes)
```
Copies data from host to device memory.

**CUDA Equivalent:**
```cuda
cudaMemcpy(ptr, host_data, size, cudaMemcpyHostToDevice);
```

**Example:**
```python
import struct
data = struct.pack('<' + 'I' * n, *host_array)
cuda.cudaMemcpyHtoD(dev_ptr, data)
```

#### cudaMemcpyDtoH
```python
host_data = cuda.cudaMemcpyDtoH(ptr, size=None)
```
Copies data from device to host memory.

**CUDA Equivalent:**
```cuda
cudaMemcpy(host_data, ptr, size, cudaMemcpyDeviceToHost);
```

**Example:**
```python
data = cuda.cudaMemcpyDtoH(C_dev, n * 4)
C_host = list(struct.unpack('<' + 'I' * n, data))
```

#### cudaFree
```python
cuda.cudaFree(ptr)
```
Frees device memory.

**CUDA Equivalent:**
```cuda
cudaFree(ptr);
```

### 2. Kernel Launch

#### launch_kernel
```python
launch_kernel(sim, kernel_func, grid_dim, block_dim, *args)
```
Launches a kernel with CUDA-style grid/block configuration.

**CUDA Equivalent:**
```cuda
kernel_func<<<grid_dim, block_dim>>>(args...);
```

**Parameters:**
- `sim`: HopperSimulator instance
- `kernel_func`: KernelFunction instance
- `grid_dim`: (grid_x, grid_y, grid_z) tuple
- `block_dim`: (block_x, block_y, block_z) tuple
- `*args`: Kernel arguments (DevicePointers, integers, etc.)

**Example:**
```python
# CUDA: vector_add<<<2, 32>>>(A_dev, B_dev, C_dev, n)
launch_kernel(sim, vector_add, (2, 1, 1), (32, 1, 1),
             A_dev, B_dev, C_dev, n)
```

### 3. Pre-built Kernels

#### vector_add_kernel
```python
kernel = vector_add_kernel()
```
Creates a vector addition kernel that computes C = A + B element-wise.

## Complete Example

### Real CUDA Code
```cuda
#include <cuda_runtime.h>

__global__ void vector_add(int *A, int *B, int *C, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    if (global_tid < n) {
        C[global_tid] = A[global_tid] + B[global_tid];
    }
}

int main() {
    int n = 64;

    // Host memory
    int *A_host = new int[n];
    int *B_host = new int[n];
    int *C_host = new int[n];

    // Initialize arrays
    for (int i = 0; i < n; i++) {
        A_host[i] = i;
        B_host[i] = i * 10;
    }

    // Device memory
    int *A_dev, *B_dev, *C_dev;
    cudaMalloc(&A_dev, n * sizeof(int));
    cudaMalloc(&B_dev, n * sizeof(int));
    cudaMalloc(&C_dev, n * sizeof(int));

    // Copy H2D
    cudaMemcpy(A_dev, A_host, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    vector_add<<<2, 32>>>(A_dev, B_dev, C_dev, n);

    // Copy D2H
    cudaMemcpy(C_host, C_dev, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < n; i++) {
        assert(C_host[i] == A_host[i] + B_host[i]);
    }

    // Free
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    delete[] A_host;
    delete[] B_host;
    delete[] C_host;

    return 0;
}
```

### Simulator Code
```python
import struct
from src.simulator import HopperSimulator, SimulatorConfig
from src.cuda_runtime import (
    cudaMalloc, cudaMemcpyHtoD, cudaMemcpyDtoH,
    launch_kernel, vector_add_kernel, CUDARuntime
)

def main():
    # Create simulator and runtime
    config = SimulatorConfig(num_sms=1, warps_per_sm=4)
    sim = HopperSimulator(config)
    cuda = CUDARuntime(sim)

    # Problem size
    n = 64

    # ========== Host Memory ==========
    A_host = [i for i in range(n)]
    B_host = [i * 10 for i in range(n)]
    C_host = [0] * n

    # ========== Device Memory Allocation ==========
    A_dev = cuda.cudaMalloc(n * 4)
    B_dev = cuda.cudaMalloc(n * 4)
    C_dev = cuda.cudaMalloc(n * 4)

    # ========== Copy H2D ==========
    A_bytes = struct.pack('<' + 'I' * n, *A_host)
    B_bytes = struct.pack('<' + 'I' * n, *B_host)
    cuda.cudaMemcpyHtoD(A_dev, A_bytes)
    cuda.cudaMemcpyHtoD(B_dev, B_bytes)

    # ========== Launch Kernel ==========
    vector_add = vector_add_kernel()
    launch_kernel(sim, vector_add, (2, 1, 1), (32, 1, 1),
                 A_dev, B_dev, C_dev, n)

    # ========== Run Simulation ==========
    result = sim.run(max_cycles=500)

    # ========== Copy D2H ==========
    C_bytes = cuda.cudaMemcpyDtoH(C_dev, n * 4)
    C_host = list(struct.unpack('<' + 'I' * n, C_bytes))

    # ========== Verify Results ==========
    for i in range(n):
        expected = A_host[i] + B_host[i]
        assert C_host[i] == expected, f"Mismatch at {i}"

    print("All results correct!")

    # ========== Free Memory ==========
    cuda.cudaFree(A_dev)
    cuda.cudaFree(B_dev)
    cuda.cudaFree(C_dev)

if __name__ == "__main__":
    main()
```

## API Mapping Table

| CUDA | Simulator | Notes |
|------|------------|-------|
| `cudaMalloc(&ptr, size)` | `cuda.cudaMalloc(size)` | Returns DevicePointer |
| `cudaMemcpy(dst, src, size, dir)` | `cuda.cudaMemcpyHtoD(ptr, data)` | Separate H2D/DtoH functions |
| `cudaFree(ptr)` | `cuda.cudaFree(ptr)` | Same |
| `kernel<<<grid, block>>>(args)` | `launch_kernel(sim, kernel, grid, block, args)` | Explicit sim parameter |
| Special registers (`%tid`, etc.) | Fully supported | Hardware-provided |

## Key Features

### 1. DevicePointer Class
Represents device memory with:
- `address`: Device memory address
- `size`: Allocation size
- Convertible to `int` for direct address use

### 2. Automatic Special Register Initialization
When `launch_kernel()` is called:
- `%tid` (threadIdx) is set based on lane position
- `%ctaid` (blockIdx) is set based on warp/block mapping
- `%ntid` (blockDim) is set to block_dim.x
- `%nctaid` (gridDim) is set to grid_dim.x

### 3. Argument Passing
Kernel arguments are automatically bound to the assembly code:
- DevicePointers → converted to device addresses
- Integers → directly embedded as immediates
- Supported types: DevicePointer, int

### 4. Memory Tracking
The runtime tracks:
- All allocations (address → size mapping)
- Host data for each allocation
- Automatic address space management

## Example Output

```
[cudaMalloc] Allocated 256 bytes at 0x10000000
[cudaMalloc] Allocated 256 bytes at 0x10000100
[cudaMalloc] Allocated 256 bytes at 0x10000200

[cudaMemcpy H->D] 256 bytes to 0x10000000
[cudaMemcpy H->D] 256 bytes to 0x10000100

[Kernel Launch] vector_add<<<(2, 1, 1), (32, 1, 1)>>>()
  Arguments: (DevicePointer(...), DevicePointer(...), DevicePointer(...), 64)

✓ Kernel completed!
  Cycles: 18
  Instructions: 34

[cudaMemcpy D->H] 256 bytes from 0x10000200

  ✓ All 64 results correct!

[cudaFree] Freed memory at 0x10000000
[cudaFree] Freed memory at 0x10000100
[cudaFree] Freed memory at 0x10000200
```

## Creating Custom Kernels

To create a custom kernel:

```python
from src.cuda_runtime import KernelFunction

class MyKernel(KernelFunction):
    def __init__(self):
        super().__init__("my_kernel")

    def generate_code(self, arg1, arg2, n) -> List[str]:
        arg1_addr = int(arg1)
        arg2_addr = int(arg2)

        return [
            "MOV R5, %tid",
            "MOV R6, %ctaid",
            # ... your kernel code ...
            "EXIT",
        ]

# Use it
kernel = MyKernel()
launch_kernel(sim, kernel, (2,1,1), (32,1,1), A_dev, B_dev, n)
```

## Benefits

1. **Familiar Syntax**: Works like real CUDA programming
2. **Type Safety**: DevicePointer prevents accidental address mix-ups
3. **Memory Safety**: Tracked allocations prevent use-after-free
4. **Clear Separation**: Host vs Device memory is explicit
5. **Debugging**: Memory operations are logged for inspection
6. **Extensible**: Easy to add new kernels and operations

## Files Modified

1. **src/cuda_runtime.py** (NEW)
   - `CUDARuntime` class
   - `DevicePointer` class
   - `KernelFunction` base class
   - `vector_add_kernel()` function
   - Helper functions

2. **examples/cuda_style_api.py** (NEW)
   - Complete demonstration of CUDA-style API

## Future Enhancements

- Async memory copy support
- Multi-dimensional grid/block (y, z dimensions)
- Multiple streams
- Event synchronization
- Shared memory allocation
- Texture memory support
- More pre-built kernels (matmul, reduction, scan, etc.)
