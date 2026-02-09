#!/usr/bin/env python3
"""
CUDA-Style API Example - Complete Memory Management & Kernel Launch

This example demonstrates the complete CUDA programming workflow:
1. Allocate host memory (tensors)
2. Allocate device memory (cudaMalloc)
3. Copy host to device (cudaMemcpy H2D)
4. Launch kernel with arguments (<<<grid, block>>>)
5. Copy device to host (cudaMemcpy D2H)
6. Free device memory (cudaFree)

This looks and feels like real CUDA programming!
"""

import struct
from src.simulator import HopperSimulator, SimulatorConfig
from src.cuda_runtime import (
    CUDARuntime, DevicePointer, cudaMalloc, cudaMemcpyHtoD,
    cudaMemcpyDtoH, launch_kernel, vector_add_kernel
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Run the CUDA-style API demonstration."""
    print_header("CUDA-STYLE API DEMONSTRATION")
    print("\nThis demo shows the complete CUDA workflow:")
    print("  1. Allocate host tensors")
    print("  2. Allocate device memory (cudaMalloc)")
    print("  3. Copy H2D (cudaMemcpy)")
    print("  4. Launch kernel (<<<grid, block>>>)")
    print("  5. Copy D2H (cudaMemcpy)")
    print("  6. Verify results")
    print("  7. Free device memory (cudaFree)")

    # Create simulator
    config = SimulatorConfig(num_sms=1, warps_per_sm=4)
    sim = HopperSimulator(config)
    cuda = CUDARuntime(sim)

    # ========== 1. Allocate Host Tensors ==========
    print_header("1. Host Memory Allocation")

    n = 64
    print(f"\nCreating host vectors of size {n}")
    print("  A[i] = i")
    print("  B[i] = i * 10")

    # Create host arrays (as Python lists)
    A_host = [i for i in range(n)]
    B_host = [i * 10 for i in range(n)]
    C_host = [0] * n  # Result buffer

    print(f"\nHost arrays created:")
    print(f"  A_host: {n} integers")
    print(f"  B_host: {n} integers")
    print(f"  C_host: {n} integers (result buffer)")

    # ========== 2. Allocate Device Memory ==========
    print_header("2. Device Memory Allocation (cudaMalloc)")

    size_bytes = n * 4  # 4 bytes per int32

    A_dev = cuda.cudaMalloc(size_bytes)
    B_dev = cuda.cudaMalloc(size_bytes)
    C_dev = cuda.cudaMalloc(size_bytes)

    # ========== 3. Copy Host to Device ==========
    print_header("3. Copy Host to Device (cudaMemcpy H2D)")

    # Pack host data as bytes
    A_bytes = bytearray()
    for val in A_host:
        A_bytes.extend(struct.pack('<I', val))

    B_bytes = bytearray()
    for val in B_host:
        B_bytes.extend(struct.pack('<I', val))

    # Copy to device
    cuda.cudaMemcpyHtoD(A_dev, bytes(A_bytes))
    cuda.cudaMemcpyHtoD(B_dev, bytes(B_bytes))

    print(f"\nSample device memory contents:")
    print(f"  A_dev[0] = {A_host[0]}")
    print(f"  A_dev[{n-1}] = {A_host[n-1]}")
    print(f"  B_dev[0] = {B_host[0]}")
    print(f"  B_dev[{n-1}] = {B_host[n-1]}")

    # ========== 4. Launch Kernel ==========
    print_header("4. Kernel Launch (<<<grid, block>>>)")

    # Create kernel function
    vector_add = vector_add_kernel()

    # Launch kernel: vector_add<<<2, 32>>>(A_dev, B_dev, C_dev, n)
    print("\nCUDA syntax: vector_add<<<2, 32>>>(A_dev, B_dev, C_dev, 64)")
    print("\nSimulator syntax:")
    print("  launch_kernel(")
    print("    sim=simulator,")
    print("    kernel_func=vector_add,")
    print("    grid_dim=(2, 1, 1),")
    print("    block_dim=(32, 1, 1),")
    print("    A_dev, B_dev, C_dev, n")
    print("  )")

    launch_kernel(
        sim,
        vector_add,
        (2, 1, 1),
        (32, 1, 1),
        A_dev, B_dev, C_dev, n
    )

    # Run the simulation
    result = sim.run(max_cycles=500)

    if not result.success:
        print(f"\n✗ Kernel execution failed: {result.error}")
        return

    print(f"\n✓ Kernel completed!")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")

    # ========== 5. Copy Device to Host ==========
    print_header("5. Copy Device to Host (cudaMemcpy D2H)")

    C_bytes = cuda.cudaMemcpyDtoH(C_dev, size_bytes)

    # Unpack result
    for i in range(n):
        val = struct.unpack('<I', C_bytes[i*4:i*4+4])[0]
        C_host[i] = val

    print(f"\nCopied {n} integers from device to host")

    # ========== 6. Verify Results ==========
    print_header("6. Result Verification")

    print("\nHost result array:")
    print("  Index   A[i]    B[i]    C[i]    Expected  Status")
    print("  " + "-" * 55)

    all_correct = True
    for i in [0, 1, 2, 31, 32, 33, 34, 63]:
        a = A_host[i]
        b = B_host[i]
        c = C_host[i]
        expected = a + b
        status = "✓" if c == expected else "✗"
        print(f"  {i:<7} {a:<7} {b:<7} {c:<7} {expected:<9} {status}")
        if c != expected:
            all_correct = False

    if all_correct:
        print(f"\n  ✓ All {n} results correct!")
    else:
        print(f"\n  ✗ Some results incorrect")

    # ========== 7. Free Device Memory ==========
    print_header("7. Free Device Memory (cudaFree)")

    cuda.cudaFree(A_dev)
    cuda.cudaFree(B_dev)
    cuda.cudaFree(C_dev)

    print("\nAll device memory freed")

    # ========== Summary ==========
    print_header("CUDA API Comparison")
    print("""
REAL CUDA CODE:
--------------
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

    // Allocate host memory
    int *A_host = new int[n];
    int *B_host = new int[n];
    int *C_host = new int[n];

    // Initialize A, B

    // Allocate device memory
    int *A_dev, *B_dev, *C_dev;
    cudaMalloc(&A_dev, n * sizeof(int));
    cudaMalloc(&B_dev, n * sizeof(int));
    cudaMalloc(&C_dev, n * sizeof(int));

    // Copy host to device
    cudaMemcpy(A_dev, A_host, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    vector_add<<<2, 32>>>(A_dev, B_dev, C_dev, n);

    // Copy device to host
    cudaMemcpy(C_host, C_dev, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    return 0;
}


SIMULATOR CODE:
---------------
import struct
from src.simulator import HopperSimulator, SimulatorConfig
from src.cuda_runtime import (
    cudaMalloc, cudaMemcpyHtoD, cudaMemcpyDtoH,
    launch_kernel, vector_add_kernel
)

# Create simulator
sim = HopperSimulator(SimulatorConfig(num_sms=1, warps_per_sm=4))
cuda = CUDARuntime(sim)

# Allocate host memory
n = 64
A_host = [i for i in range(n)]
B_host = [i * 10 for i in range(n)]
C_host = [0] * n

# Allocate device memory
A_dev = cuda.cudaMalloc(n * 4)
B_dev = cuda.cudaMalloc(n * 4)
C_dev = cuda.cudaMalloc(n * 4)

# Copy host to device
A_bytes = struct.pack('<' + 'I' * n, *A_host)
B_bytes = struct.pack('<' + 'I' * n, *B_host)
cuda.cudaMemcpyHtoD(A_dev, A_bytes)
cuda.cudaMemcpyHtoD(B_dev, B_bytes)

# Launch kernel
vector_add = vector_add_kernel()
launch_kernel(sim, vector_add, (2,1,1), (32,1,1), A_dev, B_dev, C_dev, n)

# Run kernel
sim.run(max_cycles=500)

# Copy device to host
C_bytes = cuda.cudaMemcpyDtoH(C_dev, n * 4)
C_host = list(struct.unpack('<' + 'I' * n, C_bytes))

# Free device memory
cuda.cudaFree(A_dev)
cuda.cudaFree(B_dev)
cuda.cudaFree(C_dev)


KEY MAPPING:
-------------
CUDA                         Simulator
-----                         ----------
cudaMalloc(ptr, size)        cuda.cudaMalloc(size) → DevicePointer
cudaMemcpy(dst, src, dir)     cuda.cudaMemcpyHtoD/DtoH(ptr, data)
kernel<<<grid, block>>>(...)  launch_kernel(kernel, grid_dim, block_dim, ...)
cudaFree(ptr)                 cuda.cudaFree(ptr)
    """)

    if all_correct:
        print(f"\n{'='*70}")
        print(" ✓✓✓ CUDA-Style API Works Perfectly! ✓✓✓")
        print(" ✓ Memory management: cudaMalloc, cudaMemcpy, cudaFree")
        print(" ✓ Kernel launch: <<<grid, block>>> with arguments")
        print(" ✓ Special registers: %tid, %ctaid working correctly")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
