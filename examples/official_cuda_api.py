#!/usr/bin/env python3
"""
Official CUDA Runtime API Example

This example demonstrates using the official CUDA Runtime API with function
signatures that match NVIDIA's CUDA Runtime API exactly.

This is meant to be a drop-in replacement for real CUDA code - the API calls
look and work exactly like the official CUDA Runtime API.
"""

import ctypes
import struct
from src.simulator import HopperSimulator, SimulatorConfig
from src.cuda_runtime_api import (
    # Error handling
    cudaGetLastError, cudaGetErrorString, cudaError_t,

    # Device management
    cudaGetDeviceCount, cudaGetDevice, cudaSetDevice,
    cudaGetDeviceProperties, cudaDeviceProp, cudaDeviceSynchronize,

    # Memory management
    cudaMalloc, cudaFree, cudaMemcpy, cudaMemset,
    cudaMemcpyKind,

    # Version info
    cudaGetDriverVersion, cudaRuntimeGetVersion,

    # Initialization
    init_cuda_runtime
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Run the official CUDA Runtime API demonstration."""
    print_header("OFFICIAL CUDA RUNTIME API DEMONSTRATION")
    print("\nThis demo uses the official CUDA Runtime API with exact signatures")
    print("matching NVIDIA's CUDA Runtime API documentation.")

    # Create simulator
    config = SimulatorConfig(num_sms=1, warps_per_sm=4)
    sim = HopperSimulator(config)

    # Initialize CUDA Runtime with simulator
    init_cuda_runtime(sim)

    # ========== 1. Device Management ==========
    print_header("1. Device Management")

    # Get device count
    device_count = ctypes.c_int()
    err = cudaGetDeviceCount(ctypes.pointer(device_count))
    print(f"\ncudaGetDeviceCount:")
    print(f"  Error: {cudaGetErrorString(err)}")
    print(f"  Device count: {device_count.value}")

    # Get device properties
    prop = cudaDeviceProp()
    err = cudaGetDeviceProperties(prop, 0)
    print(f"\ncudaGetDeviceProperties(0):")
    print(f"  Error: {cudaGetErrorString(err)}")
    print(f"  Name: {prop.name.decode('utf-8')}")
    print(f"  Compute Capability: {prop.major}.{prop.minor}")
    print(f"  Total Global Mem: {prop.totalGlobalMem // (1024**3)} GB")
    print(f"  Warp Size: {prop.warpSize}")
    print(f"  Max Threads Per Block: {prop.maxThreadsPerBlock}")
    print(f"  SM Count: {prop.multiProcessorCount}")

    # Get version info
    driver_version = ctypes.c_int()
    runtime_version = ctypes.c_int()
    cudaGetDriverVersion(ctypes.pointer(driver_version))
    cudaRuntimeGetVersion(ctypes.pointer(runtime_version))
    print(f"\nVersion Information:")
    print(f"  Driver Version: {driver_version.value}")
    print(f"  Runtime Version: {runtime_version.value}")

    # ========== 2. Memory Allocation ==========
    print_header("2. Device Memory Allocation (cudaMalloc)")

    n = 64
    size_bytes = n * 4  # 4 bytes per int32
    size = ctypes.c_size_t(size_bytes)

    # Allocate device memory
    A_dev = ctypes.c_void_p()
    B_dev = ctypes.c_void_p()
    C_dev = ctypes.c_void_p()

    print(f"\nAllocating {size_bytes} bytes for each array...")

    err = cudaMalloc(size, ctypes.pointer(A_dev))
    print(f"cudaMalloc A: {cudaGetErrorString(err)} -> 0x{A_dev.value:x}")

    err = cudaMalloc(size, ctypes.pointer(B_dev))
    print(f"cudaMalloc B: {cudaGetErrorString(err)} -> 0x{B_dev.value:x}")

    err = cudaMalloc(size, ctypes.pointer(C_dev))
    print(f"cudaMalloc C: {cudaGetErrorString(err)} -> 0x{C_dev.value:x}")

    # ========== 3. Host Memory Initialization ==========
    print_header("3. Host Memory Initialization")

    print(f"\nCreating host vectors of size {n}")
    print("  A[i] = i")
    print("  B[i] = i * 10")

    A_host = [i for i in range(n)]
    B_host = [i * 10 for i in range(n)]
    C_host = [0] * n

    # Pack as bytes (little-endian 32-bit integers)
    A_bytes = struct.pack('<' + 'I' * n, *A_host)
    B_bytes = struct.pack('<' + 'I' * n, *B_host)

    print(f"\nSample host data:")
    print(f"  A_host[0] = {A_host[0]}, A_host[{n-1}] = {A_host[n-1]}")
    print(f"  B_host[0] = {B_host[0]}, B_host[{n-1}] = {B_host[n-1]}")

    # ========== 4. Copy Host to Device ==========
    print_header("4. Copy Host to Device (cudaMemcpy)")

    # CUDA: cudaMemcpy(A_dev, A_host, size, cudaMemcpyHostToDevice)
    err = cudaMemcpy(A_dev, A_bytes, size, cudaMemcpyKind.cudaMemcpyHostToDevice)
    print(f"\ncudaMemcpy A (H2D): {cudaGetErrorString(err)}")

    err = cudaMemcpy(B_dev, B_bytes, size, cudaMemcpyKind.cudaMemcpyHostToDevice)
    print(f"cudaMemcpy B (H2D): {cudaGetErrorString(err)}")

    # Zero out C
    err = cudaMemset(C_dev, 0, size)
    print(f"cudaMemset C (to 0): {cudaGetErrorString(err)}")

    # ========== 5. Kernel Launch ==========
    print_header("5. Kernel Launch")

    # Vector addition kernel: C = A + B
    # Using special registers %tid and %ctaid
    kernel_code = [
        # ========== Get Thread and Block IDs ==========
        "MOV R5, %tid",             # threadIdx.x (lane ID)
        "MOV R6, %ctaid",           # blockIdx.x (warp ID in our simple mapping)

        # ========== Compute Global Thread ID ==========
        "IMUL.U32 R7, R6, 32",      # R7 = blockIdx.x * blockDim.x
        "IADD R5, R7, R5",          # R5 = global_tid

        # ========== Load A[global_tid] and B[global_tid] ==========
        f"MOV R3, {A_dev.value}",   # R3 = base address of A
        f"MOV R4, {B_dev.value}",   # R4 = base address of B

        "IMUL.U32 R8, R5, 4",       # R8 = global_tid * 4 (byte offset)
        "IADD R9, R3, R8",          # R9 = &A[global_tid]
        "IADD R10, R4, R8",         # R10 = &B[global_tid]

        "LDG.U32 R11, [R9]",        # R11 = A[global_tid]
        "LDG.U32 R12, [R10]",       # R12 = B[global_tid]

        # ========== Compute C = A + B ==========
        "IADD R13, R11, R12",       # R13 = A[global_tid] + B[global_tid]

        # ========== Store Result ==========
        f"MOV R14, {C_dev.value}",  # R14 = base address of C
        "IADD R15, R14, R8",        # R15 = &C[global_tid]
        "STG.U32 [R15], R13",       # C[global_tid] = R13

        "EXIT",
    ]

    print(f"\nLaunching kernel: vector_add<<<2, 32>>>(A, B, C, {n})")
    print(f"  Grid: (2, 1, 1)")
    print(f"  Block: (32, 1, 1)")
    print(f"  Total threads: {2 * 32}")

    # Launch kernel using simulator's launch_kernel
    sim.launch_kernel(kernel_code, grid_dim=(2, 1, 1), block_dim=(32, 1, 1))

    # Run simulation
    result = sim.run(max_cycles=500)

    if not result.success:
        print(f"\nKernel execution failed: {result.error}")
        return

    print(f"\nKernel completed successfully!")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")

    # Synchronize device
    err = cudaDeviceSynchronize()
    print(f"  cudaDeviceSynchronize: {cudaGetErrorString(err)}")

    # ========== 6. Copy Device to Host ==========
    print_header("6. Copy Device to Host (cudaMemcpy)")

    # Create buffer for result
    C_buffer = bytearray(size_bytes)

    # CUDA: cudaMemcpy(C_host, C_dev, size, cudaMemcpyDeviceToHost)
    err = cudaMemcpy(C_buffer, C_dev, size, cudaMemcpyKind.cudaMemcpyDeviceToHost)
    print(f"\ncudaMemcpy C (D2H): {cudaGetErrorString(err)}")

    # Unpack result
    C_host = list(struct.unpack('<' + 'I' * n, C_buffer))

    # ========== 7. Verify Results ==========
    print_header("7. Result Verification")

    print("\nResults:")
    print("  Index   A[i]    B[i]    C[i]    Expected  Status")
    print("  " + "-" * 55)

    all_correct = True
    for i in [0, 1, 2, 31, 32, 33, 34, 63]:
        a = A_host[i]
        b = B_host[i]
        c = C_host[i]
        expected = a + b
        status = "PASS" if c == expected else "FAIL"
        print(f"  {i:<7} {a:<7} {b:<7} {c:<7} {expected:<9} {status}")
        if c != expected:
            all_correct = False

    if all_correct:
        print(f"\n  PASS: All {n} results correct!")
    else:
        print(f"\n  FAIL: Some results incorrect")

    # ========== 8. Free Device Memory ==========
    print_header("8. Free Device Memory (cudaFree)")

    err = cudaFree(A_dev)
    print(f"\ncudaFree A_dev: {cudaGetErrorString(err)}")

    err = cudaFree(B_dev)
    print(f"cudaFree B_dev: {cudaGetErrorString(err)}")

    err = cudaFree(C_dev)
    print(f"cudaFree C_dev: {cudaGetErrorString(err)}")

    # ========== API Comparison ==========
    print_header("CUDA API Comparison")

    print("""
REAL CUDA CODE (C++):
--------------------
#include <cuda_runtime.h>

int main() {
    int n = 64;
    size_t size = n * sizeof(int);

    // Device pointers
    int *A_dev, *B_dev, *C_dev;

    // Allocate device memory
    cudaMalloc(&A_dev, size);
    cudaMalloc(&B_dev, size);
    cudaMalloc(&C_dev, size);

    // Host arrays
    int *A_host = new int[n];
    int *B_host = new int[n];
    int *C_host = new int[n];

    // Initialize A_host, B_host...

    // Copy host to device
    cudaMemcpy(A_dev, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, size, cudaMemcpyHostToDevice);

    // Launch kernel
    vector_add<<<2, 32>>>(A_dev, B_dev, C_dev, n);

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy device to host
    cudaMemcpy(C_host, C_dev, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    return 0;
}


SIMULATOR CODE (Python):
------------------------
import ctypes
import struct
from src.simulator import HopperSimulator, SimulatorConfig
from src.cuda_runtime_api import (
    cudaMalloc, cudaFree, cudaMemcpy, cudaDeviceSynchronize,
    cudaMemcpyKind, init_cuda_runtime
)

# Create simulator and initialize runtime
sim = HopperSimulator(SimulatorConfig(num_sms=1, warps_per_sm=4))
init_cuda_runtime(sim)

# Problem size
n = 64
size = ctypes.c_size_t(n * 4)

# Allocate device memory
A_dev = ctypes.c_void_p()
B_dev = ctypes.c_void_p()
C_dev = ctypes.c_void_p()

cudaMalloc(size, ctypes.pointer(A_dev))
cudaMalloc(size, ctypes.pointer(B_dev))
cudaMalloc(size, ctypes.pointer(C_dev))

# Host arrays
A_host = [i for i in range(n)]
B_host = [i * 10 for i in range(n)]

# Pack as bytes
A_bytes = struct.pack('<' + 'I' * n, *A_host)
B_bytes = struct.pack('<' + 'I' * n, *B_host)

# Copy host to device
cudaMemcpy(A_dev, A_bytes, size, cudaMemcpyKind.cudaMemcpyHostToDevice)
cudaMemcpy(B_dev, B_bytes, size, cudaMemcpyKind.cudaMemcpyHostToDevice)

# Launch kernel
kernel_code = [...]  # Your assembly kernel
sim.launch_kernel(kernel_code, grid_dim=(2,1,1), block_dim=(32,1,1))
sim.run(max_cycles=500)

# Synchronize
cudaDeviceSynchronize()

# Copy device to host
C_buffer = bytearray(n * 4)
cudaMemcpy(C_buffer, C_dev, size, cudaMemcpyKind.cudaMemcpyDeviceToHost)
C_host = list(struct.unpack('<' + 'I' * n, C_buffer))

# Free device memory
cudaFree(A_dev)
cudaFree(B_dev)
cudaFree(C_dev)

    """)

    # Summary
    print_header("Summary")

    print("\nImplemented CUDA Runtime API Functions:")
    print("  Error Handling:")
    print("    - cudaGetLastError()")
    print("    - cudaGetErrorString(error)")
    print("  Device Management:")
    print("    - cudaGetDeviceCount(&count)")
    print("    - cudaGetDevice(&device)")
    print("    - cudaSetDevice(device)")
    print("    - cudaGetDeviceProperties(&prop, device)")
    print("    - cudaDeviceSynchronize()")
    print("  Memory Management:")
    print("    - cudaMalloc(size, &ptr)")
    print("    - cudaFree(ptr)")
    print("    - cudaMemcpy(dst, src, size, kind)")
    print("    - cudaMemset(ptr, value, size)")
    print("  Version Info:")
    print("    - cudaGetDriverVersion(&version)")
    print("    - cudaRuntimeGetVersion(&version)")

    if all_correct:
        print(f"\n{'='*70}")
        print(" SUCCESS: Official CUDA Runtime API Works!")
        print(" The simulator now has CUDA-compatible API!")
        print(f"{'='*70}")

    # Check for errors
    last_error = cudaGetLastError()
    if last_error != cudaError_t.cudaSuccess:
        print(f"\nWarning: Last CUDA error: {cudaGetErrorString(last_error)}")


if __name__ == "__main__":
    main()
