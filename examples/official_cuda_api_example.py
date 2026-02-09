#!/usr/bin/env python3
"""
Official CUDA Runtime API Example

This example demonstrates how to use the CUDA Runtime API that matches
the official NVIDIA CUDA Runtime API interface. It shows the same workflow you
would use in real CUDA programming.
"""

import struct
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import HopperSimulator, SimulatorConfig
from src.cuda_runtime_api import *


def vector_add_kernel_code(A_addr, B_addr, C_addr, n):
    """
    Generate vector_add kernel code.

    Computes: C[i] = A[i] + B[i]
    """
    kernel = [
        # ========== Get Thread and Block IDs ==========
        "MOV R5, %tid",             # threadIdx.x
        "MOV R6, %ctaid",           # blockIdx.x

        # ========== Compute Global Thread ID ==========
        "IMUL.U32 R7, R6, 32",      # R7 = blockIdx.x * blockDim.x
        "IADD R5, R7, R5",          # R5 = global_tid

        # ========== Load A[global_tid] and B[global_tid] ==========
        f"MOV R3, {A_addr}",         # R3 = base address of A
        f"MOV R4, {B_addr}",         # R4 = base address of B

        "IMUL.U32 R8, R5, 4",       # R8 = global_tid * 4 (byte offset)
        "IADD R9, R3, R8",          # R9 = &A[global_tid]
        "IADD R10, R4, R8",         # R10 = &B[global_tid]

        "LDG.U32 R11, [R9]",         # R11 = A[global_tid]
        "LDG.U32 R12, [R10]",        # R12 = B[global_tid]

        # ========== Compute C = A + B ==========
        "IADD R13, R11, R12",        # R13 = A[global_tid] + B[global_tid]

        # ========== Store Result ==========
        f"MOV R14, {C_addr}",        # R14 = base address of C
        "IADD R15, R14, R8",         # R15 = &C[global_tid]
        "STG.U32 [R15], R13",       # C[global_tid] = R13

        "EXIT",
    ]

    return [line.strip() for line in kernel]


class MutableInt:
    """Mutable integer for simulating ctypes pointer behavior."""
    def __init__(self, value=0):
        self.value = value

    def __repr__(self):
        return f"MutableInt({self.value})"


def print_separator(title):
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    print("\n" + "="*60)
    print("  Official CUDA Runtime API Example")
    print("="*60)

    # ========== 1. Create Simulator and Initialize CUDA Runtime ==========
    print_separator("1. Creating Simulator and Initializing CUDA Runtime")

    config = SimulatorConfig(num_sms=1, warps_per_sm=4)
    sim = HopperSimulator(config)
    runtime = init_cuda_runtime(sim)

    print(f"✓ Simulator created: {config.num_sms} SM, {config.warps_per_sm} warps/SM")
    print(f"✓ CUDA Runtime initialized")

    # ========== 2. Query Device Information ==========
    print_separator("2. Querying Device Information")

    # Get device count
    device_count = MutableInt()
    err = cudaGetDeviceCount(device_count)
    print(f"cudaGetDeviceCount: {device_count.value} device(s)")
    print(f"  Return: {cudaGetErrorString(err)}")

    # Get current device
    current_device = MutableInt()
    err = cudaGetDevice(current_device)
    print(f"cudaGetDevice: Device {current_device.value}")
    print(f"  Return: {cudaGetErrorString(err)}")

    # Get device properties
    prop = cudaDeviceProp()
    err = cudaGetDeviceProperties(prop, 0)
    print(f"\ncudaGetDeviceProperties (Device 0):")
    print(f"  Name: {prop.name.decode()}")
    print(f"  Compute Capability: {prop.major}.{prop.minor}")
    print(f"  Total Global Mem: {prop.totalGlobalMem // (1024**3)} GB")
    print(f"  Shared Mem/Block: {prop.sharedMemPerBlock // 1024} KB")
    print(f"  Warp Size: {prop.warpSize}")
    print(f"  Max Threads/Block: {prop.maxThreadsPerBlock}")
    print(f"  SM Count: {prop.multiProcessorCount}")
    print(f"  L2 Cache: {prop.l2CacheSize // (1024**2)} MB")
    print(f"  Return: {cudaGetErrorString(err)}")

    # Get version info
    driver_version = MutableInt()
    runtime_version = MutableInt()
    cudaGetDriverVersion(driver_version)
    cudaRuntimeGetVersion(runtime_version)
    print(f"\nCUDA Driver Version: {driver_version.value // 1000}.{driver_version.value % 1000 // 10}")
    print(f"CUDA Runtime Version: {runtime_version.value // 1000}.{runtime_version.value % 1000 // 10}")

    # ========== 3. Allocate Host Memory ==========
    print_separator("3. Allocating Host Memory")

    n = 64
    print(f"Array size: {n} integers ({n * 4} bytes)")

    # Initialize host arrays
    A_host = [i for i in range(n)]
    B_host = [i * 10 for i in range(n)]
    C_host = [0] * n

    print(f"A: {A_host[:8]}...{A_host[-4:]}")
    print(f"B: {B_host[:8]}...{B_host[-4:]}")

    # ========== 4. Allocate Device Memory ==========
    print_separator("4. Allocating Device Memory (cudaMalloc)")

    size = n * 4  # 4 bytes per int

    d_A = MutableInt()
    d_B = MutableInt()
    d_C = MutableInt()

    err = cudaMalloc(size, d_A)
    print(f"cudaMalloc(d_A, {size} bytes)")
    print(f"  Device pointer: 0x{d_A.value:x}")
    print(f"  Return: {cudaGetErrorString(err)}")

    err = cudaMalloc(size, d_B)
    print(f"cudaMalloc(d_B, {size} bytes)")
    print(f"  Device pointer: 0x{d_B.value:x}")
    print(f"  Return: {cudaGetErrorString(err)}")

    err = cudaMalloc(size, d_C)
    print(f"cudaMalloc(d_C, {size} bytes)")
    print(f"  Device pointer: 0x{d_C.value:x}")
    print(f"  Return: {cudaGetErrorString(err)}")

    # ========== 5. Copy Host to Device ==========
    print_separator("5. Copying Host to Device (cudaMemcpy H2D)")

    # Pack host data as bytes
    A_bytes = struct.pack('<' + 'I' * n, *A_host)
    B_bytes = struct.pack('<' + 'I' * n, *B_host)

    err = cudaMemcpy(d_A.value, A_bytes, size, cudaMemcpyKind.cudaMemcpyHostToDevice)
    print(f"cudaMemcpy(d_A, A_host, {size}, cudaMemcpyHostToDevice)")
    print(f"  Return: {cudaGetErrorString(err)}")

    err = cudaMemcpy(d_B.value, B_bytes, size, cudaMemcpyKind.cudaMemcpyHostToDevice)
    print(f"cudaMemcpy(d_B, B_host, {size}, cudaMemcpyHostToDevice)")
    print(f"  Return: {cudaGetErrorString(err)}")

    # ========== 6. Launch Kernel ==========
    print_separator("6. Launching Kernel (cudaLaunchKernel)")

    # Generate kernel code with device addresses
    kernel_code = vector_add_kernel_code(d_A.value, d_B.value, d_C.value, n)

    # Launch kernel (simulated - in real CUDA this would be the kernel function)
    grid_dim = (2, 1, 1)  # 2 blocks
    block_dim = (32, 1, 1)  # 32 threads per block

    print(f"Kernel: vector_add<<<{grid_dim}, {block_dim}>>>(d_A, d_B, d_C, {n})")

    # Load and launch kernel in simulator
    sim.load_program(kernel_code)
    sim.launch_kernel(kernel_code, grid_dim=grid_dim, block_dim=block_dim)

    # Run simulation
    print("\nRunning simulation...")
    result = sim.run(max_cycles=500)

    status = "✓ PASSED" if result.success else "✗ FAILED"
    print(f"  {status}")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")

    # ========== 7. Copy Device to Host ==========
    print_separator("7. Copying Device to Host (cudaMemcpy D2H)")

    # Allocate buffer for result
    C_buffer = bytearray(n * 4)

    err = cudaMemcpy(C_buffer, d_C.value, size, cudaMemcpyKind.cudaMemcpyDeviceToHost)
    print(f"cudaMemcpy(C_host, d_C, {size}, cudaMemcpyDeviceToHost)")
    print(f"  Return: {cudaGetErrorString(err)}")

    # Unpack result
    C_result = list(struct.unpack('<' + 'I' * n, C_buffer))
    print(f"C: {C_result[:8]}...{C_result[-4:]}")

    # ========== 8. Verify Results ==========
    print_separator("8. Verifying Results")

    all_correct = True
    for i in range(n):
        expected = A_host[i] + B_host[i]
        if C_result[i] != expected:
            print(f"✗ Mismatch at [{i}]: {A_host[i]} + {B_host[i]} = {expected}, got {C_result[i]}")
            all_correct = False

    if all_correct:
        print(f"✓ All {n} results correct!")
        for i in range(min(8, n)):
            print(f"  C[{i}] = A[{i}] + B[{i}] = {A_host[i]} + {B_host[i]} = {C_result[i]}")

    # ========== 9. Free Device Memory ==========
    print_separator("9. Freeing Device Memory (cudaFree)")

    err = cudaFree(d_A.value)
    print(f"cudaFree(d_A) -> 0x{d_A.value:x}")
    print(f"  Return: {cudaGetErrorString(err)}")

    err = cudaFree(d_B.value)
    print(f"cudaFree(d_B) -> 0x{d_B.value:x}")
    print(f"  Return: {cudaGetErrorString(err)}")

    err = cudaFree(d_C.value)
    print(f"cudaFree(d_C) -> 0x{d_C.value:x}")
    print(f"  Return: {cudaGetErrorString(err)}")

    # ========== 10. Check for Errors ==========
    print_separator("10. Checking for Errors")

    err = cudaGetLastError()
    print(f"cudaGetLastError: {cudaGetErrorString(err)}")

    print("\n" + "="*60)
    print("  Example Complete!")
    print("="*60)

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
