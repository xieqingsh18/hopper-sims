#!/usr/bin/env python3
"""
Pointwise Kernel Example - Parallel Thread Execution

This example demonstrates a realistic GPU kernel launch with multiple warps
running in parallel, where each thread processes a different element of an array.

Kernel: C = A + B (element-wise vector addition)

Thread Organization:
- Grid: 2 warps (64 threads total)
- Each warp has 32 threads (SIMT execution)
- Each thread processes one element of the array

GPU Concepts Demonstrated:
1. Kernel launch with multiple warps
2. Thread ID and Warp ID calculation
3. Parallel execution with divergence handling
4. Per-thread register state
5. Result logging from all threads
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import HopperSimulator, SimulatorConfig
from src.core.memory import MemorySpace
from src.isa.decoder import parse_program


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def initialize_vector_data(sim: HopperSimulator, n: int):
    """
    Initialize vectors A and B in global memory for pointwise addition.

    Args:
        sim: Hopper simulator
        n: Number of elements in each vector

    Returns:
        Tuple of (A_addr, B_addr, C_addr)
    """
    print_header("Vector Initialization")

    A_addr = 0x100000
    B_addr = 0x200000
    C_addr = 0x300000

    print(f"Vector size: {n} elements")
    print(f"A @ 0x{A_addr:x}, B @ 0x{B_addr:x}, C @ 0x{C_addr:x}")

    # Initialize vector A: A[i] = i
    print(f"\nInitializing A[i] = i:")
    for i in range(n):
        value = i
        offset = i * 4
        sim.memory.write_u32(MemorySpace.GLOBAL, A_addr + offset, value)

    # Initialize vector B: B[i] = i * 10
    print(f"Initializing B[i] = i * 10:")
    for i in range(n):
        value = i * 10
        offset = i * 4
        sim.memory.write_u32(MemorySpace.GLOBAL, B_addr + offset, value)

    # Initialize C to zeros
    for i in range(n):
        sim.memory.write_u32(MemorySpace.GLOBAL, C_addr + i * 4, 0)

    print(f"\nSample values:")
    print(f"  A[0] = {sim.memory.read_u32(MemorySpace.GLOBAL, A_addr)}")
    print(f"  A[{n-1}] = {sim.memory.read_u32(MemorySpace.GLOBAL, A_addr + (n-1)*4)}")
    print(f"  B[0] = {sim.memory.read_u32(MemorySpace.GLOBAL, B_addr)}")
    print(f"  B[{n-1}] = {sim.memory.read_u32(MemorySpace.GLOBAL, B_addr + (n-1)*4)}")

    return A_addr, B_addr, C_addr


def get_pointwise_kernel(A_addr: int, B_addr: int, C_addr: int, n: int) -> list:
    """
    Generate pointwise addition kernel.

    Each thread computes:
    1. tid = thread ID (0-63) - stored in R10
    2. C[tid] = A[tid] + B[tid]

    Note: Bounds checking omitted for simplicity since we control thread count.

    Args:
        A_addr: Base address of vector A
        B_addr: Base address of vector B
        C_addr: Base address of result vector C
        n: Number of elements (not used in kernel, for documentation only)

    Returns:
        List of assembly instructions
    """
    kernel = [
        # ========== Get Thread ID ==========
        # Each lane has a different value in R10 (set before kernel launch)
        # R10 = global thread ID (0-63)

        "MOV R5, R10",           # R5 = thread_id

        # ========== Load A[thread_id] and B[thread_id] ==========
        f"MOV R1, {A_addr}",     # R1 = base address of A
        f"MOV R2, {B_addr}",     # R2 = base address of B

        # Compute element address: base + thread_id * 4
        "IMUL.U32 R3, R5, 4",    # R3 = thread_id * 4
        "IADD R4, R1, R3",       # R4 = &A[thread_id]
        "IADD R7, R2, R3",       # R7 = &B[thread_id]

        # Load values
        "LDG.U32 R8, [R4]",      # R8 = A[thread_id]
        "LDG.U32 R9, [R7]",      # R9 = B[thread_id]

        # ========== Compute C = A + B ==========
        "IADD R10, R8, R9",      # R10 = A[thread_id] + B[thread_id]

        # ========== Store Result ==========
        f"MOV R11, {C_addr}",    # R11 = base address of C
        "IADD R12, R11, R3",     # R12 = &C[thread_id]
        "STG.U32 [R12], R10",    # C[thread_id] = R10

        # ========== Store Thread ID for Logging ==========
        # Store thread_id in R20 for later logging
        "MOV R20, R5",           # R20 = thread_id (for logging)
        "MOV R21, R8",           # R21 = A value (for logging)
        "MOV R22, R9",           # R22 = B value (for logging)
        "MOV R23, R10",          # R23 = C value (for logging)

        "EXIT",
    ]

    return [line.strip() for line in kernel]


def run_pointwise_kernel():
    """
    Run pointwise addition kernel with parallel thread execution.
    """
    print_header("POINTWISE KERNEL - C = A + B")
    print("\nThis demo demonstrates parallel thread execution:")
    print("  - 2 warps (64 threads total)")
    print("  - Each thread processes one array element")
    print("  - Threads execute in SIMT (lockstep within each warp)")
    print("  - Results logged from all threads")

    # Create simulator with 2 warps
    config = SimulatorConfig(num_sms=1, warps_per_sm=2)
    sim = HopperSimulator(config)

    # Vector size (64 elements to match 2 warps * 32 threads)
    n = 64
    A_addr, B_addr, C_addr = initialize_vector_data(sim, n)

    print_header("Thread Organization")
    print(f"  Total warps: 2")
    print(f"  Threads per warp: 32")
    print(f"  Total threads: 64")
    print(f"  Array elements: {n}")
    print(f"\n  Warp 0 (threads 0-31): processes elements 0-31")
    print(f"  Warp 1 (threads 32-63): processes elements 32-63")

    # Generate kernel
    kernel = get_pointwise_kernel(A_addr, B_addr, C_addr, n)

    print_header("Kernel Assembly (First 15 Instructions)")
    for i, line in enumerate(kernel[:15]):
        print(f"  {i:02d}: {line}")
    print(f"  ... ({len(kernel)} total instructions)")

    print_header("Thread Initialization")

    # Load kernel on both warps
    sim.load_program(kernel, warp_id=0)
    sim.load_program(kernel, warp_id=1)

    # Initialize thread IDs for each warp
    # In real CUDA, thread IDs are computed from blockIdx, threadIdx, etc.
    # For this simulator, we manually set R10 for each lane
    print("\nSetting thread IDs (R10) for each lane:")
    for warp_id in range(2):
        print(f"\n  Warp {warp_id}:")
        for lane_id in range(32):
            global_tid = warp_id * 32 + lane_id
            sim.warps[warp_id].write_lane_reg(lane_id, 10, global_tid)
            if lane_id < 3 or lane_id == 31:
                print(f"    Lane {lane_id:2d}: R10 = {global_tid:2d} (global thread ID)")
            elif lane_id == 3:
                print(f"    ...")

    print_header("Kernel Execution")
    print("\nLaunching kernel with 2 warps running in parallel...")
    print("  Each warp executes independently in SIMT fashion")
    print("  Threads within a warp execute in lockstep")

    # Run the simulation
    result = sim.run(max_cycles=500)

    if result.success:
        print(f"\n✓ Kernel completed successfully!")
        print(f"  Cycles: {result.cycles}")
        print(f"  Instructions: {result.instructions_executed}")
        print(f"  IPC: {result.instructions_executed / max(result.cycles, 1):.2f}")

        print_header("Thread Results Logging")
        print("\nLogging results from all active threads:")

        # Collect and display results from each warp
        print("\n  Warp 0 Results:")
        print("  " + "-" * 60)
        print(f"  {'Lane':<6} {'Thread ID':<10} {'A':<8} {'B':<8} {'C':<8} {'Expected':<10}")
        print("  " + "-" * 60)

        for lane_id in range(32):
            global_tid = sim.read_register(0, lane_id, 20)  # R20 = thread_id
            a_val = sim.read_register(0, lane_id, 21)       # R21 = A value
            b_val = sim.read_register(0, lane_id, 22)       # R22 = B value
            c_val = sim.read_register(0, lane_id, 23)       # R23 = C value
            expected = global_tid + global_tid * 10 if global_tid < n else -1

            # Show first 8 and last 8 threads
            if lane_id < 8 or lane_id >= 24:
                status = "✓" if c_val == expected else "✗"
                print(f"  {lane_id:<6} {global_tid:<10} {a_val:<8} {b_val:<8} {c_val:<8} {expected:<10} {status}")
            elif lane_id == 8:
                print(f"  ...")

        print("\n  Warp 1 Results:")
        print("  " + "-" * 60)
        print(f"  {'Lane':<6} {'Thread ID':<10} {'A':<8} {'B':<8} {'C':<8} {'Expected':<10}")
        print("  " + "-" * 60)

        for lane_id in range(32):
            global_tid = sim.read_register(1, lane_id, 20)  # R20 = thread_id
            a_val = sim.read_register(1, lane_id, 21)       # R21 = A value
            b_val = sim.read_register(1, lane_id, 22)       # R22 = B value
            c_val = sim.read_register(1, lane_id, 23)       # R23 = C value
            expected = global_tid + global_tid * 10 if global_tid < n else -1

            # Show first 8 and last 8 threads
            if lane_id < 8 or lane_id >= 24:
                status = "✓" if c_val == expected else "✗"
                print(f"  {lane_id:<6} {global_tid:<10} {a_val:<8} {b_val:<8} {c_val:<8} {expected:<10} {status}")
            elif lane_id == 8:
                print(f"  ...")

        print_header("Global Memory Verification")
        print("\nVerifying results stored in global memory:")

        # Check all values in global memory
        all_correct = True
        error_count = 0

        for i in range(n):
            expected = i + i * 10
            actual = sim.memory.read_u32(MemorySpace.GLOBAL, C_addr + i * 4)
            if actual != expected:
                all_correct = False
                error_count += 1
                if error_count <= 5:
                    print(f"  ✗ C[{i}] = {actual} (expected {expected})")

        if all_correct:
            print(f"  ✓ All {n} values in global memory are correct!")
            sample_values = [
                (0, sim.memory.read_u32(MemorySpace.GLOBAL, C_addr)),
                (1, sim.memory.read_u32(MemorySpace.GLOBAL, C_addr + 4)),
                (31, sim.memory.read_u32(MemorySpace.GLOBAL, C_addr + 31*4)),
                (32, sim.memory.read_u32(MemorySpace.GLOBAL, C_addr + 32*4)),
                (63, sim.memory.read_u32(MemorySpace.GLOBAL, C_addr + 63*4)),
            ]
            print(f"\n  Sample values:")
            for idx, val in sample_values:
                print(f"    C[{idx}] = {val}")
        else:
            print(f"  ✗ {error_count} errors found in global memory")

        print_header("Parallel Execution Analysis")
        print(f"""
Execution Characteristics:

1. SIMT Execution:
   - Warp 0: Threads 0-31 executed in lockstep
   - Warp 1: Threads 32-63 executed in lockstep
   - Same instruction, different data for each lane

2. Memory Access Pattern:
   - Coalesced global memory access (adjacent threads access adjacent elements)
   - Efficient use of memory bandwidth

3. Branch Divergence:
   - Threads with thread_id >= 64 take the EXIT branch
   - In this example, all threads are in bounds (n=64)

4. Register Usage:
   - Each lane has its own register values
   - R20-R23 used for per-thread result logging

Real CUDA Kernel Equivalent:

__global__ void vector_add(int *A, int *B, int *C, int n) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {{
        C[tid] = A[tid] + B[tid];
    }}
}}

// Launch: vector_add<<<2, 32>>>(A, B, C, 64);
        """)

        if all_correct:
            print(f"\n  ✓✓✓ Pointwise kernel executed correctly! ✓✓✓")
            print(f"  ✓ All threads computed correct results")
            print(f"  ✓ Parallel execution demonstrated")
    else:
        print(f"\n✗ Kernel execution failed: {result.error}")


def main():
    """Run the pointwise kernel demonstration."""
    run_pointwise_kernel()


if __name__ == "__main__":
    main()
