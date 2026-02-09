#!/usr/bin/env python3
"""
Pointwise Kernel with Special Registers - CUDA-Style Launch

This example demonstrates the CUDA-style kernel launch with special registers
(%tid, %ctaid, etc.), making the simulator work much more like real CUDA.

Kernel: C = A + B (element-wise vector addition)

This demonstrates:
1. CUDA-style kernel launch: launch_kernel<<<grid, block>>>(kernel)
2. Special registers: %tid, %ctaid, %ntid, %nctaid
3. Automatic thread ID calculation by hardware
4. Per-thread result logging

Compare this to pointwise_kernel.py which manually sets thread IDs.
"""

from src.simulator import HopperSimulator, SimulatorConfig
from src.core.memory import MemorySpace


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def initialize_vector_data(sim: HopperSimulator, n: int):
    """Initialize vectors A and B in global memory."""
    print_header("Vector Initialization")

    A_addr = 0x100000
    B_addr = 0x200000
    C_addr = 0x300000

    print(f"Vector size: {n} elements")
    print(f"A @ 0x{A_addr:x}, B @ 0x{B_addr:x}, C @ 0x{C_addr:x}")

    # Initialize A[i] = i
    for i in range(n):
        sim.memory.write_u32(MemorySpace.GLOBAL, A_addr + i * 4, i)

    # Initialize B[i] = i * 10
    for i in range(n):
        sim.memory.write_u32(MemorySpace.GLOBAL, B_addr + i * 4, i * 10)

    # Initialize C to zeros
    for i in range(n):
        sim.memory.write_u32(MemorySpace.GLOBAL, C_addr + i * 4, 0)

    print(f"\nSample values:")
    print(f"  A[0] = {sim.memory.read_u32(MemorySpace.GLOBAL, A_addr)}")
    print(f"  A[{n-1}] = {sim.memory.read_u32(MemorySpace.GLOBAL, A_addr + (n-1)*4)}")
    print(f"  B[0] = {sim.memory.read_u32(MemorySpace.GLOBAL, B_addr)}")
    print(f"  B[{n-1}] = {sim.memory.read_u32(MemorySpace.GLOBAL, B_addr + (n-1)*4)}")

    return A_addr, B_addr, C_addr


def get_pointwise_kernel(A_addr: int, B_addr: int, C_addr: int) -> list:
    """
    Generate pointwise addition kernel using special registers.

    This looks much more like real PTX code!
    Each thread computes: C[global_tid] = A[global_tid] + B[global_tid]

    Args:
        A_addr: Base address of vector A
        B_addr: Base address of vector B
        C_addr: Base address of result vector C

    Returns:
        List of assembly instructions using %tid and %ctaid special registers
    """
    kernel = [
        # ========== Get Thread ID and Block ID from Special Registers ==========
        "MOV R5, %tid",            # R5 = threadIdx.x (0-31 within block)
        "MOV R6, %ctaid",          # R6 = blockIdx.x (block ID)

        # ========== Compute Global Thread ID ==========
        # global_tid = blockIdx.x * blockDim.x + threadIdx.x
        # For our launch: blockDim.x = 32
        "IMUL.U32 R7, R6, 32",     # R7 = blockIdx.x * 32
        "IADD R5, R7, R5",         # R5 = global_tid (this is our element index)

        # ========== Load A[global_tid] and B[global_tid] ==========
        f"MOV R1, {A_addr}",        # R1 = base address of A
        f"MOV R2, {B_addr}",        # R2 = base address of B

        # Compute element address: base + global_tid * 4
        "IMUL.U32 R3, R5, 4",       # R3 = global_tid * 4
        "IADD R4, R1, R3",          # R4 = &A[global_tid]
        "IADD R8, R2, R3",          # R8 = &B[global_tid]

        # Load values
        "LDG.U32 R9, [R4]",         # R9 = A[global_tid]
        "LDG.U32 R10, [R8]",        # R10 = B[global_tid]

        # ========== Compute C = A + B ==========
        "IADD R11, R9, R10",        # R11 = A[global_tid] + B[global_tid]

        # ========== Store Result ==========
        f"MOV R12, {C_addr}",       # R12 = base address of C
        "IADD R13, R12, R3",        # R13 = &C[global_tid]
        "STG.U32 [R13], R11",       # C[global_tid] = R11

        # ========== Store values for logging ==========
        "MOV R20, R5",              # R20 = global_tid (for logging)
        "MOV R21, R9",              # R21 = A value
        "MOV R22, R10",             # R22 = B value
        "MOV R23, R11",             # R23 = C value

        "EXIT",
    ]

    return [line.strip() for line in kernel]


def run_pointwise_kernel_with_special_regs():
    """
    Run pointwise addition kernel using CUDA-style launch with special registers.
    """
    print_header("POINTWISE KERNEL WITH SPECIAL REGISTERS")
    print("\nThis demo demonstrates CUDA-style kernel launching:")
    print("  - Use special registers: %tid, %ctaid, %ntid, %nctaid")
    print("  - CUDA launch API: launch_kernel(grid_dim, block_dim, kernel)")
    print("  - Automatic thread ID initialization by hardware")

    # Create simulator with enough warps
    config = SimulatorConfig(num_sms=1, warps_per_sm=4)
    sim = HopperSimulator(config)

    # Vector size (64 elements)
    n = 64
    A_addr, B_addr, C_addr = initialize_vector_data(sim, n)

    # Generate kernel using special registers
    kernel = get_pointwise_kernel(A_addr, B_addr, C_addr)

    print_header("Kernel Assembly (Using Special Registers)")
    for i, line in enumerate(kernel[:15]):
        print(f"  {i:02d}: {line}")
    print(f"  ... ({len(kernel)} total instructions)")

    print_header("CUDA-Style Kernel Launch")
    print("\nLaunch configuration: kernel<<<(2,1,1), (32,1,1)>>>(...)")
    print("  Grid: 2 blocks (blockIdx.x = 0, 1)")
    print("  Block: 32 threads each (threadIdx.x = 0-31)")
    print("  Total: 64 threads")

    # Launch kernel using CUDA-style API
    # grid_dim=(2,1,1) means 2 blocks in x-dimension
    # block_dim=(32,1,1) means 32 threads per block in x-dimension
    sim.launch_kernel(
        program=kernel,
        grid_dim=(2, 1, 1),    # 2 blocks
        block_dim=(32, 1, 1)   # 32 threads per block
    )

    print_header("Special Register Values")
    print("\nAfter kernel launch, special registers are set by hardware:")
    print("\n  Block 0 (warps 0):")
    print("    Thread 0: %tid=0,  %ctaid=0,  %ntid=32,  %nctaid=2")
    print("    Thread 1: %tid=1,  %ctaid=0,  %ntid=32,  %nctaid=2")
    print("    Thread 31: %tid=31, %ctaid=0,  %ntid=32,  %nctaid=2")
    print("\n  Block 1 (warp 1):")
    print("    Thread 0: %tid=0,  %ctaid=1,  %ntid=32,  %nctaid=2")
    print("    Thread 1: %tid=1,  %ctaid=1,  %ntid=32,  %nctaid=2")
    print("    Thread 31: %tid=31, %ctaid=1,  %ntid=32,  %nctaid=2")

    print_header("Kernel Execution")
    print("\nExecuting kernel on all warps...")

    # Run the simulation
    result = sim.run(max_cycles=500)

    if result.success:
        print(f"\n✓ Kernel completed successfully!")
        print(f"  Cycles: {result.cycles}")
        print(f"  Instructions: {result.instructions_executed}")
        print(f"  IPC: {result.instructions_executed / max(result.cycles, 1):.2f}")

        print_header("Thread Results Logging")
        print("\nLogging results from all threads (showing per-thread register state):")

        # Verify results
        all_correct = True

        # Check warp 0 (block 0, threads 0-31)
        print("\n  Block 0, Warp 0 (threadIdx.x = 0-31):")
        print("  " + "-" * 65)
        print(f"  {'Lane':<6} {'%tid':<8} {'%ctaid':<8} {'A':<8} {'B':<8} {'C':<8} {'Expected':<10}")
        print("  " + "-" * 65)

        for lane_id in [0, 1, 2, 3, 4, 5, 6, 7, 28, 29, 30, 31]:
            tid = sim.read_register(0, lane_id, 20)       # R20 = %tid
            ctaid = sim.warps[0].get_thread(lane_id).ctaid  # Read special register
            a_val = sim.read_register(0, lane_id, 21)
            b_val = sim.read_register(0, lane_id, 22)
            c_val = sim.read_register(0, lane_id, 23)
            expected = tid + tid * 10

            status = "✓" if c_val == expected else "✗"
            print(f"  {lane_id:<6} {tid:<8} {ctaid:<8} {a_val:<8} {b_val:<8} {c_val:<8} {expected:<10} {status}")
            if c_val != expected:
                all_correct = False

        # Check warp 1 (block 1, threads 32-63, but within warp they're threadIdx 0-31)
        print("\n  Block 1, Warp 1 (threadIdx.x = 0-31, global tid = 32-63):")
        print("  " + "-" * 65)
        print(f"  {'Lane':<6} {'%tid':<8} {'%ctaid':<8} {'A':<8} {'B':<8} {'C':<8} {'Expected':<10}")
        print("  " + "-" * 65)

        for lane_id in [0, 1, 2, 3, 4, 5, 6, 7, 28, 29, 30, 31]:
            local_tid = sim.read_register(1, lane_id, 20)     # R20 = global_tid
            ctaid = sim.warps[1].get_thread(lane_id).ctaid  # Read special register
            a_val = sim.read_register(1, lane_id, 21)
            b_val = sim.read_register(1, lane_id, 22)
            c_val = sim.read_register(1, lane_id, 23)
            expected = local_tid + local_tid * 10

            status = "✓" if c_val == expected else "✗"
            print(f"  {lane_id:<6} {local_tid:<8} {ctaid:<8} {a_val:<8} {b_val:<8} {c_val:<8} {expected:<10} {status}")
            if c_val != expected:
                all_correct = False

        print_header("Global Memory Verification")
        print("\nVerifying results stored in global memory:")

        for i in range(n):
            expected = i + i * 10
            actual = sim.memory.read_u32(MemorySpace.GLOBAL, C_addr + i * 4)
            if actual != expected:
                print(f"  ✗ C[{i}] = {actual} (expected {expected})")
                all_correct = False

        if all_correct:
            print(f"  ✓ All {n} values in global memory are correct!")
            print(f"\n  Sample values:")
            for i in [0, 1, 31, 32, 33, 63]:
                val = sim.memory.read_u32(MemorySpace.GLOBAL, C_addr + i * 4)
                print(f"    C[{i}] = {val}")

        print_header("Comparison: Real CUDA vs Simulator")
        print("""
REAL CUDA KERNEL:
--------------
__global__ void vector_add(int *A, int *B, int *C, int n) {
    int tid = threadIdx.x;              // %tid special register
    int bid = blockIdx.x;               // %ctaid special register
    int global_tid = bid * blockDim.x + tid;

    if (global_tid < n) {
        C[global_tid] = A[global_tid] + B[global_tid];
    }
}

// Launch:
vector_add<<<2, 32>>>(A, B, C, 64);


SIMULATOR KERNEL:
---------------
# Assembly using special registers
MOV R5, %tid             # Read thread ID from special register
MOV R6, %ctaid           # Read block ID from special register
IMUL.U32 R7, R6, 32     # Compute block offset
IADD R5, R7, R5         # global_tid = blockIdx.x * 32 + threadIdx.x
MOV R1, A_addr
...
LDG.U32 R9, [R4]         # Load A[global_tid]
LDG.U32 R10, [R8]        # Load B[global_tid]
IADD R11, R9, R10        # C = A + B
STG.U32 [R13], R11       # Store result

# Launch:
sim.launch_kernel(
    kernel,
    grid_dim=(2, 1, 1),
    block_dim=(32, 1, 1)
)

KEY DIFFERENCES:
- Real CUDA: Hardware automatically sets %tid, %ctaid based on launch config
- Simulator: Same! Special registers are initialized by launch_kernel()
- Real CUDA: Compiler generates PTX with special register references
- Simulator: You write assembly with special register references (%tid, etc.)
        """)

        if all_correct:
            print(f"\n  ✓✓✓ CUDA-Style kernel launch works! ✓✓✓")
            print(f"  ✓ Special registers (%tid, %ctaid, etc.) working correctly")
            print(f"  ✓ Automatic thread ID initialization demonstrated")
    else:
        print(f"\n✗ Kernel execution failed: {result.error}")


def main():
    """Run the pointwise kernel with special registers demonstration."""
    run_pointwise_kernel_with_special_regs()


if __name__ == "__main__":
    main()
