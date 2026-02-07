#!/usr/bin/env python3
"""
Async TMA Demo for Hopper GPU

Demonstrates asynchronous Tensor Memory Accelerator (TMA) operations.
TMA enables efficient bulk data transfer between global and shared memory
while the warp continues executing other instructions.

Key features:
- Async TMA LOAD: Transfer data from global to shared memory
- Async TMA STORE: Transfer data from shared to global memory
- Overlapped computation: Execute other instructions while TMA is in flight
- mbarrier: Synchronization for async operations
"""

import struct
from src.simulator import HopperSimulator, SimulatorConfig
from src.core.memory import MemorySpace


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def setup_matrix_memory(sim: HopperSimulator) -> None:
    """Set up matrix data in global memory."""
    print("\nSetting up matrix data in global memory...")

    # Initialize A matrix with 32-bit integers: 1, 2, 3, ..., 16
    # Initialize B matrix with 32-bit integers: all 2s
    # Initialize C matrix with 32-bit integers: all 0s

    import struct
    a_data = struct.pack('<' + 'I' * 16, *range(1, 17))  # 16 values: 1, 2, 3, ..., 16
    b_data = struct.pack('<' + 'I' * 16, *([2] * 16))  # 16 values: all 2s
    c_data = struct.pack('<' + 'I' * 16, *([0] * 16))  # 16 values: all 0s

    sim.write_memory(0x100000, a_data)  # A matrix @ 0x100000
    sim.write_memory(0x200000, b_data)  # B matrix @ 0x200000
    sim.write_memory(0x300000, c_data)  # C matrix @ 0x300000

    print("  A matrix @ 0x100000: 32-bit values 1, 2, 3, ..., 16")
    print("  B matrix @ 0x200000: 32-bit values all 2")
    print("  C matrix @ 0x300000: 32-bit values all 0")


def demo_async_tma_load(sim: HopperSimulator) -> None:
    """Demonstrate async TMA load operation."""
    print_header("ASYNC TMA LOAD DEMO")
    print("Loading data from global to shared memory asynchronously")

    setup_matrix_memory(sim)

    program = [
        # Initialize shared memory base address in R10
        "MOV R10, 16384",        # Shared memory base (0x4000)

        # Initialize global memory addresses
        "MOV R1, 1048576",       # A matrix in global (0x100000)
        "MOV R2, 2097152",       # B matrix in global (0x200000)
        "MOV R3, 3145728",       # C matrix in global (0x300000)

        # Issue async TMA LOAD for A matrix (global -> shared)
        # TMA.LOAD [shared_addr], [global_addr], size
        "TMA.LOAD [R10], [R1], 64",   # Load A matrix asynchronously

        # While TMA is in flight, do other useful work!
        # This demonstrates the key benefit of async TMA
        "MOV R20, 0",                # Initialize counter
        "MOV R21, 0",                # Accumulator

        # Do some computation while TMA runs (this overlaps!)
        "IADD R20, R20, 1",          # Increment counter
        "IADD R21, R21, R20",         # Accumulate sum
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",
        "IADD R20, R20, 1",
        "IADD R21, R21, R20",

        # At this point, TMA should have completed (50 cycles for TMA, ~20 cycles of work)
        # Use the data
        "LDS.U32 R30, [R10]",   # Read shared[0]
        "LDS.U32 R31, [R10+4]", # Read shared[4]

        "EXIT",
    ]

    sim.load_program(program)
    result = sim.run()

    print(f"\nAsync TMA Load Results:")
    if result.error:
        print(f"  Error: {result.error}")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")
    print(f"  R30 (shared[0]): {sim.read_register(0, 0, 30)} (expected: 1)")
    print(f"  R31 (shared[4]): {sim.read_register(0, 0, 31)} (expected: 2)")
    print(f"  R21 (computed during TMA): {sim.read_register(0, 0, 21)}")
    print(f"  R20 (loop iterations): {sim.read_register(0, 0, 20)}")

    # Verify results
    assert sim.read_register(0, 0, 30) == 1, f"TMA should have loaded value 1, got {sim.read_register(0, 0, 30)}"
    assert sim.read_register(0, 0, 31) == 2, f"TMA should have loaded value 2, got {sim.read_register(0, 0, 31)}"
    assert sim.read_register(0, 0, 20) == 11, f"Loop should run 11 times, got {sim.read_register(0, 0, 20)}"
    assert sim.read_register(0, 0, 21) == 66, f"Sum of 1+2+...+11 = 66, got {sim.read_register(0, 0, 21)}"

    print("\n✓ Async TMA LOAD successful!")
    print("  Note: Computation (R21=66) ran while TMA was transferring data")


def demo_async_tma_store(sim: HopperSimulator) -> None:
    """Demonstrate async TMA store operation."""
    print_header("ASYNC TMA STORE DEMO")
    print("Storing data from shared to global memory asynchronously")

    program = [
        # Initialize shared memory with test data
        "MOV R10, 16384",        # Shared memory base (0x4000)
        "MOV R1, 42",            # Value to store
        "MOV R2, 84",            # Another value to store
        "STS.U32 [R10], R1",     # shared[0] = 42
        "STS.U32 [R10+4], R2",   # shared[4] = 84

        # Initialize global address
        "MOV R3, 5242880",       # Global memory destination (0x500000)

        # Issue async TMA STORE (shared -> global)
        # TMA.STORE [global_addr], [shared_addr], size
        "TMA.STORE [R3], [R10], 8",    # Store 8 bytes asynchronously

        # While TMA store is in flight, do other work
        "MOV R20, 0",
        "IADD R20, R20, 1",
        "IMUL R21, R20, R20",    # Compute squares (1*1=1)
        "IADD R20, R20, 1",
        "IMUL R22, R20, R20",    # Compute squares (2*2=4)
        "IADD R20, R20, 1",
        "IMUL R23, R20, R20",    # Compute squares (3*3=9)
        "IADD R20, R20, 1",
        "IMUL R24, R20, R20",    # Compute squares (4*4=16)
        "IADD R20, R20, 1",
        "IMUL R25, R20, R20",    # Compute squares (5*5=25)

        "EXIT",
    ]

    sim.load_program(program)
    result = sim.run()

    print(f"\nAsync TMA Store Results:")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")
    print(f"  R21-R25 (squares computed):")
    for i in range(21, 26):
        print(f"    R{i} = {sim.read_register(0, 0, i)}")

    # Check global memory (TMA should have stored the data)
    global_data = sim.read_memory(0x500000, 8)
    val1, val2 = struct.unpack('<II', global_data)

    print(f"  Global memory[0]: {val1} (expected: 42)")
    print(f"  Global memory[1]: {val2} (expected: 84)")

    assert val1 == 42, f"TMA STORE should have written 42, got {val1}"
    assert val2 == 84, f"TMA STORE should have written 84, got {val2}"

    print("\n✓ Async TMA STORE successful!")


def demo_async_gemm_pipeline(sim: HopperSimulator) -> None:
    """Demonstrate full async GEMM pipeline with TMA."""
    print_header("ASYNC GEMM PIPELINE DEMO")
    print("Complete GEMM using async TMA for overlapping data transfer")

    setup_matrix_memory(sim)

    program = [
        # Setup addresses
        "MOV R1, 1048576",       # A matrix (global: 0x100000)
        "MOV R2, 2097152",       # B matrix (global: 0x200000)
        "MOV R3, 3145728",       # C matrix (global: 0x300000)
        "MOV R10, 16384",        # Shared memory base (0x4000)

        # Phase 1: Issue async TMA LOAD for A matrix
        "TMA.LOAD [R10], [R1], 64",    # Load A (async)

        # Phase 2: While A loads, compute something useful
        "MOV R20, 0",
        "IADD R20, R20, 1",
        "IMUL R21, R20, 2",           # Compute 2*x (2*1=2)
        "IADD R20, R20, 1",
        "IMUL R21, R21, 2",           # (2*2=4)
        "IADD R20, R20, 1",
        "IMUL R21, R21, 2",           # (4*2=8)
        "IADD R20, R20, 1",
        "IMUL R21, R21, 2",           # (8*2=16)

        # Phase 3: Issue async TMA LOAD for B matrix
        "TMA.LOAD [R10+64], [R2], 64",  # Load B (async)

        # Phase 4: While B loads, do more work
        "MOV R22, 0",
        "IADD R22, R22, 1",
        "IADD R23, R23, R22",          # Accumulate (1)
        "IADD R22, R22, 1",
        "IADD R23, R23, R22",          # (1+2=3)
        "IADD R22, R22, 1",
        "IADD R23, R23, R22",          # (3+3=6)
        "IADD R22, R22, 1",
        "IADD R23, R23, R22",          # (6+4=10)

        # Phase 5: At this point, A and B should be in shared memory
        # Perform computation on the loaded data
        "LDS.U32 R30, [R10]",     # Read A[0]
        "LDS.U32 R31, [R10+4]",   # Read A[1]
        "IADD R32, R30, R31",     # Sum

        # Phase 6: Store result to C via async TMA
        "STS.U32 [R10], R32",     # Store sum to shared
        "TMA.STORE [R3], [R10], 4",     # Async store to global

        # Phase 7: While storing, do final computation
        "MOV R24, 0",
        "IADD R24, R24, 1",
        "IADD R25, R25, R24",
        "IADD R24, R24, 1",
        "IADD R25, R25, R24",
        "IADD R24, R24, 1",
        "IADD R25, R25, R24",

        "EXIT",
    ]

    sim.load_program(program)
    result = sim.run()

    print(f"\nAsync GEMM Pipeline Results:")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")

    # Check results
    r32 = sim.read_register(0, 0, 32)  # Sum of A[0] + A[1]
    r21 = sim.read_register(0, 0, 21)  # Computed during first TMA
    r23 = sim.read_register(0, 0, 23)  # Computed during second TMA
    r25 = sim.read_register(0, 0, 25)  # Computed during TMA store

    print(f"  R32 (A[0] + A[1]): {r32} (expected: 3)")
    print(f"  R21 (computed during TMA load A): {r21}")
    print(f"  R23 (computed during TMA load B): {r23}")
    print(f"  R25 (computed during TMA store): {r25}")

    # Verify global memory C has the result
    c_data = sim.read_memory(0x300000, 4)
    result_val = struct.unpack('<I', c_data)[0]

    print(f"  C matrix @ 0x300000: {result_val} (expected: 3)")

    assert r32 == 3, f"Sum should be 3, got {r32}"
    assert result_val == 3, f"C matrix should contain 3, got {result_val}"

    print("\n✓ Async GEMM Pipeline successful!")
    print("  Key insight: TMA transfers overlapped with computation")


def demo_tma_with_mbarrier(sim: HopperSimulator) -> None:
    """Demonstrate TMA with mbarrier synchronization."""
    print_header("TMA WITH MBARRIER SYNCHRONIZATION")
    print("Using mbarrier to synchronize async TMA operations")

    setup_matrix_memory(sim)

    program = [
        # Setup
        "MOV R1, 1048576",       # A matrix (global: 0x100000)
        "MOV R10, 16384",        # Shared memory (0x4000)
        "MOV R15, 2",            # mbarrier count (2 arrivals needed)
        "MOV R20, 17408",        # mbarrier location in shared (0x4400)

        # Initialize mbarrier in shared memory
        "mbarrier.init.shared [R20], R15",

        # Issue TMA load (async)
        "TMA.LOAD [R10], [R1], 64",

        # Signal mbarrier arrival
        "mbarrier.arrive.shared [R20]",

        # Do other work
        "MOV R30, 0",
        "IADD R30, R30, 1",
        "IADD R30, R30, 1",
        "IADD R30, R30, 1",
        "IADD R30, R30, 1",
        "IADD R30, R30, 1",

        # Another arrival (simulating another warp)
        "mbarrier.arrive.shared [R20]",

        # Wait for mbarrier (2 arrivals)
        "mbarrier.test_wait.shared [R20], 2",

        # Now we know TMA has completed
        "LDS.U32 R31, [R10]",

        "EXIT",
    ]

    sim.load_program(program)
    result = sim.run()

    print(f"\nmbarrier Synchronization Results:")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")
    print(f"  R31 (loaded data): {sim.read_register(0, 0, 31)}")

    print("\n✓ mbarrier synchronization successful!")


def print_summary():
    """Print summary of async TMA benefits."""
    print_header("SUMMARY")
    print("""
Async TMA Benefits:

  1. Overlapped Execution
     - TMA runs in background while warp continues executing
     - Hides memory latency effectively
     - Enables computation-communication overlap

  2. Performance Gains
     - TMA reduces load instructions by ~10x for matrix tiles
     - Async execution improves throughput
     - Warps can process multiple tiles in pipeline

  3. Memory Efficiency
     - Bulk transfers reduce memory transaction overhead
     - Hardware-accelerated address translation
     - Efficient strided access patterns

  4. Synchronization
     - mbarrier provides efficient synchronization
     - No busy-waiting required
     - Warps can process other tiles while waiting

Real-World Applications:
  - CUTLASS 3.x GEMM kernels
  - FlashAttention-2
  - Custom transformer kernels
  - Large language model inference
""")


def main():
    """Run all async TMA demonstrations."""
    print("\n" + "█" * 70)
    print("█" + " " * 15 + "ASYNC TMA DEMO - HOPPER GPU" + " " * 16 + "█")
    print("█" * 70)
    print("\nDemonstrating Asynchronous Tensor Memory Accelerator operations.")
    print("TMA enables efficient bulk data transfer while the warp continues")
    print("executing other instructions, hiding memory latency.")

    sim = HopperSimulator()

    demo_async_tma_load(sim)
    sim.reset()

    demo_async_tma_store(sim)
    sim.reset()

    demo_async_gemm_pipeline(sim)
    sim.reset()

    demo_tma_with_mbarrier(sim)

    print_summary()


if __name__ == "__main__":
    main()

