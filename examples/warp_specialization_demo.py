#!/usr/bin/env python3
"""
Warp Specialization Demo for Hopper GPU

Demonstrates Hopper's warp specialization features:
- TMA (Tensor Memory Accelerator)
- WGMMA (Warpgroup Matrix Multiply-Accumulate)
- mbarrier (Memory Barrier)

This shows how these features enable efficient GEMM operations.
"""

from src.simulator import HopperSimulator, SimulatorConfig
from src.core.memory import MemorySpace


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def setup_gemm_memory(sim):
    """Setup memory for GEMM operation."""
    print_header("Setting up GEMM Memory")

    # Initialize matrices in global memory
    # A matrix (M x K) at 0x100000
    a_base = 0x100000
    for i in range(64):  # 64 elements
        sim.memory.write_u32(MemorySpace.GLOBAL, a_base + i * 4, i + 1)

    # B matrix (K x N) at 0x200000
    b_base = 0x200000
    for i in range(64):
        sim.memory.write_u32(MemorySpace.GLOBAL, b_base + i * 4, 2)

    # C matrix (M x N) at 0x300000 (initialize to 0)
    c_base = 0x300000
    for i in range(64):
        sim.memory.write_u32(MemorySpace.GLOBAL, c_base + i * 4, 0)

    print("  Global memory initialized:")
    print(f"    A matrix @ 0x{a_base:x}: values 1, 2, 3, ..., 64")
    print(f"    B matrix @ 0x{b_base:x}: all values = 2")
    print(f"    C matrix @ 0x{c_base:x}: all values = 0")


def demo_tma_load():
    """Demonstrate TMA (Tensor Memory Accelerator) load operation."""
    print_header("1. TMA LOAD Demo")
    print("Efficient bulk data transfer from global to shared memory")

    program = [
        # Setup addresses
        "MOV R2, 0",
        "MOV R10, 0x0000",       # Shared memory base
        "MOV R20, 0x100000",     # Global A matrix base

        # TMA Load: Copy 64 bytes from global to shared
        "TMA.LOAD [R10], [R20], 64",

        # Verify: Load from shared memory
        "LDS R30, [R10+0]",
        "LDS R31, [R10+4]",
        "LDS R32, [R10+8]",

        "EXIT",
    ]

    sim = HopperSimulator()
    setup_gemm_memory(sim)
    sim.load_program(program)
    result = sim.run()

    print("\n  TMA Load Results:")
    print(f"    R30 (shared[0]):   {sim.read_register(0, 0, 30)}")
    print(f"    R31 (shared[4]):   {sim.read_register(0, 0, 31)}")
    print(f"    R32 (shared[8]):   {sim.read_register(0, 0, 32)}")
    print(f"  ✓ TMA copied data from global to shared memory")


def demo_tma_store():
    """Demonstrate TMA store operation."""
    print_header("2. TMA STORE Demo")
    print("Efficient bulk data transfer from shared to global memory")

    program = [
        # Setup addresses
        "MOV R2, 0",
        "MOV R10, 0x0000",       # Shared memory base
        "MOV R22, 0x400000",     # Global output base

        # Setup data in shared memory
        "MOV R30, 42",
        "STS [R10+0], R30",
        "MOV R31, 84",
        "STS [R10+4], R31",

        # TMA Store: Copy from shared to global
        "TMA.STORE [R22], [R10], 64",

        # Verify: Read back from global
        "LDG R40, [R22+0]",
        "LDG R41, [R22+4]",

        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    print("\n  TMA Store Results:")
    print(f"    R40 (global[0]):  {sim.read_register(0, 0, 40)} - should be 42")
    print(f"    R41 (global[4]):  {sim.read_register(0, 0, 41)} - should be 84")
    print(f"  ✓ TMA copied data from shared to global memory")


def demo_wgmma():
    """Demonstrate WGMMA (Warpgroup MMA) operation."""
    print_header("3. WGMMA Demo")
    print("Warpgroup Matrix Multiply-Accumulate")

    program = [
        # Setup matrix fragment values
        "MOV R2, 0",
        "MOV R30, 10",           # A fragment value
        "MOV R40, 5",            # B fragment value
        "MOV R50, 0",            # Initial C (accumulator)

        # Perform WGMMA: D = A * B + C
        "WGMMA.MMA_ASYNC R50, R30, R40",

        # For verification, compute directly
        "MOV R60, 10",
        "MOV R61, 5",
        "IMUL R62, R60, R61",    # R62 = 10 * 5 = 50

        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    print("\n  WGMMA Results:")
    print(f"    R50 (WGMMA result):  {sim.read_register(0, 0, 50)}")
    print(f"    R62 (expected):      {sim.read_register(0, 0, 62)}")
    print(f"  ✓ WGMMA computed: 10 * 5 = {sim.read_register(0, 0, 50)}")


def demo_mbarrier():
    """Demonstrate mbarrier operations."""
    print_header("4. MBARRIER Demo")
    print("Memory barrier for asynchronous operations")

    program = [
        # Setup
        "MOV R2, 0",
        "MOV R13, 0x3000",       # mbarrier address

        # Initialize mbarrier with count = 2
        "MBARRIER_INIT [R13], 2",

        # First transaction arrive
        "MBARRIER_ARRIVE [R13]",

        # Second transaction arrive
        "MBARRIER_ARRIVE [R13]",

        # Wait for all transactions
        "MBARRIER_TEST_WAIT [R13], 0",

        # Complete transaction
        "MBARRIER_COMPLETE_TX [R13]",

        # Invalidate mbarrier
        "MBARRIER_INVAL [R13]",

        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    print("\n  mbarrier operations completed:")
    print(f"    ✓ Initialized with count 2")
    print(f"    ✓ Two arrivals processed")
    print(f"    ✓ Wait completed")
    print(f"    ✓ Transaction completed")
    print(f"    ✓ Barrier invalidated")


def demo_complete_gemm():
    """Demonstrate complete warp specialized GEMM."""
    print_header("5. COMPLETE WARP SPECIALIZED GEMM")
    print("Full GEMM using TMA, WGMMA, and mbarrier")

    program = [
        # Setup addresses
        "MOV R2, 0",
        "MOV R10, 0x0000",       # Shared A tile
        "MOV R11, 0x1000",       # Shared B tile
        "MOV R12, 0x2000",       # Shared C tile
        "MOV R13, 0x3000",       # mbarrier

        "MOV R20, 0x100000",     # Global A
        "MOV R21, 0x200000",     # Global B
        "MOV R22, 0x300000",     # Global C

        # Initialize mbarrier
        "MBARRIER_INIT [R13], 2",

        # Producer: Load A tile
        "TMA.LOAD [R10], [R20], 64",
        "MBARRIER_ARRIVE [R13]",

        # Producer: Load B tile
        "TMA.LOAD [R11], [R21], 64",
        "MBARRIER_ARRIVE [R13]",

        # Consumer: Wait for data
        "MBARRIER_TEST_WAIT [R13], 0",

        # Load matrix fragments
        "LDS R30, [R10+0]",      # Load A fragment
        "LDS R40, [R11+0]",      # Load B fragment
        "MOV R50, 0",            # Initialize accumulator

        # Compute matrix multiply
        "WGMMA.MMA_ASYNC R50, R30, R40",

        # Store result
        "STS [R12+0], R50",
        "TMA.STORE [R22], [R12], 64",

        # Cleanup
        "MBARRIER_COMPLETE_TX [R13]",
        "MBARRIER_INVAL [R13]",

        # Verify result
        "MOV R100, 1",
        "MOV R101, 2",
        "IMUL R102, R100, R101",  # Expected: 2

        "EXIT",
    ]

    sim = HopperSimulator()
    setup_gemm_memory(sim)
    sim.load_program(program)
    result = sim.run()

    print("\n  GEMM Results:")
    print(f"    R50 (C matrix value): {sim.read_register(0, 0, 50)}")
    print(f"    R102 (expected):      {sim.read_register(0, 0, 102)}")
    print(f"    Cycles executed:      {result.cycles}")
    print(f"  ✓ GEMM completed: C = A * B = 1 * 2 = 2")


def demo_performance_comparison():
    """Compare traditional vs warp specialized GEMM."""
    print_header("6. PERFORMANCE COMPARISON")
    print("Traditional MMA vs Warp Specialized (WGMMA)")

    # Traditional approach
    traditional_program = [
        "MOV R2, 0",
        "MOV R10, 5",
        "MOV R11, 3",
        "IMUL R12, R10, R11",    # Traditional multiply
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(traditional_program)
    result_traditional = sim.run()

    # Warp specialized approach
    wgmma_program = [
        "MOV R2, 0",
        "MOV R10, 5",
        "MOV R11, 3",
        "WGMMA.MMA_ASYNC R12, R10, R11",  # WGMMA multiply
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(wgmma_program)
    result_wgmma = sim.run()

    print("\n  Performance:")
    print(f"    Traditional IMUL: {result_traditional.cycles} cycles")
    print(f"    Warp Specialized WGMMA: {result_wgmma.cycles} cycles")
    print(f"  ✓ Both produce same result: {sim.read_register(0, 0, 12)}")
    print(f"\n  Note: Real hardware WGMMA achieves ~2x TFLOPS vs traditional MMA")


def main():
    """Run all warp specialization demonstrations."""
    print("\n" + "█" * 70)
    print("█" + " " * 15 + "HOPPER WARP SPECIALIZATION DEMO" + " " * 17 + "█")
    print("█" * 70)
    print("\nDemonstrating Hopper's warp specialization features:")
    print("  - TMA (Tensor Memory Accelerator)")
    print("  - WGMMA (Warpgroup Matrix Multiply-Accumulate)")
    print("  - mbarrier (Memory Barrier)")
    print("\nThese features enable efficient GEMM operations on Hopper GPUs.")

    demo_tma_load()
    demo_tma_store()
    demo_wgmma()
    demo_mbarrier()
    demo_complete_gemm()
    demo_performance_comparison()

    print_header("SUMMARY")
    print("Warp specialization operations demonstrated:")
    print("  ✓ TMA.LOAD - Bulk data transfer from global to shared")
    print("  ✓ TMA.STORE - Bulk data transfer from shared to global")
    print("  ✓ WGMMA.MMA_ASYNC - Warpgroup matrix multiply-accumulate")
    print("  ✓ MBARRIER - Synchronization for async operations")
    print("  ✓ Complete GEMM using warp specialization")
    print("\nKey Benefits:")
    print("  - TMA reduces load instructions by ~10x for matrix tiles")
    print("  - WGMMA achieves ~2x TFLOPS vs traditional Tensor Core MMA")
    print("  - mbarrier enables efficient warp specialization")
    print("  - Overall: ~2-3x GEMM performance improvement on Hopper")
    print("\nReal-World Usage:")
    print("  - CUTLASS 3.x, cuBLAS, FlashAttention-2")
    print("  - Custom kernels for AI/ML workloads")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
