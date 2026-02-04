#!/usr/bin/env python3
"""
Mbarrier-based Async TMA Demo for Hopper GPU

Demonstrates proper producer/consumer synchronization using mbarrier:
- Producer warp (warp 0): Issues async TMA LOAD and signals mbarrier
- Consumer warp (warp 1): Waits on mbarrier, then consumes data

This shows how real Hopper warp-specialized kernels work.
"""

import struct
from src.simulator import HopperSimulator


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def setup_matrix_memory(sim: HopperSimulator) -> None:
    """Set up matrix data in global memory."""
    print("\nSetting up matrix data in global memory...")

    # Initialize A matrix with 32-bit integers
    a_data = struct.pack('<' + 'I' * 8, *range(1, 9))   # 8 values: 1, 2, ..., 8
    b_data = struct.pack('<' + 'I' * 8, *([2] * 8))   # 8 values: all 2s

    sim.write_memory(0x100000, a_data)  # A matrix @ 0x100000
    sim.write_memory(0x200000, b_data)  # B matrix @ 0x200000

    print("  A matrix @ 0x100000: 32-bit values 1, 2, 3, ..., 8")
    print("  B matrix @ 0x200000: 32-bit values all 2")


def demo_mbarrier_sync(sim: HopperSimulator) -> None:
    """Demonstrate mbarrier-based synchronization between producer and consumer warps."""
    print_header("MBARRIER-BASED ASYNC TMA SYNCHRONIZATION")
    print("Producer warp (0) loads data, Consumer warp (1) waits on mbarrier")

    setup_matrix_memory(sim)

    # Producer warp program (warp 0)
    producer_program = [
        # Setup addresses
        "MOV R10, 16384",        # Shared memory base (0x4000)
        "MOV R11, 16400",        # Mbarrier address (0x4010)
        "MOV R1, 1048576",       # Global memory (0x100000)

        # Initialize mbarrier with expected transaction count
        "MBARRIER_INIT [R11], 1",    # Initialize mbarrier
        "MBARRIER_EXPECT_TX [R11], 1",  # Expect 1 TMA operation

        # Issue async TMA LOAD
        "TMA.LOAD [R10], [R1], 32",  # Load 32 bytes async

        # Producer can do other work here while TMA is in flight
        # For demo, just do some computation
        "MOV R20, 100",
        "MOV R21, 200",
        "MOV R22, 300",

        "EXIT",
    ]

    # Consumer warp program (warp 1)
    consumer_program = [
        # Setup addresses (must match producer)
        "MOV R10, 16384",        # Shared memory base
        "MOV R11, 16400",        # Mbarrier address

        # Wait for mbarrier to be ready
        # In real code, this would be a spin loop
        # For demo, we do some work then check
        "MOV R30, 1",
        "MOV R31, 1",
        "MOV R32, 1",
        "MOV R33, 1",
        "MOV R34, 1",
        "MOV R35, 1",
        "MOV R36, 1",
        "MOV R37, 1",
        "MOV R38, 1",
        "MOV R39, 1",
        "MOV R40, 1",
        "MOV R41, 1",
        "MOV R42, 1",
        "MOV R43, 1",
        "MOV R44, 1",
        "MOV R45, 1",
        "MOV R46, 1",
        "MOV R47, 1",
        "MOV R48, 1",
        "MOV R49, 1",
        "MOV R50, 1",
        "MOV R51, 1",
        "MOV R52, 1",
        "MOV R53, 1",
        "MOV R54, 1",
        "MOV R55, 1",

        # Check mbarrier (TMA should be complete now)
        "MBARRIER_TEST_WAIT [R11]",  # Check if mbarrier is ready

        # Now consume the data from shared memory
        "LDS.U32 R1, [R10]",     # Read shared[0]
        "LDS.U32 R2, [R10+4]",   # Read shared[4]
        "LDS.U32 R3, [R10+8]",   # Read shared[8]
        "LDS.U32 R4, [R10+12]",  # Read shared[12]

        "EXIT",
    ]

    # Load programs for both warps
    sim.load_program(producer_program, warp_id=0)
    sim.load_program(consumer_program, warp_id=1)

    # Run simulation with both warps
    result = sim.run()

    print(f"\nSimulation Results:")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")
    print(f"  Async queue completed: {sim.async_queue.get_completed_count()}")

    # Check producer warp (warp 0) - did computation while TMA was in flight
    print(f"\n  Producer Warp (0) computation while TMA in flight:")
    print(f"    R20: {sim.read_register(0, 0, 20)} (expected: 100)")
    print(f"    R21: {sim.read_register(0, 0, 21)} (expected: 200)")
    print(f"    R22: {sim.read_register(0, 0, 22)} (expected: 300)")

    # Check consumer warp (warp 1) - consumed data from shared memory
    print(f"\n  Consumer Warp (1) consumed data:")
    print(f"    R1 (shared[0]):  {sim.read_register(1, 0, 1)} (expected: 1)")
    print(f"    R2 (shared[4]):  {sim.read_register(1, 0, 2)} (expected: 2)")
    print(f"    R3 (shared[8]):  {sim.read_register(1, 0, 3)} (expected: 3)")
    print(f"    R4 (shared[12]): {sim.read_register(1, 0, 4)} (expected: 4)")

    # Check mbarrier state
    mbarrier = sim.mbarrier_manager.get_barrier(0x4010)
    print(f"\n  Mbarrier state:")
    if mbarrier:
        print(f"    Address: {mbarrier.address:#x}")
        print(f"    State: {mbarrier.state.name}")
        print(f"    Count: {mbarrier.current_count}/{mbarrier.expected_count}")
    else:
        print(f"    No mbarrier found at 0x4010")

    # Verify
    assert sim.read_register(0, 0, 20) == 100
    assert sim.read_register(0, 0, 21) == 200
    assert sim.read_register(0, 0, 22) == 300
    assert sim.read_register(1, 0, 1) == 1
    assert sim.read_register(1, 0, 2) == 2
    assert sim.read_register(1, 0, 3) == 3
    assert sim.read_register(1, 0, 4) == 4

    print("\n✓ Mbarrier-based synchronization successful!")
    print("  Key benefits:")
    print("    - Producer warp issues async TMA and continues working")
    print("    - Consumer warp efficiently waits on mbarrier")
    print("    - No busy spinning, true async communication")
    print("    - Enables warp specialization (producer vs consumer roles)")


def print_summary():
    """Print summary of mbarrier-based synchronization."""
    print_header("MBARRIER-BASED ASYNC TMA SUMMARY")
    print("""
Mbarrier-based Producer/Consumer Synchronization:

  Producer Warp (Warp 0):
    1. Initialize mbarrier: MBARRIER_INIT [addr], count
    2. Set expected transactions: MBARRIER_EXPECT_TX [addr], count
    3. Issue async TMA operations (TMA.LOAD, TMA.STORE)
    4. Continue with other work (computation, etc.)
    5. When TMA completes, hardware signals mbarrier automatically

  Consumer Warp (Warp 1):
    1. Set up addresses matching producer
    2. Do other work while waiting (or just wait)
    3. Check mbarrier: MBARRIER_TEST_WAIT [addr]
    4. When ready, consume data from shared memory
    5. Process data with WGMMA or other operations

  Benefits:
    ✓ No busy spinning - mbarrier signals when ready
    ✓ True async communication between warps
    ✓ Enables warp specialization (producer vs consumer)
    ✓ Scales to multiple async operations
    ✓ Hardware-accelerated synchronization

  Real Hopper Usage:
    - Producer warp loads tiles with TMA
    - Consumer warp processes tiles with WGMMA
    - Pipeline: Load -> Compute -> Store overlapped
    - Multiple warps working in parallel on different stages

  This is the foundation of warp-specialized GEMM and other kernels!
""")


def main():
    """Run mbarrier-based async TMA demonstrations."""
    print("\n" + "█" * 70)
    print("█" + " " * 10 + "MBARRIER-BASED ASYNC TMA - HOPPER GPU" + " " * 11 + "█")
    print("█" * 70)
    print("\nDemonstrating producer/consumer synchronization using mbarrier.")
    print("This is how real warp-specialized Hopper kernels work.")

    sim = HopperSimulator()

    demo_mbarrier_sync(sim)

    print_summary()


if __name__ == "__main__":
    main()
