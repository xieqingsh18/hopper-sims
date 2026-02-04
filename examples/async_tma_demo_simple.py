#!/usr/bin/env python3
"""
Simplified Async TMA Demo for Hopper GPU

Demonstrates that async TMA operations work correctly.
This version works around the IADD bug in the simulator.
"""

import struct
from src.simulator import HopperSimulator
from src.core.memory import MemorySpace


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def setup_matrix_memory(sim: HopperSimulator) -> None:
    """Set up matrix data in global memory."""
    print("\nSetting up matrix data in global memory...")

    # Initialize A matrix with 32-bit integers
    import struct
    a_data = struct.pack('<' + 'I' * 8, *range(1, 9))   # 8 values: 1, 2, ..., 8
    b_data = struct.pack('<' + 'I' * 8, *([2] * 8))   # 8 values: all 2s
    c_data = struct.pack('<' + 'I' * 8, *([0] * 8))   # 8 values: all 0s

    sim.write_memory(0x100000, a_data)  # A matrix @ 0x100000
    sim.write_memory(0x200000, b_data)  # B matrix @ 0x200000
    sim.write_memory(0x300000, c_data)  # C matrix @ 0x300000

    print("  A matrix @ 0x100000: 32-bit values 1, 2, 3, ..., 8")
    print("  B matrix @ 0x200000: 32-bit values all 2")
    print("  C matrix @ 0x300000: 32-bit values all 0")


def demo_async_tma_load_basic(sim: HopperSimulator) -> None:
    """Basic async TMA load demonstration."""
    print_header("ASYNC TMA LOAD - BASIC TEST")
    print("Verifying async TMA load transfers data correctly")

    setup_matrix_memory(sim)

    # Simple program: TMA load -> wait -> read
    program = [
        "MOV R10, 16384",        # Shared memory base (0x4000)
        "MOV R1, 1048576",       # Global memory (0x100000)
        "TMA.LOAD [R10], [R1], 32",  # Load 32 bytes (8 ints) async

        # Fill time to let TMA complete (TMA takes ~50 cycles)
        "MOV R20, 1",
        "MOV R21, 1",
        "MOV R22, 1",
        "MOV R23, 1",
        "MOV R24, 1",
        "MOV R25, 1",
        "MOV R26, 1",
        "MOV R27, 1",
        "MOV R28, 1",
        "MOV R29, 1",
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
        "MOV R46,1",
        "MOV R47, 1",
        "MOV R48, 1",
        "MOV R49, 1",

        # Read the loaded values from shared memory
        "LDS.U32 R1, [R10]",     # Read shared[0]
        "LDS.U32 R2, [R10+4]",   # Read shared[4]
        "LDS.U32 R3, [R10+8]",   # Read shared[8]
        "LDS.U32 R4, [R10+12]",  # Read shared[12]
        "LDS.U32 R5, [R10+16]",  # Read shared[16]
        "LDS.U32 R6, [R10+20]",  # # Read shared[20]
        "LDS.U32 R7, [R10+24]",  # # Read shared[24]
        "LDS.U32 R8, [R10+28]",  # # Read shared[28]

        "EXIT",
    ]

    sim.load_program(program)
    result = sim.run()

    print(f"\nAsync TMA Load Results:")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")
    print(f"  Async queue completed: {sim.async_queue.get_completed_count()}")

    print(f"\n  Loaded values from shared memory:")
    print(f"    R1 (shared[0]):  {sim.read_register(0, 0, 1)} (expected: 1)")
    print(f"    R2 (shared[4]):  {sim.read_register(0, 0, 2)} (expected: 2)")
    print(f"    R3 (shared[8]):  {sim.read_register(0, 0, 3)} (expected: 3)")
    print(f"    R4 (shared[12]): {sim.read_register(0, 0, 4)} (expected: 4)")
    print(f"    R5 (shared[16]): {sim.read_register(0, 0, 5)} (expected: 5)")
    print(f"    R6 (shared[20]): {sim.read_register(0, 0, 6)} (expected: 6)")
    print(f"    R7 (shared[24]): {sim.read_register(0, 0, 7)} (expected: 7)")
    print(f"    R8 (shared[28]): {sim.read_register(0, 0, 8)} (expected: 8)")

    # Verify
    for i in range(1, 9):
        assert sim.read_register(0, 0, i) == i, f"Expected {i}, got {sim.read_register(0, 0, i)}"

    print("\n✓ Async TMA LOAD successful!")


def demo_async_tma_store_basic(sim: HopperSimulator) -> None:
    """Basic async TMA store demonstration."""
    print_header("ASYNC TMA STORE - BASIC TEST")
    print("Verifying async TMA store transfers data correctly")

    # Build program with sufficient wait cycles
    program = [
        # Initialize shared memory with test values
        "MOV R10, 16384",        # Shared memory base
        "MOV R1, 100",           # Test value 1
        "MOV R2, 200",           # Test value 2

        "STS.U32 [R10], R1",     # shared[0] = 100
        "STS.U32 [R10+4], R2",   # shared[4] = 200

        # Initialize global address
        "MOV R3, 5242880",       # Global memory (0x500000)

        # Issue async TMA STORE
        "TMA.STORE [R3], [R10], 8",    # Store 8 bytes async
    ]

    # Add wait instructions (need >50 cycles for TMA operation)
    for i in range(20, 80):
        program.append(f"MOV R{i}, 1")

    program.append("EXIT")

    sim.load_program(program)
    result = sim.run()

    print(f"\nAsync TMA Store Results:")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")
    print(f"  Async queue completed: {sim.async_queue.get_completed_count()}")

    # Check global memory
    global_data = sim.read_memory(0x500000, 8)
    val1, val2 = struct.unpack('<II', global_data)

    print(f"  Global memory[0]: {val1} (expected: 100)")
    print(f"  Global memory[1]: {val2} (expected: 200)")

    assert val1 == 100, f"TMA STORE should have written 100, got {val1}"
    assert val2 == 200, f"TMA STORE should have written 200, got {val2}"

    print("\n✓ Async TMA STORE successful!")


def demo_async_tma_overlap(sim: HopperSimulator) -> None:
    """Demonstrate async TMA allowing computation overlap."""
    print_header("ASYNC TMA - COMPUTATION OVERLAP")
    print("Demonstrating that TMA runs in background while warp executes")

    # Clear shared memory to avoid interference from previous tests
    import struct
    empty_data = struct.pack('<' + 'I' * 8, *([0] * 8))
    sim.memory.write(MemorySpace.SHARED, 16384, empty_data)

    setup_matrix_memory(sim)

    # Program that shows async TMA running while computation happens
    program = [
        # Setup
        "MOV R10, 16384",        # Shared memory base
        "MOV R1, 1048576",       # Global memory (0x100000)

        # Issue async TMA LOAD
        "TMA.LOAD [R10], [R1], 32",  # Load 32 bytes async

        # While TMA is in flight, demonstrate computation overlap
        # Need >50 cycles of work for TMA to complete
        "MOV R20, 100",
        "MOV R21, 200",
        "MOV R22, 300",
        "MOV R23, 400",
        "MOV R24, 500",
        "MOV R25, 600",
        "MOV R26, 700",
        "MOV R27, 800",
        "MOV R28, 900",
        "MOV R29, 1000",
        "MOV R30, 1100",
        "MOV R31, 1200",
        "MOV R32, 1300",
        "MOV R33, 1400",
        "MOV R34, 1500",
        "MOV R35, 1600",
        "MOV R36, 1700",
        "MOV R37, 1800",
        "MOV R38, 1900",
        "MOV R39, 2000",
        "MOV R40, 2100",
        "MOV R41, 2200",
        "MOV R42, 2300",
        "MOV R43, 2400",
        "MOV R44, 2500",
        "MOV R45, 2600",
        "MOV R46, 2700",
        "MOV R47, 2800",
        "MOV R48, 2900",
        "MOV R49, 3000",
        "MOV R50, 3100",
        "MOV R51, 3200",
        "MOV R52, 3300",
        "MOV R53, 3400",
        "MOV R54, 3500",
        "MOV R55, 3600",
        "MOV R56, 3700",
        "MOV R57, 3800",
        "MOV R58, 3900",
        "MOV R59, 4000",
        "MOV R60, 4100",
        "MOV R61, 4200",
        "MOV R62, 4300",
        "MOV R63, 4400",

        # Now read the TMA-loaded data (should be complete after 50+ cycles)
        "LDS.U32 R1, [R10]",     # Read shared[0]
        "LDS.U32 R2, [R10+4]",   # Read shared[4]

        "EXIT",
    ]

    sim.load_program(program)
    result = sim.run()

    print(f"\nComputation Overlap Results:")
    print(f"  Cycles: {result.cycles}")
    print(f"  Instructions: {result.instructions_executed}")
    print(f"  Async queue completed: {sim.async_queue.get_completed_count()}")

    print(f"\n  While TMA was transferring data, warp executed:")
    print(f"    R20-R30: ", end="")
    for i in range(20, 31):
        print(f"{sim.read_register(0, 0, i)} ", end="")
    print("...")

    print(f"\n  TMA-loaded values:")
    print(f"    R1 (shared[0]):  {sim.read_register(0, 0, 1)} (expected: 1)")
    print(f"    R2 (shared[4]):  {sim.read_register(0, 0, 2)} (expected: 2)")

    assert sim.read_register(0, 0, 1) == 1
    assert sim.read_register(0, 0, 2) == 2

    print("\n✓ Async TMA computation overlap successful!")
    print("  Key benefit: Warp did useful work while TMA transferred data")


def print_summary():
    """Print summary of async TMA implementation."""
    print_header("ASYNC TMA IMPLEMENTATION SUMMARY")
    print("""
Async TMA Implementation in Hopper GPU Simulator:

  1. Async Operation Queue (src/core/async_ops.py)
     - AsyncQueue class manages pending/in-progress/completed operations
     - Supports TMA_LOAD, TMA_STORE, WGMMA operations
     - Configurable number of parallel TMA engines (default: 4)
     - tick() method advances simulation and processes completions

  2. TMA Instruction Execution (src/executor/warp.py)
     - _exec_tma() method handles TMA.LOAD and TMA.STORE
     - Creates AsyncOperation with callback
     - Callback performs actual data transfer when operation completes
     - Operations run for configurable cycle count (default: 50 cycles)

   3. Simulator Integration (src/simulator.py)
     - Shared AsyncQueue across all warps
     - Pipeline ticks async queue each cycle
     - Warp executors receive reference to async queue
     - Async operations complete in background while warp continues

  4. Key Features:
     ✓ Async TMA LOAD runs in background while warp executes
     ✓ Async TMA STORE runs in background while warp executes
     ✓ Computation-communication overlap hides memory latency
     ✓ mbarrier integration for synchronization

  Current Status:
     ✓ Async TMA operations are WORKING
     ✓ Data is correctly transferred between global and shared memory
     ✓ Async queue properly manages operation lifecycle

  Known Issues:
     - IADD instruction has a bug (registers not being updated)
     - This affects demos that use IADD for computation loops
     - Workaround: Use MOV instructions for simple demos

  Example Usage:
     # Async TMA load
     TMA.LOAD [shared_addr], [global_addr], size

     # Async TMA store
     TMA.STORE [global_addr], [shared_addr], size

     # While TMA is in flight, warp can execute other instructions
     # This enables computation-communication overlap
""")


def main():
    """Run simplified async TMA demonstrations."""
    print("\n" + "█" * 70)
    print("█" + " " * 18 + "ASYNC TMA DEMO - HOPPER GPU" + " " * 19 + "█")
    print("█" * 70)
    print("\nDemonstrating Asynchronous Tensor Memory Accelerator (TMA).")
    print("This simplified demo verifies async TMA functionality.")

    sim = HopperSimulator()

    demo_async_tma_load_basic(sim)
    sim.reset()

    demo_async_tma_store_basic(sim)
    sim.reset()

    demo_async_tma_overlap(sim)

    print_summary()


if __name__ == "__main__":
    main()
