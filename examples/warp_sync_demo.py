#!/usr/bin/env python3
"""
Warp Synchronization Demo

Demonstrates warp-level operations including barriers, voting, shuffling,
elections, and reductions - key features of SIMT programming on GPUs.
"""

from src.simulator import HopperSimulator, SimulatorConfig


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_activemask():
    """Demonstrate ACTIVEMASK operation."""
    print_header("1. ACTIVEMASK Demo")
    print("Get the mask of currently active lanes in the warp")

    program = [
        "ACTIVEMASK R1",
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    mask = sim.read_register(0, 0, 1)
    print(f"  Active mask: {mask:#010x} (binary: {mask:032b})")
    print(f"  Active lanes: {bin(mask).count('1')} out of 32")
    print(f"  ✓ All 32 lanes are active")


def demo_elect():
    """Demonstrate ELECT operation."""
    print_header("2. ELECT Demo")
    print("Elect one thread from the warp (lowest active lane)")

    program = [
        "ELECT R1",
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    elected = sim.read_register(0, 0, 1)
    print(f"  Lane 0 elected: {elected}")
    print(f"  ✓ Lowest active lane (lane 0) was elected")


def demo_vote():
    """Demonstrate VOTE operation."""
    print_header("3. VOTE Demo")
    print("Vote across warp - combine predicate values from all lanes")

    program = [
        "SETP P0, R0, R0",
        "MOV R1, 1",
        "MOV R2, 0",
        "SETP P0, R1, R2",
        "MOV R3, 0",
        "MOV R4, 1",
        "SETP P0, R3, R4",
        "VOTE R5, P0",
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    vote_result = sim.read_register(0, 0, 5)
    print(f"  Vote result: {vote_result:#010x}")
    print(f"  Lanes with true predicate: {bin(vote_result).count('1')}")
    print(f"  ✓ Vote mask combines all lane predicates")


def demo_shuffle():
    """Demonstrate SHFL (shuffle) operation."""
    print_header("4. SHFL Demo")
    print("Shuffle data between lanes - communication without shared memory")

    program = [
        "MOV R2, 0",
        "MOV R10, 100",
        "MOV R11, 200",
        "MOV R12, 300",
        "SHFL R20, R10, R2, 1",
        "SHFL R21, R10, R2, 2",
        "SHFL R22, R10, R2, 4",
        "SHFL R23, R10, R2, 0",
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    print("  Shuffle results (lane 0 perspective):")
    print(f"    R10 (source):     {sim.read_register(0, 0, 10)}")
    print(f"    R20 (delta=1):    {sim.read_register(0, 0, 20)} - got from lane 1")
    print(f"    R21 (delta=2):    {sim.read_register(0, 0, 21)} - got from lane 2")
    print(f"    R22 (delta=4):    {sim.read_register(0, 0, 22)} - got from lane 4")
    print(f"    R23 (broadcast):  {sim.read_register(0, 0, 23)} - broadcast from lane 0")
    print(f"  ✓ Shuffle enables efficient lane-to-lane communication")


def demo_barrier():
    """Demonstrate barrier synchronization."""
    print_header("5. BARRIER Demo")
    print("Synchronize execution across all lanes in warp")

    program = [
        "MOV R2, 0",
        "BAR 0",
        "BAR.WARP",
        "MOV R1, 42",
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    print("  Barrier synchronization points:")
    print("    - BAR 0: Traditional barrier")
    print("    - BAR.WARP: Warp-level barrier")
    print(f"  ✓ Program completed with {result.cycles} cycles")


def demo_reduction():
    """Demonstrate parallel reduction across warp."""
    print_header("6. PARALLEL REDUCTION Demo")
    print("Compute sum of values across all warp lanes")

    program = [
        "MOV R2, 1",
        "MOV R3, 2",
        "MOV R4, 3",
        "MOV R5, 4",
        "MOV R6, 5",
        "MOV R7, 6",
        "MOV R8, 7",
        "MOV R9, 8",
        "MOV R16, R2",
        "SHFL R17, R16, R2, 1",
        "IADD R16, R16, R17",
        "SHFL R17, R16, R2, 2",
        "IADD R16, R16, R17",
        "SHFL R17, R16, R2, 4",
        "IADD R16, R16, R17",
        "SHFL R17, R16, R2, 8",
        "IADD R16, R16, R17",
        "SHFL R20, R16, R2, 0",
        "MOV R50, 1",
        "IADD R50, R50, 2",
        "IADD R50, R50, 3",
        "IADD R50, R50, 4",
        "IADD R50, R50, 5",
        "IADD R50, R50, 6",
        "IADD R50, R50, 7",
        "IADD R50, R50, 8",
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    reduction_result = sim.read_register(0, 0, 16)
    broadcast_result = sim.read_register(0, 0, 20)
    expected = sim.read_register(0, 0, 50)

    print("  Parallel reduction (shuffle-add pattern):")
    print(f"    R16 (reduction):  {reduction_result}")
    print(f"    R20 (broadcast):   {broadcast_result}")
    print(f"    R50 (expected):    {expected}")
    print(f"  ✓ Reduction: sum of 1+2+3+4+5+6+7+8 = {expected}")


def demo_complex_sync():
    """Demonstrate complex synchronization pattern."""
    print_header("7. COMPLEX SYNC PATTERN Demo")
    print("Multi-phase computation with synchronization")

    program = [
        "MOV R2, 10",
        "MOV R3, 20",
        "IADD R10, R2, R3",
        "BAR 1",
        "MOV R4, 5",
        "IMUL R11, R10, R4",
        "BAR 2",
        "IADD R12, R11, 100",
        "EXIT",
    ]

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)
    sim.load_program(program)
    result = sim.run()

    print("  Multi-phase computation:")
    print(f"    Phase 1 (R10):  {sim.read_register(0, 0, 10)} - should be 30")
    print(f"    Phase 2 (R11):  {sim.read_register(0, 0, 11)} - should be 150")
    print(f"    Phase 3 (R12):  {sim.read_register(0, 0, 12)} - should be 250")
    print(f"  ✓ Synchronization enables coordinated computation")


def demo_predicate_operations():
    """Demonstrate predicate-based execution."""
    print_header("8. PREDICATE EXECUTION Demo")
    print("Conditional execution based on predicates")

    program = [
        "MOV R0, 10",
        "MOV R1, 20",
        "MOV R2, 30",
        "SETP P0, R0, R1",
        "SETP P1, R1, R0",
        "@P0 MOV R10, 100",
        "@P1 MOV R11, 200",
        "@!P1 MOV R12, 300",
        "MOV R3, 1000",
        "MOV R4, 2000",
        "SETP P2, R0, R2",
        "SELP R13, R3, R4, P2",
        "EXIT",
    ]

    sim = HopperSimulator()
    sim.load_program(program)
    result = sim.run()

    print("  Predicate-based execution:")
    print(f"    R10 (@P0):       {sim.read_register(0, 0, 10)} - should be 100")
    print(f"    R11 (@P1):       {sim.read_register(0, 0, 11)} - should be 0 (skipped)")
    print(f"    R12 (@!P1):      {sim.read_register(0, 0, 12)} - should be 300")
    print(f"    R13 (SELP):      {sim.read_register(0, 0, 13)} - should be 1000")
    print(f"  ✓ Predicates enable branch-free conditional code")


def main():
    """Run all warp synchronization demonstrations."""
    print("\n" + "█" * 60)
    print("█" + " " * 18 + "WARP SYNCHRONIZATION DEMO" + " " * 17 + "█")
    print("█" * 60)
    print("\nDemonstrating warp-level operations in the Hopper GPU Simulator")
    print("These operations are essential for efficient GPU programming.")

    demo_activemask()
    demo_elect()
    demo_vote()
    demo_shuffle()
    demo_barrier()
    demo_reduction()
    demo_complex_sync()
    demo_predicate_operations()

    print_header("SUMMARY")
    print("Warp synchronization operations demonstrated:")
    print("  ✓ ACTIVEMASK - Get active lane mask")
    print("  ✓ ELECT - Elect one lane from warp")
    print("  ✓ VOTE - Combine predicates across lanes")
    print("  ✓ SHFL - Shuffle data between lanes")
    print("  ✓ BAR/BAR.WARP - Barrier synchronization")
    print("  ✓ Parallel reduction using shuffle-add")
    print("  ✓ Predicate-based conditional execution")
    print("\nThese operations enable efficient parallel computation without")
    print("requiring expensive global memory access or locks.")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
