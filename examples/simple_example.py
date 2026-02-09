#!/usr/bin/env python3
"""
Simple example showing how to use the Hopper GPU Simulator.

This example demonstrates:
1. Loading and running a SASS program
2. Checking register values
3. Using memory operations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import HopperSimulator, SimulatorConfig


def main():
    print("=" * 60)
    print("Hopper GPU Simulator - Simple Example")
    print("=" * 60)

    # Define a simple program that computes some arithmetic
    program = [
        # Initialize values
        "MOV R1, 42",           # Load constant 42
        "MOV R2, 8",            # Load constant 8
        "MOV R3, 100",          # Load constant 100

        # Arithmetic operations
        "IADD R4, R1, R2",      # R4 = 42 + 8 = 50
        "IADD R5, R4, R3",      # R5 = 50 + 100 = 150

        # Store result to memory
        "MOV R10, 0x1000",      # Memory address
        "STG [R10], R5",        # Store R5 to memory

        "EXIT",
    ]

    # Create simulator
    config = SimulatorConfig(
        num_sms=1,
        warps_per_sm=1,
        global_mem_size=1024 * 1024,  # 1 MB
    )

    sim = HopperSimulator(config)

    # Load and run program
    print("\nLoading program...")
    sim.load_program(program)

    print("Running simulation...\n")
    result = sim.run()

    # Print results
    if result.success:
        print(f"✓ Simulation completed successfully!")
        print(f"  Cycles: {result.cycles}")
        print(f"  Instructions executed: {result.instructions_executed}")
        print(f"  IPC: {result.instructions_executed / max(result.cycles, 1):.2f}\n")

        # Check register values
        print("Register values (Warp 0, Lane 0):")
        print(f"  R1 = {sim.read_register(0, 0, 1)}  (expected: 42)")
        print(f"  R2 = {sim.read_register(0, 0, 2)}  (expected: 8)")
        print(f"  R3 = {sim.read_register(0, 0, 3)}  (expected: 100)")
        print(f"  R4 = {sim.read_register(0, 0, 4)}  (expected: 50)")
        print(f"  R5 = {sim.read_register(0, 0, 5)}  (expected: 150)")

        # Check memory value
        mem_value = sim.read_memory(0x1000, 4)
        stored_value = int.from_bytes(mem_value, byteorder='little')
        print(f"\nMemory value at 0x1000:")
        print(f"  {stored_value}  (expected: 150)")

        # Verify results
        assert sim.read_register(0, 0, 5) == 150, "R5 should be 150"
        assert stored_value == 150, "Memory should contain 150"

        print("\n✓ All checks passed!")

    else:
        print(f"✗ Simulation failed: {result.error}")


if __name__ == "__main__":
    main()
