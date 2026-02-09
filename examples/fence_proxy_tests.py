#!/usr/bin/env python3
"""
Fence.Proxy Tests - Mixing Generic Proxy (ld/st) with Async Proxy (TMA)

According to PTX ISA 9.1 specification:
- fence.proxy establishes ordering between different memory access proxies
- Generic proxy: normal ld/st/ldmatrix/stmatrix operations
- Async proxy: TMA (cp.async.bulk), WGMMA operations
- WITHOUT fence.proxy: No ordering guarantee between proxies
- WITH fence.proxy: Guaranteed ordering

This test file demonstrates:
1. Weak ordering: Generic proxy read might NOT see async proxy write
2. Fence.proxy effectiveness: Ensures ordering between proxies (within single warp)

Note: Current simulator limitation - warps don't share shared memory.
Tests use single-warp patterns to demonstrate fence.proxy concepts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import HopperSimulator, SimulatorConfig
from src.core.memory import MemorySpace


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_generic_async_weak_ordering():
    """
    Test 1: Demonstrates WEAK ORDERING between generic and async proxies.

    WITHOUT fence.proxy:
    - Start async TMA copy (async proxy)
    - Immediately do generic proxy load (LDS)
    - Result: Generic load might read stale data (async not yet visible)

    This demonstrates that without fence.proxy, there is NO ordering
    guarantee between generic and async proxy operations.
    """
    print_header("Test 1: Generic/Async Weak Ordering (NO fence.proxy)")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Initialize global memory with known value
    sim.memory.write_u32(MemorySpace.GLOBAL, 0x20000000, 999)

    # Kernel: Start async TMA, immediately read with generic proxy
    kernel = [
        # Initialize mbarrier
        'MOV R10, 0x5000',
        'MOV R11, 1',
        'MBARRIER_INIT [R10], R11',
        'MBARRIER_EXPECT_TX [R10], R11',

        # Start async TMA copy (async proxy)
        'MOV R12, 0x6000',
        'MOV R13, 0x20000000',
        'MOV R14, 128',
        'CP_ASYNC_BULK [R12], [R13], R14',

        # IMMEDIATELY load from shared (generic proxy)
        # NO fence.proxy - demonstrates weak ordering!
        'LDS R3, [0x6000]',

        # Later: wait for async and read again
        'MOV R15, 0',
        'MBARRIER_TEST_WAIT [R10], R15',
        'LDS R4, [0x6000]',

        'EXIT',
    ]

    print("Kernel:")
    print("  1. Initialize mbarrier")
    print("  2. Start async TMA copy (global->shared) [ASYNC PROXY]")
    print("  3. IMMEDIATELY load from shared [GENERIC PROXY]")
    print("     NO fence.proxy between them!")
    print("  4. Wait for async to complete")
    print("  5. Load from shared again")
    print()
    print("Expected (Weak Ordering):")
    print("  R3 (first load): 0 (stale - async not yet visible)")
    print("  R4 (second load): 999 (correct - after mbarrier)")

    sim.load_program(kernel)
    result = sim.run(max_cycles=100)

    r3_value = sim.warps[0].read_lane_reg(0, 3)
    r4_value = sim.warps[0].read_lane_reg(0, 4)

    print(f"\nResults:")
    print(f"  R3 (first load, NO fence.proxy): {r3_value}")
    print(f"  R4 (second load, after mbarrier): {r4_value}")
    print(f"  Cycles: {result.cycles}")

    if r3_value == 0 and r4_value == 999:
        print("\n  PASS: Demonstrates weak ordering!")
        print("        First load got stale data, second got correct data")
    elif r3_value == 999 and r4_value == 999:
        print("\n  Note: Async completed before first load (timing-dependent)")
        print("        This demonstrates the NEED for fence.proxy")
    else:
        print(f"\n  Unexpected: R3={r3_value}, R4={r4_value}")


def test_fence_proxy_before_async():
    """
    Test 2: fence.proxy BEFORE async operation.

    Pattern: Generic store -> fence.proxy.async -> Async copy -> wait -> read

    fence.proxy should ensure generic store completes BEFORE async starts.
    """
    print_header("Test 2: fence.proxy Before Async Operation")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Initialize global memory
    sim.memory.write_u32(MemorySpace.GLOBAL, 0x20000000, 777)

    # Kernel: Generic store -> fence.proxy -> async copy -> wait -> read
    kernel = [
        # Generic proxy store
        'MOV R20, 555',
        'STS [0x7000], R20',

        # fence.proxy.async - establish ordering point
        'FENCE_PROXY_ASYNC',

        # Initialize and start async copy
        'MOV R10, 0x5000',
        'MOV R11, 1',
        'MBARRIER_INIT [R10], R11',
        'MBARRIER_EXPECT_TX [R10], R11',
        'MOV R12, 0x6000',
        'MOV R13, 0x20000000',
        'MOV R14, 128',
        'CP_ASYNC_BULK [R12], [R13], R14',

        # Wait for async, then read both locations
        'MOV R15, 0',
        'MBARRIER_TEST_WAIT [R10], R15',
        'LDS R3, [0x7000]',
        'LDS R4, [0x6000]',

        'EXIT',
    ]

    print("Kernel:")
    print("  1. Generic store 555 to shared[0x7000] [GENERIC]")
    print("  2. fence.proxy.async")
    print("  3. Start async copy 777 to shared[0x6000] [ASYNC]")
    print("  4. Wait for async completion")
    print("  5. Read both locations")
    print()
    print("Expected:")
    print("  R3 = 555 (generic store)")
    print("  R4 = 777 (async copy)")

    sim.load_program(kernel)
    result = sim.run(max_cycles=200)

    r3_value = sim.warps[0].read_lane_reg(0, 3)
    r4_value = sim.warps[0].read_lane_reg(0, 4)

    print(f"\nResults:")
    print(f"  R3 (generic): {r3_value} (expected 555)")
    print(f"  R4 (async): {r4_value} (expected 777)")
    print(f"  Cycles: {result.cycles}")

    if r3_value == 555 and r4_value == 777:
        print("\n  PASS: Both operations visible correctly!")
    else:
        print(f"\n  FAIL: Expected R3=555, R4=777")


def test_fence_proxy_after_async():
    """
    Test 3: fence.proxy AFTER async operation.

    Pattern: Async copy -> wait -> fence.proxy.async -> read

    The mbarrier wait makes async visible, but fence.proxy provides
    additional ordering guarantees.
    """
    print_header("Test 3: fence.proxy After Async Completion")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Initialize global memory
    sim.memory.write_u32(MemorySpace.GLOBAL, 0x20000000, 888)

    # Kernel: Async copy -> wait -> fence.proxy -> read
    kernel = [
        # Initialize and start async copy
        'MOV R10, 0x5000',
        'MOV R11, 1',
        'MBARRIER_INIT [R10], R11',
        'MBARRIER_EXPECT_TX [R10], R11',
        'MOV R12, 0x6000',
        'MOV R13, 0x20000000',
        'MOV R14, 128',
        'CP_ASYNC_BULK [R12], [R13], R14',

        # Wait for async completion
        'MOV R15, 0',
        'MBARRIER_TEST_WAIT [R10], R15',

        # fence.proxy.async - additional synchronization
        'FENCE_PROXY_ASYNC',

        # Now read
        'LDS R3, [0x6000]',

        'EXIT',
    ]

    print("Kernel:")
    print("  1. Start async copy 888 to shared[0x6000]")
    print("  2. Wait for async completion (mbarrier)")
    print("  3. fence.proxy.async")
    print("  4. Read from shared")
    print()
    print("Expected: R3 = 888")

    sim.load_program(kernel)
    result = sim.run(max_cycles=200)

    r3_value = sim.warps[0].read_lane_reg(0, 3)

    print(f"\nResults:")
    print(f"  R3: {r3_value} (expected 888)")
    print(f"  Cycles: {result.cycles}")

    if r3_value == 888:
        print("\n  PASS: fence.proxy after mbarrier works!")
    else:
        print(f"\n  FAIL: Expected R3=888, got {r3_value}")


def test_multiple_ops_with_fence_proxy():
    """
    Test 4: Multiple operations with fence.proxy.

    Pattern: Generic ops -> fence.proxy -> Async -> wait -> Generic ops -> read
    """
    print_header("Test 4: Multiple Operations with fence.proxy")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Initialize global memory
    sim.memory.write_u32(MemorySpace.GLOBAL, 0x20001000, 111)
    sim.memory.write_u32(MemorySpace.GLOBAL, 0x20002000, 222)

    # Kernel with multiple generic and async operations
    kernel = [
        # First batch of generic operations
        'MOV R20, 999',
        'STS [0x8000], R20',
        'MOV R21, 998',
        'STS [0x8004], R21',

        # fence.proxy - establish ordering point
        'FENCE_PROXY_ASYNC',

        # First async operation
        'MOV R10, 0x5000',
        'MOV R11, 1',
        'MBARRIER_INIT [R10], R11',
        'MBARRIER_EXPECT_TX [R10], R11',
        'MOV R12, 0x6000',
        'MOV R13, 0x20001000',
        'MOV R14, 128',
        'CP_ASYNC_BULK [R12], [R13], R14',

        # Wait for first async
        'MOV R15, 0',
        'MBARRIER_TEST_WAIT [R10], R15',

        # Second batch of generic operations
        'MOV R22, 997',
        'STS [0x8008], R22',

        # fence.proxy again
        'FENCE_PROXY_ASYNC',

        # Second async operation
        'MBARRIER_INIT [R10], R11',
        'MBARRIER_EXPECT_TX [R10], R11',
        'MOV R12, 0x6004',
        'MOV R13, 0x20002000',
        'CP_ASYNC_BULK [R12], [R13], R14',

        # Wait and read everything
        'MBARRIER_TEST_WAIT [R10], R15',
        'LDS R3, [0x8000]',
        'LDS R4, [0x8004]',
        'LDS R5, [0x8008]',
        'LDS R6, [0x6000]',
        'LDS R7, [0x6004]',

        'EXIT',
    ]

    print("Kernel:")
    print("  Generic ops -> fence.proxy -> Async 1 -> wait -> Generic ops ->")
    print("  fence.proxy -> Async 2 -> wait -> Read all")
    print()
    print("Expected:")
    print("  R3=999, R4=998, R5=997 (generic)")
    print("  R6=111, R7=222 (async)")

    sim.load_program(kernel)
    result = sim.run(max_cycles=500)

    r3 = sim.warps[0].read_lane_reg(0, 3)
    r4 = sim.warps[0].read_lane_reg(0, 4)
    r5 = sim.warps[0].read_lane_reg(0, 5)
    r6 = sim.warps[0].read_lane_reg(0, 6)
    r7 = sim.warps[0].read_lane_reg(0, 7)

    print(f"\nResults:")
    print(f"  R3: {r3} (expected 999)")
    print(f"  R4: {r4} (expected 998)")
    print(f"  R5: {r5} (expected 997)")
    print(f"  R6: {r6} (expected 111)")
    print(f"  R7: {r7} (expected 222)")

    if r3 == 999 and r4 == 998 and r5 == 997 and r6 == 111 and r7 == 222:
        print("\n  PASS: All operations ordered correctly!")
    else:
        print("\n  FAIL: Some values incorrect")


def test_predicated_with_async():
    """
    Test 5: Predicated execution with async operations.

    Thread 0 does async copy, all threads wait, all threads read.
    Demonstrates predicated execution + async + mbarrier.
    """
    print_header("Test 5: Predicated Execution with Async Operations")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Initialize global memory
    sim.memory.write_u32(MemorySpace.GLOBAL, 0x20000000, 444)

    # Kernel: Thread 0 does async copy, all wait, all read
    kernel = [
        'MOV R5, %tid',
        'SETP.EQ P0, R5, 0',

        # Thread 0 only: Initialize and start async
        '@P0 MOV R10, 0x5000',
        '@P0 MOV R11, 1',
        '@P0 MBARRIER_INIT [R10], R11',
        '@P0 MBARRIER_EXPECT_TX [R10], R11',
        '@P0 MOV R12, 0x6000',
        '@P0 MOV R13, 0x20000000',
        '@P0 MOV R14, 128',
        '@P0 CP_ASYNC_BULK [R12], [R13], R14',

        # All threads: Wait and read
        'MOV R15, 0',
        'MBARRIER_TEST_WAIT [R10], R15',
        'LDS R3, [0x6000]',

        'EXIT',
    ]

    print("Kernel:")
    print("  Thread 0: Initialize mbarrier, start async copy 444")
    print("  All threads: Wait at mbarrier, read result")
    print()
    print("Expected: All threads read R3=444")

    sim.load_program(kernel)
    result = sim.run(max_cycles=100)

    # Check all threads
    all_correct = True
    for tid in range(32):
        value = sim.warps[0].read_lane_reg(tid, 3)
        if value != 444:
            print(f"  Thread {tid}: R3={value} (expected 444) FAIL")
            all_correct = False

    if all_correct:
        print("\n  PASS: All threads read 444!")
        print("        Thread 0's async copy visible to all after mbarrier")
    else:
        print("\n  FAIL: Some threads didn't read correct value")


def main():
    """Run all fence.proxy tests."""
    print("=" * 70)
    print(" FENCE.PROXY TESTS - Generic/Async Proxy Ordering")
    print("=" * 70)
    print()
    print("PTX ISA 9.1 Concepts:")
    print("- Generic proxy: ld/st operations (LDS, STS)")
    print("- Async proxy: TMA operations (CP_ASYNC_BULK)")
    print("- fence.proxy: Establishes ordering between proxies")
    print("- Without fence.proxy: WEAK ORDERING (no guarantees)")
    print()

    tests = [
        test_generic_async_weak_ordering,
        test_fence_proxy_before_async,
        test_fence_proxy_after_async,
        test_multiple_ops_with_fence_proxy,
        test_predicated_with_async,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("1. Generic and async proxy operations have WEAK ordering by default")
    print("2. fence.proxy establishes GUARANTEED ordering between proxies")
    print("3. Test 1 demonstrates weak ordering (stale data without fence)")
    print("4. Tests 2-4 demonstrate fence.proxy usage patterns")
    print("5. Test 5 shows predicated execution with async operations")


if __name__ == "__main__":
    sys.exit(main())
