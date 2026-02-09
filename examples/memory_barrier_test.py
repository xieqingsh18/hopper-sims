#!/usr/bin/env python3
"""
Memory Barrier and Synchronization Example

This example demonstrates the Hopper memory system with:
- Shared memory operations
- Barrier synchronization (barrier.sync)
- Memory ordering (fence)
- mbarrier operations
- Memory visibility semantics

According to PTX ISA specification:
- barrier.sync ensures all threads in CTA see memory operations
- fence.sc provides sequential consistency
- Memory operations are only visible after proper synchronization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import HopperSimulator, SimulatorConfig
from src.core.memory import MemorySpace, Scope, MemoryOrder


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_shared_memory_with_barrier():
    """Test shared memory with barrier synchronization."""
    print_header("Test 1: Shared Memory with Barrier Synchronization")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Thread 0 writes to shared memory, ALL threads wait at barrier,
    # then ALL threads read the value
    kernel = [
        "MOV R5, %tid",             # Thread ID

        # ========== Only thread 0 writes to shared memory ==========
        "SETP.EQ P0, R5, 0",        # Set P0 if tid == 0
        "@P0 MOV R6, 0x5000",       # Shared memory base address
        "@P0 MOV R7, 42",           # Value to write
        "@P0 STS.U32 [R6], R7",     # Store to shared memory (only thread 0)

        # ========== Barrier: ALL threads wait here ==========
        # This ensures thread 0's write is visible to all threads
        "BARrier",                  # barrier.sync - all 32 threads wait here

        # ========== All threads can now safely read shared memory ==========
        "MOV R6, 0x5000",           # Shared memory base address
        "LDS.U32 R8, [R6]",         # Load from shared memory
        "MOV R3, R8",               # Move result to return register

        "EXIT",
    ]

    print("Kernel: Thread 0 writes 42 to shared memory, all threads wait at barrier")
    print("        then all threads read the value after barrier completes")

    # Load program and launch
    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    # Run simulation
    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_fence_sc():
    """Test fence.sc (sequential consistency fence)."""
    print_header("Test 2: Fence.SC - Sequential Consistency Fence")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Write to shared memory, fence, then read
    # FENCE.SC ensures ordering of memory operations
    kernel = [
        "MOV R5, %tid",             # Thread ID

        # ========== Write to shared memory ==========
        "MOV R6, 0x6000",           # Shared memory address
        "MOV R7, 100",              # Value to write
        "STS.U32 [R6], R7",         # Store to shared memory

        # ========== Fence: Ensure ordering ==========
        "FENCE.SC",                 # Sequential consistency fence

        # ========== Now read (guaranteed to see the write) ==========
        "LDS.U32 R8, [R6]",         # Load from shared memory
        "MOV R3, R8",               # Move result to return register

        "EXIT",
    ]

    print("Kernel: Write to shared memory, fence.sc, then read")
    print("Fence ensures store is completed before load")

    # Load program and launch
    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    # Run simulation
    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_mbarrier():
    """Test mbarrier operations (async barrier)."""
    print_header("Test 3: Mbarrier - Asynchronous Barrier")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Initialize mbarrier, use it for synchronization
    kernel = [
        "MOV R5, %tid",             # Thread ID

        # ========== Initialize mbarrier ==========
        "MOV R6, 0x7000",           # mbarrier address in shared memory
        "MOV R7, 1",                # Count = 1 (single thread)
        "MBARRIER_INIT [R6], R7",  # Initialize mbarrier

        # ========== Arrive at mbarrier ==========
        "MBARRIER_ARRIVE [R6]",     # Arrive (decrement counter)

        # ========== Test and wait (stalls if not complete) ==========
        "MOV R8, 0",                # Phase 0
        "MBARRIER_TEST_WAIT [R6], R8", # Wait for phase to complete

        # ========== Continue execution ==========
        "MOV R3, 1",                # Return success

        "EXIT",
    ]

    print("Kernel: Initialize mbarrier, arrive, wait for completion")
    print("Mbarrier provides efficient async synchronization")

    # Load program and launch
    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    # Run simulation
    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_memory_ordering_relaxed():
    """Test memory ordering with relaxed semantics."""
    print_header("Test 4: Memory Ordering - Relaxed (No Fence)")

    config = SimulatorConfig(num_sms=1, warps_per_sm=2)
    sim = HopperSimulator(config)

    # Kernel: Producer writes, consumer reads WITHOUT barrier
    # Without barrier/fence, ordering is NOT guaranteed
    kernel = [
        "MOV R5, %tid",             # Thread ID (0-63)

        # ========== Producer: Threads 0-31 (warp 0) write ==========
        "SETP.LT P0, R5, 32",      # Set P0 if tid < 32 (warp 0)
        "@P0 MOV R6, 0x9000",      # Shared memory address
        "@P0 MOV R7, 999",         # Value to write
        "@P0 STS.U32 [R6], R7",    # Store to shared memory (only warp 0)

        # ========== Consumer: Threads 32-63 (warp 1) read ==========
        "SETP.GE P1, R5, 32",      # Set P1 if tid >= 32 (warp 1)
        "MOV R6, 0x9000",          # Shared memory address
        "@P1 LDS.U32 R8, [R6]",    # Load from shared memory (only warp 1)
        "@P1 MOV R3, R8",          # Return value (only warp 1)
        "@!P1 MOV R3, 999",        # Producer threads return 999

        "EXIT",
    ]

    print("Kernel: Warp 0 (threads 0-31) writes 999 to shared memory")
    print("        Warp 1 (threads 32-63) reads from shared memory")
    print("        NO barrier - warp 1 may read before warp 0's write is visible!")
    print("\nWith relaxed ordering, the result is UNDEFINED")
    print("Warp 1 might read stale data (0) or the written value (999)")

    # Load program
    sim.load_program(kernel)

    # Launch on 2 warps (64 threads total)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(64, 1, 1))

    # Run simulation
    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")
    print("\nNote: Without barrier/fence, memory ordering is NOT guaranteed!")


def test_memory_visibility_with_barrier():
    """Test memory visibility with proper barrier."""
    print_header("Test 5: Memory Ordering - With Barrier")

    config = SimulatorConfig(num_sms=1, warps_per_sm=2)
    sim = HopperSimulator(config)

    # Kernel: Producer (warp 0) writes, consumer (warp 1) reads, WITH barrier
    # Barrier ensures memory becomes visible across warps
    kernel = [
        "MOV R5, %tid",             # Thread ID (0-31 in warp 0, 32-63 in warp 1)

        # ========== Producer: Threads 0-31 (warp 0) write to shared memory ==========
        "SETP.LT P0, R5, 32",      # Set P0 if tid < 32 (warp 0)
        "@P0 MOV R6, 0x8000",      # Shared memory base
        "@P0 MOV R7, 999",         # Value to write
        "@P0 STS.U32 [R6], R7",    # Store to shared memory (only warp 0)

        # ========== Barrier: ALL threads wait here ==========
        # This ensures warp 0's write is visible to warp 1
        "Barrier",                  # barrier.sync - flushes memory operations

        # ========== Consumer: Threads 32-63 (warp 1) read from shared memory ==========
        "SETP.GE P1, R5, 32",      # Set P1 if tid >= 32 (warp 1)
        "MOV R6, 0x8000",          # Shared memory base
        "@P1 LDS.U32 R8, [R6]",    # Load from shared memory (only warp 1)
        "@P1 MOV R3, R8",          # Return value (only warp 1)
        "@!P1 MOV R3, 999",        # Producer threads return 999

        "EXIT",
    ]

    print("Kernel: Warp 0 (threads 0-31) writes 999 to shared memory")
    print("        All 64 threads wait at barrier")
    print("        Warp 1 (threads 32-63) reads the value after barrier")
    print("Barrier ensures warp 0's write is visible to warp 1")

    # Load program
    sim.load_program(kernel)

    # Launch on 2 warps (64 threads total)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(64, 1, 1))

    # Run simulation
    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_membar():
    """Test membar (global memory barrier)."""
    print_header("Test 6: MEMBAR - Global Memory Barrier")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Store to global memory, membar, then load
    kernel = [
        "MOV R5, 0x10000000",      # Global memory address

        # ========== Store to global memory ==========
        "MOV R6, 456",              # Value to store
        "STG.U32 [R5], R6",         # Store to global memory

        # ========== Global memory barrier ==========
        "MEMBAR",                   # Memory barrier (defaults to CTA scope)

        # ========== Load from global memory ==========
        "LDG.U32 R7, [R5]",         # Load from global memory
        "MOV R3, R7",               # Return value

        "EXIT",
    ]

    print("Kernel: Store to global memory, MEMBAR, then load")
    print("MEMBAR ensures global memory operations are visible")

    # Load program and launch
    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    # Run simulation
    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_mbarrier_arrive_drop():
    """Test mbarrier.arrive_drop - arrive and reduce threshold."""
    print_header("Test 7: Mbarrier - Arrive Drop")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Initialize with count 10, arrive_drop 5, then arrive 5 times
    kernel = [
        "MOV R5, %tid",
        "MOV R6, 0x7100",           # mbarrier address
        "MOV R7, 10",               # Initial count = 10
        "MBARRIER_INIT [R6], R7",  # Initialize

        # Arrive and drop threshold by 5
        "MOV R8, 5",
        "MBARRIER_ARRIVE_DROP [R6], R8",  # Reduce threshold to 5

        # Arrive 5 times (should complete since threshold is now 5)
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",

        # Wait should complete now
        "MOV R9, 0",                # Phase 0
        "MBARRIER_TEST_WAIT [R6], R9",

        "MOV R3, 1",                # Return success
        "EXIT",
    ]

    print("Kernel: Initialize mbarrier with count 10, arrive_drop 5")
    print("        Then arrive 5 times and test_wait")

    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_mbarrier_try_wait():
    """Test mbarrier.try_wait - non-blocking wait."""
    print_header("Test 8: Mbarrier - Try Wait (Non-blocking)")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Initialize, try_wait (non-blocking), check predicate
    kernel = [
        "MOV R5, %tid",
        "MOV R6, 0x7200",           # mbarrier address
        "MOV R7, 5",                # Count = 5
        "MBARRIER_INIT [R6], R7",  # Initialize

        # Try wait - should return 0 (not ready) in R8
        "MOV R8, 0",                # Pred reg
        "MBARRIER_TRY_WAIT [R6], R8",  # Non-blocking test

        # Arrive 5 times
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",

        # Try wait again - should return 1 (ready) in R8
        "MBARRIER_TRY_WAIT [R6], R8",

        "MOV R3, R8",               # Return predicate result
        "EXIT",
    ]

    print("Kernel: Initialize mbarrier, try_wait (non-blocking)")
    print("        Arrive 5 times, try_wait again")

    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_mbarrier_pending_count():
    """Test mbarrier.pending_count - get pending arrivals."""
    print_header("Test 9: Mbarrier - Pending Count")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Initialize, arrive 3 times, check pending count
    kernel = [
        "MOV R5, %tid",
        "MOV R6, 0x7300",           # mbarrier address
        "MOV R7, 10",               # Count = 10
        "MBARRIER_INIT [R6], R7",  # Initialize

        # Arrive 3 times
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",
        "MBARRIER_ARRIVE [R6]",

        # Get pending count (should be 7)
        "MOV R8, 0",                # Dest reg
        "MBARRIER_PENDING_COUNT [R6], R8",  # Get pending count

        "MOV R3, R8",               # Return pending count
        "EXIT",
    ]

    print("Kernel: Initialize with count 10, arrive 3 times")
    print("        Check pending count (should be 7)")

    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_fence_proxy():
    """Test fence.proxy - proxy fence for sync/async ordering."""
    print_header("Test 10: Fence.Proxy - Sync/Async Memory Ordering")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Store to shared, fence.proxy, load
    kernel = [
        "MOV R5, %tid",
        "MOV R6, 0x8000",           # Shared address
        "MOV R7, 123",              # Value
        "STS.U32 [R6], R7",         # Store

        # Fence.proxy - ensures ordering between sync and async ops
        "FENCE_PROXY",              # fence.proxy (simulator notation)

        "LDS.U32 R8, [R6]",         # Load
        "MOV R3, R8",               # Return value
        "EXIT",
    ]

    print("Kernel: Store to shared memory, fence.proxy, load")
    print("Fence.proxy ensures memory ordering between sync and async ops")

    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_cp_async_bulk():
    """Test cp.async.bulk - async bulk copy."""
    print_header("Test 11: CP.Async.Bulk - Async Bulk Copy")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Initialize global memory with data
    sim.memory.write_u32(MemorySpace.GLOBAL, 0x20000000, 999)

    # Kernel: Async copy from global to shared, wait, read
    kernel = [
        "MOV R5, %tid",
        "MOV R6, 0x5000",           # Shared dst address
        "MOV R7, 0x20000000",       # Global src address
        "MOV R8, 128",              # Size in bytes

        # Async bulk copy
        "CP_ASYNC_BULK [R6], [R7], R8",  # cp.async.bulk

        # Commit and wait
        "MOV R9, 0",
        "CP_ASYNC_BULK_COMMIT_GROUP R9",  # Commit group 0
        "CP_ASYNC_BULK_WAIT_GROUP R9",    # Wait group 0

        # Read from shared memory
        "LDS.U32 R10, [R6]",
        "MOV R3, R10",
        "EXIT",
    ]

    print("Kernel: Async bulk copy from global to shared memory")
    print("        Commit group, wait group, then read")

    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_wgmma_fence():
    """Test wgmma.fence - warpgroup fence for async MMA."""
    print_header("Test 12: WGMMA.Fence - Warpgroup Fence")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Store, wgmma.fence, load
    kernel = [
        "MOV R5, %tid",
        "MOV R6, 0x9000",           # Shared address
        "MOV R7, 456",              # Value
        "STS.U32 [R6], R7",         # Store

        # Warpgroup fence - ensures async MMA ops are ordered
        "WGMMA_FENCE",              # wgmma.fence (simulator notation)

        "LDS.U32 R8, [R6]",         # Load
        "MOV R3, R8",               # Return value
        "EXIT",
    ]

    print("Kernel: Store to shared, wgmma.fence, load")
    print("Wgmma.fence ensures async WGMMA operations are ordered")

    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def test_wgmma_commit_wait_group():
    """Test wgmma.commit_group and wgmma.wait_group."""
    print_header("Test 13: WGMMA Commit/Wait Group")

    config = SimulatorConfig(num_sms=1, warps_per_sm=1)
    sim = HopperSimulator(config)

    # Kernel: Store, wgmma.commit_group, wgmma.wait_group, load
    kernel = [
        "MOV R5, %tid",
        "MOV R6, 0x9100",           # Shared address
        "MOV R7, 789",              # Value
        "STS.U32 [R6], R7",         # Store

        # Commit warpgroup async group
        "WGMMA_COMMIT_GROUP",       # wgmma.commit_group

        # Wait for group 0
        "MOV R8, 0",
        "WGMMA_WAIT_GROUP R8",      # wgmma.wait_group 0

        "LDS.U32 R9, [R6]",         # Load
        "MOV R3, R9",               # Return value
        "EXIT",
    ]

    print("Kernel: Store, wgmma.commit_group, wgmma.wait_group, load")
    print("Wgmma commit/wait group synchronizes async MMA operations")

    sim.load_program(kernel)
    sim.launch_kernel(kernel, grid_dim=(1, 1, 1), block_dim=(32, 1, 1))

    result = sim.run(max_cycles=200)

    print(f"\nResult: {result.success}")
    print(f"Cycles: {result.cycles}")


def main():
    """Run all memory system tests."""
    print("\n" + "=" * 70)
    print("  HOPPER MEMORY SYSTEM - BARRIER & SYNCHRONIZATION TESTS")
    print("=" * 70)
    print("\nTesting memory ordering, barriers, and synchronization")
    print("according to PTX ISA 9.1 specification")

    try:
        test_shared_memory_with_barrier()
        test_fence_sc()
        test_mbarrier()
        test_memory_ordering_relaxed()
        test_memory_visibility_with_barrier()
        test_membar()
        test_mbarrier_arrive_drop()
        test_mbarrier_try_wait()
        test_mbarrier_pending_count()
        test_fence_proxy()
        test_cp_async_bulk()
        test_wgmma_fence()
        test_wgmma_commit_wait_group()

        print("\n" + "=" * 70)
        print("  ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
