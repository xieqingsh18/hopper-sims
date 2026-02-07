#!/usr/bin/env python3
"""
Producer-Consumer GEMM using Warp Specialization

This program demonstrates Hopper's warp specialization pattern for GEMM (General Matrix Multiply).

Key Concepts:
- Producer Warps: Use TMA (Tensor Memory Accelerator) to move data from global to shared memory
- Consumer Warps: Use WGMMA (Warpgroup MMA) to perform matrix multiply-accumulate
- mbarrier: Synchronization between producers and consumers

Architecture:
- 4 Warps total (128 threads = 1 warpgroup)
- Warps 0-1: Producers (handle data movement with TMA)
- Warps 2-3: Consumers (handle computation with WGMMA)

Memory Layout:
- Global Memory: Input matrices A, B, C and output D
- Shared Memory: Tiles for computation
- mbarrier: Synchronization primitives
"""

from src.simulator import HopperSimulator, SimulatorConfig
from src.isa.decoder import InstructionDecoder, parse_program
from src.core.memory import MemorySpace
from src.isa.suffixes import SuffixParser


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def initialize_matrices(sim: HopperSimulator, M: int, N: int, K: int):
    """
    Initialize matrices A, B, C in global memory for GEMM: D = A * B + C

    Args:
        sim: The simulator instance
        M: Rows of A and D
        N: Columns of B and D
        K: Columns of A and rows of B (shared dimension)
    """
    print_header("Matrix Initialization")
    print(f"GEMM: D = A * B + C")
    print(f"  Matrix A: {M}x{K}")
    print(f"  Matrix B: {K}x{N}")
    print(f"  Matrix C: {M}x{N}")
    print(f"  Matrix D: {M}x{N} (output)")

    # Allocate global memory addresses
    A_addr = 0x100000
    B_addr = 0x200000
    C_addr = 0x300000
    D_addr = 0x400000

    # Initialize Matrix A (M x K) - Row-major
    print(f"\nInitializing Matrix A at 0x{A_addr:x}:")
    for i in range(M):
        for j in range(K):
            # Use row-major: A[i][j] = i * K + j + 1
            value = i * K + j + 1
            offset = (i * K + j) * 4
            sim.memory.write_u32(MemorySpace.GLOBAL, A_addr + offset, value)

    # Initialize Matrix B (K x N) - Row-major
    print(f"Initializing Matrix B at 0x{B_addr:x}:")
    for i in range(K):
        for j in range(N):
            # B[i][j] = i * N + j + 1
            value = i * N + j + 1
            offset = (i * N + j) * 4
            sim.memory.write_u32(MemorySpace.GLOBAL, B_addr + offset, value)

    # Initialize Matrix C (M x N) - Row-major
    print(f"Initializing Matrix C at 0x{C_addr:x}:")
    for i in range(M):
        for j in range(N):
            # C[i][j] = 100 (constant for verification)
            value = 100
            offset = (i * N + j) * 4
            sim.memory.write_u32(MemorySpace.GLOBAL, C_addr + offset, value)

    print(f"\nMatrix A sample (first 4x4):")
    for i in range(min(4, M)):
        row = []
        for j in range(min(4, K)):
            offset = (i * K + j) * 4
            val = sim.memory.read_u32(MemorySpace.GLOBAL, A_addr + offset)
            row.append(f"{val:4d}")
        print(f"  {row}")

    print(f"\nMatrix B sample (first 4x4):")
    for i in range(min(4, K)):
        row = []
        for j in range(min(4, N)):
            offset = (i * N + j) * 4
            val = sim.memory.read_u32(MemorySpace.GLOBAL, B_addr + offset)
            row.append(f"{val:4d}")
        print(f"  {row}")

    return A_addr, B_addr, C_addr, D_addr


def producer_warp_code():
    """
    Producer warp code using TMA to load tiles from global to shared memory.

    The producer warps:
    1. Initialize mbarrier for synchronization
    2. Use TMA.LOAD to copy tiles from global memory to shared memory
    3. Signal consumer warps via mbarrier

    Key instructions:
    - mbarrier.init.shared::cta.b64: Initialize mbarrier
    - TMA.LOAD: Bulk copy from global to shared memory
    - mbarrier.arrive.shared: Signal completion
    """
    return [
        # ========== PRODUCER WARP CODE ==========
        # Initialize mbarrier for producer-consumer synchronization
        # mbarrier.init.shared::cta.b64 [mbarrier_addr], thread_count
        "mbarrier.init.shared::cta.b64 [R10], 128",  # 128 threads in warpgroup

        # Set up addresses for matrix tiles
        "MOV R0, 0x100000",  # Base address of matrix A in global memory
        "MOV R1, 0x200000",  # Base address of matrix B in global memory
        "MOV R2, 0x300000",  # Base address of matrix C in global memory
        "MOV R3, 0x000000",  # Base address of shared memory for tiles

        # ========== TMA.LOAD: Load Matrix A Tile ==========
        # TMA.LOAD copies a 64x32 tile from global to shared memory
        # This is asynchronous and handled by the TMA hardware
        "TMA.LOAD [R3 + 0x0000], [R0 + 0x0000], 4096",  # Load A tile (64x32 = 2048 elements = 8KB)
        "TMA.WAIT 0",  # Wait for TMA to complete

        # ========== TMA.LOAD: Load Matrix B Tile ==========
        "TMA.LOAD [R3 + 0x2000], [R1 + 0x0000], 4096",  # Load B tile
        "TMA.WAIT 0",

        # ========== TMA.LOAD: Load Matrix C Tile ==========
        "TMA.LOAD [R3 + 0x4000], [R2 + 0x0000], 4096",  # Load C tile
        "TMA.WAIT 0",

        # Signal consumers that data is ready
        # Decrement mbarrier counter to signal completion
        "mbarrier.arrive.shared [R10]",

        # Producer warp can now continue or exit
        "EXIT",
    ]


def consumer_warp_code():
    """
    Consumer warp code using WGMMA to perform matrix multiplication.

    The consumer warps:
    1. Wait for producer to load data via mbarrier
    2. Use WGMMA to perform matrix multiply-accumulate on tiles in shared memory
    3. Write results back or perform next iteration

    Key instructions:
    - mbarrier.test_wait.shared: Wait for producer signal
    - WGMMA.MMA: Warpgroup matrix multiply-accumulate
    - ldmatrix.sync.aligned: Load matrix fragments from shared memory
    - stmatrix: Store results
    """
    return [
        # ========== CONSUMER WARP CODE ==========
        # Wait for producer to load data
        # mbarrier.test_wait blocks until mbarrier counter reaches 0
        "mbarrier.test_wait.shared [R10], 0",

        # Set up shared memory addresses for tiles
        "MOV R4, 0x000000",  # Address of A tile in shared memory
        "MOV R5, 0x002000",  # Address of B tile in shared memory
        "MOV R6, 0x004000",  # Address of C tile in shared memory

        # Load matrix fragments from shared memory to registers
        # ldmatrix.sync.aligned loads a 8x8 matrix fragment distributed across warp
        "ldmatrix.sync.aligned.m8n8.x1.b16 {R16, R17, R18, R19}, [R4]",  # Load A fragment
        "ldmatrix.sync.aligned.m8n8.x1.b16 {R20, R21, R22, R23}, [R5]",  # Load B fragment
        "ldmatrix.sync.aligned.m8n8.x1.b16 {R24, R25, R26, R27}, [R6]",  # Load C fragment

        # Perform matrix multiply-accumulate
        # WGMMA.MMA: D = A * B + C
        # Shape: m64n8k16 (standard for Hopper WGMMA)
        # This operates on 128 threads (4 warps = 1 warpgroup)
        "WGMMA.MMA R8, R16, R20, R24",  # D = A * B + C

        # In a full implementation, we would:
        # 1. Accumulate across multiple K tiles
        # 2. Store results to global memory
        # 3. Handle edge cases and boundary conditions

        # Store result fragment back to shared memory (or global)
        # stmatrix.sync.aligned.m8n8.b16 [R7], {R8, R9, R10, R11}

        "EXIT",
    ]


def run_producer_consumer_gemm():
    """
    Run the producer-consumer GEMM demonstration.

    This demonstrates:
    1. Producer warps using TMA for efficient data movement
    2. Consumer warps using WGMMA for matrix computation
    3. mbarrier synchronization between producers and consumers
    """
    print_header("PRODUCER-CONSUMER GEMM DEMONSTRATION")
    print("\nThis demo shows Hopper's warp specialization pattern:")
    print("  - Producer Warps (0-1): Use TMA to move data (Global -> Shared)")
    print("  - Consumer Warps (2-3): Use WGMMA to compute (Matrix Multiply)")
    print("  - mbarrier: Synchronization between producers and consumers")

    # Create simulator with 4 warps (1 warpgroup = 128 threads)
    config = SimulatorConfig(num_sms=1, warps_per_sm=4)
    sim = HopperSimulator(config)

    # Initialize matrices in global memory
    # Using small dimensions for demonstration: 64x64x64 GEMM
    M, N, K = 64, 64, 64
    A_addr, B_addr, C_addr, D_addr = initialize_matrices(sim, M, N, K)

    print_header("Producer Warps (Warps 0-1)")
    print("\nLoading tiles from global memory to shared memory using TMA...")

    producer_code = producer_warp_code()
    print("\nProducer code:")
    for i, line in enumerate(producer_code):
        print(f"  {i:02d}: {line}")

    # Execute producer warps
    print("\n--- Executing Producer Warps ---")
    producer_instructions = parse_program(producer_code)

    # Warp 0 and 1 are producers
    sim.load_program(producer_code, warp_id=0)
    sim.load_program(producer_code, warp_id=1)

    # Run producer warps
    sim.run(max_cycles=1000)

    print_header("Consumer Warps (Warps 2-3)")
    print("\nPerforming matrix multiplication using WGMMA...")

    consumer_code = consumer_warp_code()
    print("\nConsumer code:")
    for i, line in enumerate(consumer_code):
        print(f"  {i:02d}: {line}")

    # Execute consumer warps
    print("\n--- Executing Consumer Warps ---")

    # Warp 2 and 3 are consumers
    sim.load_program(consumer_code, warp_id=2)
    sim.load_program(consumer_code, warp_id=3)

    # Run all warps together (producers and consumers run concurrently)
    print("\n--- Running All Warps (Producers and Consumers) ---")
    result = sim.run(max_cycles=1000)

    print_header("Execution Results")

    # Display warp statistics
    print("\nProducer Warp Statistics:")
    for warp_id in [0, 1]:
        if warp_id in result.warp_stats:
            stats = result.warp_stats[warp_id]
            print(f"  Warp {warp_id}: {stats.get('instructions_executed', 0)} instructions executed")

    print("\nConsumer Warp Statistics:")
    for warp_id in [2, 3]:
        if warp_id in result.warp_stats:
            stats = result.warp_stats[warp_id]
            print(f"  Warp {warp_id}: {stats.get('instructions_executed', 0)} instructions executed")

    # Verify shared memory contents
    print_header("Shared Memory Contents")
    print("\nAfter TMA transfer, shared memory contains matrix tiles:")

    print("\nMatrix A tile (first 16 elements):")
    for i in range(4):
        row = []
        for j in range(4):
            offset = (i * 32 + j) * 4  # 32 columns per row in tile
            val = sim.memory.read_u32(MemorySpace.SHARED, offset)
            row.append(f"{val:4d}")
        print(f"  {row}")

    print("\nMatrix B tile (first 16 elements at offset 0x2000):")
    for i in range(4):
        row = []
        for j in range(4):
            offset = 0x2000 + (i * 32 + j) * 4
            val = sim.memory.read_u32(MemorySpace.SHARED, offset)
            row.append(f"{val:4d}")
        print(f"  {row}")

    print_header("Producer-Consumer Pattern Benefits")
    print("""
The producer-consumer pattern with warp specialization provides:

1. **Latency Hiding**: Producers move data while consumers compute
   - Overlaps data movement with computation
   - Hides global memory latency

2. **Improved Register Usage**: Each warp type uses registers efficiently
   - Producers: Focus on address calculation and TMA setup
   - Consumers: Focus on matrix data and accumulation

3. **Reduced Synchronization Overhead**: mbarrier is lightweight
   - No explicit barriers between warps
   - Hardware-accelerated synchronization

4. **Better Utilization**: Each warp executes its specialized role
   - TMA hardware handles bulk transfers
   - WGMMA hardware handles matrix operations

5. **Energy Efficiency**: Specialized hardware consumes less power
   - TMA uses less energy than explicit loads
   - WGMMA is more efficient than scalar operations
    """)

    print_header("Summary")
    print("""
Producer-Consumer GEMM Execution Complete!

Warp Specialization:
  - Producer Warps (0-1): TMA.LOAD for data movement
  - Consumer Warps (2-3): WGMMA.MMA for computation
  - Synchronization: mbarrier for producer-consumer coordination

This pattern is key to achieving high performance on Hopper architecture.
Real implementations would handle:
  - Multiple K tiles (looping over K dimension)
  - Strided access patterns
  - Boundary conditions
  - Result writeback to global memory
    """)


def main():
    """Run the producer-consumer GEMM demonstration."""
    run_producer_consumer_gemm()


if __name__ == "__main__":
    main()
