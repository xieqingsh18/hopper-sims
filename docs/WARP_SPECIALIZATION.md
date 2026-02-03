# Warp Specialization in Hopper GPU Simulator

This document describes the warp specialization features implemented in the Hopper GPU Simulator, including TMA (Tensor Memory Accelerator), WGMMA (Warpgroup Matrix Multiply-Accumulate), and mbarrier operations.

## Overview

Warp specialization is a key feature of NVIDIA's Hopper architecture that enables dramatic performance improvements for GEMM and other matrix operations by:

1. **Separating producer and consumer warps** - Producer warps handle data movement while consumer warpgroups perform computation
2. **Using TMA for efficient data transfer** - Hardware-accelerated bulk transfers between global and shared memory
3. **Using WGMMA for matrix operations** - Warpgroup-level matrix multiply-accumulate on 128 threads
4. **Using mbarrier for synchronization** - Low-overhead barriers for asynchronous operations

## Implemented Operations

| Operation | Description | Status |
|-----------|-------------|--------|
| **TMA.LOAD** | Load matrix tile from global to shared memory | ✅ Implemented |
| **TMA.STORE** | Store matrix tile from shared to global memory | ✅ Implemented |
| **TMA.WAIT** | Wait for TMA operations to complete | ✅ Implemented |
| **WGMMA.MMA** | Warpgroup matrix multiply-accumulate | ✅ Implemented |
| **WGMMA.MMA_ASYNC** | Async warpgroup MMA | ✅ Implemented |
| **MBARRIER_INIT** | Initialize mbarrier with count | ✅ Implemented |
| **MBARRIER_INVAL** | Invalidate mbarrier | ✅ Implemented |
| **MBARRIER_ARRIVE** | Arrive at mbarrier (decrement) | ✅ Implemented |
| **MBARRIER_TEST_WAIT** | Test and wait for mbarrier | ✅ Implemented |
| **MBARRIER_EXPECT_TX** | Expect transaction count | ✅ Implemented |
| **MBARRIER_COMPLETE_TX** | Complete transaction | ✅ Implemented |

## Instruction Format

```
TMA.LOAD [shared_addr], [global_addr], tile_size
TMA.STORE [global_addr], [shared_addr], tile_size
TMA.WAIT barrier_id

WGMMA.MMA d, a, b, shape
WGMMA.MMA_ASYNC d, a, b

MBARRIER_INIT [mbarrier_addr], count
MBARRIER_ARRIVE [mbarrier_addr]
MBARRIER_TEST_WAIT [mbarrier_addr], barrier_id
MBARRIER_COMPLETE_TX [mbarrier_addr]
MBARRIER_INVAL [mbarrier_addr]
```

## TMA (Tensor Memory Accelerator)

### Overview

TMA provides hardware-accelerated bulk data transfer between global and shared memory. It's optimized for matrix tiles with strided access patterns.

### Key Features

- **Bulk transfer**: Move up to 256 bytes per instruction
- **Hardware address generation**: Handles strided and swizzled layouts
- **Asynchronous execution**: Doesn't block the warp
- **Cache control**: Optimized for shared memory bandwidth

### Usage Examples

```python
# Load 256-byte tile from global to shared
TMA.LOAD [R10], [R20], 256

# Store 256-byte tile from shared to global
TMA.STORE [R22], [R12], 256

# Wait for TMA completion
TMA.WAIT 0
```

### Performance Benefits

- **~10x reduction in load instructions** - One TMA.LOAD replaces hundreds of LDG instructions
- **Reduced register pressure** - No need for address calculation registers
- **Better shared memory utilization** - Hardware-managed transfers optimize bandwidth

## WGMMA (Warpgroup MMA)

### Overview

WGMMA performs matrix multiply-accumulate operations on warpgroups (128 threads = 4 warps). It's designed for efficient GEMM operations on matrix tiles stored in shared memory.

### Supported Shapes

| Shape | Description | Data Types |
|-------|-------------|------------|
| m64n8k16 | 64x8 matrix with 16 inner dimension | FP8, FP16, BF16, TF32 |
| m64n8k32 | 64x8 matrix with 32 inner dimension | FP8, FP16 |
| m64n8k64 | 64x8 matrix with 64 inner dimension | FP8 |
| m64n8k256 | 64x8 matrix with 256 inner dimension | FP8 (E4M3/E5M2) |

### Usage Examples

```python
# Synchronous WGMMA
WGMMA.MMA R50, R30, R40, 0  # shape 0 = m64n8k16

# Asynchronous WGMMA
WGMMA.MMA_ASYNC R50, R30, R40
```

### Performance Benefits

- **~2x TFLOPS vs traditional MMA** - Higher throughput through warpgroup execution
- **Efficient shared memory usage** - Operates directly on shared memory tiles
- **Asynchronous execution** - Overlap computation with data movement

## mbarrier (Memory Barrier)

### Overview

mbarrier is a low-overhead synchronization primitive for coordinating producer and consumer warps in warp-specialized kernels.

### Key Features

- **Transaction counting** - Tracks expected and completed transactions
- **Low overhead** - Faster than traditional BAR instructions
- **Async operation support** - Designed for TMA and WGMMA

### Usage Examples

```python
# Initialize mbarrier with expected transaction count
MBARRIER_INIT [R13], 2

# Producer signals completion
MBARRIER_ARRIVE [R13]

# Consumer waits for all transactions
MBARRIER_TEST_WAIT [R13], 0

# Complete and cleanup
MBARRIER_COMPLETE_TX [R13]
MBARRIER_INVAL [R13]
```

## Warp Specialized GEMM Pattern

### Producer-Consumer Model

```
┌─────────────────┐     TMA Load      ┌─────────────────┐
│  Producer Warp  │ ────────────────→ │  Shared Memory  │
│  (Load Tiles)   │                   │  (A, B tiles)   │
└─────────────────┘                   └────────┬────────┘
                                              │
                                              │ mbarrier
                                              │
                                              ▼
┌─────────────────┐                   ┌─────────────────┐
│  Shared Memory  │ ←──────────────── │Consumer Warpgroup│
│  (C tile)       │     WGMMA Store   │  (Compute)      │
└─────────────────┘                   └─────────────────┘
```

### Complete GEMM Example

```python
# Initialize mbarrier
MBARRIER_INIT [R13], 2

# Producer: Load A tile
TMA.LOAD [R10], [R20], 256
MBARRIER_ARRIVE [R13]

# Producer: Load B tile
TMA.LOAD [R11], [R21], 256
MBARRIER_ARRIVE [R13]

# Consumer: Wait for data
MBARRIER_TEST_WAIT [R13], 0

# Load fragments and compute
LDS R30, [R10+0]      # A fragment
LDS R40, [R11+0]      # B fragment
WGMMA.MMA_ASYNC R50, R30, R40

# Store result
TMA.STORE [R22], [R12], 256
MBARRIER_COMPLETE_TX [R13]
MBARRIER_INVAL [R13]
```

## Running the Demo

```bash
python3 examples/warp_specialization_demo.py
```

## Performance Comparison

| Approach | Cycles | TFLOPS | Efficiency |
|----------|--------|--------|------------|
| Traditional MMA | Baseline | 1x | 100% |
| Warp Specialized (WGMMA) | ~50% less | 2x | 200% |

## Implementation Notes

### Current Simulation Limitations

1. **Single-warp execution** - Simulator executes lane 0 only
   - Real WGMMA uses 128 threads (4 warps)
   - Simulation demonstrates semantics, not full parallelism

2. **Synchronous TMA** - TMA operations complete immediately
   - Real TMA is fully asynchronous
   - Simulation maintains logical correctness

3. **Simplified matrix layout** - Uses scalar values instead of tiles
   - Real WGMMA operates on distributed matrix fragments
   - Simulation shows the instruction flow

### Extending for Full Simulation

To implement full warpgroup simulation:

1. **Multi-warp execution model**
```python
for warpgroup_id in range(warpgroups_per_sm):
    for warp_id in warpgroup:
        # Execute instruction across all warps
        pass
```

2. **Distributed matrix storage**
```python
# Each thread holds part of the matrix fragment
for thread_id in range(128):
    fragment = compute_matrix_fragment(thread_id)
    register_file[thread_id][reg] = fragment
```

3. **Asynchronous operation tracking**
```python
class TMAQueue:
    def __init__(self):
        self.pending_operations = []

    def enqueue(self, operation):
        self.pending_operations.append(operation)

    def wait_for_completion(self):
        while self.pending_operations:
            self.process_next()
```

## Real-World Use Cases

These operations are essential for:

1. **GEMM Kernels** - General matrix multiply (CUTLASS, cuBLAS)
2. **Attention Mechanisms** - FlashAttention-2 for transformers
3. **Convolution** - Winograd and FFT-based convolutions
4. **Layer Normalization** - Fused operations with matrix reduction
5. **Custom Kernels** - AI/ML workloads requiring matrix operations

## References

- NVIDIA Hopper Architecture Whitepaper
- CUTLASS 3.x Documentation
- PTX ISA, Section 9.7.15 (WGMMA)
- PTX ISA, Section 9.7.13.15 (mbarrier)
- PTX ISA, Section 9.7.16 (Tensor Memory)
