# Hopper GPU Simulator

An instruction-level simulator for NVIDIA Hopper (H100) GPU architecture, focused on Tensor Core operations.

## Architecture Overview

```
hopper/
├── src/
│   ├── __init__.py
│   ├── core/              # Core data structures
│   │   ├── thread.py      # Thread and warp definitions
│   │   ├── register.py    # Register file implementation
│   │   └── memory.py      # Memory model
│   ├── isa/               # Instruction Set Architecture
│   │   ├── decoder.py     # SASS instruction decoder
│   │   ├── instructions.py # Instruction definitions
│   │   └── tensor.py      # Tensor Core instructions (HMMA)
│   ├── executor/          # Execution engine
│   │   ├── warp.py        # Warp execution engine
│   │   └── pipeline.py    # Execution pipeline
│   └── simulator.py       # Main simulator interface
├── tests/                 # Test programs
├── examples/              # Example kernels
└── cli.py                 # Command-line interface
```

## Design Principles

1. **Modular Design**: Each component is independent and can be tested separately
2. **Incremental**: Build and test piece by piece
3. **Educational**: Clear code structure to understand GPU architecture

## Key Components

### 1. Core Data Structures
- **Thread**: Individual GPU thread with PC, registers, state
- **Warp**: Group of 32 threads executing in lockstep
- **RegisterFile**: Per-thread register storage (255 registers per thread)
- **Memory**: Global, shared, and local memory spaces

### 2. ISA Support
- SASS instruction decoding (Hopper architecture)
- Basic arithmetic/logic instructions
- **Tensor Core**: HMMA.8816 (FP8 matrix multiply)

### 3. Execution Model
- SIMT (Single Instruction Multiple Thread) semantics
- Warp divergence handling
- Memory coalescing

## Hopper Architecture Key Facts

- **SM (Streaming Multiprocessor)**: 132 SMs per GPU
- **Warp Size**: 32 threads
- **Max Warps per SM**: 64
- **Max Threads per SM**: 2048
- **Registers per SM**: 65536 x 32-bit
- **Shared Memory per SM**: 228 KB
- **Tensor Cores**: 4th Gen Tensor Cores per SM
- **Clock**: ~1.8 GHz

## Tensor Core (Hopper)

Hopper introduces FP8 (8-bit floating point) support:
- **HMMA.8816**: Multiplies two FP8 matrices, accumulates to FP16/FP32
- Input: 16x16 FP8 matrix A × 16x16 FP8 matrix B
- Output: 16x16 FP16/FP32 matrix C
- Throughput: 4x improvement over Ampere

## Usage

```python
from src.simulator import HopperSimulator

# Create simulator instance
sim = HopperSimulator(
    num_sms=1,
    warps_per_sm=4,
    threads_per_warp=32
)

# Load and execute kernel
sim.load_kernel('kernel.sass')
sim.run()
```
