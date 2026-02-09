# Special Registers Support - CUDA-Style Kernel Launch

## Overview

The Hopper GPU Simulator now supports CUDA-style kernel launching with special registers (`%tid`, `%ctaid`, etc.), making it work much more like real CUDA.

## What's New

### 1. Special Registers

The simulator now supports PTX special registers that are automatically set by hardware:

| Special Register | Description | CUDA Equivalent |
|-----------------|-------------|------------------|
| `%tid` | Thread index within block | `threadIdx` |
| `%ctaid` | CTA (block) index within grid | `blockIdx` |
| `%ntid` | Number of threads in CTA | `blockDim` |
| `%nctaid` | Number of CTAs in grid | `gridDim` |
| `%laneid` | Lane ID within warp (0-31) | N/A |
| `%warpid` | Warp ID within CTA | N/A |
| `%nwarpid` | Number of warps in CTA | N/A |
| `%smid` | SM ID | N/A |
| `%nsmid` | Number of SMs | N/A |
| `%gridid` | Unique grid ID | N/A |

### 2. CUDA-Style Launch API

```python
sim.launch_kernel(
    program=kernel_code,
    grid_dim=(2, 1, 1),    # 2 blocks
    block_dim=(32, 1, 1)   # 32 threads per block
)
```

This mimics CUDA's:
```cuda
my_kernel<<<2, 32>>>(args...);
```

### 3. Assembly Syntax

You can now use special registers directly in assembly:

```assembly
MOV R5, %tid          # Read thread ID
MOV R6, %ctaid        # Read block ID
IMUL.U32 R7, R6, 32   # Compute block offset
IADD R5, R7, R5       # global_tid = blockIdx.x * 32 + threadIdx.x
```

## Example: Pointwise Vector Addition

### Real CUDA Kernel
```cuda
__global__ void vector_add(int *A, int *B, int *C, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    if (global_tid < n) {
        C[global_tid] = A[global_tid] + B[global_tid];
    }
}

// Launch
vector_add<<<2, 32>>>(A, B, C, 64);
```

### Simulator Assembly
```python
kernel = [
    "MOV R5, %tid",         # threadIdx.x
    "MOV R6, %ctaid",       # blockIdx.x
    "IMUL.U32 R7, R6, 32",  # bid * 32
    "IADD R5, R7, R5",      # global_tid
    "MOV R1, A_addr",
    "IMUL.U32 R3, R5, 4",   # global_tid * 4
    "IADD R4, R1, R3",      # &A[global_tid]
    "LDG.U32 R8, [R4]",     # Load A
    "LDG.U32 R9, [R7]",     # Load B
    "IADD R10, R8, R9",     # Add
    "STG.U32 [R12], R10",   # Store C
    "EXIT",
]

# Launch
sim.launch_kernel(kernel, grid_dim=(2, 1, 1), block_dim=(32, 1, 1))
```

## How It Works

### 1. Thread Initialization

When you call `launch_kernel()`, the simulator:

1. Calculates total blocks and threads needed
2. Loads the kernel program on all participating warps
3. Initializes special registers for each thread based on its position

For `launch_kernel(kernel, grid_dim=(2,1,1), block_dim=(32,1,1))`:

```
Block 0 (ctaid=0):
  Thread 0:  tid=0,  ntid=32, nctaid=2
  Thread 1:  tid=1,  ntid=32, nctaid=2
  ...
  Thread 31: tid=31, ntid=32, nctaid=2

Block 1 (ctaid=1):
  Thread 0:  tid=0,  ntid=32, nctaid=2
  Thread 1:  tid=1,  ntid=32, nctaid=2
  ...
  Thread 31: tid=31, ntid=32, nctaid=2
```

### 2. Special Register Access

The MOV instruction was extended to read from special registers:

```python
# In warp.py executor
def _exec_mov(self, instr: Instruction) -> None:
    ...
    elif src.type == OperandType.SPECIAL_REGISTER:
        special_reg = src.value  # SpecialRegister enum
        for lane_id in self.warp.get_executing_lane_ids():
            thread = self.warp.get_thread(lane_id)
            val = thread.read_special_reg(special_reg)
            self.warp.write_lane_reg(lane_id, dst, val)
```

### 3. Global Thread ID Calculation

Threads typically need a global ID to access array elements:

```assembly
# global_tid = blockIdx.x * blockDim.x + threadIdx.x
MOV R5, %tid             # R5 = threadIdx.x (0-31)
MOV R6, %ctaid           # R6 = blockIdx.x (0 or 1)
IMUL.U32 R7, R6, 32     # R7 = blockIdx.x * blockDim.x
IADD R5, R7, R5         # R5 = global_tid (0-63)
```

## Comparison: Before vs After

### Before (Manual Thread ID)
```python
# Had to manually set thread IDs in regular registers
for warp_id in range(2):
    for lane_id in range(32):
        global_tid = warp_id * 32 + lane_id
        sim.warps[warp_id].write_lane_reg(lane_id, 10, global_tid)

# Kernel used regular register
kernel = ["MOV R5, R10", ...]  # R10 manually set
```

### After (Automatic Special Registers)
```python
# Automatic initialization with launch API
sim.launch_kernel(kernel, grid_dim=(2,1,1), block_dim=(32,1,1))

# Kernel uses special register
kernel = ["MOV R5, %tid", ...]  # %tid is hardware-provided
```

## Files Modified

1. **src/core/thread.py**
   - Added `SpecialRegister` enum
   - Added special register storage
   - Added `init_special_registers()` method
   - Added properties for accessing special registers

2. **src/isa/decoder.py**
   - Added special register parsing (`%tid`, `%ctaid`, etc.)
   - Extended `_parse_generic_operand()` to recognize `%` prefix

3. **src/executor/warp.py**
   - Extended `_exec_mov()` to handle `SPECIAL_REGISTER` type

4. **src/simulator.py**
   - Added `launch_kernel()` method with CUDA-style API

5. **examples/pointwise_kernel_special_regs.py**
   - New example demonstrating special registers

## Usage

```python
from src.simulator import HopperSimulator, SimulatorConfig

# Create simulator
config = SimulatorConfig(num_sms=1, warps_per_sm=4)
sim = HopperSimulator(config)

# Define kernel using special registers
kernel = [
    "MOV R5, %tid",         # Get thread ID from hardware
    "MOV R6, %ctaid",       # Get block ID from hardware
    ...
]

# Launch kernel with CUDA-style configuration
sim.launch_kernel(
    program=kernel,
    grid_dim=(2, 1, 1),     # 2 blocks
    block_dim=(32, 1, 1)    # 32 threads per block
)

# Run simulation
result = sim.run(max_cycles=500)
```

## Benefits

1. **More Realistic**: Works like real CUDA with hardware-provided thread IDs
2. **Cleaner Code**: No manual thread ID setup required
3. **PTX Compatible**: Assembly looks more like real PTX code
4. **Easier Debugging**: Special registers are visible and queryable
5. **Scalable**: Easy to change grid/block dimensions without code changes

## Limitations

- Only supports 1D grid/block (x-dimension) for now
- Y and Z dimensions are parsed but not fully utilized
- Complex multi-dimensional indexing requires manual calculation

## Future Enhancements

- Full 3D grid/block support
- Predicate registers for branch handling
- More special registers (`%clock`, `%pm0`, etc.)
- Built-in functions (`__syncthreads()`, atomic operations)
