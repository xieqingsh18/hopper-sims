# Warp Synchronization in Hopper GPU Simulator

This document describes the warp-level synchronization and communication operations implemented in the Hopper GPU Simulator.

## Implemented Operations

| Operation | Description | Status |
|-----------|-------------|--------|
| **ACTIVEMASK** | Returns 32-bit mask of active lanes | ✅ Implemented |
| **ELECT** | Elects one thread (lowest active lane) | ✅ Implemented |
| **VOTE** | Combines predicate values across warp | ✅ Implemented |
| **VOTE.SYNC** | Synchronized vote across warp | ✅ Implemented |
| **SHFL** | Shuffle data between lanes | ✅ Implemented |
| **REDUX** | Reduction operation across warp | ✅ Implemented |
| **BAR** | Barrier synchronization | ✅ Implemented |
| **BAR.WARP** | Warp-level barrier | ✅ Implemented |
| **MEMBAR** | Memory fence | ✅ Implemented |
| **SETP/PSETP** | Set predicate from comparison | ✅ Implemented |
| **SELP** | Select based on predicate | ✅ Implemented |

## Instruction Format

```
ACTIVEMASK Rd           // Rd = active lane mask
ELECT Rd                 // Rd = 1 if elected, 0 otherwise
VOTE Rd, Ps              // Rd = mask of lanes where Ps is true
SHFL Rd, Ra, Rb, imm     // Shuffle: Rd = Ra[lane + imm]
REDUX Rd, Ra, imm        // Reduction across warp
BAR imm                  // Barrier ID
BAR.WARP                 // Warp-level barrier
MEMBAR imm               // Memory fence
SETP Pd, Ra, Rb          // Pd = (Ra cmp Rb)
SELP Rd, Ra, Rb, Ps      // Rd = Ra if Ps else Rb
@Pd instruction          // Execute if Pd is true
@!Pd instruction         // Execute if Pd is false
```

## Usage Examples

### 1. Get Active Lane Mask
```python
program = [
    "ACTIVEMASK R1",
    "EXIT",
]
```

### 2. Elect One Thread
```python
program = [
    "ELECT R1",
    "EXIT",
]
# R1 = 1 for elected thread (lane 0), 0 for others
```

### 3. Vote Across Warp
```python
program = [
    "SETP P0, R0, R1",    # Set predicate
    "VOTE R2, P0",        # Vote on P0
    "EXIT",
]
# R2 = mask of lanes where P0 is true
```

### 4. Shuffle Data
```python
program = [
    "SHFL R0, R1, R2, 1",  # R0 = R1 from lane (2+1)
    "SHFL R0, R1, R2, 0",  # Broadcast from lane 2
    "EXIT",
]
```

### 5. Barrier Synchronization
```python
program = [
    "BAR 0",              # Synchronize at barrier 0
    "MOV R1, 42",
    "EXIT",
]
```

### 6. Parallel Reduction Pattern
```python
program = [
    # Setup values in each lane
    "MOV R10, lane_value",

    # Shuffle-add reduction
    "SHFL R11, R10, R2, 1",
    "IADD R10, R10, R11",

    "SHFL R11, R10, R2, 2",
    "IADD R10, R10, R11",

    # Continue with offsets 4, 8, 16
    "SHFL R20, R10, R2, 0",  # Broadcast result
    "EXIT",
]
```

### 7. Predicated Execution
```python
program = [
    "SETP P0, R0, R1",     # P0 = (R0 < R1)
    "@P0 MOV R2, 100",     # Execute only if P0 is true
    "@!P0 MOV R2, 200",    # Execute only if P0 is false
    "EXIT",
]
```

### 8. Select Based on Predicate
```python
program = [
    "SETP P0, R0, R1",
    "SELP R2, R10, R20, P0",  # R2 = R10 if P0 else R20
    "EXIT",
]
```

## Running the Demo

```bash
python3 examples/warp_sync_demo.py
```

## Implementation Notes

### Single-Lane Limitation

The current simulator executes instructions for lane 0. This means:
- ✅ Operations that work on single lanes work correctly
- ✅ ACTIVEMASK shows all lanes as active
- ✅ ELECT correctly elects lane 0
- ⚠️ SHFL reads from other lanes (which have default values)
- ⚠️ REDUX shows simplified reduction
- ⚠️ Multi-lane patterns don't show full effect

### Full Multi-Lane Simulation

To demonstrate full shuffle/reduction:
1. Each lane needs unique values
2. All 32 lanes must execute simultaneously
3. Lane-to-lane communication must access correct lane's registers

### Extending for Full Warp Simulation

To extend for full warp simulation:

1. **Initialize all lanes with unique values:**
```python
for lane_id in range(32):
    warp.write_lane_reg(lane_id, reg_idx, lane_id * value)
```

2. **Enable multi-lane execution:**
```python
for lane_id in warp.get_executing_lane_ids():
    # Execute instruction for this lane
    pass
```

3. **Test with realistic patterns:**
   - Parallel reduction
   - Scan/prefix sum
   - Matrix transpose
   - Bitonic sort

## Real-World Use Cases

These operations are essential for:

1. **Parallel Reduction** - Sum/Min/Max across warp without shared memory
2. **Scan/Prefix Sum** - Exclusive/inclusive scan operations
3. **Load Balancing** - Elect thread to do work
4. **Divergence Handling** - ACTIVEMASK for branch handling
5. **Communication** - SHFL for low-overhead data exchange
6. **Synchronization** - BAR for coordinating across threads

## References

- NVIDIA PTX ISA Reference, Section 9.7.13
- CUDA C Programming Guide, Warp Shuffle Functions
- Hopper Architecture Whitepaper
