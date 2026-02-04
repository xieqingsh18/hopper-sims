# PTX ISA Reference

PTX (Parallel Thread Execution) is NVIDIA's low-level intermediate language for GPU programming. This document summarizes the instruction set architecture (ISA) for PTX version 9.1+.

## Table of Contents

1. [Instruction Categories](#instruction-categories)
2. [Suffixes](#suffixes)
3. [Instruction Reference](#instruction-reference)

---

## Instruction Categories

### Arithmetic Operations
| Opcode | Description | Variants |
|--------|-------------|----------|
| `add` | Addition | 11 variants: .cc, .f32, .f32x2, .f64, .ftz.f32, .sat |
| `sub` | Subtraction | 9 variants: .cc, .f32, .f32x2, .f64, .ftz.f32, .sat |
| `mul` | Multiplication | 7 variants: .f32, .f32x2, .f64, .ftz.f32, .sat.f32 |
| `mad` | Multiply-add | 8 variants: .cc, .f32, .f64, .ftz.f32, .rn.f32, .sat.f32 |
| `div` | Division | 9 variants: .approx.f32, .f32, .f64, .ftz.f32, .rn.f64 |
| `rem` | Remainder | 1 variant |
| `abs` | Absolute value | 6 variants: .bf16, .bf16x2, .f32, .f64, .ftz.f32 |
| `neg` | Negation | 6 variants: .bf16, .bf16x2, .f32, .f64, .ftz.f32 |
| `fma` | Fused multiply-add | 10 variants: .f32, .f32x2, .f64, .ftz.f32, .rn.f32 |
| `min` | Minimum | 7 variants: .f32, .f64, .ftz.f32, .relu.s32, .u16x2 |
| `max` | Maximum | 6 variants: .f32, .f64, .ftz.f32, .relu.s32, .u16x2 |

### Logic & Bitwise Operations
| Opcode | Description | Variants |
|--------|-------------|----------|
| `and` | Bitwise AND | - |
| `or` | Bitwise OR | - |
| `xor` | Bitwise XOR | - |
| `not` | Bitwise NOT | - |
| `shf` | Shuffle (lane-to-lane) | 3 variants: .l, .r |
| `shl` | Shift left | - |
| `shr` | Shift right | - |
| `bfe` | Bit field extract | - |
| `bfi` | Bit field insert | - |
| `bfind` | Find most significant bit | - |
| `prmt` | Byte permute | - |
| `lop3` | 3-input logic operation | - |

### Data Movement
| Opcode | Description | Variants |
|--------|-------------|----------|
| `ld` | Load from memory | 22 variants: .acquire, .b16, .b32, .ca, .cg, .const, .cs, .global, .lu, .nc, .proxy, .relaxed, .shared, .volatile, .wt |
| `st` | Store to memory | 18 variants: .async, .b16, .b32, .bulk, .cg, .cs, .release, .shared, .volatile, .wb, .wt |
| `mov` | Move between registers | 2 variants: .f64 |
| `ldu` | Load unaligned | 4 variants: .b16, .b32, .f64 |
| `ldmatrix` | Matrix load | - |
| `stmatrix` | Matrix store | - |
| `cp` | Copy (async/bulk) | 17 variants: .async, .async.bulk, .async.bulk.commit_group, .async.bulk.tensor, .reduce.async.bulk |
| `prefetch` | Prefetch data | - |

### Comparison & Selection
| Opcode | Description | Variants |
|--------|-------------|----------|
| `set` | Set flags | 4 variants: .dtype.f32, .dtype.f64, .ftz.dtype.f32 |
| `setp` | Set predicate | 4 variants: .dtype.f32, .dtype.f64, .ftz.dtype.f32 |
| `slct` | Select based on predicate | 4 variants: .dtype.f32, .f64, .ftz.dtype.f32 |
| `selp` | Select based on predicate value | 2 variants: .f64 |

### Control Flow
| Opcode | Description | Variants |
|--------|-------------|----------|
| `bra` | Branch | 2 variants: .uni |
| `brx` | Branch indexed | 3 variants: .idx, .idx.uni |
| `call` | Function call | - |
| `ret` | Return | - |
| `exit` | Exit kernel | - |
| `trap` | Trap to debug monitor | - |

### Warp Synchronization
| Opcode | Description | Variants |
|--------|-------------|----------|
| `bar` | Barrier | 6 variants: .arrive, .cta, .red.popc.u32, .sync, .warp.sync |
| `barrier` | CTA/Cluster barrier | 5 variants: .cluster, .cluster.arrive, .cluster.wait, .cta |
| `shfl` | Shuffle data between lanes | 2 variants: .sync |
| `vote` | Vote across warp | 4 variants: .ballot.b32, .sync, .sync.ballot.b32 |
| `activemask` | Get active lane mask | - |
| `elect` | Elect one lane from warp | 2 variants: .sync |
| `match` | Synchronize matching lanes | 2 variants: .sync |
| `fence` | Memory fence | 5 variants: .acq_rel, .proxy, .proxy.async, .sc |
| `membar` | Memory barrier | 3 variants: .proxy, .sys |
| `mbarrier` | Memory barrier (async) | 10 variants: .arrive, .arrive_drop, .complete_tx, .expect_tx, .init, .inval, .pending_count, .test_wait, .try_wait |

### Math Functions
| Opcode | Description | Variants |
|--------|-------------|----------|
| `sqrt` | Square root | 8 variants: .approx.f32, .f32, .f64, .ftz.f32, .rn.f64, .rnd.f32 |
| `rsqrt` | Reciprocal square root | 9 variants: .approx, .f32, .f64, .ftz.f32 |
| `sin` | Sine | 5 variants: .approx.f32, .f32, .ftz.f32 |
| `cos` | Cosine | 5 variants: .approx.f32, .f32, .ftz.f32 |
| `lg2` | Log base 2 | 5 variants: .approx.f32, .f32, .ftz.f32 |
| `ex2` | 2^x (exponential) | 7 variants: .approx.f16, .approx.f32, .ftz.bf16 |
| `rcp` | Reciprocal | 9 variants: .approx.f32, .f32, .f64, .ftz.f32, .rn.f64 |
| `cvt` | Convert between types | 13 variants: .f16.f32, .f32.bf16, .f32.f16, .ftz |
| `clz` | Count leading zeros | - |
| `popc` | Population count | - |
| `brev` | Bit reversal | - |

### Atomic Operations
| Opcode | Description | Variants |
|--------|-------------|----------|
| `atom` | Atomic operation | 16 variants: .add.bf16, .add.f16, .add.f32, .cas.b16, .exch, .and, .or, .xor |
| `red` | Warp reduction | 17 variants: .add.bf16, .add.f16, .add.f32, .and.popc, .min.u32, .max.u32 |

### Matrix & Tensor (Hopper)
| Opcode | Description | Variants |
|--------|-------------|----------|
| `mma` | Tensor Core MMA | 17 variants: .sync.aligned.m16n8k16, .m16n8k128, .and.popc, .xor.popc |
| `wmma` | Warp Matrix Multiply-Accumulate | 6 variants: .load, .mma, .store, .and.popc |
| `wgmma` | Warpgroup MMA (Hopper) | 6 variants: .mma_async, .mma_async.sp, .commit_group, .fence, .wait_group |
| `tcgen05` | Tensor Copy Engine | 18 variants: .alloc, .commit, .cp, .dealloc, .fence, .ld, .st, .ld.red |

### Texture & Surface (Legacy)
| Opcode | Description |
|--------|-------------|
| `tex` | Texture fetch |
| `tld4` | Texture gather |
| `suld` | Surface load |
| `sust` | Surface store |
| `sured` | Surface reduction |
| `suq` | Surface query |

### Vector Operations
| Opcode | Description |
|--------|-------------|
| `vadd` | Packed integer add |
| `vsub` | Packed integer subtract |
| `vmin` | Packed integer minimum |
| `vmax` | Packed integer maximum |
| `sad` | Sum of absolute differences |
| `dp4a` | Dot product (4 elements) |

### Memory & State Space
| Opcode | Description | Variants |
|--------|-------------|----------|
| `isspacep` | Is in memory space | 4 variants: .const, .global, .local, .param |
| `istypep` | Is of type | - |
| `cvta` | Convert address | 7 variants: .const, .global, .param, .to.gen, .to.shared, .u64 |
| `alloca` | Stack allocation | - |

---

## Suffixes

PTX instructions use suffixes to specify data types, memory spaces, synchronization semantics, and other modifiers.

### Type Suffixes
| Suffix | Description |
|--------|-------------|
| `.b8`, `.b16`, `.b32`, `.b64` | Bit types (unsigned integer) |
| `.u8`, `.u16`, `.u32`, `.u64` | Unsigned integers |
| `.s8`, `.s16`, `.s32`, `.s64` | Signed integers |
| `.f16`, `.f32`, `.f64` | Floating point |
| `.bf16` | Bfloat16 (brain float) |
| `.tf32` | Tensor float32 |
| `.f16x2`, `.f32x2`, `.bf16x2` | Packed vector types (2 elements) |
| `.s16x2`, `.u16x2` | Packed integer types (2 elements) |
| `.e4m3`, `.e5m2` | 8-bit floating point (FP8) |
| `.ue4m3`, `.ue8m0` | Unsigned 8-bit floating point |

### Memory Space Qualifiers
| Suffix | Description |
|--------|-------------|
| `.global` | Global memory |
| `.shared` | Shared memory |
| `.shared::cta` | CTA-scoped shared memory |
| `.shared::cluster` | Cluster-scoped shared memory |
| `.local` | Local memory |
| `.param` | Parameter space |
| `.const` | Constant memory |
| `.generic` | Generic addressing |

### Synchronization Modifiers
| Suffix | Description |
|--------|-------------|
| `.relaxed` | No ordering guarantees |
| `.acquire` | Load with acquire semantics |
| `.release` | Store with release semantics |
| `.acq_rel` | Both acquire and release |
| `.volatile` | Cannot be reordered |
| `.weak` | Weak consistency |
| `.mmio` | Memory-mapped I/O |

### Memory Scope
| Suffix | Description |
|--------|-------------|
| `.cta` | CTA (thread block) scope |
| `.cluster` | Cluster scope |
| `.gpu` | GPU scope |
| `.sys` | System (CPU+GPU) scope |

### Cache Operators
| Suffix | Description |
|--------|-------------|
| `.ca` | Cache at all levels |
| `.cg` | Cache at global level |
| `.cs` | Cache streaming (evict first) |
| `.lu` | Last use |
| `.cv` | Cache as volatile |
| `.wb` | Write back |
| `.wt` | Write through |

### Matrix Modifiers
| Suffix | Description |
|--------|-------------|
| `.sync` | Synchronized operation |
| `.aligned` | Aligned access |
| `.trans` | Transposed matrix |

### Matrix Shapes
| Suffix | Description |
|--------|-------------|
| `.m8n8k4` | 8x8 matrix, k=4 |
| `.m8n8k16` | 8x8 matrix, k=16 |
| `.m8n8k32` | 8x8 matrix, k=32 |
| `.m8n8k128` | 8x8 matrix, k=128 |
| `.m16n8k4` | 16x8 matrix, k=4 |
| `.m16n8k8` | 16x8 matrix, k=8 |
| `.m16n8k16` | 16x8 matrix, k=16 |
| `.m16n8k32` | 16x8 matrix, k=32 |
| `.m16n8k64` | 16x8 matrix, k=64 |
| `.m16n8k128` | 16x8 matrix, k=128 |
| `.m16n8k256` | 16x8 matrix, k=256 |
| `.m64n8k16` | 64x8 warpgroup matrix (WGMMA) |
| `.m64n72k16` | 64x72 warpgroup matrix (WGMMA) |
| `.m64n120k16` | 64x120 warpgroup matrix (WGMMA) |

### Vector Width
| Suffix | Description |
|--------|-------------|
| `.x1` | 1 vector |
| `.x2` | 2 vectors |
| `.x4` | 4 vectors |
| `.x8` | 8 vectors |

### Other Modifiers
| Suffix | Description |
|--------|-------------|
| `.ftz` | Flush-to-zero (denormals → 0) |
| `.nofutz` | Don't flush-to-zero |
| `.sat` | Saturating arithmetic |
| `.satfinite` | Saturate only finite values |
| `.rn` | Rounding: round-to-nearest |
| `.rnd` | Rounding mode |
| `.approx` | Approximate (faster, less accurate) |
| `.uni` | Uniform (all lanes agree) |
| `.async` | Asynchronous operation |
| `.bulk` | Bulk operation |
| `.cc` | Set carry condition code |
| `.pred` | Predicate register |
| `.relu` | ReLU activation (min/max) |
| `.pack` | Packed data operation |
| `.popc` | Population count |
| `.mmio` | Memory-mapped I/O |
| `.commit_group` | Commit async group |
| `.wait_group` | Wait for async group |
| `.trap` | Trap on error |
| `.init` | Initialize |
| `.inval` | Invalidate |

---

## Instruction Format

PTX instructions follow this general format:

```
[opcode][.suffixes] [destination], [source(s)]
```

### Examples

```ptx
// Arithmetic with type suffix
add.f32 R1, R2, R3;        // Floating point add
add.sat.s32 R1, R2, R3;    // Saturating signed add

// Memory load with space and sync
ld.global.f32 R1, [R2];           // Global load
ld.shared.acquire.u32 R1, [R2];   // Shared load with acquire

// Atomic operation
atom.add.relaxed.cta.u32 R1, [R2], R3;  // Atomic add with relaxed ordering, CTA scope

// Matrix operation (Tensor Core)
mma.sync.aligned.m16n8k16.row.col.f32.f32.f32.f32
    {R0,R1,R2,R3}, {R4,R5,R6,R7}, {R8,R9,R10,R11}, R12;

// Warpgroup MMA (Hopper)
wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16
    R0, R1, R2, R3;

// Memory barrier
mbarrier.init.shared::cta.b64 [R1], R2;

// Warp sync
shfl.sync.b32 R1, R2, R3, R4;  // Shuffle with sync
vote.sync.ballot.b32 R1, P1;   // Vote with ballot
```

---

## Statistics

- **Total unique base opcodes**: 395
- **Total instruction variants**: 768+
- **Type suffixes**: 15+
- **Memory spaces**: 6
- **Sync modifiers**: 6
- **Memory scopes**: 4
- **Matrix shapes**: 11+

---

## References

- NVIDIA PTX ISA Version 9.1+ Documentation
- CUDA C++ Programming Guide
- Hopper Architecture Whitepaper
