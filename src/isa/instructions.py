"""
Instruction Definitions for Hopper SASS

Hopper SASS (Shader Assembly) is the native instruction set for NVIDIA GPUs.
This module defines the instruction formats, opcodes, and operand types.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Union, Optional, List


class InstructionType(Enum):
    """Categories of instructions."""
    ARITHMETIC = auto()      # Integer arithmetic
    FLOATING = auto()        # Floating point operations
    MEMORY = auto()          # Memory load/store
    CONTROL_FLOW = auto()    # Branches, jumps
    LOGICAL = auto()         # Bitwise operations
    MOVE = auto()            # Data movement
    TENSOR_CORE = auto()     # Tensor Core operations
    WARP_LEVEL = auto()      # Warp-wide operations (SYNC, etc.)
    SPECIAL = auto()         # Special instructions


class Opcode(Enum):
    """
    PTX ISA Opcodes for Hopper GPU Simulator.

    Based on NVIDIA PTX ISA Reference.
    Includes instructions from multiple categories for comprehensive simulation.
    """

    # ==================== Integer Arithmetic ====================
    IADD = "IADD"        # Integer add
    ISUB = "ISUB"        # Integer subtract
    IMUL = "IMUL"        # Integer multiply
    IMAD = "IMAD"        # Integer multiply-add
    IDIV = "IDIV"        # Integer divide
    IREM = "IREM"        # Integer remainder
    IABS = "IABS"        # Integer absolute
    INEG = "INEG"        # Integer negate
    IMIN = "IMIN"        # Integer minimum
    IMAX = "IMAX"        # Integer maximum
    IADD3 = "IADD3"      # Integer add with carry (3-operand)
    ISULd = "ISULd"      # Signed integer multiply (24-bit)
    IMMA = "IMMA"        # Integer matrix multiply-accumulate

    # Bit manipulation
    POPC = "POPC"        # Population count (count set bits)
    CLZ = "CLZ"          # Count leading zeros
    BREV = "BREV"        # Bit reverse
    BFE = "BFE"          # Bit field extract
    BFI = "BFI"          # Bit field insert
    BMSK = "BMSK"        # Bit mask
    FNS = "FNS"          # Find nth set

    # Dot product (4-way)
    DP4A = "DP4A"        # Dot product 4-way (signed)
    DP2A = "DP2A"        # Dot product 2-way

    # ==================== Floating Point ====================
    FADD = "FADD"        # Floating point add
    FSUB = "FSUB"        # Floating point subtract
    FMUL = "FMUL"        # Floating point multiply
    FFMA = "FFMA"        # Floating point fused multiply-add
    FMAD = "FMAD"        # Floating point multiply-add
    FDIV = "FDIV"        # Floating point divide
    FABS = "FABS"        # Floating point absolute
    FNEG = "FNEG"        # Floating point negate
    FMIN = "FMIN"        # Floating point minimum
    FMAX = "FMAX"        # Floating point maximum
    FRCP = "FRCP"        # Floating point reciprocal
    FSQRT = "FSQRT"      # Floating point square root
    FRSQRT = "FRSQRT"    # Floating point reciprocal square root
    FSIN = "FSIN"        # Floating point sine
    FCOS = "FCOS"        # Floating point cosine
    FLOG2 = "FLOG2"      # Floating point log base 2
    FEXP2 = "FEXP2"      # Floating point exponent base 2
    FTANH = "FTANH"      # Floating point hyperbolic tangent
    FSETP = "FSETP"      # Floating point comparison and set predicate
    TESTP = "TESTP"      # Test floating point properties

    # Half precision floating point
    HADD = "HADD"        # Half precision add
    HSUB = "HSUB"        # Half precision sub
    HMUL = "HMUL"        # Half precision mul
    HFMA = "HFMA"        # Half precision FMA
    HNEG = "HNEG"        # Half precision negate
    HABS = "HABS"        # Half precision abs
    HMIN = "HMIN"        # Half precision min
    HMAX = "HMAX"        # Half precision max

    # ==================== Comparison & Selection ====================
    SETP = "SETP"        # Set predicate from comparison
    SET = "SET"          # Set register from comparison
    SELP = "SELP"        # Select based on predicate
    SLCT = "SLCT"        # Select based on predicate (register)

    # ==================== Logic & Shift ====================
    LOP = "LOP"          # Logical operations (AND, OR, XOR)
    LOP3 = "LOP3"        # Three-input logical operation
    AND = "AND"          # Bitwise AND
    OR = "OR"            # Bitwise OR
    XOR = "XOR"          # Bitwise XOR
    NOT = "NOT"          # Bitwise NOT
    CNOT = "CNOT"        # Conditional NOT
    SHL = "SHL"          # Shift left
    SHR = "SHR"          # Shift right
    SHF = "SHF"          # Shift funnel (shuffle)
    PRMT = "PRMT"        # Permute bytes

    # ==================== Data Movement ====================
    MOV = "MOV"          # Move register to register
    MVA = "MVA"          # Move between register spaces
    S2R = "S2R"          # Move from special register
    R2R = "R2R"          # Move between register files
    CVT = "CVT"          # Convert data type
    CVTA = "CVTA"        # Convert address space

    # Load instructions
    LD = "LD"            # Generic load
    LDG = "LDG"          # Load from global memory
    LDG_E = "LDG.E"      # Load from global (evict-first)
    LDS = "LDS"          # Load from shared memory
    LDSM = "LDSM"        # Load from shared memory with broadcast
    LDC = "LDC"          # Load from constant memory
    LDL = "LDL"          # Load from local memory
    LDU = "LDU"          # Load unaligned

    # Store instructions
    ST = "ST"            # Generic store
    STG = "STG"          # Store to global memory
    STS = "STS"          # Store to shared memory
    STL = "STL"          # Store to local memory

    # Async copy
    CPASYNC = "CPASYNC"  # Async copy
    PREFETCH = "PREFETCH"  # Prefetch
    PREFETCHU = "PREFETCHU"  # Prefetch unaligned

    # Matrix load/store
    LDMATRIX = "LDMATRIX"  # Load matrix fragment
    STMATRIX = "STMATRIX"  # Store matrix fragment

    # ==================== Control Flow ====================
    BRA = "BRA"          # Branch
    BRX = "BRX"          # Indexed branch
    CALL = "CALL"        # Call subroutine
    RET = "RET"          # Return
    EXIT = "EXIT"        # Exit kernel
    JMP = "JMP"          # Jump
    CAL = "CAL"          # Call (SASS variant)
    BSS = "BSS"          # Branch and sync
    SSY = "SSY"          # Set synchronization point

    # ==================== Predicates ====================
    PSETP = "PSETP"      # Set predicate
    P2R = "P2R"          # Predicate to register
    R2P = "R2P"          # Register to predicate

    # ==================== Warp Level Operations ====================
    BAR = "BAR"          # Barrier synchronization
    BAR_WARP = "BAR.WARP"  # Warp-level barrier
    MEMBAR = "MEMBAR"    # Memory barrier
    FENCE = "FENCE"      # Memory fence
    VOTE = "VOTE"        # Vote across warp
    ACTIVEMASK = "ACTIVEMASK"  # Get active lane mask
    SHFL = "SHFL"        # Shuffle data between lanes
    ELECT = "ELECT"      # Elect one thread
    MATCH = "MATCH"      # Match sync

    # Reduction across warp
    REDUX = "REDUX"      # Reduction across warp

    # ==================== Atomic Operations ====================
    ATOM = "ATOM"        # Atomic operation
    RED = "RED"          # Reduction operation

    # Atomic operations (specific)
    ATOM_ADD = "ATOM.ADD"      # Atomic add
    ATOM_SUB = "ATOM.SUB"      # Atomic sub
    ATOM_MIN = "ATOM.MIN"      # Atomic min
    ATOM_MAX = "ATOM.MAX"      # Atomic max
    ATOM_AND = "ATOM.AND"      # Atomic and
    ATOM_OR = "ATOM.OR"        # Atomic or
    ATOM_XOR = "ATOM.XOR"      # Atomic xor
    ATOM_XCHG = "ATOM.XCHG"    # Atomic exchange
    ATOM_CAS = "ATOM.CAS"      # Atomic compare-and-swap

    # ==================== Tensor Core ====================
    HMMA = "HMMA"        # Matrix multiply-accumulate (FP8/FP16/FP32)
    MMA = "MMA"          # Matrix multiply-accumulate (general)
    WMMA = "WMMA"        # Warp matrix multiply-accumulate

    # Tensor Core shapes
    MMA_M16N8K16 = "MMA.M16N8K16"  # 16x8x16 matrix multiply
    MMA_M8N8K32 = "MMA.M8N8K32"    # 8x8x32 matrix multiply

    # Sparse Tensor Core
    MMA_SP = "MMA.SP"    # Sparse matrix multiply

    # ==================== PTX Instruction Set (from isa.md) ====================
    # Note: These use PTX naming convention (lowercase, with suffixes)
    # Arithmetic operations
    ADD = "ADD"          # PTX add
    SUB = "SUB"          # PTX sub
    MUL = "MUL"          # PTX mul
    MAD = "MAD"          # PTX mad
    DIV = "DIV"          # PTX div
    REM = "REM"          # PTX remainder
    ABS = "ABS"          # PTX absolute
    NEG = "NEG"          # PTX negate
    MIN = "MIN"          # PTX minimum
    MAX = "MAX"          # PTX maximum
    FMA = "FMA"          # PTX fused multiply-add
    ADDC = "ADDC"        # Add with carry
    SUBC = "SUBC"        # Subtract with carry
    MUL24 = "MUL24"      # 24-bit multiply
    MAD24 = "MAD24"      # 24-bit multiply-add
    SAD = "SAD"          # Sum of absolute differences

    # Math functions
    SQRT = "SQRT"        # Square root
    RSQRT = "RSQRT"      # Reciprocal square root
    SIN = "SIN"          # Sine
    COS = "COS"          # Cosine
    LG2 = "LG2"          # Log base 2
    EX2 = "EX2"          # Exponent base 2
    RCP = "RCP"          # Reciprocal
    COPYSIGN = "COPYSIGN"  # Copy sign
    TANH = "TANH"        # Hyperbolic tangent

    # Bit manipulation (PTX variants)
    BFIND = "BFIND"      # Find bit (PTX variant)
    RBITS = "RBITS"      # Reverse bits (PTX variant)

    # Vector operations
    VADD = "VADD"        # Vector add
    VSUB = "VSUB"        # Vector subtract
    VMIN = "VMIN"        # Vector minimum
    VMAX = "VMAX"        # Vector maximum
    VADDDIFF = "VABSDIFF"  # Vector absolute difference
    VADDDIFF2 = "VABSDIFF2"  # Vector absolute difference 2
    VADDDIFF4 = "VABSDIFF4"  # Vector absolute difference 4
    VADD2 = "VADD2"      # Vector add 2
    VSUB2 = "VSUB2"      # Vector subtract 2
    VMIN2 = "VMIN2"      # Vector minimum 2
    VMAX2 = "VMAX2"      # Vector maximum 2
    VADD4 = "VADD4"      # Vector add 4
    VSUB4 = "VSUB4"      # Vector subtract 4
    VMIN4 = "VMIN4"      # Vector minimum 4
    VMAX4 = "VMAX4"      # Vector maximum 4
    VSET = "VSET"        # Vector set
    VSET2 = "VSET2"      # Vector set 2
    VSET4 = "VSET4"      # Vector set 4
    VSHL = "VSHL"        # Vector shift left
    VSHR = "VSHR"        # Vector shift right
    VARVG2 = "VARVG2"    # Vector average
    VARVG4 = "VARVG4"    # Vector average
    VAVRG2 = "VAVRG2"    # Vector average round
    VAVRG4 = "VAVRG4"    # Vector average round
    VMAD = "VMAD"        # Vector multiply-add

    # DP4A and DP2A already defined above

    # Memory operations
    ISSPACEP = "ISSPACEP"  # Is in space
    ISTYPEP = "ISTYPEP"    # Is of type

    # Barrier operations (PTX variants)
    BARRIER = "BARRIER"    # Cluster/CTA barrier
    BARRIER_CLUSTER = "BARRIER.CLUSTER"  # Cluster barrier
    BARRIER_CTA = "BARRIER.CTA"          # CTA barrier

    # FENCE operations (PTX variants)
    FENCE_SC = "FENCE.SC"      # Sequential consistency fence
    FENCE_ACQ_REL = "FENCE.ACQ_REL"  # Acquire-release fence

    # Texture operations (PTX)
    # TEX already defined above
    TLD4 = "TLD4"          # Texture gather (PTX)
    TXQ = "TXQ"            # Texture query (PTX)

    # Surface operations (PTX)
    # SULD, SUST, SURED already defined above
    SUQ = "SUQ"            # Surface query (PTX)

    # Multimem operations
    MULTIMEM = "MULTIMEM"  # Multi-memory operations

    # Tensormap
    TENSORMAP = "TENSORMAP"  # Tensor map operations

    # Cluster launch control
    CLUSTERLAUNCHCONTROL = "CLUSTERLAUNCHCONTROL"  # Cluster launch

    # Grid dependency control
    GRIDDEPCONTROL = "GRIDDEPCONTROL"  # Grid dependency

    # Get CTA rank
    GETCTARANK = "GETCTARANK"  # Get CTA rank

    # B4x16 operations
    B4E = "B4E"            # B4x16 extract
    B4X16_P64 = "B4X16_P64"  # B4x16 with p64
    B6X16_P32 = "B6X16_P32"  # B6x16 with p32

    # Memory address operations
    ASEL = "ASEL"          # Address select
    BSEL = "BSEL"          # Byte select

    # Mx4 operations
    MXF4 = "MXF4"          # Matrix operation
    MXF4NVF4 = "MXF4NVF4"  # Matrix operation
    MXF8F6F4 = "MXF8F6F4"  # Matrix operation

    # Extended operations
    A2D = "A2D"            # Add 2D
    A2DMS = "A2DMS"        # Add 2D with multiplier and shuffle

    # Nanosleep
    NANOSLEEP = "NANOSLEEP"  # Nanosleep

    # Apply priority
    APPLYPRIORITY = "APPLYPRIORITY"  # Apply priority

    # Enable SMEM spilling
    ENABLE_SMEM_SPILLING = "ENABLE_SMEM_SPILLING"  # Enable shared memory spilling

    # Member mask
    MEMBERMASK = "MEMBERMASK"  # Member mask

    # Map operations
    MAPA = "MAPA"          # Map A

    # Comparison operators (PTX)
    EQ = "EQ"              # Equal
    NE = "NE"              # Not equal
    LT = "LT"              # Less than
    LE = "LE"              # Less than or equal
    GT = "GT"              # Greater than
    GE = "GE"              # Greater than or equal
    LTU = "LTU"            # Less than unsigned
    LEU = "LEU"            # Less than or equal unsigned
    GTU = "GTU"            # Greater than unsigned
    GEU = "GEU"            # Greater than or equal unsigned
    LO = "LO"              # Low (for sub)
    HI = "HI"              # High (for sub)
    LS = "LS"              # Low or same
    HS = "HS"              # High or same

    # Equate
    EQU = "EQU"            # Equate

    # Extended comparison
    NUM = "NUM"            # Number comparison
    NAN = "NAN"            # NaN comparison

    # Exit
    RETD = "RETD"          # Return with delay (deprecated)

    # ==================== Warpgroup Matrix (Hopper) ====================
    WGMMA = "WGMMA"              # Warpgroup matrix multiply
    WGMMA_MMA = "WGMMA.MMA"      # Warpgroup matrix multiply-accumulate
    WGMMA_MMA_ASYNC = "WGMMA.MMA_ASYNC"  # Async warpgroup MMA

    # Warpgroup shapes for WGMMA
    # m64nNk16, m64nNk8, m64nNk32, m64nNk64, m64nNk256

    # ==================== Tensor Memory Accelerator (TMA) ====================
    TMA = "TMA"                  # Tensor Memory Accelerator
    TMA_ALLOC = "TMA.ALLOC"      # Allocate TMA descriptor
    TMA_LOAD = "TMA.LOAD"        # TMA load from global to shared
    TMA_STORE = "TMA.STORE"      # TMA store from shared to global
    TMA_WAIT = "TMA.WAIT"        # Wait for TMA completion

    # ==================== Memory Barrier (mbarrier) ====================
    MBARRIER_INIT = "MBARRIER_INIT"      # Initialize mbarrier
    MBARRIER_INIT_DOT = "mbarrier.init"  # PTX dot notation
    MBARRIER_INVAL = "MBARRIER_INVAL"    # Invalidate mbarrier
    MBARRIER_INVAL_DOT = "mbarrier.inval"  # PTX dot notation
    MBARRIER_ARRIVE = "MBARRIER_ARRIVE"  # Arrive at mbarrier
    MBARRIER_ARRIVE_DOT = "mbarrier.arrive"  # PTX dot notation
    MBARRIER_ARRIVE_DROP = "MBARRIER_ARRIVE_DROP"  # Arrive and drop threshold
    MBARRIER_ARRIVE_DROP_DOT = "mbarrier.arrive_drop"  # PTX dot notation
    MBARRIER_TEST_WAIT = "MBARRIER_TEST_WAIT"  # Test and wait mbarrier
    MBARRIER_TEST_WAIT_DOT = "mbarrier.test_wait"  # PTX dot notation
    MBARRIER_TRY_WAIT = "MBARRIER_TRY_WAIT"  # Try wait mbarrier (non-blocking)
    MBARRIER_TRY_WAIT_DOT = "mbarrier.try_wait"  # PTX dot notation
    MBARRIER_PENDING_COUNT = "MBARRIER_PENDING_COUNT"  # Get pending count
    MBARRIER_PENDING_COUNT_DOT = "mbarrier.pending_count"  # PTX dot notation
    MBARRIER_EXPECT_TX = "MBARRIER_EXPECT_TX"  # Expect transaction
    MBARRIER_EXPECT_TX_DOT = "mbarrier.expect_tx"  # PTX dot notation
    MBARRIER_COMPLETE_TX = "MBARRIER_COMPLETE_TX"  # Complete transaction
    MBARRIER_COMPLETE_TX_DOT = "mbarrier.complete_tx"  # PTX dot notation
    CP_ASYNC_MBARRIER_ARRIVE = "CP_ASYNC_MBARRIER_ARRIVE"  # Async copy with mbarrier arrive
    CP_ASYNC_MBARRIER_ARRIVE_DOT = "cp.async.mbarrier.arrive"  # PTX dot notation

    # ==================== Proxy Fence ====================
    FENCE_PROXY = "FENCE_PROXY"  # Proxy fence
    FENCE_PROXY_DOT = "fence.proxy"  # PTX dot notation
    FENCE_PROXY_ASYNC = "FENCE_PROXY_ASYNC"  # Async proxy fence
    FENCE_PROXY_ASYNC_DOT = "fence.proxy.async"  # PTX dot notation
    FENCE_PROXY_TENSORMAP = "FENCE_PROXY_TENSORMAP"  # Tensormap proxy fence
    FENCE_PROXY_TENSORMAP_DOT = "fence.proxy.tensormap"  # PTX dot notation

    # ==================== TMA (Tensor Memory Accelerator) ====================
    CP_ASYNC_BULK = "CP_ASYNC_BULK"  # Async bulk copy
    CP_ASYNC_BULK_DOT = "cp.async.bulk"  # PTX dot notation
    CP_ASYNC_BULK_TENSOR = "CP_ASYNC_BULK_TENSOR"  # Async bulk tensor copy
    CP_ASYNC_BULK_TENSOR_DOT = "cp.async.bulk.tensor"  # PTX dot notation
    CP_ASYNC_BULK_PREFETCH = "CP_ASYNC_BULK_PREFETCH"  # Async bulk prefetch
    CP_ASYNC_BULK_PREFETCH_DOT = "cp.async.bulk.prefetch"  # PTX dot notation
    CP_ASYNC_BULK_PREFETCH_TENSOR = "CP_ASYNC_BULK_PREFETCH_TENSOR"  # Async bulk prefetch tensor
    CP_ASYNC_BULK_PREFETCH_TENSOR_DOT = "cp.async.bulk.prefetch.tensor"  # PTX dot notation
    CP_REDUCE_ASYNC_BULK = "CP_REDUCE_ASYNC_BULK"  # Async bulk reduction
    CP_REDUCE_ASYNC_BULK_DOT = "cp.reduce.async.bulk"  # PTX dot notation
    CP_REDUCE_ASYNC_BULK_TENSOR = "CP_REDUCE_ASYNC_BULK_TENSOR"  # Async bulk tensor reduction
    CP_REDUCE_ASYNC_BULK_TENSOR_DOT = "cp.reduce.async.bulk.tensor"  # PTX dot notation
    CP_ASYNC_BULK_COMMIT_GROUP = "CP_ASYNC_BULK_COMMIT_GROUP"  # Commit async bulk group
    CP_ASYNC_BULK_COMMIT_GROUP_DOT = "cp.async.bulk.commit_group"  # PTX dot notation
    CP_ASYNC_BULK_WAIT_GROUP = "CP_ASYNC_BULK_WAIT_GROUP"  # Wait for async bulk group
    CP_ASYNC_BULK_WAIT_GROUP_DOT = "cp.async.bulk.wait_group"  # PTX dot notation
    MULTIMEM_CP_ASYNC_BULK = "MULTIMEM_CP_ASYNC_BULK"  # Multimem async bulk copy
    MULTIMEM_CP_ASYNC_BULK_DOT = "multimem.cp.async.bulk"  # PTX dot notation
    MULTIMEM_CP_REDUCE_ASYNC_BULK = "MULTIMEM_CP_REDUCE_ASYNC_BULK"  # Multimem async bulk reduction
    MULTIMEM_CP_REDUCE_ASYNC_BULK_DOT = "multimem.cp.reduce.async.bulk"  # PTX dot notation

    # ==================== Warpgroup Operations ====================
    WGMMA_FENCE = "WGMMA_FENCE"  # Warpgroup fence
    WGMMA_FENCE_DOT = "wgmma.fence"  # PTX dot notation
    WGMMA_COMMIT_GROUP = "WGMMA_COMMIT_GROUP"  # Warpgroup commit group
    WGMMA_COMMIT_GROUP_DOT = "wgmma.commit_group"  # PTX dot notation
    WGMMA_WAIT_GROUP = "WGMMA_WAIT_GROUP"  # Warpgroup wait group
    WGMMA_WAIT_GROUP_DOT = "wgmma.wait_group"  # PTX dot notation
    WGMMA_MMA_ASYNC_SP = "WGMMA_MMA_ASYNC_SP"  # Sparse async warpgroup MMA
    WGMMA_MMA_ASYNC_SP_DOT = "wgmma.mma_async.sp"  # PTX dot notation

    # ==================== Texture / Surface ====================
    TEX = "TEX"          # Texture fetch
    TEXFETCH = "TEXFETCH"  # Texture fetch with grad
    SULD = "SULD"        # Surface load
    SUST = "SUST"        # Surface store
    SURED = "SURED"      # Surface reduction


class OperandType(Enum):
    """Types of operands."""
    REGISTER = auto()
    PREDICATE = auto()
    IMMEDIATE = auto()
    MEMORY = auto()
    SPECIAL_REGISTER = auto()
    NONE = auto()


@dataclass
class Operand:
    """
    Instruction operand.

    Can be a register, immediate value, predicate, memory reference, etc.
    """
    type: OperandType
    value: Union[str, int, float]  # For register: "R5", for immediate: 123

    # Optional modifiers
    negate: bool = False           # Negate the operand
    absolute: bool = False         # Absolute value
    is_dest: bool = False   # Is this a destination operand?

    def __repr__(self) -> str:
        prefix = ""
        if self.negate:
            prefix = "-"
        elif self.absolute:
            prefix = "|"

        return f"{prefix}{self.value}"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class InstructionFormat:
    """
    SASS instruction format specification.

    Describes the format of an instruction for encoding/decoding.
    """
    name: str
    opcode: Opcode
    type: InstructionType
    operands: List[OperandType]  # [dst, src1, src2, ...]

    # Format-specific flags
    has_predicates: bool = False
    has_wide_operands: bool = False  # 64-bit operands
    is_tensor: bool = False

    def __repr__(self) -> str:
        return f"InstructionFormat({self.name}, opcode={self.opcode.value})"


# Common instruction formats
INSTRUCTION_FORMATS = {
    # ==================== Integer Arithmetic ====================
    # Binary operations: Rd = R1 op R2
    Opcode.IADD: InstructionFormat("IADD", Opcode.IADD, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.ISUB: InstructionFormat("ISUB", Opcode.ISUB, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.IMUL: InstructionFormat("IMUL", Opcode.IMUL, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.IMIN: InstructionFormat("IMIN", Opcode.IMIN, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.IMAX: InstructionFormat("IMAX", Opcode.IMAX, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.IABS: InstructionFormat("IABS", Opcode.IABS, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.INEG: InstructionFormat("INEG", Opcode.INEG, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER]),

    # Ternary operations: Rd = R1 op R2 op R3
    Opcode.IMAD: InstructionFormat("IMAD", Opcode.IMAD, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.IADD3: InstructionFormat("IADD3", Opcode.IADD3, InstructionType.ARITHMETIC,
                                    [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),

    # Bit manipulation
    Opcode.POPC: InstructionFormat("POPC", Opcode.POPC, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.CLZ: InstructionFormat("CLZ", Opcode.CLZ, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.BFE: InstructionFormat("BFE", Opcode.BFE, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.IMMEDIATE, OperandType.IMMEDIATE]),

    # Dot product
    Opcode.DP4A: InstructionFormat("DP4A", Opcode.DP4A, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),

    # ==================== Floating Point ====================
    Opcode.FADD: InstructionFormat("FADD", Opcode.FADD, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FSUB: InstructionFormat("FSUB", Opcode.FSUB, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FMUL: InstructionFormat("FMUL", Opcode.FMUL, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FFMA: InstructionFormat("FFMA", Opcode.FFMA, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FABS: InstructionFormat("FABS", Opcode.FABS, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FNEG: InstructionFormat("FNEG", Opcode.FNEG, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FMIN: InstructionFormat("FMIN", Opcode.FMIN, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FMAX: InstructionFormat("FMAX", Opcode.FMAX, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FSETP: InstructionFormat("FSETP", Opcode.FSETP, InstructionType.FLOATING,
                                    [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.TESTP: InstructionFormat("TESTP", Opcode.TESTP, InstructionType.FLOATING,
                                    [OperandType.REGISTER, OperandType.REGISTER]),

    # ==================== Logic & Shift ====================
    Opcode.AND: InstructionFormat("AND", Opcode.AND, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.OR: InstructionFormat("OR", Opcode.OR, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.XOR: InstructionFormat("XOR", Opcode.XOR, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.NOT: InstructionFormat("NOT", Opcode.NOT, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.LOP3: InstructionFormat("LOP3", Opcode.LOP3, InstructionType.LOGICAL,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.IMMEDIATE]),
    Opcode.SHL: InstructionFormat("SHL", Opcode.SHL, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.SHR: InstructionFormat("SHR", Opcode.SHR, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.SHF: InstructionFormat("SHF", Opcode.SHF, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.IMMEDIATE]),
    Opcode.PRMT: InstructionFormat("PRMT", Opcode.PRMT, InstructionType.LOGICAL,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.IMMEDIATE]),

    # ==================== Data Movement ====================
    Opcode.MOV: InstructionFormat("MOV", Opcode.MOV, InstructionType.MOVE,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.CVT: InstructionFormat("CVT", Opcode.CVT, InstructionType.MOVE,
                                  [OperandType.REGISTER, OperandType.REGISTER]),

    # Load/Store
    Opcode.LDG: InstructionFormat("LDG", Opcode.LDG, InstructionType.MEMORY,
                                  [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.LDS: InstructionFormat("LDS", Opcode.LDS, InstructionType.MEMORY,
                                  [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.LDC: InstructionFormat("LDC", Opcode.LDC, InstructionType.MEMORY,
                                  [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.LDL: InstructionFormat("LDL", Opcode.LDL, InstructionType.MEMORY,
                                  [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.LDU: InstructionFormat("LDU", Opcode.LDU, InstructionType.MEMORY,
                                  [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.LD: InstructionFormat("LD", Opcode.LD, InstructionType.MEMORY,
                                [OperandType.REGISTER, OperandType.MEMORY]),

    Opcode.STG: InstructionFormat("STG", Opcode.STG, InstructionType.MEMORY,
                                  [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.STS: InstructionFormat("STS", Opcode.STS, InstructionType.MEMORY,
                                  [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.STL: InstructionFormat("STL", Opcode.STL, InstructionType.MEMORY,
                                  [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.ST: InstructionFormat("ST", Opcode.ST, InstructionType.MEMORY,
                                 [OperandType.MEMORY, OperandType.REGISTER]),

    # Matrix load/store
    Opcode.LDMATRIX: InstructionFormat("LDMATRIX", Opcode.LDMATRIX, InstructionType.MEMORY,
                                       [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.STMATRIX: InstructionFormat("STMATRIX", Opcode.STMATRIX, InstructionType.MEMORY,
                                       [OperandType.MEMORY, OperandType.REGISTER]),

    # ==================== Control Flow ====================
    Opcode.BRA: InstructionFormat("BRA", Opcode.BRA, InstructionType.CONTROL_FLOW,
                                  [OperandType.IMMEDIATE]),
    Opcode.BRX: InstructionFormat("BRX", Opcode.BRX, InstructionType.CONTROL_FLOW,
                                  [OperandType.REGISTER]),
    Opcode.CALL: InstructionFormat("CALL", Opcode.CALL, InstructionType.CONTROL_FLOW,
                                   [OperandType.IMMEDIATE]),
    Opcode.RET: InstructionFormat("RET", Opcode.RET, InstructionType.CONTROL_FLOW,
                                  []),
    Opcode.EXIT: InstructionFormat("EXIT", Opcode.EXIT, InstructionType.CONTROL_FLOW,
                                   []),
    Opcode.CAL: InstructionFormat("CAL", Opcode.CAL, InstructionType.CONTROL_FLOW,
                                  [OperandType.IMMEDIATE]),

    # ==================== Predicates ====================
    Opcode.PSETP: InstructionFormat("PSETP", Opcode.PSETP, InstructionType.LOGICAL,
                                    [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.SETP: InstructionFormat("SETP", Opcode.SETP, InstructionType.LOGICAL,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.SELP: InstructionFormat("SELP", Opcode.SELP, InstructionType.LOGICAL,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.PREDICATE]),

    # ==================== Warp Level ====================
    Opcode.BAR: InstructionFormat("BAR", Opcode.BAR, InstructionType.WARP_LEVEL,
                                  [OperandType.IMMEDIATE]),
    Opcode.BAR_WARP: InstructionFormat("BAR.WARP", Opcode.BAR_WARP, InstructionType.WARP_LEVEL,
                                        []),
    Opcode.MEMBAR: InstructionFormat("MEMBAR", Opcode.MEMBAR, InstructionType.WARP_LEVEL,
                                     [OperandType.IMMEDIATE]),
    Opcode.VOTE: InstructionFormat("VOTE", Opcode.VOTE, InstructionType.WARP_LEVEL,
                                   [OperandType.REGISTER, OperandType.PREDICATE]),
    Opcode.ACTIVEMASK: InstructionFormat("ACTIVEMASK", Opcode.ACTIVEMASK, InstructionType.WARP_LEVEL,
                                         [OperandType.REGISTER]),
    Opcode.ELECT: InstructionFormat("ELECT", Opcode.ELECT, InstructionType.WARP_LEVEL,
                                    [OperandType.REGISTER]),
    Opcode.SHFL: InstructionFormat("SHFL", Opcode.SHFL, InstructionType.WARP_LEVEL,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.IMMEDIATE]),
    Opcode.REDUX: InstructionFormat("REDUX", Opcode.REDUX, InstructionType.WARP_LEVEL,
                                    [OperandType.REGISTER, OperandType.REGISTER, OperandType.IMMEDIATE]),

    # ==================== Atomic Operations ====================
    Opcode.ATOM: InstructionFormat("ATOM", Opcode.ATOM, InstructionType.WARP_LEVEL,
                                   [OperandType.REGISTER, OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.ATOM_ADD: InstructionFormat("ATOM.ADD", Opcode.ATOM_ADD, InstructionType.WARP_LEVEL,
                                       [OperandType.REGISTER, OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.RED: InstructionFormat("RED", Opcode.RED, InstructionType.WARP_LEVEL,
                                  [OperandType.MEMORY, OperandType.REGISTER]),

    # ==================== Tensor Core ====================
    Opcode.HMMA: InstructionFormat("HMMA", Opcode.HMMA, InstructionType.TENSOR_CORE,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER],
                                   is_tensor=True),
    Opcode.MMA: InstructionFormat("MMA", Opcode.MMA, InstructionType.TENSOR_CORE,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER],
                                  is_tensor=True),
    Opcode.WMMA: InstructionFormat("WMMA", Opcode.WMMA, InstructionType.TENSOR_CORE,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER],
                                   is_tensor=True),

    # ==================== Warpgroup Matrix (WGMMA) ====================
    Opcode.WGMMA: InstructionFormat("WGMMA", Opcode.WGMMA, InstructionType.TENSOR_CORE,
                                    [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER],
                                    is_tensor=True),
    Opcode.WGMMA_MMA: InstructionFormat("WGMMA.MMA", Opcode.WGMMA_MMA, InstructionType.TENSOR_CORE,
                                        [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.IMMEDIATE],
                                        is_tensor=True),
    Opcode.WGMMA_MMA_ASYNC: InstructionFormat("WGMMA.MMA_ASYNC", Opcode.WGMMA_MMA_ASYNC, InstructionType.TENSOR_CORE,
                                              [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER],
                                              is_tensor=True),

    # ==================== Tensor Memory Accelerator (TMA) ====================
    Opcode.TMA: InstructionFormat("TMA", Opcode.TMA, InstructionType.MEMORY,
                                  [OperandType.MEMORY, OperandType.MEMORY]),
    Opcode.TMA_LOAD: InstructionFormat("TMA.LOAD", Opcode.TMA_LOAD, InstructionType.MEMORY,
                                       [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.TMA_STORE: InstructionFormat("TMA.STORE", Opcode.TMA_STORE, InstructionType.MEMORY,
                                        [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.TMA_WAIT: InstructionFormat("TMA.WAIT", Opcode.TMA_WAIT, InstructionType.WARP_LEVEL,
                                       [OperandType.IMMEDIATE]),

    # ==================== Memory Barrier (mbarrier) ====================
    Opcode.MBARRIER_INIT: InstructionFormat("MBARRIER_INIT", Opcode.MBARRIER_INIT, InstructionType.WARP_LEVEL,
                                            [OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MBARRIER_INIT_DOT: InstructionFormat("mbarrier.init", Opcode.MBARRIER_INIT_DOT, InstructionType.WARP_LEVEL,
                                                [OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MBARRIER_INVAL: InstructionFormat("MBARRIER_INVAL", Opcode.MBARRIER_INVAL, InstructionType.WARP_LEVEL,
                                             [OperandType.MEMORY]),
    Opcode.MBARRIER_INVAL_DOT: InstructionFormat("mbarrier.inval", Opcode.MBARRIER_INVAL_DOT, InstructionType.WARP_LEVEL,
                                                 [OperandType.MEMORY]),
    Opcode.MBARRIER_ARRIVE: InstructionFormat("MBARRIER_ARRIVE", Opcode.MBARRIER_ARRIVE, InstructionType.WARP_LEVEL,
                                              [OperandType.MEMORY]),
    Opcode.MBARRIER_ARRIVE_DOT: InstructionFormat("mbarrier.arrive", Opcode.MBARRIER_ARRIVE_DOT, InstructionType.WARP_LEVEL,
                                                  [OperandType.MEMORY]),
    Opcode.MBARRIER_TEST_WAIT: InstructionFormat("MBARRIER_TEST_WAIT", Opcode.MBARRIER_TEST_WAIT, InstructionType.WARP_LEVEL,
                                                 [OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MBARRIER_TEST_WAIT_DOT: InstructionFormat("mbarrier.test_wait", Opcode.MBARRIER_TEST_WAIT_DOT, InstructionType.WARP_LEVEL,
                                                     [OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MBARRIER_EXPECT_TX: InstructionFormat("MBARRIER_EXPECT_TX", Opcode.MBARRIER_EXPECT_TX, InstructionType.WARP_LEVEL,
                                                 [OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MBARRIER_EXPECT_TX_DOT: InstructionFormat("mbarrier.expect_tx", Opcode.MBARRIER_EXPECT_TX_DOT, InstructionType.WARP_LEVEL,
                                                     [OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MBARRIER_COMPLETE_TX: InstructionFormat("MBARRIER_COMPLETE_TX", Opcode.MBARRIER_COMPLETE_TX, InstructionType.WARP_LEVEL,
                                                   [OperandType.MEMORY]),
    Opcode.MBARRIER_COMPLETE_TX_DOT: InstructionFormat("mbarrier.complete_tx", Opcode.MBARRIER_COMPLETE_TX_DOT, InstructionType.WARP_LEVEL,
                                                       [OperandType.MEMORY]),
    # Additional mbarrier instructions
    Opcode.MBARRIER_ARRIVE_DROP: InstructionFormat("MBARRIER_ARRIVE_DROP", Opcode.MBARRIER_ARRIVE_DROP, InstructionType.WARP_LEVEL,
                                                   [OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MBARRIER_ARRIVE_DROP_DOT: InstructionFormat("mbarrier.arrive_drop", Opcode.MBARRIER_ARRIVE_DROP_DOT, InstructionType.WARP_LEVEL,
                                                       [OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MBARRIER_TRY_WAIT: InstructionFormat("MBARRIER_TRY_WAIT", Opcode.MBARRIER_TRY_WAIT, InstructionType.WARP_LEVEL,
                                                [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.MBARRIER_TRY_WAIT_DOT: InstructionFormat("mbarrier.try_wait", Opcode.MBARRIER_TRY_WAIT_DOT, InstructionType.WARP_LEVEL,
                                                    [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.MBARRIER_PENDING_COUNT: InstructionFormat("MBARRIER_PENDING_COUNT", Opcode.MBARRIER_PENDING_COUNT, InstructionType.WARP_LEVEL,
                                                     [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.MBARRIER_PENDING_COUNT_DOT: InstructionFormat("mbarrier.pending_count", Opcode.MBARRIER_PENDING_COUNT_DOT, InstructionType.WARP_LEVEL,
                                                         [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.CP_ASYNC_MBARRIER_ARRIVE: InstructionFormat("CP_ASYNC_MBARRIER_ARRIVE", Opcode.CP_ASYNC_MBARRIER_ARRIVE, InstructionType.MEMORY,
                                                       [OperandType.MEMORY, OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.CP_ASYNC_MBARRIER_ARRIVE_DOT: InstructionFormat("cp.async.mbarrier.arrive", Opcode.CP_ASYNC_MBARRIER_ARRIVE_DOT, InstructionType.MEMORY,
                                                           [OperandType.MEMORY, OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),

    # ==================== Proxy Fence ====================
    Opcode.FENCE_PROXY: InstructionFormat("FENCE_PROXY", Opcode.FENCE_PROXY, InstructionType.WARP_LEVEL, []),
    Opcode.FENCE_PROXY_DOT: InstructionFormat("fence.proxy", Opcode.FENCE_PROXY_DOT, InstructionType.WARP_LEVEL, []),
    Opcode.FENCE_PROXY_ASYNC: InstructionFormat("FENCE_PROXY_ASYNC", Opcode.FENCE_PROXY_ASYNC, InstructionType.WARP_LEVEL, []),
    Opcode.FENCE_PROXY_ASYNC_DOT: InstructionFormat("fence.proxy.async", Opcode.FENCE_PROXY_ASYNC_DOT, InstructionType.WARP_LEVEL, []),
    Opcode.FENCE_PROXY_TENSORMAP: InstructionFormat("FENCE_PROXY_TENSORMAP", Opcode.FENCE_PROXY_TENSORMAP, InstructionType.WARP_LEVEL, []),
    Opcode.FENCE_PROXY_TENSORMAP_DOT: InstructionFormat("fence.proxy.tensormap", Opcode.FENCE_PROXY_TENSORMAP_DOT, InstructionType.WARP_LEVEL, []),

    # ==================== TMA (Tensor Memory Accelerator) ====================
    Opcode.CP_ASYNC_BULK: InstructionFormat("CP_ASYNC_BULK", Opcode.CP_ASYNC_BULK, InstructionType.MEMORY,
                                           [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.CP_ASYNC_BULK_DOT: InstructionFormat("cp.async.bulk", Opcode.CP_ASYNC_BULK_DOT, InstructionType.MEMORY,
                                                [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.CP_ASYNC_BULK_TENSOR: InstructionFormat("CP_ASYNC_BULK_TENSOR", Opcode.CP_ASYNC_BULK_TENSOR, InstructionType.MEMORY,
                                                   [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE, OperandType.MEMORY]),
    Opcode.CP_ASYNC_BULK_TENSOR_DOT: InstructionFormat("cp.async.bulk.tensor", Opcode.CP_ASYNC_BULK_TENSOR_DOT, InstructionType.MEMORY,
                                                       [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE, OperandType.MEMORY]),
    Opcode.CP_ASYNC_BULK_PREFETCH: InstructionFormat("CP_ASYNC_BULK_PREFETCH", Opcode.CP_ASYNC_BULK_PREFETCH, InstructionType.MEMORY,
                                                      [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.CP_ASYNC_BULK_PREFETCH_DOT: InstructionFormat("cp.async.bulk.prefetch", Opcode.CP_ASYNC_BULK_PREFETCH_DOT, InstructionType.MEMORY,
                                                          [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.CP_ASYNC_BULK_PREFETCH_TENSOR: InstructionFormat("CP_ASYNC_BULK_PREFETCH_TENSOR", Opcode.CP_ASYNC_BULK_PREFETCH_TENSOR, InstructionType.MEMORY,
                                                             [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE, OperandType.MEMORY]),
    Opcode.CP_ASYNC_BULK_PREFETCH_TENSOR_DOT: InstructionFormat("cp.async.bulk.prefetch.tensor", Opcode.CP_ASYNC_BULK_PREFETCH_TENSOR_DOT, InstructionType.MEMORY,
                                                                 [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE, OperandType.MEMORY]),
    Opcode.CP_REDUCE_ASYNC_BULK: InstructionFormat("CP_REDUCE_ASYNC_BULK", Opcode.CP_REDUCE_ASYNC_BULK, InstructionType.MEMORY,
                                                   [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.CP_REDUCE_ASYNC_BULK_DOT: InstructionFormat("cp.reduce.async.bulk", Opcode.CP_REDUCE_ASYNC_BULK_DOT, InstructionType.MEMORY,
                                                       [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.CP_REDUCE_ASYNC_BULK_TENSOR: InstructionFormat("CP_REDUCE_ASYNC_BULK_TENSOR", Opcode.CP_REDUCE_ASYNC_BULK_TENSOR, InstructionType.MEMORY,
                                                          [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE, OperandType.MEMORY]),
    Opcode.CP_REDUCE_ASYNC_BULK_TENSOR_DOT: InstructionFormat("cp.reduce.async.bulk.tensor", Opcode.CP_REDUCE_ASYNC_BULK_TENSOR_DOT, InstructionType.MEMORY,
                                                              [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE, OperandType.MEMORY]),
    Opcode.CP_ASYNC_BULK_COMMIT_GROUP: InstructionFormat("CP_ASYNC_BULK_COMMIT_GROUP", Opcode.CP_ASYNC_BULK_COMMIT_GROUP, InstructionType.WARP_LEVEL,
                                                         [OperandType.IMMEDIATE]),
    Opcode.CP_ASYNC_BULK_COMMIT_GROUP_DOT: InstructionFormat("cp.async.bulk.commit_group", Opcode.CP_ASYNC_BULK_COMMIT_GROUP_DOT, InstructionType.WARP_LEVEL,
                                                             [OperandType.IMMEDIATE]),
    Opcode.CP_ASYNC_BULK_WAIT_GROUP: InstructionFormat("CP_ASYNC_BULK_WAIT_GROUP", Opcode.CP_ASYNC_BULK_WAIT_GROUP, InstructionType.WARP_LEVEL,
                                                       [OperandType.IMMEDIATE]),
    Opcode.CP_ASYNC_BULK_WAIT_GROUP_DOT: InstructionFormat("cp.async.bulk.wait_group", Opcode.CP_ASYNC_BULK_WAIT_GROUP_DOT, InstructionType.WARP_LEVEL,
                                                           [OperandType.IMMEDIATE]),
    Opcode.MULTIMEM_CP_ASYNC_BULK: InstructionFormat("MULTIMEM_CP_ASYNC_BULK", Opcode.MULTIMEM_CP_ASYNC_BULK, InstructionType.MEMORY,
                                                     [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MULTIMEM_CP_ASYNC_BULK_DOT: InstructionFormat("multimem.cp.async.bulk", Opcode.MULTIMEM_CP_ASYNC_BULK_DOT, InstructionType.MEMORY,
                                                         [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MULTIMEM_CP_REDUCE_ASYNC_BULK: InstructionFormat("MULTIMEM_CP_REDUCE_ASYNC_BULK", Opcode.MULTIMEM_CP_REDUCE_ASYNC_BULK, InstructionType.MEMORY,
                                                            [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),
    Opcode.MULTIMEM_CP_REDUCE_ASYNC_BULK_DOT: InstructionFormat("multimem.cp.reduce.async.bulk", Opcode.MULTIMEM_CP_REDUCE_ASYNC_BULK_DOT, InstructionType.MEMORY,
                                                                [OperandType.MEMORY, OperandType.MEMORY, OperandType.IMMEDIATE]),

    # ==================== Warpgroup Operations ====================
    Opcode.WGMMA_FENCE: InstructionFormat("WGMMA_FENCE", Opcode.WGMMA_FENCE, InstructionType.WARP_LEVEL, []),
    Opcode.WGMMA_FENCE_DOT: InstructionFormat("wgmma.fence", Opcode.WGMMA_FENCE_DOT, InstructionType.WARP_LEVEL, []),
    Opcode.WGMMA_COMMIT_GROUP: InstructionFormat("WGMMA_COMMIT_GROUP", Opcode.WGMMA_COMMIT_GROUP, InstructionType.WARP_LEVEL, []),
    Opcode.WGMMA_COMMIT_GROUP_DOT: InstructionFormat("wgmma.commit_group", Opcode.WGMMA_COMMIT_GROUP_DOT, InstructionType.WARP_LEVEL, []),
    Opcode.WGMMA_WAIT_GROUP: InstructionFormat("WGMMA_WAIT_GROUP", Opcode.WGMMA_WAIT_GROUP, InstructionType.WARP_LEVEL,
                                              [OperandType.IMMEDIATE]),
    Opcode.WGMMA_WAIT_GROUP_DOT: InstructionFormat("wgmma.wait_group", Opcode.WGMMA_WAIT_GROUP_DOT, InstructionType.WARP_LEVEL,
                                                  [OperandType.IMMEDIATE]),
    Opcode.WGMMA_MMA_ASYNC_SP: InstructionFormat("WGMMA_MMA_ASYNC_SP", Opcode.WGMMA_MMA_ASYNC_SP, InstructionType.TENSOR_CORE,
                                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER],
                                                 is_tensor=True),
    Opcode.WGMMA_MMA_ASYNC_SP_DOT: InstructionFormat("wgmma.mma_async.sp", Opcode.WGMMA_MMA_ASYNC_SP_DOT, InstructionType.TENSOR_CORE,
                                                     [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER],
                                                     is_tensor=True),

    # ==================== PTX Instructions (from isa.md) ====================
    # Arithmetic
    Opcode.ADD: InstructionFormat("ADD", Opcode.ADD, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.SUB: InstructionFormat("SUB", Opcode.SUB, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.MUL: InstructionFormat("MUL", Opcode.MUL, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.MAD: InstructionFormat("MAD", Opcode.MAD, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.DIV: InstructionFormat("DIV", Opcode.DIV, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.REM: InstructionFormat("REM", Opcode.REM, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.ABS: InstructionFormat("ABS", Opcode.ABS, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.NEG: InstructionFormat("NEG", Opcode.NEG, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.MIN: InstructionFormat("MIN", Opcode.MIN, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.MAX: InstructionFormat("MAX", Opcode.MAX, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.FMA: InstructionFormat("FMA", Opcode.FMA, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.MUL24: InstructionFormat("MUL24", Opcode.MUL24, InstructionType.ARITHMETIC,
                                    [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.MAD24: InstructionFormat("MAD24", Opcode.MAD24, InstructionType.ARITHMETIC,
                                    [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.SAD: InstructionFormat("SAD", Opcode.SAD, InstructionType.ARITHMETIC,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),

    # Math functions
    Opcode.SQRT: InstructionFormat("SQRT", Opcode.SQRT, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.RSQRT: InstructionFormat("RSQRT", Opcode.RSQRT, InstructionType.FLOATING,
                                    [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.SIN: InstructionFormat("SIN", Opcode.SIN, InstructionType.FLOATING,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.COS: InstructionFormat("COS", Opcode.COS, InstructionType.FLOATING,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.LG2: InstructionFormat("LG2", Opcode.LG2, InstructionType.FLOATING,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.EX2: InstructionFormat("EX2", Opcode.EX2, InstructionType.FLOATING,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.RCP: InstructionFormat("RCP", Opcode.RCP, InstructionType.FLOATING,
                                  [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.COPYSIGN: InstructionFormat("COPYSIGN", Opcode.COPYSIGN, InstructionType.FLOATING,
                                       [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.TANH: InstructionFormat("TANH", Opcode.TANH, InstructionType.FLOATING,
                                   [OperandType.REGISTER, OperandType.REGISTER]),

    # Bit manipulation (PTX)
    Opcode.BFIND: InstructionFormat("BFIND", Opcode.BFIND, InstructionType.LOGICAL,
                                    [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.BMSK: InstructionFormat("BMSK", Opcode.BMSK, InstructionType.LOGICAL,
                                   [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.RBITS: InstructionFormat("RBITS", Opcode.RBITS, InstructionType.LOGICAL,
                                    [OperandType.REGISTER, OperandType.REGISTER]),

    # Vector operations
    Opcode.VADD: InstructionFormat("VADD", Opcode.VADD, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.VSUB: InstructionFormat("VSUB", Opcode.VSUB, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.VMIN: InstructionFormat("VMIN", Opcode.VMIN, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.VMAX: InstructionFormat("VMAX", Opcode.VMAX, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.DP2A: InstructionFormat("DP2A", Opcode.DP2A, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),

    # Memory operations
    Opcode.ISSPACEP: InstructionFormat("ISSPACEP", Opcode.ISSPACEP, InstructionType.LOGICAL,
                                       [OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.ISTYPEP: InstructionFormat("ISTYPEP", Opcode.ISTYPEP, InstructionType.LOGICAL,
                                      [OperandType.REGISTER, OperandType.REGISTER]),

    # Barrier operations (PTX)
    Opcode.BARRIER: InstructionFormat("BARRIER", Opcode.BARRIER, InstructionType.WARP_LEVEL,
                                      [OperandType.IMMEDIATE]),
    Opcode.BARRIER_CLUSTER: InstructionFormat("BARRIER.CLUSTER", Opcode.BARRIER_CLUSTER, InstructionType.WARP_LEVEL,
                                              [OperandType.IMMEDIATE]),
    Opcode.BARRIER_CTA: InstructionFormat("BARRIER.CTA", Opcode.BARRIER_CTA, InstructionType.WARP_LEVEL,
                                         [OperandType.IMMEDIATE]),

    # FENCE operations (PTX)
    Opcode.FENCE_SC: InstructionFormat("FENCE.SC", Opcode.FENCE_SC, InstructionType.WARP_LEVEL,
                                       [OperandType.IMMEDIATE]),
    Opcode.FENCE_ACQ_REL: InstructionFormat("FENCE.ACQ_REL", Opcode.FENCE_ACQ_REL, InstructionType.WARP_LEVEL,
                                            [OperandType.IMMEDIATE]),

    # Texture operations (PTX)
    Opcode.TEX: InstructionFormat("TEX", Opcode.TEX, InstructionType.MEMORY,
                                  [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.TLD4: InstructionFormat("TLD4", Opcode.TLD4, InstructionType.MEMORY,
                                   [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.TXQ: InstructionFormat("TXQ", Opcode.TXQ, InstructionType.MEMORY,
                                  [OperandType.REGISTER, OperandType.IMMEDIATE]),

    # Surface operations (PTX)
    Opcode.SULD: InstructionFormat("SULD", Opcode.SULD, InstructionType.MEMORY,
                                   [OperandType.REGISTER, OperandType.MEMORY]),
    Opcode.SUST: InstructionFormat("SUST", Opcode.SUST, InstructionType.MEMORY,
                                   [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.SURED: InstructionFormat("SURED", Opcode.SURED, InstructionType.MEMORY,
                                    [OperandType.MEMORY, OperandType.REGISTER]),
    Opcode.SUQ: InstructionFormat("SUQ", Opcode.SUQ, InstructionType.MEMORY,
                                  [OperandType.REGISTER, OperandType.MEMORY]),

    # Multimem
    Opcode.MULTIMEM: InstructionFormat("MULTIMEM", Opcode.MULTIMEM, InstructionType.MEMORY,
                                       [OperandType.REGISTER, OperandType.MEMORY]),

    # Tensormap
    Opcode.TENSORMAP: InstructionFormat("TENSORMAP", Opcode.TENSORMAP, InstructionType.MEMORY,
                                        [OperandType.REGISTER, OperandType.MEMORY]),

    # Cluster launch
    Opcode.CLUSTERLAUNCHCONTROL: InstructionFormat("CLUSTERLAUNCHCONTROL", Opcode.CLUSTERLAUNCHCONTROL, InstructionType.CONTROL_FLOW,
                                                   [OperandType.IMMEDIATE]),

    # Grid dependency
    Opcode.GRIDDEPCONTROL: InstructionFormat("GRIDDEPCONTROL", Opcode.GRIDDEPCONTROL, InstructionType.CONTROL_FLOW,
                                             [OperandType.IMMEDIATE]),

    # Get CTA rank
    Opcode.GETCTARANK: InstructionFormat("GETCTARANK", Opcode.GETCTARANK, InstructionType.WARP_LEVEL,
                                         [OperandType.REGISTER]),

    # Nanosleep
    Opcode.NANOSLEEP: InstructionFormat("NANOSLEEP", Opcode.NANOSLEEP, InstructionType.SPECIAL,
                                        [OperandType.IMMEDIATE]),

    # Member mask
    Opcode.MEMBERMASK: InstructionFormat("MEMBERMASK", Opcode.MEMBERMASK, InstructionType.WARP_LEVEL,
                                        [OperandType.REGISTER]),

    # Map operations
    Opcode.MAPA: InstructionFormat("MAPA", Opcode.MAPA, InstructionType.ARITHMETIC,
                                   [OperandType.REGISTER, OperandType.REGISTER]),

    # Comparison operators (PTX)
    Opcode.EQ: InstructionFormat("EQ", Opcode.EQ, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.NE: InstructionFormat("NE", Opcode.NE, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.LT: InstructionFormat("LT", Opcode.LT, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.LE: InstructionFormat("LE", Opcode.LE, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.GT: InstructionFormat("GT", Opcode.GT, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.GE: InstructionFormat("GE", Opcode.GE, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.LTU: InstructionFormat("LTU", Opcode.LTU, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.LEU: InstructionFormat("LEU", Opcode.LEU, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.GTU: InstructionFormat("GTU", Opcode.GTU, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.GEU: InstructionFormat("GEU", Opcode.GEU, InstructionType.LOGICAL,
                                  [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.LO: InstructionFormat("LO", Opcode.LO, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.HI: InstructionFormat("HI", Opcode.HI, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.LS: InstructionFormat("LS", Opcode.LS, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.HS: InstructionFormat("HS", Opcode.HS, InstructionType.LOGICAL,
                                 [OperandType.REGISTER, OperandType.REGISTER, OperandType.REGISTER]),
    Opcode.EQU: InstructionFormat("EQU", Opcode.EQU, InstructionType.LOGICAL,
                                  [OperandType.IMMEDIATE]),
}


def get_format(opcode: Opcode) -> Optional[InstructionFormat]:
    """Get the format specification for an opcode."""
    return INSTRUCTION_FORMATS.get(opcode)


def is_load_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a load instruction."""
    return opcode in {
        Opcode.LDG, Opcode.LD, Opcode.LDS, Opcode.LDSM,
        Opcode.LDC, Opcode.LDL, Opcode.LDU,
        Opcode.LDMATRIX, Opcode.PREFETCH, Opcode.PREFETCHU
    }


def is_store_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a store instruction."""
    return opcode in {
        Opcode.STG, Opcode.ST, Opcode.STS, Opcode.STL,
        Opcode.STMATRIX
    }


def is_branch_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a branch instruction."""
    return opcode in {Opcode.BRA, Opcode.BRX, Opcode.BSS, Opcode.JMP}


def is_call_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a call instruction."""
    return opcode in {Opcode.CALL, Opcode.CAL}


def is_tensor_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a Tensor Core instruction."""
    return opcode in {Opcode.HMMA, Opcode.MMA, Opcode.WMMA, Opcode.MMA_SP, Opcode.WGMMA}


def is_atomic_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is an atomic instruction."""
    return opcode in {
        Opcode.ATOM, Opcode.ATOM_ADD, Opcode.ATOM_SUB,
        Opcode.ATOM_MIN, Opcode.ATOM_MAX, Opcode.ATOM_AND,
        Opcode.ATOM_OR, Opcode.ATOM_XOR, Opcode.ATOM_XCHG,
        Opcode.ATOM_CAS, Opcode.RED
    }


def is_barrier_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a barrier instruction."""
    return opcode in {
        Opcode.BAR, Opcode.BAR_WARP, Opcode.MEMBAR,
        Opcode.BARRIER, Opcode.BARRIER_CTA, Opcode.BARRIER_CLUSTER,
        Opcode.FENCE_SC, Opcode.FENCE_ACQ_REL,
        Opcode.FENCE_PROXY, Opcode.FENCE_PROXY_DOT,
        Opcode.FENCE_PROXY_ASYNC, Opcode.FENCE_PROXY_ASYNC_DOT,
        Opcode.FENCE_PROXY_TENSORMAP, Opcode.FENCE_PROXY_TENSORMAP_DOT,
        Opcode.MBARRIER_INIT, Opcode.MBARRIER_INIT_DOT,
        Opcode.MBARRIER_INVAL, Opcode.MBARRIER_INVAL_DOT,
        Opcode.MBARRIER_ARRIVE, Opcode.MBARRIER_ARRIVE_DOT,
        Opcode.MBARRIER_ARRIVE_DROP, Opcode.MBARRIER_ARRIVE_DROP_DOT,
        Opcode.MBARRIER_TEST_WAIT, Opcode.MBARRIER_TEST_WAIT_DOT,
        Opcode.MBARRIER_TRY_WAIT, Opcode.MBARRIER_TRY_WAIT_DOT,
        Opcode.MBARRIER_PENDING_COUNT, Opcode.MBARRIER_PENDING_COUNT_DOT,
        Opcode.MBARRIER_EXPECT_TX, Opcode.MBARRIER_EXPECT_TX_DOT,
        Opcode.MBARRIER_COMPLETE_TX, Opcode.MBARRIER_COMPLETE_TX_DOT,
        Opcode.CP_ASYNC_MBARRIER_ARRIVE, Opcode.CP_ASYNC_MBARRIER_ARRIVE_DOT,
        Opcode.WGMMA_FENCE, Opcode.WGMMA_FENCE_DOT,
        Opcode.WGMMA_COMMIT_GROUP, Opcode.WGMMA_COMMIT_GROUP_DOT,
        Opcode.WGMMA_WAIT_GROUP, Opcode.WGMMA_WAIT_GROUP_DOT,
    }


def is_tma_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a TMA (Tensor Memory Accelerator) instruction."""
    return opcode in {
        Opcode.TMA, Opcode.TMA_LOAD, Opcode.TMA_STORE, Opcode.TMA_WAIT,
        Opcode.CP_ASYNC_BULK, Opcode.CP_ASYNC_BULK_DOT,
        Opcode.CP_ASYNC_BULK_TENSOR, Opcode.CP_ASYNC_BULK_TENSOR_DOT,
        Opcode.CP_ASYNC_BULK_PREFETCH, Opcode.CP_ASYNC_BULK_PREFETCH_DOT,
        Opcode.CP_ASYNC_BULK_PREFETCH_TENSOR, Opcode.CP_ASYNC_BULK_PREFETCH_TENSOR_DOT,
        Opcode.CP_REDUCE_ASYNC_BULK, Opcode.CP_REDUCE_ASYNC_BULK_DOT,
        Opcode.CP_REDUCE_ASYNC_BULK_TENSOR, Opcode.CP_REDUCE_ASYNC_BULK_TENSOR_DOT,
        Opcode.CP_ASYNC_BULK_COMMIT_GROUP, Opcode.CP_ASYNC_BULK_COMMIT_GROUP_DOT,
        Opcode.CP_ASYNC_BULK_WAIT_GROUP, Opcode.CP_ASYNC_BULK_WAIT_GROUP_DOT,
        Opcode.MULTIMEM_CP_ASYNC_BULK, Opcode.MULTIMEM_CP_ASYNC_BULK_DOT,
        Opcode.MULTIMEM_CP_REDUCE_ASYNC_BULK, Opcode.MULTIMEM_CP_REDUCE_ASYNC_BULK_DOT,
        Opcode.CP_ASYNC_MBARRIER_ARRIVE, Opcode.CP_ASYNC_MBARRIER_ARRIVE_DOT,
    }


def is_wgmma_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a WGMMA (Warpgroup MMA) instruction."""
    return opcode in {
        Opcode.WGMMA, Opcode.WGMMA_MMA, Opcode.WGMMA_MMA_ASYNC,
        Opcode.WGMMA_MMA_ASYNC_SP, Opcode.WGMMA_MMA_ASYNC_SP_DOT,
        # Note: WGMMA_FENCE, WGMMA_COMMIT_GROUP, WGMMA_WAIT_GROUP are barrier instructions
    }


if __name__ == "__main__":
    # Test instruction format lookup
    fmt = get_format(Opcode.FADD)
    if fmt:
        print(f"Format: {fmt.name}")
        print(f"Operands: {fmt.operands}")

    # Test instruction type checks
    print(f"\nIs LDG a load? {is_load_instruction(Opcode.LDG)}")
    print(f"Is STG a store? {is_store_instruction(Opcode.STG)}")
    print(f"Is BRA a branch? {is_branch_instruction(Opcode.BRA)}")
    print(f"Is HMMA a tensor instruction? {is_tensor_instruction(Opcode.HMMA)}")
