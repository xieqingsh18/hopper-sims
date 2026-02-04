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
    MBARRIER_TEST_WAIT = "MBARRIER_TEST_WAIT"  # Test and wait mbarrier
    MBARRIER_TEST_WAIT_DOT = "mbarrier.test_wait"  # PTX dot notation
    MBARRIER_EXPECT_TX = "MBARRIER_EXPECT_TX"  # Expect transaction
    MBARRIER_EXPECT_TX_DOT = "mbarrier.expect_tx"  # PTX dot notation
    MBARRIER_COMPLETE_TX = "MBARRIER_COMPLETE_TX"  # Complete transaction
    MBARRIER_COMPLETE_TX_DOT = "mbarrier.complete_tx"  # PTX dot notation

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
                                       [OperandType.MEMORY, OperandType.REGISTER, OperandType.IMMEDIATE]),
    Opcode.TMA_STORE: InstructionFormat("TMA.STORE", Opcode.TMA_STORE, InstructionType.MEMORY,
                                        [OperandType.MEMORY, OperandType.REGISTER, OperandType.IMMEDIATE]),
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
        Opcode.MBARRIER_INIT, Opcode.MBARRIER_INIT_DOT,
        Opcode.MBARRIER_INVAL, Opcode.MBARRIER_INVAL_DOT,
        Opcode.MBARRIER_ARRIVE, Opcode.MBARRIER_ARRIVE_DOT,
        Opcode.MBARRIER_TEST_WAIT, Opcode.MBARRIER_TEST_WAIT_DOT,
        Opcode.MBARRIER_EXPECT_TX, Opcode.MBARRIER_EXPECT_TX_DOT,
        Opcode.MBARRIER_COMPLETE_TX, Opcode.MBARRIER_COMPLETE_TX_DOT
    }


def is_tma_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a TMA (Tensor Memory Accelerator) instruction."""
    return opcode in {Opcode.TMA, Opcode.TMA_LOAD, Opcode.TMA_STORE, Opcode.TMA_WAIT}


def is_wgmma_instruction(opcode: Opcode) -> bool:
    """Check if an opcode is a WGMMA (Warpgroup MMA) instruction."""
    return opcode in {Opcode.WGMMA, Opcode.WGMMA_MMA, Opcode.WGMMA_MMA_ASYNC}


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
