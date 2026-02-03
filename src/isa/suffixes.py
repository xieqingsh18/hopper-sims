"""
PTX Instruction Suffixes and Modifiers

This module defines the various suffixes that modify PTX instruction behavior,
including type suffixes, memory spaces, scope, and synchronization modifiers.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List


class TypeSuffix(Enum):
    """Data type suffixes for PTX instructions."""
    B8 = "b8"
    B16 = "b16"
    B32 = "b32"
    B64 = "b64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    S8 = "s8"
    S16 = "s16"
    S32 = "s32"
    S64 = "s64"
    F16 = "f16"
    F16X2 = "f16x2"
    F32 = "f32"
    F64 = "f64"
    BF16 = "bf16"
    TF32 = "tf32"


class MemorySpace(Enum):
    """Memory space qualifiers."""
    GLOBAL = "global"
    SHARED = "shared"
    LOCAL = "local"
    PARAM = "param"
    CONST = "const"
    REG = "reg"


class SharedScope(Enum):
    """Shared memory sub-qualifiers."""
    NONE = ""          # Default (no qualifier)
    CTA = "cta"        # CTA-wide shared
    CLUSTER = "cluster"  # Cluster-wide shared


class ParamScope(Enum):
    """Parameter space qualifiers."""
    NONE = ""          # Default
    ENTRY = "entry"    # Kernel entry parameter
    FUNC = "func"      # Function parameter


class MemoryScope(Enum):
    """Memory scope qualifiers."""
    CTA = "cta"
    CLUSTER = "cluster"
    GPU = "gpu"
    SYS = "sys"


class SyncModifier(Enum):
    """Synchronization modifiers."""
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    VOLATILE = "volatile"
    WEAK = "weak"
    MMIO = "mmio"


class CacheOp(Enum):
    """Cache operations."""
    CV = "cv"          # Cache volatile
    CG = "cg"          # Cache at global level
    CS = "cs"          # Cache streaming
    LU = "lu"          # Last use
    WB = "wb"          # Write-back


class MatrixShape(Enum):
    """Matrix shapes for ldmatrix/stmatrix/wgmma."""
    M8N8 = "m8n8"
    M8N16 = "m8n16"
    M16N8 = "m16n8"
    M16N16 = "m16n16"
    M32N8 = "m32n8"
    M64N8 = "m64n8"


class MatrixModifier(Enum):
    """Matrix instruction modifiers."""
    SYNC = "sync"
    ALIGNED = "aligned"
    TRANS = "trans"
    SPARSE = "sparse"


class VectorWidth(Enum):
    """Vector width modifiers."""
    X1 = "x1"
    X2 = "x2"
    X4 = "x4"


@dataclass
class InstructionSuffixes:
    """
    Collection of all suffixes for a PTX instruction.

    Example: "ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16"
    - modifier: [MatrixModifier.SYNC, MatrixModifier.ALIGNED]
    - shape: MatrixShape.M8N8
    - vector_width: VectorWidth.X1
    - memory_space: MemorySpace.SHARED
    - shared_scope: SharedScope.CTA
    - type_suffix: TypeSuffix.B16
    """
    # Type suffix (e.g., .b32, .f32)
    type_suffix: Optional[TypeSuffix] = None

    # Memory space (e.g., .global, .shared)
    memory_space: Optional[MemorySpace] = None

    # Shared memory sub-qualifier (e.g., ::cta, ::cluster)
    shared_scope: SharedScope = SharedScope.NONE

    # Parameter space qualifier (e.g., ::entry, ::func)
    param_scope: ParamScope = ParamScope.NONE

    # Memory scope (e.g., .cta, .cluster, .gpu, .sys)
    memory_scope: Optional[MemoryScope] = None

    # Synchronization modifier (e.g., .relaxed, .acquire, .release)
    sync_modifier: Optional[SyncModifier] = None

    # Cache operation (e.g., .cv, .cg, .cs)
    cache_op: Optional[CacheOp] = None

    # Matrix shape (e.g., .m8n8, .m16n8, .m64n8)
    matrix_shape: Optional[MatrixShape] = None

    # Matrix modifiers (e.g., .sync, .aligned, .trans)
    matrix_modifiers: List[MatrixModifier] = None

    # Vector width (e.g., .x1, .x2, .x4)
    vector_width: Optional[VectorWidth] = None

    # Raw suffix string for unhandled suffixes
    raw_suffix: Optional[str] = None

    def __post_init__(self):
        if self.matrix_modifiers is None:
            self.matrix_modifiers = []

    def __str__(self) -> str:
        """Convert suffixes to string representation."""
        parts = []

        # Add sync modifier (e.g., .relaxed, .acquire, .release)
        if self.sync_modifier:
            parts.append(f".{self.sync_modifier.value}")

        # Add matrix modifiers
        for mod in self.matrix_modifiers:
            parts.append(f".{mod.value}")

        # Add matrix shape
        if self.matrix_shape:
            parts.append(f".{self.matrix_shape.value}")

        # Add vector width
        if self.vector_width:
            parts.append(f".{self.vector_width.value}")

        # Add memory space
        if self.memory_space:
            mem_str = self.memory_space.value
            # Add sub-qualifiers
            if self.memory_space == MemorySpace.SHARED and self.shared_scope != SharedScope.NONE:
                mem_str += f"::{self.shared_scope.value}"
            elif self.memory_space == MemorySpace.PARAM and self.param_scope != ParamScope.NONE:
                mem_str += f"::{self.param_scope.value}"
            parts.append(f".{mem_str}")

        # Add memory scope
        if self.memory_scope:
            parts.append(f".{self.memory_scope.value}")

        # Add cache op
        if self.cache_op:
            parts.append(f".{self.cache_op.value}")

        # Add type suffix
        if self.type_suffix:
            parts.append(f".{self.type_suffix.value}")

        # Add raw suffix if present
        if self.raw_suffix:
            parts.append(self.raw_suffix)

        return "".join(parts)

    def __repr__(self) -> str:
        return f"InstructionSuffixes({self.__str__()})"


class SuffixParser:
    """Parser for PTX instruction suffixes."""

    # Maps suffix strings to enums
    TYPE_SUFFIXES = {t.value: t for t in TypeSuffix}
    MEMORY_SPACES = {s.value: s for s in MemorySpace}
    SHARED_SCOPES = {s.value: s for s in SharedScope if s.value}
    PARAM_SCOPES = {s.value: s for s in ParamScope if s.value}
    MEMORY_SCOPES = {s.value: s for s in MemoryScope}
    SYNC_MODIFIERS = {s.value: s for s in SyncModifier}
    CACHE_OPS = {c.value: c for c in CacheOp}
    MATRIX_SHAPES = {m.value: m for m in MatrixShape}
    MATRIX_MODIFIERS = {m.value: m for m in MatrixModifier}
    VECTOR_WIDTHS = {v.value: v for v in VectorWidth}

    @classmethod
    def parse(cls, opcode: str, full_opcode: str) -> InstructionSuffixes:
        """
        Parse suffixes from a full opcode string.

        Args:
            opcode: Base opcode (e.g., "ldmatrix")
            full_opcode: Full opcode with suffixes (e.g., "ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16")

        Returns:
            InstructionSuffixes object
        """
        suffixes = InstructionSuffixes()

        if full_opcode == opcode:
            return suffixes

        # Extract suffix part (everything after base opcode)
        suffix_part = full_opcode[len(opcode):]
        if not suffix_part.startswith('.'):
            # Invalid format, return empty
            return suffixes

        # Split by '.' and process each part
        parts = suffix_part.split('.')
        i = 1  # Skip the first empty part before '.'

        while i < len(parts):
            part = parts[i]

            # Handle ::cta or ::cluster (shared memory sub-qualifiers)
            if '::' in part:
                base, scope = part.split('::', 1)
                if base == 'shared' and scope in cls.SHARED_SCOPES:
                    suffixes.memory_space = MemorySpace.SHARED
                    suffixes.shared_scope = cls.SHARED_SCOPES[scope]
                elif base == 'param' and scope in cls.PARAM_SCOPES:
                    suffixes.memory_space = MemorySpace.PARAM
                    suffixes.param_scope = cls.PARAM_SCOPES[scope]
                else:
                    # Unknown qualifier, store as raw
                    suffixes.raw_suffix = f".{part}"
                i += 1
                continue

            # Check for sync modifier
            if part in cls.SYNC_MODIFIERS:
                suffixes.sync_modifier = cls.SYNC_MODIFIERS[part]
                i += 1
                continue

            # Check for memory space
            if part in cls.MEMORY_SPACES:
                # Only set if not already set (shared::cta case handled above)
                if suffixes.memory_space is None:
                    suffixes.memory_space = cls.MEMORY_SPACES[part]
                i += 1
                continue

            # Check for memory scope
            if part in cls.MEMORY_SCOPES:
                suffixes.memory_scope = cls.MEMORY_SCOPES[part]
                i += 1
                continue

            # Check for cache op
            if part in cls.CACHE_OPS:
                suffixes.cache_op = cls.CACHE_OPS[part]
                i += 1
                continue

            # Check for matrix shape
            if part in cls.MATRIX_SHAPES:
                suffixes.matrix_shape = cls.MATRIX_SHAPES[part]
                i += 1
                continue

            # Check for matrix modifier
            if part in cls.MATRIX_MODIFIERS:
                suffixes.matrix_modifiers.append(cls.MATRIX_MODIFIERS[part])
                i += 1
                continue

            # Check for vector width
            if part in cls.VECTOR_WIDTHS:
                suffixes.vector_width = cls.VECTOR_WIDTHS[part]
                i += 1
                continue

            # Check for type suffix (check last)
            if part in cls.TYPE_SUFFIXES:
                suffixes.type_suffix = cls.TYPE_SUFFIXES[part]
                i += 1
                continue

            # Unknown suffix, store as raw
            if suffixes.raw_suffix is None:
                suffixes.raw_suffix = ""
            suffixes.raw_suffix += f".{part}"
            i += 1

        return suffixes


# Example usage and tests
if __name__ == "__main__":
    # Test parsing various instruction formats
    test_cases = [
        ("ldmatrix", "ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16"),
        ("ld", "ld.global.relaxed.gpu.u32"),
        ("mbarrier.init", "mbarrier.init.shared::cta.b64"),
        ("stmatrix", "stmatrix.sync.aligned.m8n8.x1.trans.shared.b16"),
        ("wgmma.mma", "wgmma.mma.m64n8k16.f32.tf32"),
    ]

    for base, full in test_cases:
        suffixes = SuffixParser.parse(base, full)
        print(f"Base: {base}")
        print(f"Full: {full}")
        print(f"Parsed: {suffixes}")
        print(f"Type: {suffixes.type_suffix}")
        print(f"Memory: {suffixes.memory_space}")
        print(f"Sync: {suffixes.sync_modifier}")
        print()
