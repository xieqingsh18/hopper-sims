#!/usr/bin/env python3
"""
PTX Instruction Suffixes Demo

Demonstrates the parsing and handling of PTX instruction suffixes including:
- Type suffixes (.b32, .f32, .u64, etc.)
- Memory space qualifiers (.global, .shared, .shared::cta, .shared::cluster, etc.)
- Memory synchronization (.relaxed, .acquire, .release, etc.)
- Memory scope (.cta, .cluster, .gpu, .sys)
- Matrix modifiers (.sync, .aligned, .trans, .m8n8, .m16n8, .m64n8, etc.)
- Vector width (.x1, .x2, .x4)
"""

from src.isa.decoder import InstructionDecoder
from src.isa.suffixes import (
    TypeSuffix, MemorySpace, SyncModifier, MemoryScope,
    MatrixShape, MatrixModifier, SharedScope, SuffixParser
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_type_suffixes():
    """Demonstrate type suffix parsing."""
    print_header("1. Type Suffixes")
    print("Data type qualifiers for instructions")

    examples = [
        ("mov.b32", "32-bit bit-type operation"),
        ("add.f32", "32-bit floating point"),
        ("ld.u64", "64-bit unsigned integer load"),
        ("cvt.s32.f32", "Convert float to signed int"),
        ("wgmma.mma.f32.tf32", "Matrix multiply with f32/tf32 types"),
    ]

    for full_opcode, description in examples:
        suffixes = SuffixParser.parse(full_opcode.split('.')[0], full_opcode)
        print(f"\n  {full_opcode}")
        print(f"    Description: {description}")
        print(f"    Type suffix: {suffixes.type_suffix}")


def demo_memory_spaces():
    """Demonstrate memory space qualifiers."""
    print_header("2. Memory Space Qualifiers")
    print("Memory spaces and sub-qualifiers")

    examples = [
        ("ld.global.u32", "Global memory load"),
        ("ld.shared.b32", "Shared memory load"),
        ("ld.shared::cta.u32", "CTA-scoped shared memory"),
        ("ld.shared::cluster.u32", "Cluster-scoped shared memory"),
        ("st.local.u32", "Local memory store"),
        ("ld.param.f32", "Parameter space load"),
    ]

    for full_opcode, description in examples:
        suffixes = SuffixParser.parse(full_opcode.split('.')[0], full_opcode)
        print(f"\n  {full_opcode}")
        print(f"    Description: {description}")
        print(f"    Memory space: {suffixes.memory_space}")
        if suffixes.shared_scope != SharedScope.NONE:
            print(f"    Shared scope: {suffixes.shared_scope}")


def demo_sync_modifiers():
    """Demonstrate synchronization modifiers."""
    print_header("3. Memory Synchronization Modifiers")
    print("Memory consistency and synchronization")

    examples = [
        ("ld.relaxed.u32", "Relaxed load (no ordering)"),
        ("ld.acquire.u32", "Load with acquire semantics"),
        ("st.release.u32", "Store with release semantics"),
        ("atom.add.acq_rel.u32", "Atomic with acquire-release"),
        ("ld.volatile.u32", "Volatile load (cannot be reordered)"),
    ]

    for full_opcode, description in examples:
        suffixes = SuffixParser.parse(full_opcode.split('.')[0], full_opcode)
        print(f"\n  {full_opcode}")
        print(f"    Description: {description}")
        print(f"    Sync modifier: {suffixes.sync_modifier}")


def demo_memory_scope():
    """Demonstrate memory scope qualifiers."""
    print_header("4. Memory Scope Qualifiers")
    print("Visibility scope of memory operations")

    examples = [
        ("ld.relaxed.cta.u32", "CTA (thread block) scope"),
        ("ld.relaxed.cluster.u32", "Cluster scope"),
        ("ld.relaxed.gpu.u32", "GPU scope"),
        ("ld.relaxed.sys.u32", "System (CPU+GPU) scope"),
    ]

    for full_opcode, description in examples:
        suffixes = SuffixParser.parse(full_opcode.split('.')[0], full_opcode)
        print(f"\n  {full_opcode}")
        print(f"    Description: {description}")
        print(f"    Memory scope: {suffixes.memory_scope}")


def demo_matrix_instructions():
    """Demonstrate matrix instruction modifiers."""
    print_header("5. Matrix Instruction Modifiers")
    print("Modifiers for ldmatrix, stmatrix, wgmma, etc.")

    examples = [
        ("ldmatrix.sync.aligned.m8n8.x1.b16", "Synchronized aligned 8x8 matrix load"),
        ("ldmatrix.sync.m16n8.x2.trans.b16", "Transposed 16x8 matrix load, x2"),
        ("stmatrix.sync.aligned.m8n8.b16", "Matrix store"),
        ("wgmma.mma.m64n8k16", "Warpgroup MMA 64x8x16"),
        ("mma.m16n8k16.f32", "MMA 16x8x16"),
    ]

    for full_opcode, description in examples:
        suffixes = SuffixParser.parse(full_opcode.split('.')[0], full_opcode)
        print(f"\n  {full_opcode}")
        print(f"    Description: {description}")
        print(f"    Matrix shape: {suffixes.matrix_shape}")
        print(f"    Matrix modifiers: {suffixes.matrix_modifiers}")
        print(f"    Vector width: {suffixes.vector_width}")
        print(f"    Type: {suffixes.type_suffix}")


def demo_mbarrier_instructions():
    """Demonstrate mbarrier instruction suffixes."""
    print_header("6. mbarrier Instructions")
    print("Memory barrier operations with suffixes")

    examples = [
        ("mbarrier.init.shared.b64", "Initialize mbarrier"),
        ("mbarrier.init.shared::cta.b64", "Initialize with CTA scope"),
        ("mbarrier.init.shared::cluster.b64", "Initialize with cluster scope"),
        ("mbarrier.arrive.shared", "Arrive at barrier"),
        ("mbarrier.test_wait.shared", "Wait for barrier"),
    ]

    for full_opcode, description in examples:
        suffixes = SuffixParser.parse(full_opcode.split('.')[0], full_opcode)
        print(f"\n  {full_opcode}")
        print(f"    Description: {description}")
        print(f"    Memory space: {suffixes.memory_space}")
        print(f"    Shared scope: {suffixes.shared_scope}")
        print(f"    Type: {suffixes.type_suffix}")


def demo_complete_instructions():
    """Demonstrate complete instruction decoding with suffixes."""
    print_header("7. Complete Instruction Decoding")
    print("Full instructions with operands and suffixes")

    decoder = InstructionDecoder()

    instructions = [
        "ldmatrix.sync.aligned.m8n8.b16 {R0, R1, R2, R3}, [R4]",
        "ld.global.relaxed.gpu.u32 R5, [R6]",
        "st.shared.release.u32 [R7], R8",
        "wgmma.mma.m64n8k16.f32.tf32 R9, R10, R11, R12",
        "mbarrier.init.shared::cta.b64 [R13], 32",
        "atom.add.relaxed.cta.u32 R14, [R15], R16",
    ]

    for instr_str in instructions:
        try:
            instr = decoder.decode(instr_str)
            print(f"\n  Input: {instr_str}")
            print(f"  Opcode: {instr.opcode.value}")
            if instr.full_opcode:
                print(f"  Full: {instr.full_opcode}")
            if instr.suffixes:
                print(f"  Suffixes: {instr.suffixes}")
        except Exception as e:
            print(f"\n  Error: {instr_str}")
            print(f"  {e}")


def demo_suffix_structure():
    """Show the structure of parsed suffixes."""
    print_header("8. Suffix Structure Breakdown")
    print("Detailed breakdown of suffix components")

    complex_example = "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16"
    suffixes = SuffixParser.parse("ldmatrix", complex_example)

    print(f"\n  Full opcode: {complex_example}")
    print(f"  Base opcode: ldmatrix")
    print(f"\n  Parsed components:")
    print(f"    Type suffix:        {suffixes.type_suffix}")
    print(f"    Memory space:       {suffixes.memory_space}")
    print(f"    Shared scope:       {suffixes.shared_scope}")
    print(f"    Memory scope:       {suffixes.memory_scope}")
    print(f"    Sync modifier:      {suffixes.sync_modifier}")
    print(f"    Cache op:           {suffixes.cache_op}")
    print(f"    Matrix shape:       {suffixes.matrix_shape}")
    print(f"    Matrix modifiers:   {suffixes.matrix_modifiers}")
    print(f"    Vector width:       {suffixes.vector_width}")
    print(f"    Raw suffix:         {suffixes.raw_suffix}")
    print(f"\n  Reconstructed: ldmatrix{suffixes}")


def main():
    """Run all suffix demonstrations."""
    print("\n" + "█" * 70)
    print("█" + " " * 18 + "PTX INSTRUCTION SUFFIXES DEMO" + " " * 19 + "█")
    print("█" * 70)
    print("\nDemonstrating PTX instruction suffix parsing and handling.")
    print("These suffixes modify instruction behavior and are essential")
    print("for writing correct PTX code.")

    demo_type_suffixes()
    demo_memory_spaces()
    demo_sync_modifiers()
    demo_memory_scope()
    demo_matrix_instructions()
    demo_mbarrier_instructions()
    demo_complete_instructions()
    demo_suffix_structure()

    print_header("SUMMARY")
    print("PTX Instruction Suffix Categories:")
    print("  1. Type Suffixes")
    print("     - .b8, .b16, .b32, .b64 (bit types)")
    print("     - .u8, .u16, .u32, .u64 (unsigned)")
    print("     - .s8, .s16, .s32, .s64 (signed)")
    print("     - .f16, .f32, .f64 (floating point)")
    print("     - .bf16, .tf32 (brain float, tensor float)")
    print()
    print("  2. Memory Spaces")
    print("     - .global, .shared, .local, .param, .const")
    print("     - .shared::cta, .shared::cluster (sub-scopes)")
    print()
    print("  3. Synchronization")
    print("     - .relaxed, .acquire, .release, .acq_rel")
    print("     - .volatile, .weak, .mmio")
    print()
    print("  4. Memory Scope")
    print("     - .cta, .cluster, .gpu, .sys")
    print()
    print("  5. Matrix Modifiers")
    print("     - .sync, .aligned, .trans")
    print("     - .m8n8, .m16n8, .m32n8, .m64n8 (shapes)")
    print("     - .x1, .x2, .x4 (vector width)")
    print()
    print("These suffixes enable precise control over:")
    print("  - Memory operations and visibility")
    print("  - Synchronization and ordering")
    print("  - Data type conversions")
    print("  - Matrix operation parameters")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
