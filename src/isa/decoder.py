"""
Instruction Decoder for Hopper SASS

Decodes SASS instructions from various formats (assembly, binary).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
import re
from .instructions import (
    Opcode, InstructionType, Operand, OperandType,
    get_format, is_load_instruction, is_store_instruction,
    is_branch_instruction, is_tensor_instruction
)
from .suffixes import InstructionSuffixes, SuffixParser


@dataclass
class Instruction:
    """
    Decoded instruction.

    Contains all information needed for execution.
    """
    opcode: Opcode
    type: InstructionType
    operands: List[Operand]
    pc: int = 0  # Program counter of this instruction

    # Predication (optional)
    predicate_reg: Optional[int] = None  # Predicate register number
    predicate_condition: bool = True  # True if execute when pred=1, False if pred=0

    # Instruction modifiers
    saturate: bool = False  # Saturate result
    round_mode: Optional[str] = None  # Rounding mode (RN, RM, RP, RZ)

    # PTX suffixes (type, memory space, sync, etc.)
    suffixes: Optional[InstructionSuffixes] = None

    # For branch instructions
    branch_target: Optional[int] = None

    # Original encoding (for debugging)
    raw_encoding: Optional[str] = None

    # Full opcode string with suffixes (e.g., "ldmatrix.sync.aligned.m8n8.b16")
    full_opcode: Optional[str] = None

    def __repr__(self) -> str:
        """String representation of the instruction."""
        pred_str = ""
        if self.predicate_reg is not None:
            pred_str = f"@P{self.predicate_reg} "
            if not self.predicate_condition:
                pred_str = f"@!P{self.predicate_reg} "

        # Use full opcode if available
        op_str = self.full_opcode if self.full_opcode else self.opcode.value
        operands_str = ", ".join(str(op) for op in self.operands)
        return f"{self.pc:04x}: {pred_str}{op_str} {operands_str}"

    def __str__(self) -> str:
        return self.__repr__()


class InstructionDecoder:
    """
    Decodes Hopper SASS instructions.

    Supports:
    - Assembly text format (e.g., "IADD R1, R2, R3")
    - Basic binary encoding (simplified)
    """

    # Regex patterns for parsing assembly
    REGISTER_PATTERN = r'R(\d+)'
    PREDICATE_PATTERN = r'P(\d+)'
    IMMEDIATE_PATTERN = r'(-?0x[0-9A-Fa-f]+|-?\d+)'
    MEMORY_PATTERN = r'\[([^\]]+)\]'
    PREDICATE_PREFIX = r'@(!?)P(\d+)\s*'

    def __init__(self) -> None:
        """Initialize the decoder."""
        # Cache for parsed instructions
        self._instruction_cache: Dict[str, Instruction] = {}

    def decode(self, instruction_str: str, pc: int = 0) -> Instruction:
        """
        Decode an assembly instruction.

        Args:
            instruction_str: Assembly instruction (e.g., "IADD R1, R2, R3" or "ldmatrix.sync.aligned.m8n8.b16")
            pc: Program counter of this instruction

        Returns:
            Decoded Instruction

        Raises:
            ValueError: If instruction cannot be decoded
        """
        # Check cache
        cache_key = f"{instruction_str}_{pc}"
        if cache_key in self._instruction_cache:
            return self._instruction_cache[cache_key]

        # Strip whitespace and comments
        instr = instruction_str.strip()
        if '//' in instr:
            instr = instr.split('//')[0].strip()

        if not instr:
            raise ValueError("Empty instruction")

        # Parse predicate
        predicate_reg = None
        predicate_condition = True

        pred_match = re.match(self.PREDICATE_PREFIX, instr)
        if pred_match:
            neg = pred_match.group(1) == '!'
            pred_num = int(pred_match.group(2))
            predicate_reg = pred_num
            predicate_condition = not neg  # @!P0 means execute if P0=0
            instr = instr[pred_match.end():]

        # Split into opcode and operands
        parts = instr.split(None, 1)
        full_opcode_str = parts[0].lower()  # Keep original case for suffix parsing
        base_opcode_str = full_opcode_str.upper()

        # Extract base opcode (handle suffixes like ".sync.aligned.m8n8.b16")
        # Split by '.' and take the first part as base opcode
        base_opcode_parts = base_opcode_str.split('.')
        base_opcode_only = base_opcode_parts[0]

        # Try to look up opcode with base name first
        opcode = None
        suffixes = None
        matched_opcode_str = None

        # Try with progressively fewer suffix parts until we find a match
        for i in range(len(base_opcode_parts), 0, -1):
            try_opcode = '.'.join(base_opcode_parts[:i]).upper()
            try:
                opcode = Opcode(try_opcode)
                matched_opcode_str = try_opcode
                # Parse the suffixes - pass the matched opcode as base
                suffixes = SuffixParser.parse(matched_opcode_str.lower(), full_opcode_str)
                break
            except ValueError:
                continue

        # If no match found, try the original full string
        if opcode is None:
            # Special handling for compound opcodes with dots (like mbarrier.init)
            # Try to find a base opcode that matches
            for known_opcode in Opcode:
                known_str = known_opcode.value.lower().replace('.', '')
                test_str = full_opcode_str.replace('.', '')
                if test_str.startswith(known_str):
                    # Found a potential match - parse suffixes from full string
                    suffixes = SuffixParser.parse(known_opcode.value.lower(), full_opcode_str)
                    opcode = known_opcode
                    matched_opcode_str = known_opcode.value
                    break

        if opcode is None:
            try:
                opcode = Opcode(base_opcode_only)
                matched_opcode_str = base_opcode_only
                suffixes = SuffixParser.parse(base_opcode_only.lower(), full_opcode_str)
            except ValueError:
                raise ValueError(f"Unknown opcode: {full_opcode_str}")

        # Get format
        fmt = get_format(opcode)
        if fmt is None:
            raise ValueError(f"No format defined for opcode: {opcode.value}")

        # Parse operands
        operands: List[Operand] = []
        if len(parts) > 1 and parts[1].strip():
            operand_strs = [s.strip() for s in parts[1].split(',')]

            for i, op_str in enumerate(operand_strs):
                # Determine if this is a destination or source
                # First operand is typically destination (except for load/store)
                is_dest = (i == 0 and opcode not in {Opcode.LDG, Opcode.LDS,
                                                      Opcode.LDSM, Opcode.STG, Opcode.STS,
                                                      Opcode.TMA_LOAD, Opcode.TMA_STORE})

                # Handle memory operands for load/store
                if is_load_instruction(opcode) and i == 1:
                    op = self._parse_memory_operand(op_str)
                elif is_store_instruction(opcode) and i == 0:
                    op = self._parse_memory_operand(op_str)
                elif is_branch_instruction(opcode):
                    op = self._parse_immediate_operand(op_str)
                else:
                    op = self._parse_generic_operand(op_str)

                if op:
                    op.is_dest = is_dest
                    operands.append(op)

        # Create instruction
        instr_obj = Instruction(
            opcode=opcode,
            type=fmt.type,
            operands=operands,
            pc=pc,
            predicate_reg=predicate_reg,
            predicate_condition=predicate_condition,
            suffixes=suffixes,
            raw_encoding=instruction_str,
            full_opcode=full_opcode_str if suffixes and (str(suffixes) != "") else None
        )

        # Cache and return
        self._instruction_cache[cache_key] = instr_obj
        return instr_obj

    def _parse_generic_operand(self, op_str: str) -> Optional[Operand]:
        """Parse a generic operand (register, immediate, memory, etc.)."""
        op_str = op_str.strip()

        # Check for memory operand (brackets) first - for TMA, mbarrier, etc.
        if op_str.startswith('[') and op_str.endswith(']'):
            return self._parse_memory_operand(op_str)

        # Check for negation
        negate = op_str.startswith('-')
        if negate:
            op_str = op_str[1:].strip()

        # Check for absolute value
        absolute = op_str.startswith('|') and op_str.endswith('|')
        if absolute:
            op_str = op_str[1:-1].strip()

        # Try register
        reg_match = re.fullmatch(self.REGISTER_PATTERN, op_str)
        if reg_match:
            reg_num = int(reg_match.group(1))
            return Operand(OperandType.REGISTER, reg_num,
                          negate=negate, absolute=absolute)

        # Try immediate
        imm_match = re.fullmatch(self.IMMEDIATE_PATTERN, op_str)
        if imm_match:
            imm_str = imm_match.group(1)
            # Parse as hex or decimal
            if imm_str.startswith('0x') or imm_str.startswith('-0x'):
                value = int(imm_str, 16)
            else:
                value = int(imm_str)
            return Operand(OperandType.IMMEDIATE, value,
                          negate=negate, absolute=absolute)

        # Try predicate
        pred_match = re.fullmatch(self.PREDICATE_PATTERN, op_str)
        if pred_match:
            pred_num = int(pred_match.group(1))
            return Operand(OperandType.PREDICATE, pred_num,
                          negate=negate)

        raise ValueError(f"Cannot parse operand: {op_str}")

    def _parse_memory_operand(self, op_str: str) -> Operand:
        """Parse a memory operand (e.g., "[R1 + 0x10]")."""
        # Strip brackets
        inner = op_str.strip().strip('[]')

        # Simple format: just a register (e.g., "[R1]")
        reg_match = re.fullmatch(self.REGISTER_PATTERN, inner)
        if reg_match:
            reg_num = int(reg_match.group(1))
            return Operand(OperandType.MEMORY, f"[R{reg_num}]")

        # Format: register + offset (e.g., "[R1 + 0x10]" or "[R1+0x10]")
        offset_match = re.match(rf'{self.REGISTER_PATTERN}\s*\+\s*{self.IMMEDIATE_PATTERN}', inner)
        if offset_match:
            reg_num = int(offset_match.group(1))
            offset_str = offset_match.group(2)
            if offset_str.startswith('0x'):
                offset = int(offset_str, 16)
            else:
                offset = int(offset_str)
            return Operand(OperandType.MEMORY, f"[R{reg_num}+{offset}]")

        # Store as-is for complex addressing modes
        return Operand(OperandType.MEMORY, f"[{inner}]")

    def _parse_immediate_operand(self, op_str: str) -> Operand:
        """Parse an immediate operand (for branch targets)."""
        op_str = op_str.strip()

        # Try hex
        if op_str.startswith('0x'):
            value = int(op_str, 16)
        else:
            value = int(op_str)

        return Operand(OperandType.IMMEDIATE, value)

    def decode_binary(self, encoding: int, pc: int = 0) -> Instruction:
        """
        Decode an instruction from binary encoding.

        Note: This is a simplified implementation.
        Real SASS encoding is much more complex.

        Args:
            encoding: 64-bit instruction encoding
            pc: Program counter

        Returns:
            Decoded Instruction
        """
        # Simplified: Extract opcode from high bits
        # In real implementation, this would be a full decode table

        # For now, raise an error - binary decoding not fully implemented
        raise NotImplementedError("Binary decoding not fully implemented. Use assembly format.")


def parse_program(asm_lines: List[str]) -> List[Instruction]:
    """
    Parse a complete program from assembly lines.

    Args:
        asm_lines: List of assembly instruction strings

    Returns:
        List of decoded instructions
    """
    decoder = InstructionDecoder()
    instructions = []

    pc = 0
    for line in asm_lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue  # Skip empty lines and comments

        # Handle labels
        if line.endswith(':'):
            continue  # Skip labels for now (would need label table)

        try:
            instr = decoder.decode(line, pc)
            instructions.append(instr)
            pc += 4  # Instructions are 4-byte aligned
        except ValueError as e:
            print(f"Warning: Failed to decode instruction '{line}': {e}")

    return instructions


if __name__ == "__main__":
    # Test the decoder
    decoder = InstructionDecoder()

    # Test simple instruction
    instr1 = decoder.decode("IADD R1, R2, R3", pc=0x100)
    print(instr1)

    # Test predicated instruction
    instr2 = decoder.decode("@P2 FADD R5, R6, R7", pc=0x104)
    print(instr2)

    # Test negated predicate
    instr3 = decoder.decode("@!P1 BRA 0x200", pc=0x108)
    print(instr3)

    # Test load instruction
    instr4 = decoder.decode("LDG R10, [R1 + 0x10]", pc=0x10C)
    print(instr4)

    # Test immediate
    instr5 = decoder.decode("MOV R0, 0x1234", pc=0x110)
    print(instr5)

    # Test program parsing
    program = [
        "MOV R1, 0x1000",
        "LDG R2, [R1]",
        "IADD R3, R2, 5",
        "STG [R1], R3",
        "EXIT",
    ]

    print("\nParsed program:")
    for instr in parse_program(program):
        print(instr)
