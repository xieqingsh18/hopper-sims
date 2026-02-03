"""
Register File Implementation for Hopper GPU

Each thread has its own register file with:
- 255 general-purpose registers (R0-R254)
- R0 is always zero (reads return 0, writes are ignored)
- Each register is 32-bit (can be extended for 64-bit operations)
"""

from typing import Union
import struct


class RegisterFile:
    """
    Per-thread register file for Hopper GPU.

    Hopper Architecture Specs:
    - 255 general-purpose 32-bit registers (R0-R254)
    - R0 is hardwired to zero
    - Registers can be paired for 64-bit operations
    """

    # Number of registers per thread
    NUM_REGISTERS = 255

    # Register width in bits
    REGISTER_WIDTH = 32

    # Zero register index
    ZERO_REG = 0

    def __init__(self) -> None:
        """Initialize a new register file with all zeros."""
        # Use list of 32-bit unsigned integers
        self._registers: list[int] = [0] * self.NUM_REGISTERS

    def read(self, reg_idx: int) -> int:
        """
        Read a 32-bit value from a register.

        Args:
            reg_idx: Register index (0-254)

        Returns:
            32-bit unsigned integer value

        Raises:
            ValueError: If register index is out of range
        """
        if not 0 <= reg_idx < self.NUM_REGISTERS:
            raise ValueError(f"Register index {reg_idx} out of range [0, {self.NUM_REGISTERS-1}]")

        # R0 always reads as zero
        if reg_idx == self.ZERO_REG:
            return 0

        return self._registers[reg_idx]

    def write(self, reg_idx: int, value: int) -> None:
        """
        Write a 32-bit value to a register.

        Args:
            reg_idx: Register index (0-254)
            value: 32-bit value to write

        Raises:
            ValueError: If register index is out of range

        Note:
            Writes to R0 are silently ignored (it's always zero)
        """
        if not 0 <= reg_idx < self.NUM_REGISTERS:
            raise ValueError(f"Register index {reg_idx} out of range [0, {self.NUM_REGISTERS-1}]")

        # Ignore writes to R0 (zero register)
        if reg_idx == self.ZERO_REG:
            return

        # Mask to 32 bits
        self._registers[reg_idx] = value & 0xFFFFFFFF

    def read_pair(self, reg_idx: int) -> int:
        """
        Read a 64-bit value from a register pair.

        For 64-bit operations, registers are paired:
        - Even register holds lower 32 bits
        - Next odd register holds upper 32 bits

        Args:
            reg_idx: First register index (must be even)

        Returns:
            64-bit unsigned integer value
        """
        if reg_idx % 2 != 0:
            raise ValueError(f"Register pair must start at even index, got {reg_idx}")

        if reg_idx + 1 >= self.NUM_REGISTERS:
            raise ValueError(f"Register pair {reg_idx}:{reg_idx+1} out of range")

        low = self.read(reg_idx)
        high = self.read(reg_idx + 1)

        return (high << 32) | low

    def write_pair(self, reg_idx: int, value: int) -> None:
        """
        Write a 64-bit value to a register pair.

        Args:
            reg_idx: First register index (must be even)
            value: 64-bit value to write
        """
        if reg_idx % 2 != 0:
            raise ValueError(f"Register pair must start at even index, got {reg_idx}")

        if reg_idx + 1 >= self.NUM_REGISTERS:
            raise ValueError(f"Register pair {reg_idx}:{reg_idx+1} out of range")

        low = value & 0xFFFFFFFF
        high = (value >> 32) & 0xFFFFFFFF

        self.write(reg_idx, low)
        self.write(reg_idx + 1, high)

    def read_f32(self, reg_idx: int) -> float:
        """
        Read a register as a 32-bit floating point value.

        Args:
            reg_idx: Register index

        Returns:
            Float value
        """
        bits = self.read(reg_idx)
        return struct.unpack('>f', struct.pack('>I', bits))[0]

    def write_f32(self, reg_idx: int, value: float) -> None:
        """
        Write a 32-bit floating point value to a register.

        Args:
            reg_idx: Register index
            value: Float value to write
        """
        bits = struct.unpack('>I', struct.pack('>f', value))[0]
        self.write(reg_idx, bits)

    def read_bits(self, reg_idx: int, bit_offset: int, num_bits: int) -> int:
        """
        Read a bit field from a register.

        Args:
            reg_idx: Register index
            bit_offset: Starting bit position (0 = LSB)
            num_bits: Number of bits to read

        Returns:
            Extracted bit field
        """
        if num_bits > 32:
            raise ValueError(f"Cannot read {num_bits} bits from 32-bit register")

        value = self.read(reg_idx)
        mask = (1 << num_bits) - 1
        return (value >> bit_offset) & mask

    def __repr__(self) -> str:
        """String representation showing non-zero registers."""
        non_zero = [(i, self._registers[i])
                    for i in range(self.NUM_REGISTERS)
                    if self._registers[i] != 0]

        if not non_zero:
            return "RegisterFile(all zeros)"

        return f"RegisterFile({', '.join(f'R{i}={v:#x}' for i, v in non_zero[:5])}{'...' if len(non_zero) > 5 else ''})"


if __name__ == "__main__":
    # Test the register file
    rf = RegisterFile()

    # Test basic read/write
    rf.write(1, 0xDEADBEEF)
    print(f"R1 = {rf.read(1):#x}")  # Should be 0xdeadbeef

    # Test zero register
    rf.write(0, 0xFFFFFFFF)
    print(f"R0 = {rf.read(0):#x}")  # Should be 0x0

    # Test register pair
    rf.write_pair(2, 0x123456789ABCDEF0)
    print(f"R2:R3 = {rf.read_pair(2):#x}")  # Should be 0x123456789abcdef0

    # Test float operations
    rf.write_f32(4, 3.14159)
    print(f"R4 (float) = {rf.read_f32(4)}")  # Should be ~3.14159

    print(rf)
