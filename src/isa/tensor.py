"""
Tensor Core Instructions for Hopper GPU

Hopper introduces 4th generation Tensor Cores with FP8 (8-bit floating point) support.
This module implements the key Tensor Core instructions, particularly HMMA (Matrix Multiply-Accumulate).
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
import struct


class HMMAOp(Enum):
    """
    HMMA (Half Matrix Multiply-Accumulate) operation types.

    Hopper supports various matrix formats:
    - FP8: 8-bit floating point (E4M3 or E5M2 formats)
    - FP16: 16-bit floating point
    - FP32: 32-bit floating point (accumulation)
    - INT8: 8-bit integer
    """

    # FP8 variants (Hopper key feature)
    FP8_FP16 = "8816"    # FP8 x FP8 -> FP16 accumulation
    FP8_FP32 = "8832"    # FP8 x FP8 -> FP32 accumulation

    # FP16 variants
    FP16_FP16 = "1616"   # FP16 x FP16 -> FP16
    FP16_FP32 = "1632"   # FP16 x FP16 -> FP32

    # INT8 variants
    INT8_INT32 = "8832"  # INT8 x INT8 -> INT32


@dataclass
class Fragment:
    """
    Represents a matrix fragment stored in registers.

    Tensor Core operations work on matrix fragments (tiles):
    - Each thread holds a portion of the matrix in registers
    - Fragment size depends on the mma instruction
    """

    registers: List[int]  # List of register numbers
    rows: int             # Number of rows in fragment
    cols: int             # Number of columns in fragment
    data_type: str        # Data type (fp8, fp16, fp32, int8)

    def size(self) -> int:
        """Total number of elements in fragment."""
        return self.rows * self.cols


class TensorCoreInstruction:
    """
    Tensor Core instruction implementation for Hopper.

    HMMA instruction format:
    HMMA.{dtype} Rd, Ra, Rb, Rc

    Where:
    - Rd: Destination registers (accumulator/fragment D)
    - Ra: Source fragment A
    - Rb: Source fragment B
    - Rc: Source accumulator (for FMA operations)

    Matrix dimensions vary by data type:
    - FP8: 16x16 x 16x16 -> 16x16 (per 8 threads)
    - FP16: 16x16 x 16x16 -> 16x16 (per 16 threads)
    """

    # Matrix dimensions for different data types
    DIMENSIONS = {
        HMMAOp.FP8_FP16: (16, 16, 16),  # M=16, N=16, K=16
        HMMAOp.FP8_FP32: (16, 16, 16),
        HMMAOp.FP16_FP16: (16, 16, 16),
        HMMAOp.FP16_FP32: (16, 16, 16),
    }

    # Threads per operation (how many threads collaborate on one MMA)
    THREADS_PER_MMA = {
        HMMAOp.FP8_FP16: 8,
        HMMAOp.FP8_FP32: 8,
        HMMAOp.FP16_FP16: 16,
        HMMAOp.FP16_FP32: 16,
    }

    def __init__(self, op_type: HMMAOp = HMMAOp.FP8_FP16) -> None:
        """
        Initialize Tensor Core instruction handler.

        Args:
            op_type: Type of HMMA operation
        """
        self.op_type = op_type
        self.m, self.n, self.k = self.DIMENSIONS[op_type]
        self.threads_per_mma = self.THREADS_PER_MMA[op_type]

    def execute(self,
                d_regs: List[int],
                a_regs: List[int],
                b_regs: List[int],
                c_regs: List[int],
                register_files: List) -> None:
        """
        Execute Tensor Core matrix multiply-accumulate.

        Computes: D = A × B + C

        Where A is M×K, B is K×N, C and D are M×N

        Args:
            d_regs: Register numbers for destination fragment D
            a_regs: Register numbers for source fragment A
            b_regs: Register numbers for source fragment B
            c_regs: Register numbers for accumulator C
            register_files: List of register files (one per thread)
        """
        # Simplified implementation - each thread independently computes
        # In real hardware, threads collaborate to compute different portions

        num_threads = min(len(register_files), self.threads_per_mma)

        for thread_idx in range(num_threads):
            rf = register_files[thread_idx]

            # Read accumulator (C)
            c_val = 0.0
            if thread_idx < len(c_regs):
                c_val = rf.read_f32(c_regs[thread_idx])

            # Simplified: Each thread reads one element from A and B
            # In reality, threads hold fragments and collaborate
            a_val = self._read_fragment_element(a_regs, thread_idx, rf, is_fp8=True)
            b_val = self._read_fragment_element(b_regs, thread_idx, rf, is_fp8=True)

            # Compute: D = A * B + C
            result = a_val * b_val + c_val

            # Write result
            if thread_idx < len(d_regs):
                rf.write_f32(d_regs[thread_idx], result)

    def _read_fragment_element(self,
                                regs: List[int],
                                elem_idx: int,
                                rf,
                                is_fp8: bool) -> float:
        """
        Read an element from a matrix fragment.

        Args:
            regs: Register numbers holding the fragment
            elem_idx: Element index within fragment
            rf: Register file to read from
            is_fp8: Whether data is FP8 format

        Returns:
            Floating point value
        """
        if elem_idx >= len(regs):
            return 0.0

        reg_num = regs[elem_idx]

        if is_fp8:
            # FP8 is stored in lower 8 bits of 32-bit register
            raw = rf.read(reg_num) & 0xFF
            return self._fp8_to_float(raw)
        else:
            # Read as FP16 or FP32
            return rf.read_f32(reg_num)

    def _fp8_to_float(self, fp8_bits: int) -> float:
        """
        Convert FP8 (E4M3 format) to Python float.

        FP8 E4M3 format:
        - 1 sign bit
        - 4 exponent bits (bias=7)
        - 3 mantissa bits

        Args:
            fp8_bits: 8-bit FP8 value

        Returns:
            Python float
        """
        # Extract components
        sign = (fp8_bits >> 7) & 0x1
        exponent = (fp8_bits >> 3) & 0xF
        mantissa = fp8_bits & 0x7

        if exponent == 0:
            if mantissa == 0:
                # Zero
                return 0.0
            # Subnormal number
            return float((-1) ** sign * (mantissa / 8.0) * (2 ** -6))
        elif exponent == 0xF:
            # Infinity or NaN
            if mantissa == 0:
                return float('-inf') if sign else float('inf')
            else:
                return float('nan')

        # Normal number
        value = (-1) ** sign * (1.0 + mantissa / 8.0) * (2 ** (exponent - 7))
        return float(value)

    def _float_to_fp8(self, value: float) -> int:
        """
        Convert Python float to FP8 (E4M3 format).

        Args:
            value: Floating point value

        Returns:
            8-bit FP8 value
        """
        # Simplified conversion (not fully IEEE compliant)
        if value == 0.0:
            return 0

        # Handle sign
        sign = 1 if value < 0 else 0
        value = abs(value)

        # Find exponent and mantissa
        import math
        if value < 2 ** -6:
            # Subnormal range
            mantissa = int(value * 8.0 * (2 ** 6))
            return (sign << 7) | mantissa
        else:
            exp = int(math.log2(value))
            mantissa = int((value / (2 ** exp) - 1.0) * 8.0)

            # Clamp values
            exp = min(max(exp + 7, 0), 14)
            mantissa = min(mantissa, 7)

            return (sign << 7) | (exp << 3) | mantissa


# FP8 conversion utilities
def pack_fp8_e4m3(value: float) -> int:
    """Pack a float into FP8 E4M3 format."""
    converter = TensorCoreInstruction()
    return converter._float_to_fp8(value)


def unpack_fp8_e4m3(fp8: int) -> float:
    """Unpack FP8 E4M3 to float."""
    converter = TensorCoreInstruction()
    return converter._fp8_to_float(fp8)


if __name__ == "__main__":
    # Test FP8 conversion
    print("Testing FP8 E4M3 conversion:")

    test_values = [0.0, 1.0, -1.0, 2.5, -2.5, 0.125, -0.125]

    for val in test_values:
        fp8 = pack_fp8_e4m3(val)
        recovered = unpack_fp8_e4m3(fp8)
        print(f"{val:8.4f} -> FP8: 0x{fp8:02x} -> {recovered:8.4f} (error: {abs(val - recovered):.6f})")

    # Test Tensor Core instruction
    print("\nTesting Tensor Core instruction:")

    class MockRegFile:
        def __init__(self):
            self.regs = [0] * 255

        def read(self, idx):
            return self.regs[idx]

        def read_f32(self, idx):
            import struct
            return struct.unpack('>f', struct.pack('>I', self.regs[idx]))[0]

        def write_f32(self, idx, val):
            import struct
            self.regs[idx] = struct.unpack('>I', struct.pack('>f', val))[0]

    # Create mock register files for 8 threads
    reg_files = [MockRegFile() for _ in range(8)]

    # Initialize A and B fragments with test values
    for i, rf in enumerate(reg_files):
        rf.write_f32(i, float(i + 1))  # A[i] = i+1
        rf.write_f32(i + 10, float(i + 2))  # B[i] = i+2
        rf.write_f32(i + 20, 0.0)  # C[i] = 0

    # Execute HMMA
    instr = TensorCoreInstruction(HMMAOp.FP8_FP32)
    instr.execute(
        d_regs=list(range(30, 38)),    # D0-D7
        a_regs=list(range(0, 8)),      # A0-A7
        b_regs=list(range(10, 18)),    # B0-B7
        c_regs=list(range(20, 28)),    # C0-C7
        register_files=reg_files
    )

    # Print results
    print("\nResults:")
    for i in range(8):
        result = reg_files[i].read_f32(30 + i)
        expected = float((i + 1) * (i + 2))
        print(f"Thread {i}: D[{i}] = {result:.2f} (expected: {expected:.2f})")
