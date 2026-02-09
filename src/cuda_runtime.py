"""
CUDA-Like API for Hopper GPU Simulator

This module provides CUDA-like functions for memory management and kernel launches,
making the simulator work much more like real CUDA programming.
"""

from typing import List, Tuple, Any, Union
from ctypes import c_void_p, c_int32, addressof
import struct


class DevicePointer:
    """Represents a device memory pointer (like cudaMalloc result)."""

    def __init__(self, address: int, size: int):
        self.address = address  # Device memory address
        self.size = size        # Allocation size in bytes

    def __repr__(self):
        return f"DevicePointer(addr=0x{self.address:x}, size={self.size})"

    def __int__(self):
        return self.address


class CUDARuntime:
    """
    CUDA Runtime API simulator.

    Provides cudaMalloc, cudaMemcpy, cudaFree, and kernel launching.
    """

    def __init__(self, simulator):
        """
        Initialize CUDA runtime.

        Args:
            simulator: HopperSimulator instance
        """
        self.sim = simulator
        self.allocations = {}  # Track allocations: ptr -> (size, host_data)
        self.next_address = 0x10000000  # Start device allocations at high address

        # Counter for generating unique IDs
        self.alloc_id = 0

    def cudaMalloc(self, size: int) -> DevicePointer:
        """
        Allocate device memory (like cudaMalloc).

        Args:
            size: Size in bytes to allocate

        Returns:
            DevicePointer to allocated memory
        """
        # Allocate at next available address
        ptr = DevicePointer(self.next_address, size)
        self.allocations[ptr.address] = (size, None)
        self.next_address += size

        print(f"[cudaMalloc] Allocated {size} bytes at 0x{ptr.address:x}")
        return ptr

    def cudaFree(self, ptr: DevicePointer) -> None:
        """
        Free device memory (like cudaFree).

        Args:
            ptr: DevicePointer to free
        """
        if ptr.address in self.allocations:
            del self.allocations[ptr.address]
            print(f"[cudaFree] Freed memory at 0x{ptr.address:x}")
        else:
            raise RuntimeError(f"Invalid pointer: 0x{ptr.address:x}")

    def cudaMemcpyHtoD(self, ptr: DevicePointer, host_data: bytes) -> None:
        """
        Copy data from host to device (like cudaMemcpy with cudaMemcpyHostToDevice).

        Args:
            ptr: Device pointer
            host_data: Host data to copy (bytes object)
        """
        if ptr.address not in self.allocations:
            raise RuntimeError(f"Invalid pointer: 0x{ptr.address:x}")

        size, _ = self.allocations[ptr.address]

        if len(host_data) > size:
            raise RuntimeError(f"Data size ({len(host_data)}) exceeds allocation ({size})")

        # Write to simulator's global memory
        from src.core.memory import MemorySpace
        for i, byte in enumerate(host_data):
            self.sim.memory.write(MemorySpace.GLOBAL, ptr.address + i, bytes([byte]))

        # Track host data for debugging
        self.allocations[ptr.address] = (size, host_data)

        # Show sample data
        if len(host_data) <= 64:
            sample = list(host_data[:16])
            print(f"[cudaMemcpy H->D] {len(host_data)} bytes to 0x{ptr.address:x}")
            print(f"  Data: {sample}")
        else:
            print(f"[cudaMemcpy H->D] {len(host_data)} bytes to 0x{ptr.address:x}")

    def cudaMemcpyDtoH(self, ptr: DevicePointer, size: int = None) -> bytes:
        """
        Copy data from device to host (like cudaMemcpy with cudaMemcpyDeviceToHost).

        Args:
            ptr: Device pointer
            size: Number of bytes to copy (uses allocation size if None)

        Returns:
            Host data as bytes
        """
        if ptr.address not in self.allocations:
            raise RuntimeError(f"Invalid pointer: 0x{ptr.address:x}")

        alloc_size, _ = self.allocations[ptr.address]
        copy_size = size if size is not None else alloc_size

        if copy_size > alloc_size:
            raise RuntimeError(f"Copy size ({copy_size}) exceeds allocation ({alloc_size})")

        # Read from simulator's global memory
        from src.core.memory import MemorySpace
        data = bytearray()
        for i in range(copy_size):
            byte = self.sim.memory.read(MemorySpace.GLOBAL, ptr.address + i, 1)
            data.extend(byte)

        print(f"[cudaMemcpy D->H] {copy_size} bytes from 0x{ptr.address:x}")

        # Show sample data
        if copy_size <= 64:
            sample = list(data[:16])
            print(f"  Data: {sample}")

        return bytes(data)

    def launch_kernel(self,
                      kernel_func: 'KernelFunction',
                      grid_dim: Tuple[int, int, int],
                      block_dim: Tuple[int, int, int],
                      *args) -> None:
        """
        Launch a kernel with CUDA-style syntax.

        This mimics: kernel<<<grid_dim, block_dim>>>(args...)

        Args:
            kernel_func: KernelFunction instance
            grid_dim: (grid_x, grid_y, grid_z) - number of blocks
            block_dim: (block_x, block_y, block_z) - threads per block
            *args: Kernel arguments (DevicePointers, integers, etc.)

        Example:
            cuda.launch_kernel(vector_add, (2,1,1), (32,1,1), A_dev, B_dev, C_dev, n)
            # Equivalent to: vector_add<<<2, 32>>>(A_dev, B_dev, C_dev, n)
        """
        print(f"\n[Kernel Launch] {kernel_func.__name__}<<<{grid_dim}, {block_dim}>>>()")
        print(f"  Arguments: {args}")

        # Generate kernel code with argument substitution
        kernel_code = kernel_func.generate_code(*args)

        # Launch the kernel
        self.sim.launch_kernel(
            program=kernel_code,
            grid_dim=grid_dim,
            block_dim=block_dim
        )


class KernelFunction:
    """
    Base class for kernel functions.

    Allows defining kernels in a Pythonic way that get converted
    to assembly with proper argument handling.
    """

    def __init__(self, name: str):
        self.name = name
        self.__name__ = name  # For display purposes

    def generate_code(self, *args) -> List[str]:
        """
        Generate assembly code for this kernel with arguments bound.

        Args:
            *args: Kernel arguments (DevicePointers, integers, etc.)

        Returns:
            List of assembly instructions
        """
        raise NotImplementedError("Subclasses must implement generate_code")


def vector_add_kernel() -> KernelFunction:
    """
    Create a vector_add kernel function.

    Usage:
        kernel = vector_add_kernel()
        cuda.launch_kernel(kernel, (2,1,1), (32,1,1), A_dev, B_dev, C_dev, n)
    """
    class VectorAddKernel(KernelFunction):
        def __init__(self):
            super().__init__("vector_add")

        def generate_code(self, A_ptr, B_ptr, C_ptr, n) -> List[str]:
            """
            Generate kernel code for: C = A + B

            Args:
                A_ptr: DevicePointer to vector A
                B_ptr: DevicePointer to vector B
                C_ptr: DevicePointer to result vector C
                n: Number of elements

            Returns:
                Assembly instructions
            """
            A_addr = int(A_ptr)
            B_addr = int(B_ptr)
            C_addr = int(C_ptr)

            kernel = [
                # ========== Get Thread and Block IDs ==========
                "MOV R5, %tid",             # threadIdx.x
                "MOV R6, %ctaid",           # blockIdx.x

                # ========== Compute Global Thread ID ==========
                "IMUL.U32 R7, R6, 32",      # R7 = blockIdx.x * blockDim.x
                "IADD R5, R7, R5",          # R5 = global_tid

                # ========== Load A[global_tid] and B[global_tid] ==========
                f"MOV R3, {A_addr}",         # R3 = base address of A
                f"MOV R4, {B_addr}",         # R4 = base address of B

                "IMUL.U32 R8, R5, 4",       # R8 = global_tid * 4
                "IADD R9, R3, R8",          # R9 = &A[global_tid]
                "IADD R10, R4, R8",         # R10 = &B[global_tid]

                "LDG.U32 R11, [R9]",         # R11 = A[global_tid]
                "LDG.U32 R12, [R10]",        # R12 = B[global_tid]

                # ========== Compute C = A + B ==========
                "IADD R13, R11, R12",        # R13 = A[global_tid] + B[global_tid]

                # ========== Store Result ==========
                f"MOV R14, {C_addr}",        # R14 = base address of C
                "IADD R15, R14, R8",         # R15 = &C[global_tid]
                "STG.U32 [R15], R13",       # C[global_tid] = R13

                "EXIT",
            ]

            return [line.strip() for line in kernel]

    return VectorAddKernel()


# Helper functions for common operations

def to_device(array: List[int]) -> Tuple[DevicePointer, bytes]:
    """
    Convert a host array to device memory.

    Args:
        array: List of integers

    Returns:
        (DevicePointer, host_data) tuple
    """
    # Pack as 32-bit integers
    host_data = bytearray()
    for val in array:
        host_data.extend(struct.pack('<I', val))

    return (host_data, len(host_data))


def from_device(data: bytes, count: int) -> List[int]:
    """
    Convert device data to host array.

    Args:
        data: Bytes from device
        count: Number of integers to unpack

    Returns:
        List of integers
    """
    result = []
    for i in range(count):
        if i * 4 + 4 <= len(data):
            val = struct.unpack('<I', data[i*4:i*4+4])[0]
            result.append(val)
    return result


# Convenience functions matching CUDA API naming

def cudaMalloc(simulator, size: int) -> DevicePointer:
    """Allocate device memory."""
    runtime = CUDARuntime(simulator)
    return runtime.cudaMalloc(size)


def cudaMemcpyHtoD(simulator, ptr: DevicePointer, host_data: bytes) -> None:
    """Copy host to device."""
    runtime = CUDARuntime(simulator)
    runtime.cudaMemcpyHtoD(ptr, host_data)


def cudaMemcpyDtoH(simulator, ptr: DevicePointer, size: int = None) -> bytes:
    """Copy device to host."""
    runtime = CUDARuntime(simulator)
    return runtime.cudaMemcpyDtoH(ptr, size)


def launch_kernel(simulator, kernel_func, grid_dim, block_dim, *args):
    """Launch a kernel."""
    runtime = CUDARuntime(simulator)
    runtime.launch_kernel(kernel_func, grid_dim, block_dim, *args)
