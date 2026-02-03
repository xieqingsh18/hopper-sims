"""
Memory Model for Hopper GPU

Implements the different memory spaces in Hopper architecture.
"""

from enum import Enum, auto
from typing import Union, Optional
import struct


class MemorySpace(Enum):
    """Types of memory spaces in GPU."""
    GLOBAL = auto()      # Global memory (device memory) - HBM
    SHARED = auto()      # Shared memory (on-chip, per-block)
    LOCAL = auto()       # Local memory (per-thread, private)
    CONSTANT = auto()    # Constant memory (read-only, cached)
    TEXTURE = auto()     # Texture memory (read-only, cached)


class MemoryRegion:
    """A contiguous region of memory."""

    def __init__(self, size: int, name: str = "") -> None:
        """
        Initialize a memory region.

        Args:
            size: Size in bytes
            name: Optional name for debugging
        """
        self.size = size
        self.name = name
        self._data = bytearray(size)

    def read(self, offset: int, size: int) -> bytes:
        """
        Read bytes from memory region.

        Args:
            offset: Byte offset within region
            size: Number of bytes to read

        Returns:
            Data as bytes

        Raises:
            ValueError: If offset or size is out of bounds
        """
        if offset < 0 or offset + size > self.size:
            raise ValueError(f"Memory access out of bounds: offset={offset}, size={size}, region_size={self.size}")

        return bytes(self._data[offset:offset + size])

    def write(self, offset: int, data: bytes) -> None:
        """
        Write bytes to memory region.

        Args:
            offset: Byte offset within region
            data: Data to write

        Raises:
            ValueError: If offset is out of bounds
        """
        size = len(data)
        if offset < 0 or offset + size > self.size:
            raise ValueError(f"Memory access out of bounds: offset={offset}, size={size}, region_size={self.size}")

        self._data[offset:offset + size] = data

    def read_u8(self, offset: int) -> int:
        """Read an unsigned 8-bit value."""
        data = self.read(offset, 1)
        return data[0]

    def read_u16(self, offset: int) -> int:
        """Read an unsigned 16-bit value (little-endian)."""
        data = self.read(offset, 2)
        return int.from_bytes(data, byteorder='little')

    def read_u32(self, offset: int) -> int:
        """Read an unsigned 32-bit value (little-endian)."""
        data = self.read(offset, 4)
        return int.from_bytes(data, byteorder='little')

    def read_u64(self, offset: int) -> int:
        """Read an unsigned 64-bit value (little-endian)."""
        data = self.read(offset, 8)
        return int.from_bytes(data, byteorder='little')

    def read_f32(self, offset: int) -> float:
        """Read a 32-bit floating point value."""
        data = self.read(offset, 4)
        return struct.unpack('<f', data)[0]

    def read_f64(self, offset: int) -> float:
        """Read a 64-bit floating point value."""
        data = self.read(offset, 8)
        return struct.unpack('<d', data)[0]

    def write_u8(self, offset: int, value: int) -> None:
        """Write an unsigned 8-bit value."""
        self.write(offset, bytes([value & 0xFF]))

    def write_u16(self, offset: int, value: int) -> None:
        """Write an unsigned 16-bit value (little-endian)."""
        self.write(offset, value.to_bytes(2, byteorder='little'))

    def write_u32(self, offset: int, value: int) -> None:
        """Write an unsigned 32-bit value (little-endian)."""
        self.write(offset, value.to_bytes(4, byteorder='little'))

    def write_u64(self, offset: int, value: int) -> None:
        """Write an unsigned 64-bit value (little-endian)."""
        self.write(offset, value.to_bytes(8, byteorder='little'))

    def write_f32(self, offset: int, value: float) -> None:
        """Write a 32-bit floating point value."""
        data = struct.pack('<f', value)
        self.write(offset, data)

    def write_f64(self, offset: int, value: float) -> None:
        """Write a 64-bit floating point value."""
        data = struct.pack('<d', value)
        self.write(offset, data)

    def __repr__(self) -> str:
        """String representation."""
        name_str = f" '{self.name}'" if self.name else ""
        return f"MemoryRegion(size={self.size}{name_str})"


class Memory:
    """
    GPU Memory subsystem.

    Hopper Memory Hierarchy (simplified):
    - Global Memory: HBM3 (up to 80GB, ~3.35 TB/s bandwidth)
    - Shared Memory: On-chip, 228 KB per SM
    - Local Memory: Per-thread, resides in global memory
    - Constant Memory: 64 KB, read-only, cached
    - Texture Memory: Read-only, cached
    """

    # Memory sizes for a single SM (simplified)
    SHARED_MEM_SIZE = 228 * 1024  # 228 KB per SM
    CONSTANT_MEM_SIZE = 64 * 1024  # 64 KB constant cache
    DEFAULT_GLOBAL_MEM_SIZE = 1024 * 1024 * 1024  # 1 GB default

    def __init__(self,
                 global_size: int = DEFAULT_GLOBAL_MEM_SIZE,
                 shared_size: int = SHARED_MEM_SIZE,
                 constant_size: int = CONSTANT_MEM_SIZE) -> None:
        """
        Initialize GPU memory.

        Args:
            global_size: Size of global memory in bytes
            shared_size: Size of shared memory in bytes
            constant_size: Size of constant memory in bytes
        """
        # Global memory
        self.global_memory = MemoryRegion(global_size, "Global")

        # Shared memory (per-block/shared by all threads in block)
        self.shared_memory = MemoryRegion(shared_size, "Shared")

        # Constant memory (read-only)
        self.constant_memory = MemoryRegion(constant_size, "Constant")

        # Local memory (per-thread, we allocate on demand)
        self.local_memory: dict[int, MemoryRegion] = {}

    def read(self, space: MemorySpace, offset: int, size: int, thread_id: Optional[int] = None) -> bytes:
        """
        Read from a memory space.

        Args:
            space: Which memory space to read from
            offset: Byte offset within the memory space
            size: Number of bytes to read
            thread_id: Thread ID (required for LOCAL memory space)

        Returns:
            Data as bytes
        """
        if space == MemorySpace.GLOBAL:
            return self.global_memory.read(offset, size)
        elif space == MemorySpace.SHARED:
            return self.shared_memory.read(offset, size)
        elif space == MemorySpace.CONSTANT:
            return self.constant_memory.read(offset, size)
        elif space == MemorySpace.LOCAL:
            if thread_id is None:
                raise ValueError("thread_id required for LOCAL memory access")
            if thread_id not in self.local_memory:
                raise ValueError(f"No local memory allocated for thread {thread_id}")
            return self.local_memory[thread_id].read(offset, size)
        else:
            raise ValueError(f"Unsupported memory space: {space}")

    def write(self, space: MemorySpace, offset: int, data: bytes, thread_id: Optional[int] = None) -> None:
        """
        Write to a memory space.

        Args:
            space: Which memory space to write to
            offset: Byte offset within the memory space
            data: Data to write
            thread_id: Thread ID (required for LOCAL memory space)
        """
        if space == MemorySpace.GLOBAL:
            self.global_memory.write(offset, data)
        elif space == MemorySpace.SHARED:
            self.shared_memory.write(offset, data)
        elif space == MemorySpace.CONSTANT:
            raise ValueError("Cannot write to constant memory (read-only)")
        elif space == MemorySpace.LOCAL:
            if thread_id is None:
                raise ValueError("thread_id required for LOCAL memory access")
            if thread_id not in self.local_memory:
                # Allocate local memory on demand (default 256 bytes)
                self.allocate_local_memory(thread_id, 256)
            self.local_memory[thread_id].write(offset, data)
        else:
            raise ValueError(f"Unsupported memory space: {space}")

    def allocate_local_memory(self, thread_id: int, size: int) -> None:
        """
        Allocate local memory for a thread.

        Args:
            thread_id: Thread identifier
            size: Size in bytes
        """
        if thread_id in self.local_memory:
            raise ValueError(f"Local memory already allocated for thread {thread_id}")

        self.local_memory[thread_id] = MemoryRegion(size, f"Local_T{thread_id}")

    def read_u32(self, space: MemorySpace, offset: int, thread_id: Optional[int] = None) -> int:
        """Read a 32-bit unsigned integer."""
        data = self.read(space, offset, 4, thread_id)
        return int.from_bytes(data, byteorder='little')

    def write_u32(self, space: MemorySpace, offset: int, value: int, thread_id: Optional[int] = None) -> None:
        """Write a 32-bit unsigned integer."""
        data = value.to_bytes(4, byteorder='little')
        self.write(space, offset, data, thread_id)

    def read_f32(self, space: MemorySpace, offset: int, thread_id: Optional[int] = None) -> float:
        """Read a 32-bit float."""
        data = self.read(space, offset, 4, thread_id)
        return struct.unpack('<f', data)[0]

    def write_f32(self, space: MemorySpace, offset: int, value: float, thread_id: Optional[int] = None) -> None:
        """Write a 32-bit float."""
        data = struct.pack('<f', value)
        self.write(space, offset, data, thread_id)

    def __repr__(self) -> str:
        """String representation."""
        return (f"Memory(global={self.global_memory.size >> 20}MB, "
                f"shared={self.shared_memory.size >> 10}KB, "
                f"local_threads={len(self.local_memory)})")


if __name__ == "__main__":
    # Test memory
    mem = Memory()

    # Test global memory
    mem.write_u32(MemorySpace.GLOBAL, 0x100, 0xDEADBEEF)
    val = mem.read_u32(MemorySpace.GLOBAL, 0x100)
    print(f"Global memory read: {val:#x}")

    # Test shared memory
    mem.write_f32(MemorySpace.SHARED, 0, 3.14159)
    fval = mem.read_f32(MemorySpace.SHARED, 0)
    print(f"Shared memory float: {fval}")

    # Test local memory
    mem.allocate_local_memory(0, 1024)
    mem.write_u32(MemorySpace.LOCAL, 0, 0xCAFEBABE, thread_id=0)
    lval = mem.read_u32(MemorySpace.LOCAL, 0, thread_id=0)
    print(f"Local memory read: {lval:#x}")

    print(mem)
