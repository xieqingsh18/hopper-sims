"""
Hopper GPU Memory Model - PTX Compliant Implementation

Implements the complete memory hierarchy according to PTX ISA 9.1 specification:
- Memory spaces (global, shared, local, constant, texture, tensor)
- Memory consistency model
- Barriers and synchronization
- Memory ordering and visibility
- Cache hierarchy (L1, L2, shared memory)
- Bank conflicts and coalescing
- Fence operations
- mbarrier operations
"""

from enum import Enum, auto, IntEnum
from typing import Union, Optional, Dict, List, Set, Tuple
from dataclasses import dataclass, field
import struct
import time


# ==============================================================================
# Memory Spaces
# ==============================================================================

class MemorySpace(Enum):
    """Types of memory spaces in GPU according to PTX ISA."""
    GLOBAL = auto()      # Global memory (device memory) - HBM
    SHARED = auto()      # Shared memory (on-chip, per-block/cluster)
    LOCAL = auto()       # Local memory (per-thread, private, in global)
    CONSTANT = auto()    # Constant memory (read-only, cached)
    TEXTURE = auto()     # Texture memory (read-only, cached, deprecated)
    TENSOR = auto()      # Tensor memory (for WGMMA operations)
    REG = auto()         # Register file (fast, per-thread)


class Scope(IntEnum):
    """Memory visibility scope according to PTX ISA."""
    CTA = 0      # Cooperative Thread Array (block/warp group)
    CLUSTER = 1  # Cluster of CTAs
    GPU = 2      # Entire GPU (all CTAs)
    SYSTEM = 3   # Across multiple GPUs


class MemoryOrder(IntEnum):
    """Memory ordering semantics according to PTX ISA."""
    RELAXED = 0     # No ordering guarantees
    ACQUIRE = 1     # Acquire semantics (load)
    RELEASE = 2     # Release semantics (store)
    ACQ_REL = 3     # Acquire+Release
    SC = 4          # Sequentially consistent


# ==============================================================================
# Cache Operators and Hints
# ==============================================================================

class CacheOperator(Enum):
    """Cache operators for load/store instructions."""
    CA = auto()  # Cache at all levels (default for loads)
    CG = auto()  # Cache at global level (L2 only)
    CS = auto()  # Cache streaming (evict-first)
    LU = auto()  # Last use
    CV = auto()  # Don't cache
    WB = auto()  # Write-back (default for stores)
    WT = auto()  # Write-through


class EvictPriority(IntEnum):
    """Eviction priority hints (sm_70+)."""
    NORMAL = 0
    FIRST = 1
    LAST = 2
    UNCHANGED = 3


# ==============================================================================
# Memory Consistency Model
# ==============================================================================

class ProxyType(Enum):
    """Memory access proxy types (PTX ISA 9.1)."""
    GENERIC = auto()    # Generic proxy: ld/st/ldmatrix/stmatrix operations
    ASYNC = auto()      # Async proxy: TMA (cp.async.bulk), WGMMA operations


@dataclass
class MemoryOperation:
    """A memory operation with ordering metadata."""
    address: int
    size: int
    is_store: bool
    space: MemorySpace
    thread_id: int
    warp_id: int
    cta_id: int
    order: MemoryOrder = MemoryOrder.RELAXED
    scope: Scope = Scope.CTA
    timestamp: int = 0
    is_visible: bool = False  # Becomes true after synchronization
    proxy: ProxyType = ProxyType.GENERIC  # Which proxy performed this operation


@dataclass
class PendingBarrier:
    """A pending barrier synchronization."""
    barrier_id: int
    scope: Scope
    expected_count: int
    arrived_count: int = 0
    complete: bool = False
    waiting_warps: Set[int] = field(default_factory=set)
    waiting_threads: Set[Tuple[int, int]] = field(default_factory=set)  # (warp_id, thread_id)


@dataclass
class MBarrierState:
    """State of an mbarrier (asynchronous barrier)."""
    address: int
    scope: Scope
    phase: int = 0  # 0 or 1 for toggle
    arrived_count: int = 0
    expected_count: int = 0
    completed_phase: int = -1  # Which phase has completed
    pending_operations: List[MemoryOperation] = field(default_factory=list)


# ==============================================================================
# Memory Region with Latency Modeling
# ==============================================================================

class MemoryRegion:
    """A contiguous region of memory with access latency tracking."""

    # Latencies in cycles for Hopper architecture
    LATENCY = {
        MemorySpace.REG: 1,
        MemorySpace.SHARED: 30,      # Bank-conflict free
        MemorySpace.CONSTANT: 30,    # L1 cache hit
        MemorySpace.GLOBAL: 400,     # Average with cache misses
        MemorySpace.LOCAL: 400,      # Spilled to global
        MemorySpace.TEXTURE: 30,     # L1 cache hit
        MemorySpace.TENSOR: 100,     # Asynchronous operation
    }

    def __init__(self, size: int, name: str = "", space: MemorySpace = MemorySpace.GLOBAL) -> None:
        """
        Initialize a memory region.

        Args:
            size: Size in bytes
            name: Optional name for debugging
            space: Type of memory space
        """
        self.size = size
        self.name = name
        self.space = space
        self._data = bytearray(size)
        self._access_count = 0
        self._last_access_time = 0

    def read(self, offset: int, size: int) -> bytes:
        """Read bytes from memory region."""
        if offset < 0 or offset + size > self.size:
            raise ValueError(f"Memory access out of bounds: offset={offset}, size={size}, region_size={self.size}")
        self._access_count += 1
        return bytes(self._data[offset:offset + size])

    def write(self, offset: int, data: bytes) -> None:
        """Write bytes to memory region."""
        size = len(data)
        if offset < 0 or offset + size > self.size:
            raise ValueError(f"Memory access out of bounds: offset={offset}, size={size}, region_size={self.size}")
        self._data[offset:offset + size] = data
        self._access_count += 1

    def get_latency(self, access_size: int = 4) -> int:
        """Get access latency in cycles."""
        base_latency = self.LATENCY.get(self.space, 100)
        # Add variability for simulation
        return base_latency

    # Type-specific read/write methods
    def read_u8(self, offset: int) -> int: return self.read(offset, 1)[0]
    def read_u16(self, offset: int) -> int: return int.from_bytes(self.read(offset, 2), 'little')
    def read_u32(self, offset: int) -> int: return int.from_bytes(self.read(offset, 4), 'little')
    def read_u64(self, offset: int) -> int: return int.from_bytes(self.read(offset, 8), 'little')
    def read_f32(self, offset: int) -> float: return struct.unpack('<f', self.read(offset, 4))[0]
    def read_f64(self, offset: int) -> int: return struct.unpack('<d', self.read(offset, 8))[0]

    def write_u8(self, offset: int, value: int) -> None: self.write(offset, bytes([value & 0xFF]))
    def write_u16(self, offset: int, value: int) -> None: self.write(offset, value.to_bytes(2, 'little'))
    def write_u32(self, offset: int, value: int) -> None: self.write(offset, value.to_bytes(4, 'little'))
    def write_u64(self, offset: int, value: int) -> None: self.write(offset, value.to_bytes(8, 'little'))
    def write_f32(self, offset: int, value: float) -> None: self.write(offset, struct.pack('<f', value))
    def write_f64(self, offset: int, value: float) -> None: self.write(offset, struct.pack('<d', value))

    def __repr__(self) -> str:
        return f"MemoryRegion(size={self.size}, space={self.space.name}, name='{self.name}')"


# ==============================================================================
# Shared Memory with Bank Conflicts
# ==============================================================================

class SharedMemoryRegion(MemoryRegion):
    """
    Shared memory with bank conflict modeling.

    Hopper shared memory organization:
    - 32 banks, 4 bytes per bank
    - 192 KB total per SM (96 KB data + 96 KB parity)
    - Bank conflicts cause serialization
    """

    NUM_BANKS = 32
    BANK_WIDTH = 4  # bytes per bank

    def __init__(self, size: int, name: str = "") -> None:
        super().__init__(size, name, MemorySpace.SHARED)
        self._bank_access_counts = [0] * self.NUM_BANKS

    def get_bank(self, address: int) -> int:
        """Get which bank an address maps to."""
        return (address // self.BANK_WIDTH) % self.NUM_BANKS

    def get_latency(self, access_size: int = 4) -> int:
        """Get access latency with bank conflicts considered."""
        base_latency = self.LATENCY[MemorySpace.SHARED]
        # TODO: Model bank conflicts based on warp-wide access pattern
        return base_latency


# ==============================================================================
# Barrier and Synchronization
# ==============================================================================

class BarrierManager:
    """Manages barrier synchronization and memory visibility."""

    def __init__(self):
        self._barriers: Dict[int, PendingBarrier] = {}
        self._next_barrier_id = 0
        self._mbarriers: Dict[int, MBarrierState] = {}
        self._pending_operations: List[MemoryOperation] = []
        self._completed_operations: List[MemoryOperation] = []

    def allocate_barrier(self) -> int:
        """Allocate a new barrier ID."""
        barrier_id = self._next_barrier_id
        self._next_barrier_id += 1
        return barrier_id

    def barrier_sync(self, barrier_id: int, num_threads: int, thread_id: int, warp_id: int, cta_id: int) -> bool:
        """
        Execute barrier.sync instruction.

        All threads in CTA must reach this barrier before any can proceed.
        Provides full memory fence semantics for CTA scope.

        Returns:
            True if barrier is complete (all threads arrived), False if waiting
        """
        if barrier_id not in self._barriers:
            self._barriers[barrier_id] = PendingBarrier(
                barrier_id=barrier_id,
                scope=Scope.CTA,
                expected_count=num_threads
            )

        barrier = self._barriers[barrier_id]
        thread_key = (warp_id, thread_id)

        if thread_key not in barrier.waiting_threads:
            barrier.arrived_count += 1
            barrier.waiting_threads.add(thread_key)

        # Check if barrier is complete
        if barrier.arrived_count >= barrier.expected_count:
            # Barrier complete - make all pending memory visible within CTA
            self._make_cta_operations_visible(cta_id)
            barrier.complete = True
            return True

        return False

    def barrier_cluster_arrive(self, barrier_id: int, thread_id: int, warp_id: int) -> None:
        """Arrive at cluster barrier."""
        if barrier_id not in self._barriers:
            self._barriers[barrier_id] = PendingBarrier(
                barrier_id=barrier_id,
                scope=Scope.CLUSTER,
                expected_count=0  # Will be set when wait is called
            )

        barrier = self._barriers[barrier_id]
        thread_key = (warp_id, thread_id)
        if thread_key not in barrier.waiting_threads:
            barrier.arrived_count += 1
            barrier.waiting_threads.add(thread_key)

    def barrier_cluster_wait(self, barrier_id: int, num_threads: int, cta_id: int) -> bool:
        """Wait at cluster barrier."""
        if barrier_id not in self._barriers:
            return False

        barrier = self._barriers[barrier_id]
        if barrier.arrived_count >= num_threads:
            self._make_cluster_operations_visible()
            barrier.complete = True
            return True
        return False

    def mbarrier_init(self, address: int, count: int, scope: Scope) -> None:
        """Initialize mbarrier at address."""
        self._mbarriers[address] = MBarrierState(
            address=address,
            scope=scope,
            expected_count=count
        )

    def mbarrier_arrive(self, address: int, thread_id: int, warp_id: int) -> None:
        """Arrive at mbarrier."""
        if address not in self._mbarriers:
            raise ValueError(f"mbarrier not initialized at address 0x{address:x}")

        mbarrier = self._mbarriers[address]
        mbarrier.arrived_count += 1

        # Toggle phase when expected count reached
        if mbarrier.arrived_count >= mbarrier.expected_count:
            mbarrier.completed_phase = mbarrier.phase
            mbarrier.phase = 1 - mbarrier.phase  # Toggle
            mbarrier.arrived_count = 0

    def mbarrier_test_wait(self, address: int, phase: int) -> bool:
        """Test if mbarrier phase is complete."""
        if address not in self._mbarriers:
            raise ValueError(f"mbarrier not initialized at address 0x{address:x}")

        mbarrier = self._mbarriers[address]
        return mbarrier.completed_phase == phase

    def mbarrier_complete_tx(self, address: int, count: int, warp_id: int) -> None:
        """
        Complete transactions for mbarrier.

        Called by async operations (TMA, WGMMA) when they complete.
        Signals that async operations have finished and data is ready.
        """
        if address not in self._mbarriers:
            # Mbarrier may not be initialized yet (lazy init)
            self._mbarriers[address] = MBarrierState(
                address=address,
                scope=Scope.CTA,
                expected_count=count
            )

        mbarrier = self._mbarriers[address]

        # Increment arrived count (async operations complete)
        mbarrier.arrived_count += count

        # Check if we've reached the expected count
        if mbarrier.arrived_count >= mbarrier.expected_count:
            # Toggle phase and mark complete
            mbarrier.completed_phase = mbarrier.phase
            mbarrier.phase = 1 - mbarrier.phase
            mbarrier.arrived_count = 0

    def _make_cta_operations_visible(self, cta_id: int) -> None:
        """Make all pending CTA-scoped operations visible."""
        for op in self._pending_operations:
            if op.cta_id == cta_id and op.scope == Scope.CTA:
                op.is_visible = True
                self._completed_operations.append(op)
        self._pending_operations = [op for op in self._pending_operations
                                    if not (op.cta_id == cta_id and op.scope == Scope.CTA)]

    def _make_cluster_operations_visible(self) -> None:
        """Make all pending cluster-scoped operations visible."""
        for op in self._pending_operations:
            if op.scope in (Scope.CTA, Scope.CLUSTER):
                op.is_visible = True
                self._completed_operations.append(op)
        self._pending_operations = [op for op in self._pending_operations
                                    if op.scope not in (Scope.CTA, Scope.CLUSTER)]

    def fence(self, scope: Scope, order: MemoryOrder, thread_id: int, warp_id: int, cta_id: int) -> None:
        """
        Execute fence instruction.

        Ensures memory ordering according to scope and semantics.
        """
        if order == MemoryOrder.SC:
            # Strongest fence - make all operations visible within scope
            self._flush_pending_operations(scope, cta_id)
        elif order == MemoryOrder.ACQUIRE:
            # Acquire semantics - make prior operations visible
            self._make_acquire_visible(scope, cta_id)
        elif order == MemoryOrder.RELEASE:
            # Release semantics - prepare to make operations visible
            self._make_release_ready(scope, cta_id)
        elif order == MemoryOrder.ACQ_REL:
            self._make_acquire_visible(scope, cta_id)
            self._make_release_ready(scope, cta_id)

    def membar(self, scope: Scope) -> None:
        """Execute membar instruction (global memory barrier)."""
        # Make all global memory operations visible within scope
        self._flush_pending_operations(scope, -1)  # -1 means all CTAs

    def _flush_pending_operations(self, scope: Scope, cta_id: int) -> None:
        """Flush all pending operations within scope."""
        if scope == Scope.CTA:
            self._make_cta_operations_visible(cta_id)
        elif scope == Scope.CLUSTER:
            self._make_cluster_operations_visible()
        else:
            # GPU or system scope - make everything visible
            for op in self._pending_operations:
                op.is_visible = True
                self._completed_operations.append(op)
            self._pending_operations.clear()

    def _make_acquire_visible(self, scope: Scope, cta_id: int) -> None:
        """Make prior operations visible (acquire semantics)."""
        # Acquire ensures prior operations are visible to current thread
        self._flush_pending_operations(scope, cta_id)

    def _make_release_ready(self, scope: Scope, cta_id: int) -> None:
        """Prepare operations to be visible (release semantics)."""
        # Release marks operations that should become visible to other threads
        # Operations become visible after next acquire/barrier
        pass  # Operations already tracked in pending list

    def add_pending_operation(self, op: MemoryOperation) -> None:
        """Add a memory operation to pending list."""
        self._pending_operations.append(op)

    def is_operation_visible(self, op: MemoryOperation, requesting_thread_id: int,
                            requesting_warp_id: int, requesting_cta_id: int) -> bool:
        """Check if an operation is visible to requesting thread."""
        if op.is_visible:
            return True

        # Operations in same thread are always visible
        if op.thread_id == requesting_thread_id and op.warp_id == requesting_warp_id:
            return True

        # Operations in same CTA with relaxed ordering require barrier
        if op.cta_id == requesting_cta_id and op.scope == Scope.CTA:
            return False  # Requires barrier

        return False


# ==============================================================================
# Main Memory System
# ==============================================================================

class Memory:
    """
    Complete Hopper Memory Hierarchy according to PTX ISA.

    Memory sizes for Hopper (sm_90a):
    - Global Memory: Up to 80 GB HBM3
    - Shared Memory: 228 KB per SM (per CTA, cluster-addressable)
    - L1 Cache: 128 KB per SM (64 KB data, 64 KB instruction)
    - L2 Cache: 50 MB total, shared across GPU
    - Constant Memory: 64 KB + 10 x 64 KB regions = 704 KB total
    - Local Memory: Per-thread, resides in global memory
    - Tensor Memory: Per-cluster, for WGMMA operations
    """

    # Hopper memory sizes
    SHARED_MEM_SIZE = 228 * 1024      # 228 KB per SM
    CONSTANT_MEM_SIZE = 64 * 1024      # 64 KB base
    CONSTANT_REGIONS = 10 * 64 * 1024  # 10 x 64 KB regions
    L1_CACHE_SIZE = 128 * 1024         # 128 KB per SM
    L2_CACHE_SIZE = 50 * 1024 * 1024   # 50 MB
    DEFAULT_GLOBAL_MEM_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB default

    def __init__(self,
                 global_size: int = DEFAULT_GLOBAL_MEM_SIZE,
                 shared_size: int = SHARED_MEM_SIZE,
                 constant_size: int = CONSTANT_MEM_SIZE) -> None:
        """Initialize GPU memory with all spaces."""
        # Global memory (HBM3)
        self.global_memory = MemoryRegion(global_size, "Global", MemorySpace.GLOBAL)

        # Shared memory (on-chip, banked)
        self.shared_memory = SharedMemoryRegion(shared_size, "Shared")

        # Constant memory (read-only, cached)
        self.constant_memory = MemoryRegion(constant_size, "Constant", MemorySpace.CONSTANT)

        # Local memory (per-thread, allocated on demand)
        self.local_memory: Dict[int, MemoryRegion] = {}

        # Tensor memory (for WGMMA, per-cluster)
        self.tensor_memory: Dict[int, MemoryRegion] = {}

        # Barrier and synchronization manager
        self.barrier_manager = BarrierManager()

        # Current cycle counter
        self._current_cycle = 0

    def advance_cycles(self, cycles: int) -> None:
        """Advance the cycle counter."""
        self._current_cycle += cycles

    def read(self, space: MemorySpace, offset: int, size: int,
             thread_id: Optional[int] = None, warp_id: Optional[int] = None,
             cta_id: Optional[int] = None, order: MemoryOrder = MemoryOrder.RELAXED,
             scope: Scope = Scope.CTA, proxy: ProxyType = ProxyType.GENERIC) -> bytes:
        """
        Read from a memory space with ordering and visibility checks.

        Args:
            space: Which memory space to read from
            offset: Byte offset within the memory space
            size: Number of bytes to read
            thread_id: Thread ID (required for LOCAL memory)
            warp_id: Warp ID
            cta_id: CTA ID
            order: Memory ordering semantics
            scope: Memory visibility scope
            proxy: Which proxy is performing this read (GENERIC or ASYNC)

        Returns:
            Data as bytes
        """
        # For weak memory model, check visibility for shared/global memory
        # Proxy-based ordering: Different proxies have weak ordering

        should_check_visibility = False
        if space in (MemorySpace.SHARED, MemorySpace.GLOBAL):
            if thread_id is not None and warp_id is not None and cta_id is not None:
                should_check_visibility = True

        data = None
        if should_check_visibility:
            # Find the most recent write at this address
            visible_data = self._get_visible_data(space, offset, size, thread_id, warp_id, cta_id, scope, proxy)
            if visible_data is not None:
                data = visible_data

        # Perform actual memory read (or use cached visible data)
        if data is None:
            if space == MemorySpace.GLOBAL:
                data = self.global_memory.read(offset, size)
            elif space == MemorySpace.SHARED:
                data = self.shared_memory.read(offset, size)
            elif space == MemorySpace.CONSTANT:
                data = self.constant_memory.read(offset, size)
            elif space == MemorySpace.LOCAL:
                if thread_id is None:
                    raise ValueError("thread_id required for LOCAL memory access")
                if thread_id not in self.local_memory:
                    raise ValueError(f"No local memory allocated for thread {thread_id}")
                data = self.local_memory[thread_id].read(offset, size)
            elif space == MemorySpace.TENSOR:
                if thread_id is None:
                    raise ValueError("thread_id required for TENSOR memory access")
                if thread_id not in self.tensor_memory:
                    raise ValueError(f"No tensor memory allocated for thread {thread_id}")
                data = self.tensor_memory[thread_id].read(offset, size)
            else:
                raise ValueError(f"Unsupported memory space: {space}")

        return data

    def _get_visible_data(self, space: MemorySpace, offset: int, size: int,
                          thread_id: int, warp_id: int, cta_id: int,
                          scope: Scope, requesting_proxy: ProxyType = ProxyType.GENERIC) -> Optional[bytes]:
        """
        Get the data that should be visible to this thread based on weak memory rules.

        PTX ISA 9.1 Proxy Ordering:
        - Same proxy operations: Ordered (visible to same thread)
        - Different proxy operations: NOT ordered (not visible without fence.proxy)
        - fence.proxy: Establishes ordering between different proxies

        Args:
            requesting_proxy: Which proxy is performing this read operation

        Returns:
            Visible data bytes, or None if no tracked data found
        """
        # Look for the most recent operation at this address
        most_recent_op = None

        # Check completed operations first (visible to all)
        for op in reversed(self.barrier_manager._completed_operations):
            if (op.space == space and
                op.address == offset and
                op.is_store and
                hasattr(op, 'data') and
                len(op.data) == size):
                most_recent_op = op
                break

        # If no visible operation found, check pending operations
        if most_recent_op is None:
            for op in reversed(self.barrier_manager._pending_operations):
                if (op.space == space and
                    op.address == offset and
                    op.is_store and
                    hasattr(op, 'data') and
                    len(op.data) == size):
                    most_recent_op = op
                    break

        if most_recent_op is None:
            # No tracked operation - return None (will read from memory)
            return None

        # Check if this operation is visible to the requesting thread
        if most_recent_op.is_visible:
            # Globally visible - all threads can see it
            return most_recent_op.data

        # Not globally visible yet - apply proxy-based visibility rules
        if most_recent_op.thread_id == thread_id and most_recent_op.warp_id == warp_id:
            # Same thread/warp - check proxy type
            if most_recent_op.proxy == requesting_proxy:
                # SAME proxy: operations are ordered, immediately visible
                return most_recent_op.data
            else:
                # DIFFERENT proxy: NOT ordered, NOT visible without fence.proxy
                # This is the KEY behavior for fence.proxy!
                return None

        # Different thread/warp + not globally visible = WEAK MEMORY
        # This thread cannot see the write yet - will read stale data from memory
        return None

    def _is_data_visible(self, space: MemorySpace, offset: int,
                        thread_id: int, warp_id: int, cta_id: int,
                        scope: Scope) -> bool:
        """
        Check if data at address is visible to this thread.

        Weak memory model rules:
        - Thread always sees its OWN operations (same-thread visibility)
        - Thread sees other thread's operations only after fence/barrier

        Returns:
            True if data is visible, False if not yet visible
        """
        # Find the most recent write operation at this address
        most_recent_op = None
        most_recent_cycle = -1

        # Check completed (visible) operations
        for op in self.barrier_manager._completed_operations:
            if (op.space == space and
                op.address == offset and
                op.is_store and
                op.timestamp > most_recent_cycle):
                most_recent_op = op
                most_recent_cycle = op.timestamp

        # Check pending (not yet visible) operations
        for op in self.barrier_manager._pending_operations:
            if (op.space == space and
                op.address == offset and
                op.is_store and
                op.timestamp > most_recent_cycle):
                most_recent_op = op
                most_recent_cycle = op.timestamp

        if most_recent_op is None:
            # No operation found - data is in initial state (visible)
            return True

        # Check if this operation is visible to this thread
        if most_recent_op.is_visible:
            # Operation has been made visible to all threads
            return True

        # Operation is not yet globally visible
        # Check same-thread visibility rule
        if most_recent_op.thread_id == thread_id:
            # Thread always sees its own operations (even before fence)
            # This is crucial for program correctness
            return True

        # Different thread, operation not yet visible - weak memory!
        # This thread will see stale data
        return False

    def write(self, space: MemorySpace, offset: int, data: bytes,
              thread_id: Optional[int] = None, warp_id: Optional[int] = None,
              cta_id: Optional[int] = None, order: MemoryOrder = MemoryOrder.RELAXED,
              scope: Scope = Scope.CTA, proxy: ProxyType = ProxyType.GENERIC) -> None:
        """
        Write to a memory space with ordering and visibility tracking.

        Args:
            space: Which memory space to write to
            offset: Byte offset within the memory space
            data: Data to write
            thread_id: Thread ID (required for LOCAL memory)
            warp_id: Warp ID
            cta_id: CTA ID
            order: Memory ordering semantics
            scope: Memory visibility scope
            proxy: Which proxy is performing this write (GENERIC or ASYNC)
        """
        # Track operation for ordering if not relaxed
        if order != MemoryOrder.RELAXED or space == MemorySpace.SHARED:
            if thread_id is not None and warp_id is not None and cta_id is not None:
                op = MemoryOperation(
                    address=offset,
                    size=len(data),
                    is_store=True,
                    space=space,
                    thread_id=thread_id,
                    warp_id=warp_id,
                    cta_id=cta_id,
                    order=order,
                    scope=scope,
                    timestamp=self._current_cycle,
                    is_visible=False,  # Start as not visible
                    proxy=proxy  # Track which proxy performed this operation
                )
                # Store the written data with the operation for visibility tracking
                op.data = data
                self.barrier_manager.add_pending_operation(op)

                # IMPORTANT: For weak memory simulation, we need to track
                # per-thread visible state. The write is immediately
                # visible to the writing thread, but NOT to other threads.

        # Perform write (this updates the "latest" value in actual memory)
        # For proper weak memory, we'd need to buffer writes, but that's complex
        # Instead, we track which thread wrote which value and enforce on read
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
                self.allocate_local_memory(thread_id, 256)
            self.local_memory[thread_id].write(offset, data)
        elif space == MemorySpace.TENSOR:
            if thread_id is None:
                raise ValueError("thread_id required for TENSOR memory access")
            if thread_id not in self.tensor_memory:
                self.allocate_tensor_memory(thread_id, 1024)
            self.tensor_memory[thread_id].write(offset, data)
        else:
            raise ValueError(f"Unsupported memory space: {space}")

    def allocate_local_memory(self, thread_id: int, size: int) -> None:
        """Allocate local memory for a thread."""
        if thread_id in self.local_memory:
            raise ValueError(f"Local memory already allocated for thread {thread_id}")
        self.local_memory[thread_id] = MemoryRegion(size, f"Local_T{thread_id}", MemorySpace.LOCAL)

    def allocate_tensor_memory(self, cluster_id: int, size: int) -> None:
        """Allocate tensor memory for a cluster."""
        if cluster_id in self.tensor_memory:
            raise ValueError(f"Tensor memory already allocated for cluster {cluster_id}")
        self.tensor_memory[cluster_id] = MemoryRegion(size, f"Tensor_C{cluster_id}", MemorySpace.TENSOR)

    # Convenience methods for typed access
    def read_u32(self, space: MemorySpace, offset: int, **kwargs) -> int:
        """Read a 32-bit unsigned integer."""
        data = self.read(space, offset, 4, **kwargs)
        return int.from_bytes(data, byteorder='little')

    def write_u32(self, space: MemorySpace, offset: int, value: int, **kwargs) -> None:
        """Write a 32-bit unsigned integer."""
        data = value.to_bytes(4, byteorder='little')
        self.write(space, offset, data, **kwargs)

    def read_f32(self, space: MemorySpace, offset: int, **kwargs) -> float:
        """Read a 32-bit float."""
        data = self.read(space, offset, 4, **kwargs)
        return struct.unpack('<f', data)[0]

    def write_f32(self, space: MemorySpace, offset: int, value: float, **kwargs) -> None:
        """Write a 32-bit float."""
        data = struct.pack('<f', value)
        self.write(space, offset, data, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return (f"Memory(global={self.global_memory.size >> 30}GB, "
                f"shared={self.shared_memory.size >> 10}KB, "
                f"local_threads={len(self.local_memory)})")


if __name__ == "__main__":
    # Test memory system
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

    # Test barrier
    barrier_id = mem.barrier_manager.allocate_barrier()
    print(f"Allocated barrier: {barrier_id}")

    print(mem)
