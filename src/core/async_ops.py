"""
Asynchronous Operations for Hopper GPU

Implements async operation queue for TMA (Tensor Memory Accelerator)
and other asynchronous operations.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from queue import Queue
from threading import Lock


class AsyncOpType(Enum):
    """Types of asynchronous operations."""
    TMA_LOAD = auto()      # TMA load from global to shared
    TMA_STORE = auto()     # TMA store from shared to global
    WGMMA = auto()         # Warpgroup matrix multiply
    CP_ASYNC = auto()      # Async copy


class AsyncOpState(Enum):
    """States of an asynchronous operation."""
    PENDING = auto()       # Operation is queued
    IN_PROGRESS = auto()   # Operation is executing
    COMPLETED = auto()     # Operation completed successfully
    FAILED = auto()        # Operation failed


@dataclass
class AsyncOperation:
    """
    Represents an asynchronous operation.

    Async operations (like TMA) run in the background while the warp
    continues executing other instructions.
    """
    op_id: int                               # Unique operation ID
    op_type: AsyncOpType                      # Type of operation
    state: AsyncOpState = AsyncOpState.PENDING

    # Operation parameters
    src_addr: int = 0                         # Source address
    dst_addr: int = 0                         # Destination address
    size: int = 0                             # Size in bytes
    warp_id: int = 0                          # Warp that initiated this

    # Completion callback (for simulation)
    callback: Optional[Callable] = None

    # Operation metadata
    group_id: Optional[int] = None            # Operation group ID
    cycles_remaining: int = 0                 # Cycles until completion (for simulation)
    result: Optional[Any] = None              # Operation result

    def __repr__(self) -> str:
        return (f"AsyncOperation(id={self.op_id}, type={self.op_type.name}, "
                f"state={self.state.name}, group={self.group_id})")


class AsyncQueue:
    """
    Queue for managing asynchronous operations.

    In real hardware, async operations (TMA, WGMMA) run in parallel
    with warp execution. This simulator emulates this behavior.
    """

    def __init__(self, num_units: int = 4) -> None:
        """
        Initialize async operation queue.

        Args:
            num_units: Number of parallel execution units (TMA engines, etc.)
        """
        self.num_units = num_units
        self.operations: List[AsyncOperation] = []
        self.next_id = 0
        self._lock = Lock()

        # Statistics
        self.total_submitted = 0
        self.total_completed = 0
        self.total_failed = 0

    def submit(self, op: AsyncOperation) -> int:
        """
        Submit an asynchronous operation.

        Args:
            op: Operation to submit

        Returns:
            Operation ID
        """
        with self._lock:
            op.op_id = self.next_id
            self.next_id += 1
            self.operations.append(op)
            self.total_submitted += 1
            return op.op_id

    def create_tma_load(self, dst_addr: int, src_addr: int, size: int,
                       warp_id: int, cycles: int = 100) -> AsyncOperation:
        """
        Create a TMA load operation.

        Args:
            dst_addr: Destination address (shared memory)
            src_addr: Source address (global memory)
            size: Size in bytes
            warp_id: Warp ID initiating the operation
            cycles: Simulation cycles until completion

        Returns:
            AsyncOperation ready to submit
        """
        return AsyncOperation(
            op_id=0,  # Will be assigned on submit
            op_type=AsyncOpType.TMA_LOAD,
            src_addr=src_addr,
            dst_addr=dst_addr,
            size=size,
            warp_id=warp_id,
            cycles_remaining=cycles
        )

    def create_tma_store(self, dst_addr: int, src_addr: int, size: int,
                        warp_id: int, cycles: int = 100) -> AsyncOperation:
        """
        Create a TMA store operation.

        Args:
            dst_addr: Destination address (global memory)
            src_addr: Source address (shared memory)
            size: Size in bytes
            warp_id: Warp ID initiating the operation
            cycles: Simulation cycles until completion

        Returns:
            AsyncOperation ready to submit
        """
        return AsyncOperation(
            op_id=0,
            op_type=AsyncOpType.TMA_STORE,
            src_addr=src_addr,
            dst_addr=dst_addr,
            size=size,
            warp_id=warp_id,
            cycles_remaining=cycles
        )

    def create_wgmma(self, warp_id: int, cycles: int = 50) -> AsyncOperation:
        """
        Create a WGMMA operation.

        Args:
            warp_id: Warp ID initiating the operation
            cycles: Simulation cycles until completion

        Returns:
            AsyncOperation ready to submit
        """
        return AsyncOperation(
            op_id=0,
            op_type=AsyncOpType.WGMMA,
            warp_id=warp_id,
            cycles_remaining=cycles
        )

    def tick(self) -> int:
        """
        Advance simulation by one cycle.

        Processes pending async operations and returns the number
        of operations that completed in this cycle.

        Returns:
            Number of operations completed this cycle
        """
        completed = 0

        with self._lock:
            # Process all operations
            for op in self.operations:
                if op.state == AsyncOpState.PENDING:
                    # Start operation if we have free units
                    in_progress = sum(1 for o in self.operations
                                    if o.state == AsyncOpState.IN_PROGRESS)
                    if in_progress < self.num_units:
                        op.state = AsyncOpState.IN_PROGRESS

                elif op.state == AsyncOpState.IN_PROGRESS:
                    # Advance operation
                    op.cycles_remaining -= 1
                    if op.cycles_remaining <= 0:
                        op.state = AsyncOpState.COMPLETED
                        completed += 1
                        self.total_completed += 1

                        # Call completion callback if registered
                        if op.callback:
                            op.callback(op)

            # Clean up old completed operations
            # (keep recent ones for debugging)
            self.operations = [op for op in self.operations
                             if op.state != AsyncOpState.COMPLETED or
                             op.op_id > self.next_id - 100]

        return completed

    def wait_for_group(self, group_id: int) -> bool:
        """
        Check if all operations in a group are complete.

        Args:
            group_id: Group ID to check

        Returns:
            True if all operations in group are complete
        """
        with self._lock:
            group_ops = [op for op in self.operations
                        if op.group_id == group_id]
            return all(op.state == AsyncOpState.COMPLETED for op in group_ops)

    def wait_for_warp(self, warp_id: int) -> bool:
        """
        Check if all operations for a warp are complete.

        Args:
            warp_id: Warp ID to check

        Returns:
            True if all operations for warp are complete
        """
        with self._lock:
            warp_ops = [op for op in self.operations
                       if op.warp_id == warp_id]
            return all(op.state == AsyncOpState.COMPLETED for op in warp_ops)

    def get_pending_count(self) -> int:
        """Get number of pending operations."""
        with self._lock:
            return sum(1 for op in self.operations
                      if op.state in (AsyncOpState.PENDING, AsyncOpState.IN_PROGRESS))

    def get_completed_count(self) -> int:
        """Get number of completed operations."""
        return self.total_completed

    def __repr__(self) -> str:
        pending = sum(1 for op in self.operations
                     if op.state in (AsyncOpState.PENDING, AsyncOpState.IN_PROGRESS))
        return (f"AsyncQueue(units={self.num_units}, pending={pending}, "
                f"completed={self.total_completed})")
