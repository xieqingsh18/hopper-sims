"""
Warp Implementation for Hopper GPU

A warp is a group of 32 threads that execute instructions in lockstep (SIMT model).
"""

from typing import List, Callable, Optional
from dataclasses import dataclass, field
from .thread import Thread, ThreadState


@dataclass
class WarpState:
    """State of a warp."""
    active_mask: int = 0xFFFFFFFF  # Which lanes are active (1 = active)
    execution_mask: int = 0xFFFFFFFF  # Which lanes execute current instruction
    divergence_stack: List[tuple[int, int]] = field(default_factory=list)
    # Divergence stack: (target_pc, reconvergence_pc)


class Warp:
    """
    Represents a warp of 32 threads executing in lockstep.

    In SIMT (Single Instruction, Multiple Threads) model:
    - All threads in a warp execute the same instruction
    - Individual threads can be masked out (predicated execution)
    - Warp divergence occurs when threads branch to different targets
    - Threads execute until they reconverge
    """

    WARP_SIZE = 32
    FULL_MASK = 0xFFFFFFFF  # All 32 lanes active

    def __init__(self, warp_id: int, start_pc: int = 0) -> None:
        """
        Initialize a new warp.

        Args:
            warp_id: Unique warp identifier
            start_pc: Initial program counter for all threads
        """
        self.warp_id = warp_id
        self.pc = start_pc

        # Create 32 threads
        base_thread_id = warp_id * self.WARP_SIZE
        self.threads: List[Thread] = [
            Thread(thread_id=base_thread_id + i, start_pc=start_pc)
            for i in range(self.WARP_SIZE)
        ]

        # Warp state
        self.state = WarpState()

    @property
    def lane_masks(self) -> int:
        """Get the current active lane mask."""
        return self.state.active_mask

    @property
    def execution_masks(self) -> int:
        """Get the execution mask for current instruction."""
        return self.state.execution_mask

    def get_thread(self, lane_id: int) -> Thread:
        """
        Get a thread by lane ID.

        Args:
            lane_id: Lane ID (0-31)

        Returns:
            Thread at that lane
        """
        return self.threads[lane_id]

    def set_lane_active(self, lane_id: int, active: bool = True) -> None:
        """
        Set whether a lane is active.

        Args:
            lane_id: Lane ID (0-31)
            active: True to activate, False to deactivate
        """
        if active:
            self.state.active_mask |= (1 << lane_id)
            self.threads[lane_id].activate_lane()
        else:
            self.state.active_mask &= ~(1 << lane_id)
            self.threads[lane_id].deactivate_lane()

    def set_execution_mask(self, mask: int) -> None:
        """
        Set the execution mask directly.

        Args:
            mask: 32-bit mask where bit i = 1 means lane i executes
        """
        self.state.execution_mask = mask & self.FULL_MASK

    def update_execution_mask(self) -> None:
        """Update execution mask based on thread predicates and active lanes."""
        mask = 0
        for lane_id in range(self.WARP_SIZE):
            thread = self.threads[lane_id]
            if (self.state.active_mask & (1 << lane_id)) and thread.pred:
                mask |= (1 << lane_id)

        self.state.execution_mask = mask

    def any_active(self) -> bool:
        """Check if any lanes are active."""
        return self.state.active_mask != 0

    def any_executing(self) -> bool:
        """Check if any lanes will execute current instruction."""
        return self.state.execution_mask != 0

    def count_active(self) -> int:
        """Count number of active lanes."""
        return bin(self.state.active_mask).count('1')

    def count_executing(self) -> int:
        """Count number of lanes that will execute current instruction."""
        return bin(self.state.execution_mask).count('1')

    def broadcast(self, value: int, source_lane: int) -> int:
        """
        Broadcast a value from one lane to all lanes.

        Used by SHFL (shuffle) instructions.

        Args:
            value: Value to broadcast
            source_lane: Lane ID to broadcast from

        Returns:
            The broadcast value (same for all lanes)
        """
        return value  # Simplified - in real GPU, this moves data between lanes

    def diverge(self, then_mask: int, else_mask: int, then_pc: int, else_pc: int) -> None:
        """
        Handle warp divergence.

        When threads in a warp take different paths:
        - Push divergence info to stack
        - Execute one path first (threads taking other path are disabled)
        - Later reconverge

        Args:
            then_mask: Lanes taking the 'then' path
            else_mask: Lanes taking the 'else' path
            then_pc: PC for 'then' path
            else_pc: PC for 'else' path (other path)
        """
        # Simplified divergence handling
        # In real implementation, we'd manage a divergence stack
        # For now, just note the divergence
        pass

    def advance_pc(self, offset: int = 1) -> None:
        """
        Advance PC for all active threads.

        Args:
            offset: Number of instructions to advance
        """
        self.pc += offset
        for thread in self.threads:
            if thread.is_active():
                thread.advance_pc(offset)

    def branch(self, target_pc: int) -> None:
        """
        Branch to target PC for all active threads.

        Args:
            target_pc: Target program counter
        """
        self.pc = target_pc
        for thread in self.threads:
            if thread.is_active():
                thread.branch(target_pc)

    def read_lane_reg(self, lane_id: int, reg_idx: int) -> int:
        """Read a register from a specific lane."""
        return self.threads[lane_id].read_reg(reg_idx)

    def write_lane_reg(self, lane_id: int, reg_idx: int, value: int) -> None:
        """Write a value to a register in a specific lane."""
        self.threads[lane_id].write_reg(reg_idx, value)

    def read_lane_reg_f32(self, lane_id: int, reg_idx: int) -> float:
        """Read a float32 register from a specific lane."""
        return self.threads[lane_id].read_reg_f32(reg_idx)

    def write_lane_reg_f32(self, lane_id: int, reg_idx: int, value: float) -> None:
        """Write a float32 value to a register in a specific lane."""
        self.threads[lane_id].write_reg_f32(reg_idx, value)

    def get_active_lane_ids(self) -> List[int]:
        """Get list of active lane IDs."""
        return [i for i in range(self.WARP_SIZE) if self.state.active_mask & (1 << i)]

    def get_executing_lane_ids(self) -> List[int]:
        """Get list of lane IDs that will execute current instruction."""
        return [i for i in range(self.WARP_SIZE) if self.state.execution_mask & (1 << i)]

    def all_threads_done(self) -> bool:
        """Check if all threads have exited."""
        return all(t.is_done() for t in self.threads)

    def __repr__(self) -> str:
        """String representation of the warp."""
        active_lanes = self.count_active()
        return (f"Warp(id={self.warp_id}, "
                f"pc={self.pc:#x}, "
                f"active_lanes={active_lanes}/{self.WARP_SIZE})")


if __name__ == "__main__":
    # Test the warp
    w = Warp(warp_id=0, start_pc=0x1000)
    print(w)

    # Test lane operations
    print("\nInitial active lanes:", w.get_active_lane_ids())

    # Deactivate lane 5
    w.set_lane_active(5, False)
    print(f"After disabling lane 5: {w.get_active_lane_ids()}")

    # Register operations
    for lane in range(4):
        w.write_lane_reg(lane, 1, lane * 10)

    print("\nRegister R1 values:")
    for lane in range(4):
        print(f"  Lane {lane}: R1 = {w.read_lane_reg(lane, 1)}")

    # Test predicates
    w.get_thread(0).set_pred(True)
    w.get_thread(1).set_pred(False)
    w.get_thread(2).set_pred(True)
    w.get_thread(3).set_pred(False)

    w.update_execution_mask()
    print(f"\nExecution mask: {w.execution_masks:#010b}")
    print(f"Executing lanes: {w.get_executing_lane_ids()}")
