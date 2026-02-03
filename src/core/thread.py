"""
Thread Implementation for Hopper GPU

A thread represents a single CUDA thread executing on the GPU.
"""

from enum import Enum, auto
from typing import Optional
from .register import RegisterFile


class ThreadState(Enum):
    """Possible states of a GPU thread."""
    ACTIVE = auto()      # Thread is actively executing
    SUSPENDED = auto()   # Thread is waiting (e.g., at barrier)
    EXITED = auto()      # Thread has completed execution
    DIVERGED = auto()    # Thread has diverged from warp (inactive lane)


class Thread:
    """
    Represents a single GPU thread in the Hopper architecture.

    Each thread maintains:
    - Program Counter (PC): Points to current instruction
    - Register File: 255 general-purpose registers
    - State: Current execution state
    - Thread ID: Unique identifier within block
    - Active predicate: For predicated execution
    """

    def __init__(self, thread_id: int, start_pc: int = 0) -> None:
        """
        Initialize a new thread.

        Args:
            thread_id: Unique thread identifier
            start_pc: Initial program counter value
        """
        self.thread_id = thread_id
        self.pc = start_pc
        self.register_file = RegisterFile()
        self.state = ThreadState.ACTIVE
        self.active = True  # For predicated execution (lane active mask)
        self.pred = True    # Predicate register (for conditional execution)

        # Special registers (simplified model)
        self._lane_id = thread_id % 32  # Lane ID within warp (0-31)
        self._warp_id = thread_id // 32  # Warp ID

    @property
    def lane_id(self) -> int:
        """Get the lane ID (position within warp)."""
        return self._lane_id

    @property
    def warp_id(self) -> int:
        """Get the warp ID."""
        return self._warp_id

    def read_reg(self, reg_idx: int) -> int:
        """Read from a register."""
        return self.register_file.read(reg_idx)

    def write_reg(self, reg_idx: int, value: int) -> None:
        """Write to a register."""
        self.register_file.write(reg_idx, value)

    def read_reg_f32(self, reg_idx: int) -> float:
        """Read a register as float32."""
        return self.register_file.read_f32(reg_idx)

    def write_reg_f32(self, reg_idx: int, value: float) -> None:
        """Write a float32 value to a register."""
        self.register_file.write_f32(reg_idx, value)

    def set_pred(self, value: bool) -> None:
        """Set the predicate register."""
        self.pred = value

    def is_active_lane(self) -> bool:
        """Check if this thread's lane is active (for divergent warps)."""
        return self.active

    def activate_lane(self) -> None:
        """Activate this thread's lane."""
        self.active = True

    def deactivate_lane(self) -> None:
        """Deactivate this thread's lane (due to divergence)."""
        self.active = False

    def advance_pc(self, offset: int = 1) -> None:
        """
        Advance the program counter.

        Args:
            offset: Number of instructions to advance (default: 1)
        """
        self.pc += offset

    def branch(self, target_pc: int) -> None:
        """Branch to a target address."""
        self.pc = target_pc

    def set_state(self, state: ThreadState) -> None:
        """Set the thread state."""
        self.state = state

    def is_active(self) -> bool:
        """Check if thread is actively executing."""
        return self.state == ThreadState.ACTIVE

    def is_done(self) -> bool:
        """Check if thread has finished execution."""
        return self.state == ThreadState.EXITED

    def __repr__(self) -> str:
        """String representation of the thread."""
        return (f"Thread(id={self.thread_id}, "
                f"pc={self.pc}, state={self.state.name}, "
                f"active={self.active})")


if __name__ == "__main__":
    # Test the thread
    t = Thread(thread_id=42)
    print(t)

    # Test register operations
    t.write_reg(1, 0xCAFEBABE)
    print(f"R1 = {t.read_reg(1):#x}")

    # Test float operations
    t.write_reg_f32(2, 2.71828)
    print(f"R2 (float) = {t.read_reg_f32(2)}")

    # Test PC manipulation
    t.advance_pc(5)
    print(f"PC = {t.pc}")
    t.branch(0x100)
    print(f"PC after branch = {t.pc:#x}")

    # Test lane info
    print(f"Lane ID: {t.lane_id}, Warp ID: {t.warp_id}")
