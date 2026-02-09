"""
Main Simulator Interface for Hopper GPU

Provides the main interface for running Hopper GPU simulations.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .core.warp import Warp
from .core.memory import Memory, MemorySpace
from .core.thread import ThreadState
from .core.async_ops import AsyncQueue
from .core.mbarrier import MbarrierManager
from .executor.warp import WarpExecutor
from .executor.pipeline import ExecutionPipeline, PipelineConfig
from .isa.decoder import parse_program


@dataclass
class SimulatorConfig:
    """Configuration for Hopper simulator."""
    num_sms: int = 1  # Number of streaming multiprocessors
    warps_per_sm: int = 4  # Warps per SM
    threads_per_warp: int = 32  # Threads per warp (always 32 for NVIDIA)
    global_mem_size: int = 1024 * 1024 * 1024  # 1 GB
    shared_mem_size: int = 228 * 1024  # 228 KB per SM
    max_cycles: int = 10000  # Maximum execution cycles


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    success: bool
    cycles: int
    instructions_executed: int
    warp_stats: Dict[int, Dict[str, Any]]
    error: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return (f"SimulationResult(success=True, cycles={self.cycles}, "
                    f"instructions={self.instructions_executed})")
        else:
            return f"SimulationResult(success=False, error={self.error})"


class HopperSimulator:
    """
    Main simulator class for Hopper GPU.

    This is the primary interface for running GPU simulations.
    """

    def __init__(self, config: SimulatorConfig = None) -> None:
        """
        Initialize the Hopper simulator.

        Args:
            config: Simulator configuration
        """
        self.config = config or SimulatorConfig()

        # Create memory system (shared across all SMs)
        self.memory = Memory(
            global_size=self.config.global_mem_size,
            shared_size=self.config.shared_mem_size
        )

        # Create async operation queue (shared across all warps for TMA, WGMMA)
        self.async_queue = AsyncQueue(num_units=4)

        # Create mbarrier manager for synchronizing async operations
        self.mbarrier_manager = MbarrierManager()

        # Create warps
        self.warps: List[Warp] = []
        self.executors: List[WarpExecutor] = []

        for sm_id in range(self.config.num_sms):
            for warp_id in range(self.config.warps_per_sm):
                global_warp_id = sm_id * self.config.warps_per_sm + warp_id
                warp = Warp(warp_id=global_warp_id, start_pc=0)
                executor = WarpExecutor(warp, self.memory, self.async_queue, self.mbarrier_manager)

                self.warps.append(warp)
                self.executors.append(executor)

        # Execution pipeline
        self.pipeline = ExecutionPipeline(PipelineConfig(
            issue_width=1,
            max_warps=self.config.num_sms * self.config.warps_per_sm,
        ))

        # Loaded programs
        self.programs: Dict[int, List] = {}

    def load_program(self, program: List[str], warp_id: int = 0) -> None:
        """
        Load a program for a specific warp.

        Args:
            program: List of assembly instructions
            warp_id: Warp ID to load program for (0 by default)
        """
        instructions = parse_program(program)
        self.programs[warp_id] = instructions

    def load_program_from_file(self, filename: str, warp_id: int = 0) -> None:
        """
        Load a program from a file.

        Args:
            filename: Path to assembly file
            warp_id: Warp ID to load program for
        """
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        instructions = parse_program(lines)
        self.programs[warp_id] = instructions

    def launch_kernel(self,
                      program: List[str],
                      grid_dim: tuple,
                      block_dim: tuple) -> None:
        """
        Launch a kernel with CUDA-style grid/block configuration.

        This mimics the CUDA launch syntax:
            kernel<<<grid_dim, block_dim>>>(args...)

        The kernel is loaded on all warps needed to cover the grid.
        Special registers (%tid, %ctaid, etc.) are initialized for each thread.

        Args:
            program: List of assembly instructions (the kernel code)
            grid_dim: (grid_x, grid_y, grid_z) - number of blocks
            block_dim: (block_x, block_y, block_z) - threads per block

        Example:
            # Launch: my_kernel<<<(2, 1, 1), (32, 1, 1)>>>
            sim.launch_kernel(kernel_code, grid_dim=(2, 1, 1), block_dim=(32, 1, 1))
            # This creates 2 blocks with 32 threads each (64 total threads)
        """
        from .core.thread import SpecialRegister

        grid_x, grid_y, grid_z = grid_dim
        block_x, block_y, block_z = block_dim

        # Calculate total blocks and threads per block
        total_blocks = grid_x * grid_y * grid_z
        threads_per_block = block_x * block_y * block_z
        total_threads = total_blocks * threads_per_block

        # Calculate how many warps we need
        warps_per_block = (threads_per_block + 31) // 32  # Ceiling division
        total_warps = total_blocks * warps_per_block

        print(f"Kernel Launch Configuration:")
        print(f"  Grid: {grid_dim} ({total_blocks} blocks)")
        print(f"  Block: {block_dim} ({threads_per_block} threads per block)")
        print(f"  Total threads: {total_threads}")
        print(f"  Warps per block: {warps_per_block}")
        print(f"  Total warps: {total_warps}")

        # Check if we have enough warps in the simulator
        available_warps = len(self.warps)
        if total_warps > available_warps:
            print(f"  Warning: Need {total_warps} warps but only {available_warps} available")
            total_warps = available_warps

        # Parse the kernel program once
        instructions = parse_program(program)

        # Load kernel on all participating warps and initialize special registers
        for warp_idx in range(total_warps):
            # Calculate which block this warp belongs to
            block_idx = warp_idx // warps_per_block
            warp_in_block = warp_idx % warps_per_block

            # Calculate block coordinates (simplified: only x-dimension for now)
            block_id_x = block_idx % grid_x
            block_id_y = (block_idx // grid_x) % grid_y
            block_id_z = block_idx // (grid_x * grid_y)

            # Load program on this warp
            self.programs[warp_idx] = instructions

            # Initialize special registers for each thread in this warp
            for lane_id in range(32):
                # Calculate global thread ID
                thread_in_block = warp_in_block * 32 + lane_id

                # Skip if this lane doesn't correspond to a valid thread
                if thread_in_block >= threads_per_block:
                    continue

                # Thread coordinates within block
                thread_id_x = thread_in_block % block_x
                thread_id_y = (thread_in_block // block_x) % block_y
                thread_id_z = thread_in_block // (block_x * block_y)

                # Calculate 1D thread ID (common in CUDA)
                tid = thread_id_z * (block_x * block_y) + thread_id_y * block_x + thread_id_x

                # Initialize special registers for this thread
                thread = self.warps[warp_idx].get_thread(lane_id)
                thread.init_special_registers(
                    tid=tid,                         # Thread index within block
                    ctaid=block_idx,                 # CTA (block) index
                    ntid=threads_per_block,          # Threads per block
                    nctaid=total_blocks,             # Total blocks
                    laneid=lane_id,                  # Lane ID within warp
                    warpid=warp_in_block,            # Warp ID within CTA
                    nwarpid=warps_per_block,         # Warps per CTA
                    smid=0,                          # SM ID (simplified)
                    nsmid=1,                         # Number of SMs (simplified)
                    gridid=0                         # Grid ID (simplified)
                )

        print(f"  Loaded kernel on {total_warps} warps")
        print(f"  Initialized special registers for all threads")

    def set_memory(self, data: bytes, offset: int = 0) -> None:
        """
        Set initial memory contents.

        Args:
            data: Data to write
            offset: Offset in global memory
        """
        self.memory.write(MemorySpace.GLOBAL, offset, data)

    def run(self, max_cycles: Optional[int] = None) -> SimulationResult:
        """
        Run the simulation.

        Args:
            max_cycles: Maximum cycles to execute (overrides config)

        Returns:
            Simulation results
        """
        if not self.programs:
            return SimulationResult(
                success=False,
                cycles=0,
                instructions_executed=0,
                warp_stats={},
                error="No program loaded"
            )

        max_cycles = max_cycles or self.config.max_cycles

        try:
            # Execute using pipeline
            stats = self.pipeline.execute_warps(
                self.executors,
                self.programs,
                max_cycles=max_cycles,
                async_queue=self.async_queue
            )

            return SimulationResult(
                success=True,
                cycles=stats.total_cycles,
                instructions_executed=stats.total_instructions,
                warp_stats=stats.warp_stats
            )

        except Exception as e:
            return SimulationResult(
                success=False,
                cycles=0,
                instructions_executed=0,
                warp_stats={},
                error=str(e)
            )

    def read_register(self, warp_id: int, lane_id: int, reg_idx: int) -> int:
        """
        Read a register from a specific thread.

        Args:
            warp_id: Warp ID
            lane_id: Lane ID within warp (0-31)
            reg_idx: Register index

        Returns:
            Register value
        """
        warp = self.warps[warp_id]
        return warp.read_lane_reg(lane_id, reg_idx)

    def write_register(self, warp_id: int, lane_id: int, reg_idx: int, value: int) -> None:
        """
        Write to a register in a specific thread.

        Args:
            warp_id: Warp ID
            lane_id: Lane ID within warp (0-31)
            reg_idx: Register index
            value: Value to write
        """
        warp = self.warps[warp_id]
        warp.write_lane_reg(lane_id, reg_idx, value)

    def read_memory(self, address: int, size: int = 4) -> bytes:
        """
        Read from global memory.

        Args:
            address: Memory address
            size: Number of bytes to read

        Returns:
            Data from memory
        """
        return self.memory.read(MemorySpace.GLOBAL, address, size)

    def write_memory(self, address: int, data: bytes) -> None:
        """
        Write to global memory.

        Args:
            address: Memory address
            data: Data to write
        """
        self.memory.write(MemorySpace.GLOBAL, address, data)

    def reset(self) -> None:
        """Reset the simulator to initial state."""
        # Reset async queue and mbarrier manager FIRST (before recreating executors)
        from .core.async_ops import AsyncQueue
        from .core.mbarrier import MbarrierManager
        self.async_queue = AsyncQueue(num_units=4)
        self.mbarrier_manager = MbarrierManager()

        # Re-create warps and executors with new async queue
        self.warps.clear()
        self.executors.clear()

        for sm_id in range(self.config.num_sms):
            for warp_id in range(self.config.warps_per_sm):
                global_warp_id = sm_id * self.config.warps_per_sm + warp_id
                warp = Warp(warp_id=global_warp_id, start_pc=0)
                executor = WarpExecutor(warp, self.memory, self.async_queue, self.mbarrier_manager)

                self.warps.append(warp)
                self.executors.append(executor)

        # Clear programs
        self.programs.clear()

    def get_state(self) -> Dict[str, Any]:
        """
        Get current simulator state.

        Returns:
            Dictionary with current state
        """
        return {
            'config': {
                'num_sms': self.config.num_sms,
                'warps_per_sm': self.config.warps_per_sm,
                'threads_per_warp': self.config.threads_per_warp,
            },
            'warps': [
                {
                    'warp_id': w.warp_id,
                    'pc': w.pc,
                    'active_lanes': w.count_active(),
                }
                for w in self.warps
            ],
            'memory': str(self.memory),
        }

    def __repr__(self) -> str:
        return (f"HopperSimulator(sms={self.config.num_sms}, "
                f"warps={len(self.warps)})")


# Convenience function for quick simulations
def simulate(program: List[str],
             init_memory: Optional[Dict[int, int]] = None,
             config: SimulatorConfig = None) -> SimulationResult:
    """
    Run a quick simulation.

    Args:
        program: List of assembly instructions
        init_memory: Optional initial memory values {address: value}
        config: Simulator configuration

    Returns:
        Simulation results
    """
    sim = HopperSimulator(config)

    # Initialize memory if provided
    if init_memory:
        for addr, val in init_memory.items():
            sim.write_memory(addr, val.to_bytes(4, byteorder='little'))

    # Load and run
    sim.load_program(program)
    result = sim.run()

    return result


if __name__ == "__main__":
    # Test the simulator
    program = [
        "MOV R1, 10",
        "MOV R2, 20",
        "IADD R3, R1, R2",
        "IADD R4, R3, 5",
        "EXIT",
    ]

    result = simulate(program)

    print(f"Result: {result}")
    print(f"Success: {result.success}")
    print(f"Cycles: {result.cycles}")
    print(f"Instructions: {result.instructions_executed}")
    print(f"Warp stats: {result.warp_stats}")
