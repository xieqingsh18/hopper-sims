"""
Warp Executor for Hopper GPU

Executes instructions for a warp of 32 threads in SIMT fashion.
"""

from typing import List, Optional, Dict, Any
import struct
from ctypes import c_int32, c_uint32
from ..core.warp import Warp
from ..core.memory import Memory, MemorySpace, ProxyType
from ..core.thread import ThreadState
from ..core.async_ops import AsyncQueue, AsyncOperation, AsyncOpType
from ..core.mbarrier import MbarrierManager
from ..isa.decoder import Instruction
from ..isa.instructions import (
    Opcode, is_tensor_instruction, is_load_instruction, is_store_instruction,
    is_branch_instruction, is_atomic_instruction, is_barrier_instruction,
    is_call_instruction, is_tma_instruction, is_wgmma_instruction, OperandType
)
from ..isa.tensor import TensorCoreInstruction, HMMAOp


class ExecutionException(Exception):
    """Exception raised during instruction execution."""
    pass


class WarpExecutor:
    """
    Executes instructions for a warp.

    Handles:
    - SIMT execution (single instruction, multiple threads)
    - Predicate evaluation
    - Arithmetic and logical operations
    - Memory operations (load/store)
    - Control flow (branches)
    - Tensor Core operations
    """

    def __init__(self, warp: Warp, memory: Memory,
                 async_queue: Optional[AsyncQueue] = None,
                 mbarrier_manager: Optional[MbarrierManager] = None) -> None:
        """
        Initialize warp executor.

        Args:
            warp: The warp to execute
            memory: GPU memory subsystem
            async_queue: Optional async operation queue for TMA/WGMMA
            mbarrier_manager: Optional mbarrier manager for synchronization
        """
        self.warp = warp
        self.memory = memory

        # Async operation queue (shared across all executors)
        self.async_queue = async_queue or AsyncQueue(num_units=4)

        # Mbarrier manager for synchronizing async operations
        self.mbarrier_manager = mbarrier_manager or MbarrierManager()

        # Track active mbarrier for async operations (set by MBARRIER_EXPECT_TX)
        self.active_mbarrier_addr: Optional[int] = None

        # Statistics
        self.instructions_executed = 0
        self.branches_taken = 0

        # Tensor Core instruction handler
        self.tensor_handler = TensorCoreInstruction(HMMAOp.FP8_FP32)

    def execute(self, instruction: Instruction) -> bool:
        """
        Execute an instruction on the warp.

        Args:
            instruction: Instruction to execute

        Returns:
            True if execution should continue, False if warp should exit
        """
        # Check for per-lane predicate (PTX @P0, @!P0 syntax)
        if instruction.predicate_reg is not None:
            pred_reg = instruction.predicate_reg
            pred_condition = instruction.predicate_condition  # True for @P, False for @!P

            # Update execution mask based on per-lane predicate values
            for lane_id in self.warp.get_executing_lane_ids():
                # Get predicate value for this lane
                lane_pred = self.warp.get_thread(lane_id).pred
                # Lane executes if its predicate matches the condition
                should_execute_lane = lane_pred == pred_condition
                # Update lane's active status
                if not should_execute_lane:
                    self.warp.deactivate_lane(lane_id)

        # Update execution mask based on predicates and active lanes
        self.warp.update_execution_mask()

        # If no lanes should execute, skip but still advance PC
        if not self.warp.any_executing():
            self.warp.advance_pc(4)
            self.instructions_executed += 1
            return True

        # Execute based on opcode
        opcode = instruction.opcode

        try:
            # ==================== Integer Arithmetic ====================
            if opcode == Opcode.IADD:
                self._exec_iadd(instruction)
            elif opcode == Opcode.ISUB:
                self._exec_isub(instruction)
            elif opcode == Opcode.IMUL:
                self._exec_imul(instruction)
            elif opcode == Opcode.IMAD:
                self._exec_imad(instruction)
            elif opcode in {Opcode.IMIN, Opcode.IMAX}:
                self._exec_iminmax(instruction)
            elif opcode == Opcode.IABS:
                self._exec_iabs(instruction)
            elif opcode == Opcode.INEG:
                self._exec_ineg(instruction)
            elif opcode == Opcode.POPC:
                self._exec_popc(instruction)
            elif opcode == Opcode.CLZ:
                self._exec_clz(instruction)
            elif opcode == Opcode.BFE:
                self._exec_bfe(instruction)

            # ==================== Floating Point ====================
            elif opcode == Opcode.FADD:
                self._exec_fadd(instruction)
            elif opcode == Opcode.FSUB:
                self._exec_fsub(instruction)
            elif opcode == Opcode.FMUL:
                self._exec_fmul(instruction)
            elif opcode == Opcode.FFMA:
                self._exec_ffma(instruction)
            elif opcode in {Opcode.FMIN, Opcode.FMAX}:
                self._exec_fminmax(instruction)
            elif opcode == Opcode.FABS:
                self._exec_fabs(instruction)
            elif opcode == Opcode.FNEG:
                self._exec_fneg(instruction)

            # ==================== Logic & Shift ====================
            elif opcode == Opcode.AND:
                self._exec_and(instruction)
            elif opcode == Opcode.OR:
                self._exec_or(instruction)
            elif opcode == Opcode.XOR:
                self._exec_xor(instruction)
            elif opcode == Opcode.NOT:
                self._exec_not(instruction)
            elif opcode in {Opcode.SHL, Opcode.SHR}:
                self._exec_shift(instruction)
            elif opcode == Opcode.LOP:
                self._exec_lop(instruction)

            # ==================== Data Movement ====================
            elif opcode == Opcode.MOV:
                self._exec_mov(instruction)
            elif opcode == Opcode.CVT:
                self._exec_cvt(instruction)

            # ==================== Memory Operations ====================
            elif is_load_instruction(opcode):
                self._exec_load(instruction)
            elif is_store_instruction(opcode):
                self._exec_store(instruction)

            # ==================== Control Flow ====================
            elif is_branch_instruction(opcode):
                self._exec_branch(instruction)
            elif is_call_instruction(opcode):
                self._exec_call(instruction)
            elif opcode == Opcode.RET:
                self._exec_ret(instruction)
            elif opcode == Opcode.EXIT:
                return False  # Signal to stop execution

            # ==================== Predicates ====================
            elif opcode in {Opcode.PSETP, Opcode.SETP}:
                self._exec_setp(instruction)
            elif opcode == Opcode.SELP:
                self._exec_selp(instruction)

            # ==================== Warp Specialization (Hopper) ====================
            # Check these BEFORE generic barrier instruction check
            elif is_tma_instruction(opcode):
                self._exec_tma(instruction)
            elif is_wgmma_instruction(opcode):
                self._exec_wgmma(instruction)
            elif opcode in {Opcode.MBARRIER_INIT, Opcode.MBARRIER_INIT_DOT,
                           Opcode.MBARRIER_INVAL, Opcode.MBARRIER_INVAL_DOT,
                           Opcode.MBARRIER_ARRIVE, Opcode.MBARRIER_ARRIVE_DOT,
                           Opcode.MBARRIER_TEST_WAIT, Opcode.MBARRIER_TEST_WAIT_DOT,
                           Opcode.MBARRIER_EXPECT_TX, Opcode.MBARRIER_EXPECT_TX_DOT,
                           Opcode.MBARRIER_COMPLETE_TX, Opcode.MBARRIER_COMPLETE_TX_DOT}:
                self._exec_mbarrier(instruction)

            # ==================== Warp Level ====================
            elif opcode in {Opcode.FENCE_SC, Opcode.FENCE_ACQ_REL}:
                self._exec_fence(instruction)
            elif is_barrier_instruction(opcode):
                self._exec_barrier(instruction)
            elif opcode == Opcode.VOTE:
                self._exec_vote(instruction)
            elif opcode == Opcode.ACTIVEMASK:
                self._exec_activemask(instruction)
            elif opcode == Opcode.ELECT:
                self._exec_elect(instruction)
            elif opcode == Opcode.SHFL:
                self._exec_shfl(instruction)
            elif opcode == Opcode.REDUX:
                self._exec_redux(instruction)

            # ==================== Atomic Operations ====================
            elif is_atomic_instruction(opcode):
                self._exec_atomic(instruction)

            # ==================== Tensor Core ====================
            elif is_tensor_instruction(opcode):
                self._exec_tensor(instruction)

            # ==================== Existing Fallback ====================
            elif opcode == Opcode.FSETP:
                self._exec_fsetp(instruction)
            else:
                raise ExecutionException(f"Unsupported opcode: {opcode}")

        except ExecutionException as e:
            print(f"Execution error at PC={instruction.pc:#x}: {e}")
            # Continue execution despite error
            self.warp.advance_pc(4)
            # Reactivate all lanes for next instruction
            self.warp.reactivate_all_lanes()
            return True

        # Advance PC (unless instruction modified it or warp is stalled)
        # Check if warp is stalled after instruction execution (e.g., mbarrier.test_wait)
        if not self.warp.is_stalled():
            if opcode not in {Opcode.BRA, Opcode.BRX, Opcode.CALL, Opcode.CAL, Opcode.RET, Opcode.EXIT}:
                self.warp.advance_pc(4)

        # Reactivate all lanes for next instruction (predicated execution is per-instruction)
        # Do this AFTER advancing PC so next instruction starts with all lanes active
        self.warp.reactivate_all_lanes()

        self.instructions_executed += 1
        return True

    def _evaluate_predicate(self, instr: Instruction) -> bool:
        """Evaluate predicate register."""
        if instr.predicate_reg is None:
            return True

        # Check first active lane's predicate
        for lane_id in self.warp.get_executing_lane_ids():
            return self.warp.get_thread(lane_id).pred

        return True

    def _exec_add(self, instr: Instruction) -> None:
        """Execute IADD/FADD: Rd = Ra + Rb (for each active lane)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)

            if instr.opcode == Opcode.FADD:
                # Floating point addition
                import struct
                fval1 = struct.unpack('>f', struct.pack('>I', val1))[0]
                fval2 = struct.unpack('>f', struct.pack('>I', val2))[0]
                result = struct.unpack('>I', struct.pack('>f', fval1 + fval2))[0]
            else:
                # Integer addition
                result = (val1 + val2) & 0xFFFFFFFF

            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_ffma(self, instr: Instruction) -> None:
        """Execute FFMA: Rd = Ra * Rb + Rc (fused multiply-add)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value
        src3 = instr.operands[3].value

        for lane_id in self.warp.get_executing_lane_ids():
            import struct
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            val3 = self.warp.read_lane_reg(lane_id, src3)

            fval1 = struct.unpack('>f', struct.pack('>I', val1))[0]
            fval2 = struct.unpack('>f', struct.pack('>I', val2))[0]
            fval3 = struct.unpack('>f', struct.pack('>I', val3))[0]

            result = fval1 * fval2 + fval3
            packed = struct.unpack('>I', struct.pack('>f', result))[0]

            self.warp.write_lane_reg(lane_id, dst, packed)

    def _exec_imad(self, instr: Instruction) -> None:
        """Execute IMAD: Rd = Ra * Rb + Rc (integer multiply-add)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value
        src3 = instr.operands[3].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            val3 = self.warp.read_lane_reg(lane_id, src3)

            # 24-bit multiply (truncate to 24 bits before multiply)
            val1_24 = val1 & 0xFFFFFF
            val2_24 = val2 & 0xFFFFFF

            # Sign extend for signed multiplication
            if val1_24 & 0x800000:
                val1_24 |= 0xFF000000
            if val2_24 & 0x800000:
                val2_24 |= 0xFF000000

            # Compute (signed)
            from ctypes import c_int32
            product = c_int32(val1_24).value * c_int32(val2_24).value
            result = (product + val3) & 0xFFFFFFFF

            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_mov(self, instr: Instruction) -> None:
        """Execute MOV: Rd = src"""
        dst = instr.operands[0].value
        src = instr.operands[1]

        from ..isa.instructions import OperandType
        from ..core.thread import SpecialRegister

        if src.type == OperandType.REGISTER:
            src_val = src.value
            for lane_id in self.warp.get_executing_lane_ids():
                val = self.warp.read_lane_reg(lane_id, src_val)
                self.warp.write_lane_reg(lane_id, dst, val)
        elif src.type == OperandType.IMMEDIATE:
            for lane_id in self.warp.get_executing_lane_ids():
                self.warp.write_lane_reg(lane_id, dst, src.value)
        elif src.type == OperandType.SPECIAL_REGISTER:
            # Read from special register (%tid, %ctaid, etc.)
            special_reg = src.value  # This is a SpecialRegister enum
            for lane_id in self.warp.get_executing_lane_ids():
                thread = self.warp.get_thread(lane_id)
                val = thread.read_special_reg(special_reg)
                self.warp.write_lane_reg(lane_id, dst, val)

    def _exec_ldg(self, instr: Instruction) -> None:
        """Execute LDG: Load from global memory"""
        dst = instr.operands[0].value
        addr_str = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            addr = self._compute_address(addr_str, lane_id)
            value = self.memory.read_u32(MemorySpace.GLOBAL, addr)
            self.warp.write_lane_reg(lane_id, dst, value)

    def _exec_lds(self, instr: Instruction) -> None:
        """Execute LDS: Load from shared memory"""
        dst = instr.operands[0].value
        addr_str = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            addr = self._compute_address(addr_str, lane_id)
            value = self.memory.read_u32(MemorySpace.SHARED, addr)
            self.warp.write_lane_reg(lane_id, dst, value)

    def _exec_stg(self, instr: Instruction) -> None:
        """Execute STG: Store to global memory"""
        addr_str = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            addr = self._compute_address(addr_str, lane_id)
            value = self.warp.read_lane_reg(lane_id, src)
            self.memory.write_u32(MemorySpace.GLOBAL, addr, value)

    def _exec_sts(self, instr: Instruction) -> None:
        """Execute STS: Store to shared memory"""
        addr_str = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            addr = self._compute_address(addr_str, lane_id)
            value = self.warp.read_lane_reg(lane_id, src)
            self.memory.write_u32(MemorySpace.SHARED, addr, value)

    def _exec_bra(self, instr: Instruction) -> None:
        """Execute BRA: Branch"""
        target = instr.operands[0].value
        self.warp.branch(target)
        self.branches_taken += 1

    def _exec_hmma(self, instr: Instruction) -> None:
        """Execute HMMA: Tensor Core matrix multiply-accumulate"""
        # Parse operand lists for matrix fragments
        d_regs = [instr.operands[0].value]
        a_regs = [instr.operands[1].value]
        b_regs = [instr.operands[2].value]
        c_regs = [instr.operands[3].value]

        # Get register files for active threads
        reg_files = []
        for lane_id in self.warp.get_executing_lane_ids():
            reg_files.append(self.warp.get_thread(lane_id).register_file)

        # Execute Tensor Core operation
        self.tensor_handler.execute(d_regs, a_regs, b_regs, c_regs, reg_files)

    def _exec_fsetp(self, instr: Instruction) -> None:
        """Execute FSETP: Floating point comparison, set predicate"""
        # Simplified: compare two registers and set predicate
        pred_dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            import struct
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)

            fval1 = struct.unpack('>f', struct.pack('>I', val1))[0]
            fval2 = struct.unpack('>f', struct.pack('>I', val2))[0]

            result = fval1 < fval2  # Simplified: just less-than comparison
            self.warp.get_thread(lane_id).set_pred(result)

    def _exec_psetp(self, instr: Instruction) -> None:
        """Execute PSETP: Set predicate from predicate(s)"""
        # Simplified: copy or combine predicates
        dst = instr.operands[0].value
        src1 = instr.operands[1].value if len(instr.operands) > 1 else 0

        for lane_id in self.warp.get_executing_lane_ids():
            # Just copy predicate for now
            pred = self.warp.get_thread(lane_id).pred
            self.warp.get_thread(lane_id).set_pred(pred)

    def _exec_lop(self, instr: Instruction) -> None:
        """Execute LOP: Logical operation (AND, OR, XOR)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)

            # Default to AND (simplified)
            result = val1 & val2
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_bar(self, instr: Instruction) -> None:
        """Execute BAR: Barrier synchronization"""
        # Simplified: barriers are no-ops in single-warp simulation
        pass

    def _exec_barrier(self, instr: Instruction) -> None:
        """
        Execute barrier instructions (barrier.sync, barrier.cluster, membar).

        barrier.sync - Synchronize all threads in CTA
        barrier.cluster - Cluster barrier (arrive/wait pattern)
        membar - Memory barrier
        fence.proxy - Proxy fence for sync/async memory ordering
        wgmma.fence - Warpgroup fence for async MMA operations

        PTX specification:
        - barrier.sync provides full memory fence for CTA scope
        - All pending memory operations become visible
        - Threads wait until all threads arrive
        """
        if instr.opcode == Opcode.MEMBAR:
            # MEMBAR instruction - handle in _exec_membar
            self._exec_membar(instr)

        elif instr.opcode == Opcode.BARRIER or instr.opcode == Opcode.BARRIER_CTA:
            # barrier.sync or barrier.cta
            # Format: BARRIER or BARRIER.CTA

            # For simplicity in current simulation, assume:
            # - Single warp = all threads arrive simultaneously
            # - Just need to flush pending memory operations

            # Get thread info for barrier manager
            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            thread = self.warp.threads[lane_id]
            thread_id = thread.thread_id
            warp_id = self.warp.warp_id
            # Compute CTA ID (simplified: treat each warp as separate CTA for now)
            cta_id = warp_id

            # Call barrier sync - returns True if complete
            complete = self.memory.barrier_manager.barrier_sync(
                barrier_id=0,  # Single barrier ID
                num_threads=self.warp.WARP_SIZE,
                thread_id=thread_id,
                warp_id=warp_id,
                cta_id=cta_id
            )

            if complete:
                # All threads arrived, operations are now visible
                pass

        elif instr.opcode == Opcode.BARRIER_CLUSTER:
            # barrier.cluster - arrive/wait pattern
            # Format: BARRIER.CLUSTER
            # For simplified simulation, treat same as barrier.sync
            self._exec_barrier(instr)

        elif instr.opcode in {Opcode.BAR, Opcode.BAR_WARP}:
            # Old PTX BAR instruction - simplified
            pass

        # ==================== Proxy Fence Instructions ====================
        elif instr.opcode in {Opcode.FENCE_PROXY, Opcode.FENCE_PROXY_DOT}:
            # fence.proxy - Generic proxy fence
            # Establishes memory ordering between different memory proxies
            # Ensures ordering between sync and async operations
            from ..core.memory import Scope, MemoryOrder
            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            thread = self.warp.threads[lane_id]
            thread_id = thread.thread_id
            warp_id = self.warp.warp_id
            cta_id = warp_id

            # Flush all pending async operations
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.ACQ_REL,
                thread_id=thread_id,
                warp_id=warp_id,
                cta_id=cta_id
            )

        elif instr.opcode in {Opcode.FENCE_PROXY_ASYNC, Opcode.FENCE_PROXY_ASYNC_DOT}:
            # fence.proxy.async - Async proxy fence
            # Synchronizes between generic proxy and async proxy
            # Ensures async TMA/WGMMA operations are visible
            from ..core.memory import Scope, MemoryOrder
            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            thread = self.warp.threads[lane_id]
            thread_id = thread.thread_id
            warp_id = self.warp.warp_id
            cta_id = warp_id

            # Flush async operations and make them visible
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.SC,  # Strong ordering for async
                thread_id=thread_id,
                warp_id=warp_id,
                cta_id=cta_id
            )

        elif instr.opcode in {Opcode.FENCE_PROXY_TENSORMAP, Opcode.FENCE_PROXY_TENSORMAP_DOT}:
            # fence.proxy.tensormap - Tensormap proxy fence
            # Synchronizes tensormap proxy operations
            # Used with WGMMA tensor operations
            from ..core.memory import Scope, MemoryOrder
            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            thread = self.warp.threads[lane_id]
            thread_id = thread.thread_id
            warp_id = self.warp.warp_id
            cta_id = warp_id

            # Flush tensor operations
            self.memory.barrier_manager.fence(
                scope=Scope.GPU,  # Tensor ops typically use GPU scope
                order=MemoryOrder.ACQ_REL,
                thread_id=thread_id,
                warp_id=warp_id,
                cta_id=cta_id
            )

        # ==================== Warpgroup Fence Instructions ====================
        elif instr.opcode in {Opcode.WGMMA_FENCE, Opcode.WGMMA_FENCE_DOT}:
            # wgmma.fence - Warpgroup fence
            # Ensures ordering of async warpgroup MMA operations
            # Prevents subsequent WGMMA from starting until previous complete
            from ..core.memory import Scope, MemoryOrder
            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            thread = self.warp.threads[lane_id]
            thread_id = thread.thread_id
            warp_id = self.warp.warp_id
            cta_id = warp_id

            # Flush warpgroup operations
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.SC,
                thread_id=thread_id,
                warp_id=warp_id,
                cta_id=cta_id
            )

        elif instr.opcode in {Opcode.WGMMA_COMMIT_GROUP, Opcode.WGMMA_COMMIT_GROUP_DOT}:
            # wgmma.commit_group - Commit warpgroup async group
            # Marks completion of a group of async WGMMA operations
            # This allows dependent operations to proceed
            from ..core.memory import Scope, MemoryOrder
            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)

            # Commit all pending WGMMA operations in current group
            # For simulation: mark async queue operations as complete
            if hasattr(self, 'wgmma_group_id'):
                self.wgmma_group_id += 1
            else:
                self.wgmma_group_id = 1

            # Flush operations to make them visible
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.RELEASE,
                thread_id=0,
                warp_id=self.warp.warp_id,
                cta_id=self.warp.warp_id
            )

        elif instr.opcode in {Opcode.WGMMA_WAIT_GROUP, Opcode.WGMMA_WAIT_GROUP_DOT}:
            # wgmma.wait_group - Wait for warpgroup async group
            # Stalls until specified group of WGMMA operations complete
            # Format: wgmma.wait_group group_id
            from ..core.memory import Scope, MemoryOrder
            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            thread = self.warp.threads[lane_id]
            thread_id = thread.thread_id
            warp_id = self.warp.warp_id
            cta_id = warp_id

            # Get group ID to wait for (default: 0 = current group)
            wait_group = instr.operands[0].value if len(instr.operands) > 0 else 0

            # For simulation: check if group is complete
            # In real hardware, this would stall the warp
            # For now, we'll just acquire the fence
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.ACQUIRE,
                thread_id=thread_id,
                warp_id=warp_id,
                cta_id=cta_id
            )

    def _exec_fence(self, instr: Instruction) -> None:
        """
        Execute fence instructions for memory ordering.

        FENCE.SC - Sequential consistency fence
        FENCE.ACQ_REL - Acquire-release fence

        These ensure memory ordering according to PTX specification.
        """
        # Get thread info for barrier manager
        lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
        thread = self.warp.threads[lane_id]
        thread_id = thread.thread_id
        warp_id = self.warp.warp_id
        cta_id = warp_id  # Simplified

        from ..core.memory import MemoryOrder, Scope

        if instr.opcode == Opcode.FENCE_SC:
            # Sequential consistency fence - strongest ordering
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.SC,
                thread_id=thread_id,
                warp_id=warp_id,
                cta_id=cta_id
            )

        elif instr.opcode == Opcode.FENCE_ACQ_REL:
            # Acquire-release fence
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.ACQ_REL,
                thread_id=thread_id,
                warp_id=warp_id,
                cta_id=cta_id
            )

    def _exec_membar(self, instr: Instruction) -> None:
        """
        Execute MEMBAR instruction (global memory barrier).

        MEMBAR - Memory barrier for global memory visibility
        MEMBAR.CTA - CTA scope
        MEMBAR.GL - GPU scope
        MEMBAR.SYS - System scope
        """
        # Get thread info
        lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
        thread = self.warp.threads[lane_id]
        thread_id = thread.thread_id
        warp_id = self.warp.warp_id
        cta_id = warp_id  # Simplified

        from ..core.memory import Scope

        # Default to CTA scope
        scope = Scope.CTA

        # Check if scope is specified in instruction
        if len(instr.operands) > 0:
            scope_val = instr.operands[0].value
            if isinstance(scope_val, str):
                if "CTA" in scope_val.upper():
                    scope = Scope.CTA
                elif "GPU" in scope_val.upper() or "GL" in scope_val.upper():
                    scope = Scope.GPU
                elif "SYS" in scope_val.upper():
                    scope = Scope.SYSTEM
            elif isinstance(scope_val, int):
                # Numeric scope encoding
                scope = Scope(scope_val)

        self.memory.barrier_manager.membar(scope=scope)

    # ==================== New Integer Arithmetic Functions ====================

    def _exec_iadd(self, instr: Instruction) -> None:
        """Execute IADD: Rd = Ra + Rb"""
        dst = instr.operands[0].value

        for lane_id in self.warp.get_executing_lane_ids():
            # Get src1 value (register or immediate)
            if instr.operands[1].type == OperandType.IMMEDIATE:
                val1 = instr.operands[1].value
            else:
                val1 = self.warp.read_lane_reg(lane_id, instr.operands[1].value)

            # Get src2 value (register or immediate)
            if instr.operands[2].type == OperandType.IMMEDIATE:
                val2 = instr.operands[2].value
            else:
                val2 = self.warp.read_lane_reg(lane_id, instr.operands[2].value)

            result = (val1 + val2) & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_isub(self, instr: Instruction) -> None:
        """Execute ISUB: Rd = Ra - Rb"""
        dst = instr.operands[0].value

        for lane_id in self.warp.get_executing_lane_ids():
            # Get src1 value (register or immediate)
            if instr.operands[1].type == OperandType.IMMEDIATE:
                val1 = instr.operands[1].value
            else:
                val1 = self.warp.read_lane_reg(lane_id, instr.operands[1].value)

            # Get src2 value (register or immediate)
            if instr.operands[2].type == OperandType.IMMEDIATE:
                val2 = instr.operands[2].value
            else:
                val2 = self.warp.read_lane_reg(lane_id, instr.operands[2].value)

            result = (val1 - val2) & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_imul(self, instr: Instruction) -> None:
        """Execute IMUL: Rd = Ra * Rb"""
        dst = instr.operands[0].value

        for lane_id in self.warp.get_executing_lane_ids():
            # Get src1 value (register or immediate)
            if instr.operands[1].type == OperandType.IMMEDIATE:
                val1 = instr.operands[1].value
            else:
                val1 = self.warp.read_lane_reg(lane_id, instr.operands[1].value)

            # Get src2 value (register or immediate)
            if instr.operands[2].type == OperandType.IMMEDIATE:
                val2 = instr.operands[2].value
            else:
                val2 = self.warp.read_lane_reg(lane_id, instr.operands[2].value)

            # Sign-extend and multiply
            prod = c_int32(val1).value * c_int32(val2).value
            result = prod & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_iminmax(self, instr: Instruction) -> None:
        """Execute IMIN/IMAX: Rd = min(Ra, Rb) or max(Ra, Rb)"""
        dst = instr.operands[0].value
        is_max = (instr.opcode == Opcode.IMAX)

        for lane_id in self.warp.get_executing_lane_ids():
            # Get src1 value (register or immediate)
            if instr.operands[1].type == OperandType.IMMEDIATE:
                val1 = instr.operands[1].value
            else:
                val1 = c_int32(self.warp.read_lane_reg(lane_id, instr.operands[1].value)).value

            # Get src2 value (register or immediate)
            if instr.operands[2].type == OperandType.IMMEDIATE:
                val2 = instr.operands[2].value
            else:
                val2 = c_int32(self.warp.read_lane_reg(lane_id, instr.operands[2].value)).value

            result = max(val1, val2) if is_max else min(val1, val2)
            self.warp.write_lane_reg(lane_id, dst, result & 0xFFFFFFFF)

    def _exec_iabs(self, instr: Instruction) -> None:
        """Execute IABS: Rd = |Ra|"""
        dst = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            val = c_int32(self.warp.read_lane_reg(lane_id, src)).value
            result = abs(val) & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_ineg(self, instr: Instruction) -> None:
        """Execute INEG: Rd = -Ra"""
        dst = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            val = c_int32(self.warp.read_lane_reg(lane_id, src)).value
            result = (-val) & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_popc(self, instr: Instruction) -> None:
        """Execute POPC: Rd = population_count(Ra)"""
        dst = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            val = self.warp.read_lane_reg(lane_id, src)
            result = bin(val).count('1')
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_clz(self, instr: Instruction) -> None:
        """Execute CLZ: Rd = count_leading_zeros(Ra)"""
        dst = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            val = self.warp.read_lane_reg(lane_id, src)
            if val == 0:
                result = 32
            else:
                # Count leading zeros: 32 - bit_length
                result = 32 - val.bit_length()
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_bfe(self, instr: Instruction) -> None:
        """Execute BFE: Bit field extract"""
        dst = instr.operands[0].value
        src = instr.operands[1].value
        offset = instr.operands[2].value
        count = instr.operands[3].value if len(instr.operands) > 3 else 32

        for lane_id in self.warp.get_executing_lane_ids():
            val = self.warp.read_lane_reg(lane_id, src)
            # Extract bits [offset:offset+count]
            mask = (1 << count) - 1
            result = (val >> offset) & mask
            self.warp.write_lane_reg(lane_id, dst, result)

    # ==================== New Floating Point Functions ====================

    def _exec_fadd(self, instr: Instruction) -> None:
        """Execute FADD: Rd = Ra + Rb (float)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            fval1 = struct.unpack('>f', struct.pack('>I', val1))[0]
            fval2 = struct.unpack('>f', struct.pack('>I', val2))[0]
            result = struct.unpack('>I', struct.pack('>f', fval1 + fval2))[0]
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_fsub(self, instr: Instruction) -> None:
        """Execute FSUB: Rd = Ra - Rb (float)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            fval1 = struct.unpack('>f', struct.pack('>I', val1))[0]
            fval2 = struct.unpack('>f', struct.pack('>I', val2))[0]
            result = struct.unpack('>I', struct.pack('>f', fval1 - fval2))[0]
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_fmul(self, instr: Instruction) -> None:
        """Execute FMUL: Rd = Ra * Rb (float)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            fval1 = struct.unpack('>f', struct.pack('>I', val1))[0]
            fval2 = struct.unpack('>f', struct.pack('>I', val2))[0]
            result = struct.unpack('>I', struct.pack('>f', fval1 * fval2))[0]
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_fminmax(self, instr: Instruction) -> None:
        """Execute FMIN/FMAX: Rd = min(Ra, Rb) or max(Ra, Rb) (float)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value
        is_max = (instr.opcode == Opcode.FMAX)

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            fval1 = struct.unpack('>f', struct.pack('>I', val1))[0]
            fval2 = struct.unpack('>f', struct.pack('>I', val2))[0]
            result = max(fval1, fval2) if is_max else min(fval1, fval2)
            packed = struct.unpack('>I', struct.pack('>f', result))[0]
            self.warp.write_lane_reg(lane_id, dst, packed)

    def _exec_fabs(self, instr: Instruction) -> None:
        """Execute FABS: Rd = |Ra| (float)"""
        dst = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            val = self.warp.read_lane_reg(lane_id, src)
            # Clear sign bit
            result = val & 0x7FFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_fneg(self, instr: Instruction) -> None:
        """Execute FNEG: Rd = -Ra (float)"""
        dst = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            val = self.warp.read_lane_reg(lane_id, src)
            # Flip sign bit
            result = val ^ 0x80000000
            self.warp.write_lane_reg(lane_id, dst, result)

    # ==================== New Logic & Shift Functions ====================

    def _exec_and(self, instr: Instruction) -> None:
        """Execute AND: Rd = Ra & Rb"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            result = val1 & val2
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_or(self, instr: Instruction) -> None:
        """Execute OR: Rd = Ra | Rb"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            result = val1 | val2
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_xor(self, instr: Instruction) -> None:
        """Execute XOR: Rd = Ra ^ Rb"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            result = val1 ^ val2
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_not(self, instr: Instruction) -> None:
        """Execute NOT: Rd = ~Ra"""
        dst = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            val = self.warp.read_lane_reg(lane_id, src)
            result = (~val) & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_shift(self, instr: Instruction) -> None:
        """Execute SHL/SHR: Rd = Ra << Rb or Rd = Ra >> Rb"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2]
        is_left = (instr.opcode == Opcode.SHL)

        # Get shift amount (can be register or immediate)
        if src2.type == OperandType.IMMEDIATE:
            shift_amt = src2.value & 0x1F  # Mask to 5 bits
            use_reg = False
        else:
            shift_amt = src2.value
            use_reg = True

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            if use_reg:
                val2 = self.warp.read_lane_reg(lane_id, shift_amt) & 0x1F
            else:
                val2 = shift_amt
            result = (val1 << val2) if is_left else (val1 >> val2)
            result &= 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    # ==================== New Data Movement Functions ====================

    def _exec_cvt(self, instr: Instruction) -> None:
        """Execute CVT: Convert data type (simplified)"""
        dst = instr.operands[0].value
        src = instr.operands[1].value

        for lane_id in self.warp.get_executing_lane_ids():
            # Simplified: just move the value
            # Real implementation would handle type conversions
            val = self.warp.read_lane_reg(lane_id, src)
            self.warp.write_lane_reg(lane_id, dst, val)

    # ==================== New Memory Functions ====================

    def _exec_load(self, instr: Instruction) -> None:
        """Generic load handler"""
        dst = instr.operands[0].value
        addr_str = instr.operands[1].value

        # Determine memory space
        if instr.opcode in {Opcode.LDG, Opcode.LD}:
            space = MemorySpace.GLOBAL
        elif instr.opcode == Opcode.LDS:
            space = MemorySpace.SHARED
        elif instr.opcode == Opcode.LDC:
            space = MemorySpace.CONSTANT
        elif instr.opcode == Opcode.LDL:
            space = MemorySpace.LOCAL
        else:
            space = MemorySpace.GLOBAL  # Default

        for lane_id in self.warp.get_executing_lane_ids():
            addr = self._compute_address(addr_str, lane_id)
            value = self.memory.read_u32(space, addr)
            self.warp.write_lane_reg(lane_id, dst, value)

    def _exec_store(self, instr: Instruction) -> None:
        """Generic store handler"""
        addr_str = instr.operands[0].value
        src_operand = instr.operands[1]

        # Determine memory space
        if instr.opcode in {Opcode.STG, Opcode.ST}:
            space = MemorySpace.GLOBAL
        elif instr.opcode == Opcode.STS:
            space = MemorySpace.SHARED
        elif instr.opcode == Opcode.STL:
            space = MemorySpace.LOCAL
        else:
            space = MemorySpace.GLOBAL  # Default

        for lane_id in self.warp.get_executing_lane_ids():
            addr = self._compute_address(addr_str, lane_id)
            # Get value - can be register or immediate
            if src_operand.type == OperandType.IMMEDIATE:
                value = src_operand.value
            else:
                value = self.warp.read_lane_reg(lane_id, src_operand.value)
            self.memory.write_u32(space, addr, value)

    # ==================== New Control Flow Functions ====================

    def _exec_branch(self, instr: Instruction) -> None:
        """Generic branch handler"""
        if instr.opcode == Opcode.BRX:
            # Indexed branch
            idx_reg = instr.operands[0].value
            idx = self.warp.read_lane_reg(0, idx_reg)
            # Simplified: just use the index directly
            self.warp.branch(idx)
        else:
            # Regular branch
            target = instr.operands[0].value
            self.warp.branch(target)
            self.branches_taken += 1

    def _exec_call(self, instr: Instruction) -> None:
        """Execute CALL: Call subroutine (simplified)"""
        target = instr.operands[0].value
        # Simplified: just branch (no return stack)
        self.warp.branch(target)

    def _exec_ret(self, instr: Instruction) -> None:
        """Execute RET: Return (simplified)"""
        # Simplified: treat as exit
        # Real implementation would use return stack
        pass

    # ==================== New Predicate Functions ====================

    def _exec_setp(self, instr: Instruction) -> None:
        """Execute SETP/PSETP: Set predicate from comparison"""
        pred_dst = instr.operands[0].value
        src1 = instr.operands[1].value if len(instr.operands) > 1 else 0
        src2_or_imm = instr.operands[2].value if len(instr.operands) > 2 else 0

        # Extract comparison type from full_opcode (e.g., "setp.eq" -> "eq")
        comparison_op = "lt"  # Default
        if instr.full_opcode:
            parts = instr.full_opcode.split(".")
            if len(parts) > 1:
                comparison_op = parts[1]

        # Define comparison operations
        comparisons = {
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "lt": lambda a, b: a < b,
            "le": lambda a, b: a <= b,
            "gt": lambda a, b: a > b,
            "ge": lambda a, b: a >= b,
            "lo": lambda a, b: a < b,  # unsigned less than
            "ls": lambda a, b: a <= b, # unsigned less than or equal
            "hi": lambda a, b: a > b,  # unsigned greater than
            "hs": lambda a, b: a >= b, # unsigned greater than or equal
        }

        # Get comparison function
        compare = comparisons.get(comparison_op, comparisons["lt"])

        # Check if src2 is an immediate
        if len(instr.operands) > 2 and instr.operands[2].type == OperandType.IMMEDIATE:
            val2 = src2_or_imm
            for lane_id in self.warp.get_executing_lane_ids():
                val1 = self.warp.read_lane_reg(lane_id, src1)
                result = compare(val1, val2)
                self.warp.get_thread(lane_id).set_pred(result)
        else:
            # Both operands are registers
            for lane_id in self.warp.get_executing_lane_ids():
                val1 = self.warp.read_lane_reg(lane_id, src1)
                val2 = self.warp.read_lane_reg(lane_id, src2_or_imm)
                result = compare(val1, val2)
                self.warp.get_thread(lane_id).set_pred(result)

    def _exec_selp(self, instr: Instruction) -> None:
        """Execute SELP: Select based on predicate"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            pred = self.warp.get_thread(lane_id).pred
            val = self.warp.read_lane_reg(lane_id, src1 if pred else src2)
            self.warp.write_lane_reg(lane_id, dst, val)

    # ==================== New Warp Level Functions ====================

    def _exec_vote(self, instr: Instruction) -> None:
        """Execute VOTE: Vote across warp"""
        dst = instr.operands[0].value
        pred_src = instr.operands[1].value if len(instr.operands) > 1 else None

        # Collect vote result
        vote_mask = 0
        for lane_id in self.warp.get_executing_lane_ids():
            if pred_src is not None:
                # Vote on predicate value
                # Simplified: check if lane's predicate is true
                if self.warp.get_thread(lane_id).pred:
                    vote_mask |= (1 << lane_id)
            else:
                # Vote on active lanes
                vote_mask |= (1 << lane_id)

        # Broadcast result to all lanes
        for lane_id in self.warp.get_executing_lane_ids():
            self.warp.write_lane_reg(lane_id, dst, vote_mask)

    def _exec_activemask(self, instr: Instruction) -> None:
        """Execute ACTIVEMASK: Get active lane mask"""
        dst = instr.operands[0].value

        for lane_id in self.warp.get_executing_lane_ids():
            self.warp.write_lane_reg(lane_id, dst, self.warp.lane_masks)

    def _exec_elect(self, instr: Instruction) -> None:
        """Execute ELECT: Elect one thread from warp"""
        dst = instr.operands[0].value

        # Elect lowest active lane
        active_lanes = self.warp.get_executing_lane_ids()
        elected = min(active_lanes) if active_lanes else 0

        for lane_id in self.warp.get_executing_lane_ids():
            result = 1 if lane_id == elected else 0
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_shfl(self, instr: Instruction) -> None:
        """Execute SHFL: Shuffle data between lanes"""
        dst = instr.operands[0].value
        src = instr.operands[1].value
        delta = instr.operands[2].value if len(instr.operands) > 2 else 0

        for lane_id in self.warp.get_executing_lane_ids():
            # Simplified: shuffle with fixed delta
            src_lane = (lane_id + delta) & 0x1F
            val = self.warp.read_lane_reg(src_lane, src)
            self.warp.write_lane_reg(lane_id, dst, val)

    def _exec_redux(self, instr: Instruction) -> None:
        """Execute REDUX: Reduction across warp"""
        dst = instr.operands[0].value
        src = instr.operands[1].value
        mask_op = instr.operands[2].value if len(instr.operands) > 2 else 0

        # Collect values from all active lanes
        values = {}
        for lane_id in self.warp.get_executing_lane_ids():
            values[lane_id] = self.warp.read_lane_reg(lane_id, src)

        # Simplified: just broadcast lane 0's value
        if values:
            result = values[min(values.keys())]
            for lane_id in self.warp.get_executing_lane_ids():
                self.warp.write_lane_reg(lane_id, dst, result)

    # ==================== New Atomic Functions ====================

    def _exec_atomic(self, instr: Instruction) -> None:
        """Execute atomic operation (simplified)"""
        # Simplified: just do regular operation without atomicity
        if instr.opcode in {Opcode.ATOM, Opcode.ATOM_ADD}:
            # Atomic add
            dst = instr.operands[0].value
            addr_str = instr.operands[1].value
            src = instr.operands[2].value

            for lane_id in self.warp.get_executing_lane_ids():
                addr = self._compute_address(addr_str, lane_id)
                old_val = self.memory.read_u32(MemorySpace.GLOBAL, addr)
                new_val = self.warp.read_lane_reg(lane_id, src)
                result = (old_val + new_val) & 0xFFFFFFFF
                self.memory.write_u32(MemorySpace.GLOBAL, addr, result)
                self.warp.write_lane_reg(lane_id, dst, old_val)

    # ==================== New Tensor Functions ====================

    def _exec_tensor(self, instr: Instruction) -> None:
        """Generic tensor instruction handler"""
        if instr.opcode == Opcode.HMMA:
            self._exec_hmma(instr)
        else:
            # Other tensor instructions (MMA, WMMA, etc.)
            # Simplified: treat as no-op for now
            pass

    # ==================== Warp Specialization Functions ====================

    def _exec_tma(self, instr: Instruction) -> None:
        """
        Execute TMA (Tensor Memory Accelerator) instruction.

        TMA enables efficient bulk data transfer between global and shared memory
        with hardware-accelerated address translation and strided access.

        TMA operations are ASYNCHRONOUS - they run in the background while
        the warp continues executing other instructions.

        When paired with mbarrier, TMA operations signal completion via
        mbarrier.complete_tx(), allowing consumer warps to wait efficiently.
        """
        if instr.opcode == Opcode.TMA_LOAD:
            # TMA.LOAD: Load matrix tile from global to shared memory
            # Format: TMA.LOAD [shared_addr], [global_addr], tile_size
            shared_addr_str = instr.operands[0].value
            global_addr_str = instr.operands[1].value
            tile_size = instr.operands[2].value if len(instr.operands) > 2 else 256

            # Get the first executing lane's addresses (TMA is warp-wide)
            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            shared_base = self._compute_address(shared_addr_str, lane_id)
            global_base = self._compute_address(global_addr_str, lane_id)

            # Create async TMA load operation with mbarrier signaling
            def tma_load_complete(op: AsyncOperation) -> None:
                """Callback when TMA load completes."""
                # Perform the actual data transfer
                for offset in range(0, op.size, 4):
                    global_addr = op.src_addr + offset
                    shared_addr = op.dst_addr + offset
                    value = self.memory.read_u32(MemorySpace.GLOBAL, global_addr)
                    # Write via ASYNC proxy (TMA operation)
                    self.memory.write_u32(MemorySpace.SHARED, shared_addr, value, proxy=ProxyType.ASYNC)

                # Signal mbarrier completion if mbarrier_addr is set
                if hasattr(op, 'mbarrier_addr') and op.mbarrier_addr is not None:
                    self.mbarrier_manager.complete_tx(op.mbarrier_addr)

            # Create and submit async operation
            tma_op = self.async_queue.create_tma_load(
                dst_addr=shared_base,
                src_addr=global_base,
                size=tile_size,
                warp_id=self.warp.warp_id,
                cycles=1  # Simulation: 1 cycle to complete
            )
            tma_op.callback = tma_load_complete
            # Link to active mbarrier if set
            if self.active_mbarrier_addr is not None:
                tma_op.mbarrier_addr = self.active_mbarrier_addr
            self.async_queue.submit(tma_op)

        elif instr.opcode == Opcode.TMA_STORE:
            # TMA.STORE: Store matrix tile from shared to global memory
            # Format: TMA.STORE [global_dst], [shared_src], size
            global_addr_str = instr.operands[0].value
            shared_addr_str = instr.operands[1].value
            tile_size = instr.operands[2].value if len(instr.operands) > 2 else 256

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            global_base = self._compute_address(global_addr_str, lane_id)
            shared_base = self._compute_address(shared_addr_str, lane_id)

            # Create async TMA store operation with mbarrier signaling
            def tma_store_complete(op: AsyncOperation) -> None:
                """Callback when TMA store completes."""
                for offset in range(0, op.size, 4):
                    shared_addr = op.src_addr + offset
                    global_addr = op.dst_addr + offset
                    value = self.memory.read_u32(MemorySpace.SHARED, shared_addr)
                    # Write via ASYNC proxy (TMA operation)
                    self.memory.write_u32(MemorySpace.GLOBAL, global_addr, value, proxy=ProxyType.ASYNC)

                # Signal mbarrier completion if mbarrier_addr is set
                if hasattr(op, 'mbarrier_addr') and op.mbarrier_addr is not None:
                    self.mbarrier_manager.complete_tx(op.mbarrier_addr)

            tma_op = self.async_queue.create_tma_store(
                dst_addr=global_base,
                src_addr=shared_base,
                size=tile_size,
                warp_id=self.warp.warp_id,
                cycles=1  # Simulation: 1 cycle to complete
            )
            tma_op.callback = tma_store_complete
            # Link to active mbarrier if set
            if self.active_mbarrier_addr is not None:
                tma_op.mbarrier_addr = self.active_mbarrier_addr
            self.async_queue.submit(tma_op)

        elif instr.opcode == Opcode.TMA_WAIT:
            # TMA.WAIT: Wait for TMA operations to complete
            # Wait for all pending async operations for this warp to complete
            # The pipeline will tick the async queue each cycle until done
            # For now, we don't block here - the async operations complete
            # in the background based on their cycle count
            pass

        # ==================== CP.ASYNC.BULK Instructions ====================
        elif instr.opcode in {Opcode.CP_ASYNC_BULK, Opcode.CP_ASYNC_BULK_DOT}:
            # cp.async.bulk: Async bulk copy from global to shared memory
            # Format: cp.async.bulk.shared::cta [dst], [src], size
            dst_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value

            # Get size - can be register or immediate
            if len(instr.operands) > 2:
                size_operand = instr.operands[2]
                if size_operand.type == OperandType.IMMEDIATE:
                    size = size_operand.value
                else:
                    # Read from register
                    lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
                    size = self.warp.read_lane_reg(lane_id, size_operand.value)
            else:
                size = 128

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            dst_addr = self._compute_address(dst_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            # Queue async bulk copy operation
            def bulk_copy_complete(op: AsyncOperation) -> None:
                """Callback when bulk copy completes."""
                for offset in range(0, op.size, 4):
                    global_addr = op.src_addr + offset
                    shared_addr = op.dst_addr + offset
                    # Read from global (generic proxy is fine for source)
                    value = self.memory.read_u32(MemorySpace.GLOBAL, global_addr)
                    # Write to shared via ASYNC proxy (TMA operation)
                    self.memory.write_u32(MemorySpace.SHARED, shared_addr, value, proxy=ProxyType.ASYNC)

                # Signal mbarrier completion if linked
                if hasattr(op, 'mbarrier_addr') and op.mbarrier_addr is not None:
                    self.mbarrier_manager.complete_tx(op.mbarrier_addr)

            bulk_op = self.async_queue.create_tma_load(
                dst_addr=dst_addr,
                src_addr=src_addr,
                size=size,
                warp_id=self.warp.warp_id,
                cycles=10  # Simulated latency
            )
            bulk_op.callback = bulk_copy_complete
            if self.active_mbarrier_addr is not None:
                bulk_op.mbarrier_addr = self.active_mbarrier_addr
            self.async_queue.submit(bulk_op)

        elif instr.opcode in {Opcode.CP_ASYNC_BULK_TENSOR, Opcode.CP_ASYNC_BULK_TENSOR_DOT}:
            # cp.async.bulk.tensor: Async bulk tensor copy
            # Format: cp.async.bulk.tensor.shared::cta [dst], [src], size, [stride]
            dst_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value
            size = instr.operands[2].value if len(instr.operands) > 2 else 128
            stride_str = instr.operands[3].value if len(instr.operands) > 3 else None

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            dst_addr = self._compute_address(dst_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            # Queue async tensor copy operation with stride support
            def tensor_copy_complete(op: AsyncOperation) -> None:
                """Callback when tensor copy completes."""
                # Tensor memory layout with stride
                stride = getattr(op, 'stride', size)
                for i in range(op.size // 16):  # Assume 16-byte tiles
                    tile_src = op.src_addr + i * stride
                    tile_dst = op.dst_addr + i * 16
                    for offset in range(0, min(16, op.size - i * 16), 4):
                        global_addr = tile_src + offset
                        shared_addr = tile_dst + offset
                        value = self.memory.read_u32(MemorySpace.GLOBAL, global_addr)
                        # Write via ASYNC proxy (TMA operation)
                        self.memory.write_u32(MemorySpace.SHARED, shared_addr, value, proxy=ProxyType.ASYNC)

                if hasattr(op, 'mbarrier_addr') and op.mbarrier_addr is not None:
                    self.memory.barrier_manager.mbarrier_complete_tx(
                        op.mbarrier_addr, 1, self.warp.warp_id
                    )

            tensor_op = self.async_queue.create_tma_load(
                dst_addr=dst_addr,
                src_addr=src_addr,
                size=size,
                warp_id=self.warp.warp_id,
                cycles=15  # Higher latency for tensor operations
            )
            tensor_op.callback = tensor_copy_complete
            if stride_str is not None:
                tensor_op.stride = self._compute_address(stride_str, lane_id) - src_addr
            if self.active_mbarrier_addr is not None:
                tensor_op.mbarrier_addr = self.active_mbarrier_addr
            self.async_queue.submit(tensor_op)

        elif instr.opcode in {Opcode.CP_ASYNC_BULK_PREFETCH, Opcode.CP_ASYNC_BULK_PREFETCH_DOT}:
            # cp.async.bulk.prefetch: Async bulk prefetch
            # Similar to cp.async.bulk but with prefetch semantics (no immediate use)
            dst_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value
            size = instr.operands[2].value if len(instr.operands) > 2 else 128

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            dst_addr = self._compute_address(dst_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            # Prefetch - same as bulk copy but with lower priority
            prefetch_op = self.async_queue.create_tma_load(
                dst_addr=dst_addr,
                src_addr=src_addr,
                size=size,
                warp_id=self.warp.warp_id,
                cycles=8  # Slightly lower latency
            )
            self.async_queue.submit(prefetch_op)

        elif instr.opcode in {Opcode.CP_ASYNC_BULK_PREFETCH_TENSOR, Opcode.CP_ASYNC_BULK_PREFETCH_TENSOR_DOT}:
            # cp.async.bulk.prefetch.tensor: Async bulk tensor prefetch
            dst_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value
            size = instr.operands[2].value if len(instr.operands) > 2 else 128
            stride_str = instr.operands[3].value if len(instr.operands) > 3 else None

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            dst_addr = self._compute_address(dst_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            prefetch_tensor_op = self.async_queue.create_tma_load(
                dst_addr=dst_addr,
                src_addr=src_addr,
                size=size,
                warp_id=self.warp.warp_id,
                cycles=12  # Latency for tensor prefetch
            )
            if stride_str is not None:
                prefetch_tensor_op.stride = self._compute_address(stride_str, lane_id) - src_addr
            self.async_queue.submit(prefetch_tensor_op)

        elif instr.opcode in {Opcode.CP_REDUCE_ASYNC_BULK, Opcode.CP_REDUCE_ASYNC_BULK_DOT}:
            # cp.reduce.async.bulk: Async bulk reduction
            # Performs reduction operation while copying
            dst_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value
            size = instr.operands[2].value if len(instr.operands) > 2 else 128

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            dst_addr = self._compute_address(dst_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            def reduce_complete(op: AsyncOperation) -> None:
                """Callback when reduction completes."""
                # Read-Modify-Write reduction (add)
                for offset in range(0, op.size, 4):
                    global_addr = op.src_addr + offset
                    shared_addr = op.dst_addr + offset
                    src_val = self.memory.read_u32(MemorySpace.GLOBAL, global_addr)
                    dst_val = self.memory.read_u32(MemorySpace.SHARED, shared_addr)
                    # Write via ASYNC proxy (TMA reduction operation)
                    self.memory.write_u32(MemorySpace.SHARED, shared_addr, dst_val + src_val, proxy=ProxyType.ASYNC)

            reduce_op = self.async_queue.create_tma_load(
                dst_addr=dst_addr,
                src_addr=src_addr,
                size=size,
                warp_id=self.warp.warp_id,
                cycles=20  # Higher latency for reductions
            )
            reduce_op.callback = reduce_complete
            self.async_queue.submit(reduce_op)

        elif instr.opcode in {Opcode.CP_REDUCE_ASYNC_BULK_TENSOR, Opcode.CP_REDUCE_ASYNC_BULK_TENSOR_DOT}:
            # cp.reduce.async.bulk.tensor: Async bulk tensor reduction
            dst_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value
            size = instr.operands[2].value if len(instr.operands) > 2 else 128
            stride_str = instr.operands[3].value if len(instr.operands) > 3 else None

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            dst_addr = self._compute_address(dst_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            def tensor_reduce_complete(op: AsyncOperation) -> None:
                """Callback when tensor reduction completes."""
                stride = getattr(op, 'stride', size)
                for i in range(op.size // 16):
                    tile_src = op.src_addr + i * stride
                    tile_dst = op.dst_addr + i * 16
                    for offset in range(0, min(16, op.size - i * 16), 4):
                        global_addr = tile_src + offset
                        shared_addr = tile_dst + offset
                        src_val = self.memory.read_u32(MemorySpace.GLOBAL, global_addr)
                        dst_val = self.memory.read_u32(MemorySpace.SHARED, shared_addr)
                        # Write via ASYNC proxy (TMA reduction operation)
                        self.memory.write_u32(MemorySpace.SHARED, shared_addr, dst_val + src_val, proxy=ProxyType.ASYNC)

            reduce_tensor_op = self.async_queue.create_tma_load(
                dst_addr=dst_addr,
                src_addr=src_addr,
                size=size,
                warp_id=self.warp.warp_id,
                cycles=25  # Highest latency for tensor reductions
            )
            reduce_tensor_op.callback = tensor_reduce_complete
            if stride_str is not None:
                reduce_tensor_op.stride = self._compute_address(stride_str, lane_id) - src_addr
            self.async_queue.submit(reduce_tensor_op)

        elif instr.opcode in {Opcode.CP_ASYNC_BULK_COMMIT_GROUP, Opcode.CP_ASYNC_BULK_COMMIT_GROUP_DOT}:
            # cp.async.bulk.commit_group: Commit async bulk group
            # Marks completion of a group of async bulk operations
            group_id = instr.operands[0].value if len(instr.operands) > 0 else 0

            # For simulation: track group completion
            if not hasattr(self, 'cp_async_group_id'):
                self.cp_async_group_id = 0
            self.cp_async_group_id = max(self.cp_async_group_id, group_id + 1)

            # Flush pending operations in this group
            from ..core.memory import Scope, MemoryOrder
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.RELEASE,
                thread_id=0,
                warp_id=self.warp.warp_id,
                cta_id=self.warp.warp_id
            )

        elif instr.opcode in {Opcode.CP_ASYNC_BULK_WAIT_GROUP, Opcode.CP_ASYNC_BULK_WAIT_GROUP_DOT}:
            # cp.async.bulk.wait_group: Wait for async bulk group
            # Stalls until specified group completes
            wait_group = instr.operands[0].value if len(instr.operands) > 0 else 0

            # For simulation: acquire fence
            from ..core.memory import Scope, MemoryOrder
            self.memory.barrier_manager.fence(
                scope=Scope.CTA,
                order=MemoryOrder.ACQUIRE,
                thread_id=0,
                warp_id=self.warp.warp_id,
                cta_id=self.warp.warp_id
            )

        elif instr.opcode in {Opcode.MULTIMEM_CP_ASYNC_BULK, Opcode.MULTIMEM_CP_ASYNC_BULK_DOT}:
            # multimem.cp.async.bulk: Multimem async bulk copy
            # Similar to cp.async.bulk but uses multimem (multiple memory channels)
            dst_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value
            size = instr.operands[2].value if len(instr.operands) > 2 else 128

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            dst_addr = self._compute_address(dst_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            # Multimem - higher bandwidth, lower latency
            multimem_op = self.async_queue.create_tma_load(
                dst_addr=dst_addr,
                src_addr=src_addr,
                size=size,
                warp_id=self.warp.warp_id,
                cycles=5  # Lower latency due to multimem
            )
            self.async_queue.submit(multimem_op)

        elif instr.opcode in {Opcode.MULTIMEM_CP_REDUCE_ASYNC_BULK, Opcode.MULTIMEM_CP_REDUCE_ASYNC_BULK_DOT}:
            # multimem.cp.reduce.async.bulk: Multimem async bulk reduction
            dst_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value
            size = instr.operands[2].value if len(instr.operands) > 2 else 128

            lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
            dst_addr = self._compute_address(dst_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            def multimem_reduce_complete(op: AsyncOperation) -> None:
                """Callback when multimem reduction completes."""
                for offset in range(0, op.size, 4):
                    global_addr = op.src_addr + offset
                    shared_addr = op.dst_addr + offset
                    src_val = self.memory.read_u32(MemorySpace.GLOBAL, global_addr)
                    dst_val = self.memory.read_u32(MemorySpace.SHARED, shared_addr)
                    # Write via ASYNC proxy (multimem reduction operation)
                    self.memory.write_u32(MemorySpace.SHARED, shared_addr, dst_val + src_val, proxy=ProxyType.ASYNC)

            multimem_reduce_op = self.async_queue.create_tma_load(
                dst_addr=dst_addr,
                src_addr=src_addr,
                size=size,
                warp_id=self.warp.warp_id,
                cycles=10  # Lower latency than regular reduce
            )
            multimem_reduce_op.callback = multimem_reduce_complete
            self.async_queue.submit(multimem_reduce_op)

    def _exec_wgmma(self, instr: Instruction) -> None:
        """
        Execute WGMMA (Warpgroup Matrix Multiply-Accumulate) instruction.

        WGMMA operates on warpgroups (128 threads = 4 warps) and performs
        asynchronous matrix multiply-accumulate operations on matrix tiles.
        This is the key instruction for Hopper's warp specialization.
        """
        # WGMMA formats:
        # WGMMA.MMA d, a, b, shape - Matrix multiply-accumulate
        # WGMMA.MMA_ASYNC d, a, b - Async version

        dst = instr.operands[0].value
        src_a = instr.operands[1].value
        src_b = instr.operands[2].value
        src_c = instr.operands[3].value if len(instr.operands) > 3 else None

        # WGMMA operates on warpgroup (128 threads)
        # For simulation, we'll use the current warp's registers
        # Common WGMMA shapes: m64n8k16, m64n8k32, m64n8k64, m64n8k256
        # (full implementation would use these for proper matrix tiling)

        # Simulate matrix multiplication
        # For simplicity, we'll do a scalar multiply-accumulate
        # Real WGMMA would operate on matrix fragments distributed across warpgroup

        for lane_id in self.warp.get_executing_lane_ids():
            # Read matrix fragment values
            a_val = self.warp.read_lane_reg(lane_id, src_a)
            b_val = self.warp.read_lane_reg(lane_id, src_b)
            c_val = self.warp.read_lane_reg(lane_id, src_c) if src_c is not None else 0

            # Perform: D = A * B + C
            result = (a_val * b_val + c_val) & 0xFFFFFFFF

            # Write result
            self.warp.write_lane_reg(lane_id, dst, result)

        # Note: Real WGMMA would:
        # 1. Read matrix tiles from shared memory
        # 2. Perform 64x8x16 matrix multiply across 128 threads
        # 3. Write result tile to registers
        # 4. Support various data types (FP8, FP16, BF16, FP32, TF32)
        # 5. Execute asynchronously with WGMMA.MMA_ASYNC

    def _exec_mbarrier(self, instr: Instruction) -> None:
        """
        Execute mbarrier (memory barrier) instruction.

        mbarrier is used for synchronizing asynchronous operations (TMA, WGMMA)
        in warp-specialized kernels.

        Producer workflow:
        1. mbarrier.init [addr], count
        2. mbarrier.expect_tx [addr], count
        3. Issue async operations (TMA, WGMMA)

        Consumer workflow:
        1. mbarrier.try_wait [addr] - spins until ready
        2. Consume data
        """
        # Get thread info
        lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)
        thread = self.warp.threads[lane_id]
        thread_id = thread.thread_id
        warp_id = self.warp.warp_id

        if instr.opcode in {Opcode.MBARRIER_INIT, Opcode.MBARRIER_INIT_DOT}:
            # MBARRIER_INIT: Initialize mbarrier
            # Format: MBARRIER_INIT [mbarrier_addr], count
            mbarrier_addr_str = instr.operands[0].value

            # Get count - can be register or immediate
            if len(instr.operands) > 1:
                count_operand = instr.operands[1]
                if count_operand.type == OperandType.IMMEDIATE:
                    count = count_operand.value
                else:
                    # Read from register
                    count = self.warp.read_lane_reg(lane_id, count_operand.value)
            else:
                count = 1

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            self.mbarrier_manager.init_barrier(addr, count, self.warp.warp_id)

            # Also write to shared memory for debugging/visibility
            self.memory.write_u32(MemorySpace.SHARED, addr, count)

        elif instr.opcode in {Opcode.MBARRIER_INVAL, Opcode.MBARRIER_INVAL_DOT}:
            # MBARRIER_INVAL: Invalidate mbarrier
            mbarrier_addr_str = instr.operands[0].value

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            self.mbarrier_manager.invalidate(addr)
            self.memory.write_u32(MemorySpace.SHARED, addr, 0)

        elif instr.opcode in {Opcode.MBARRIER_ARRIVE, Opcode.MBARRIER_ARRIVE_DOT}:
            # MBARRIER_ARRIVE: Arrive at mbarrier (decrement counter)
            # This signals that one thread/warp has reached the barrier
            mbarrier_addr_str = instr.operands[0].value

            addr = self._compute_address(mbarrier_addr_str, lane_id)

            # Decrement counter in mbarrier
            self.mbarrier_manager.complete_tx(addr)

            # Update shared memory for visibility
            barrier = self.mbarrier_manager.get_barrier(addr)
            if barrier:
                self.memory.write_u32(MemorySpace.SHARED, addr, barrier.current_count)

        elif instr.opcode in {Opcode.MBARRIER_TEST_WAIT, Opcode.MBARRIER_TEST_WAIT_DOT}:
            # MBARRIER_TEST_WAIT: Test and wait for mbarrier
            # Stalls warp at this instruction until mbarrier is ready
            mbarrier_addr_str = instr.operands[0].value
            addr = self._compute_address(mbarrier_addr_str, lane_id)

            # Test if barrier is ready
            ready = self.mbarrier_manager.try_wait(addr)

            if ready:
                # Barrier is satisfied, continue execution
                # Set predicate register (R0) to true
                self.warp.write_lane_reg(lane_id, 0, 1)
                # Unstall the warp so it can continue
                self.warp.unstall()
            else:
                # Barrier not ready, stall the warp
                # The warp will not advance PC this cycle
                self.warp.stall("mbarrier_test_wait")
                # Don't advance PC - warp stays at this instruction
                return True

        elif instr.opcode in {Opcode.MBARRIER_EXPECT_TX, Opcode.MBARRIER_EXPECT_TX_DOT}:
            # MBARRIER_EXPECT_TX: Expect transaction (producer)
            # Tells mbarrier how many async operations to expect
            # Also links subsequent TMA operations to this mbarrier
            mbarrier_addr_str = instr.operands[0].value

            # Get count - can be register or immediate
            if len(instr.operands) > 1:
                count_operand = instr.operands[1]
                if count_operand.type == OperandType.IMMEDIATE:
                    count = count_operand.value
                else:
                    # Read from register
                    count = self.warp.read_lane_reg(lane_id, count_operand.value)
            else:
                count = 1

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            # Set expected transaction count
            self.mbarrier_manager.expect_tx(addr, count)
            self.memory.write_u32(MemorySpace.SHARED, addr, count)

            # Track this mbarrier for subsequent TMA operations
            self.active_mbarrier_addr = addr

        elif instr.opcode in {Opcode.MBARRIER_COMPLETE_TX, Opcode.MBARRIER_COMPLETE_TX_DOT}:
            # MBARRIER_COMPLETE_TX: Complete transaction
            # Called manually or by async operation callback
            mbarrier_addr_str = instr.operands[0].value

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            # Mark transaction as complete in memory system's barrier manager
            if hasattr(self.memory.barrier_manager, '_mbarriers') and addr in self.memory.barrier_manager._mbarriers:
                mbarrier = self.memory.barrier_manager._mbarriers[addr]
                mbarrier.completed_phase = mbarrier.phase
            current = self.memory.read_u32(MemorySpace.SHARED, addr)
            self.memory.write_u32(MemorySpace.SHARED, addr, current)

        elif instr.opcode in {Opcode.MBARRIER_ARRIVE_DROP, Opcode.MBARRIER_ARRIVE_DROP_DOT}:
            # MBARRIER_ARRIVE_DROP: Arrive at mbarrier and reduce threshold
            # Reduces the expected count (useful for dynamic thread counts)
            mbarrier_addr_str = instr.operands[0].value
            drop_count = instr.operands[1].value if len(instr.operands) > 1 else 1

            addr = self._compute_address(mbarrier_addr_str, lane_id)

            # Update both shared memory and mbarrier manager
            current = self.memory.read_u32(MemorySpace.SHARED, addr)
            new_count = max(0, current - drop_count)
            self.memory.write_u32(MemorySpace.SHARED, addr, new_count)

            # Update mbarrier manager's expected count
            if hasattr(self.memory.barrier_manager, '_mbarriers') and addr in self.memory.barrier_manager._mbarriers:
                mbarrier = self.memory.barrier_manager._mbarriers[addr]
                mbarrier.expected_count = max(0, mbarrier.expected_count - drop_count)

        elif instr.opcode in {Opcode.MBARRIER_TRY_WAIT, Opcode.MBARRIER_TRY_WAIT_DOT}:
            # MBARRIER_TRY_WAIT: Try wait for mbarrier (non-blocking)
            # Returns a predicate indicating if barrier is ready
            # Unlike test_wait, this doesn't stall - it sets a predicate
            mbarrier_addr_str = instr.operands[0].value
            pred_reg = instr.operands[1].value if len(instr.operands) > 1 else 0

            addr = self._compute_address(mbarrier_addr_str, lane_id)

            # Get current phase from shared memory
            current = self.memory.read_u32(MemorySpace.SHARED, addr)
            phase = current % 2

            # Test if barrier is ready (non-blocking)
            ready = self.mbarrier_manager.try_wait(addr)

            # Set predicate register to indicate readiness
            # Note: PTX uses predicate registers, here we use R0 as a simple predicate
            self.warp.write_lane_reg(lane_id, pred_reg, 1 if ready else 0)

        elif instr.opcode in {Opcode.MBARRIER_PENDING_COUNT, Opcode.MBARRIER_PENDING_COUNT_DOT}:
            # MBARRIER_PENDING_COUNT: Get number of pending arrivals
            # Returns how many arrivals are still needed
            mbarrier_addr_str = instr.operands[0].value
            dst_reg = instr.operands[1].value

            addr = self._compute_address(mbarrier_addr_str, lane_id)

            # Get pending count from mbarrier manager
            barrier = self.mbarrier_manager.get_barrier(addr)
            if barrier:
                pending = barrier.current_count
            else:
                pending = 0

            # Write pending count to destination register
            self.warp.write_lane_reg(lane_id, dst_reg, pending)

        elif instr.opcode in {Opcode.CP_ASYNC_MBARRIER_ARRIVE, Opcode.CP_ASYNC_MBARRIER_ARRIVE_DOT}:
            # CP_ASYNC_MBARRIER_ARRIVE: Async copy with mbarrier arrive
            # Combines async copy operation with mbarrier arrival
            # Format: cp.async.mbarrier.arrive [mbar], [src], dst, count
            mbarrier_addr_str = instr.operands[0].value
            src_addr_str = instr.operands[1].value
            dst_reg = instr.operands[2].value if len(instr.operands) > 2 else None
            count = instr.operands[3].value if len(instr.operands) > 3 else 1

            mbarrier_addr = self._compute_address(mbarrier_addr_str, lane_id)
            src_addr = self._compute_address(src_addr_str, lane_id)

            # Queue async copy operation
            self.async_queue.enqueue_copy(src_addr, dst_reg if dst_reg is not None else 0, count)

            # Arrive at mbarrier
            self.mbarrier_manager.complete_tx(mbarrier_addr)

    def _compute_address(self, addr_str: str, lane_id: int) -> int:
        """
        Compute memory address from address string.

        Supports formats:
        - "[0x...]" - literal hexadecimal address
        - "[0...]" - literal decimal address
        - "[Rn]" - register indirect
        - "[Rn + offset]" - register + immediate offset

        Args:
            addr_str: Address string (e.g., "[R1+16]" or "[0x6000]")
            lane_id: Lane ID for reading register

        Returns:
            Computed address
        """
        import re
        # Remove brackets
        inner = addr_str.strip().strip('[]')

        # Try [0x...] or [0...] - literal addresses (hex or decimal)
        match = re.match(r'(0x[0-9a-fA-F]+|\d+)', inner)
        if match:
            # Literal address
            return int(match.group(1), 0)  # Auto-detect base

        # Try [Rn + offset]
        match = re.match(r'R(\d+)\s*\+\s*(\d+)', inner)
        if match:
            reg_num = int(match.group(1))
            offset = int(match.group(2))
            base = self.warp.read_lane_reg(lane_id, reg_num)
            return base + offset

        # Try [Rn]
        match = re.match(r'R(\d+)', inner)
        if match:
            reg_num = int(match.group(1))
            return self.warp.read_lane_reg(lane_id, reg_num)

        raise ExecutionException(f"Invalid address format: {addr_str}")

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'instructions_executed': self.instructions_executed,
            'branches_taken': self.branches_taken,
            'active_lanes': self.warp.count_active(),
        }

    def __repr__(self) -> str:
        return f"WarpExecutor(warp={self.warp.warp_id}, executed={self.instructions_executed})"
