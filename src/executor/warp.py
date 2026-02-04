"""
Warp Executor for Hopper GPU

Executes instructions for a warp of 32 threads in SIMT fashion.
"""

from typing import List, Optional, Dict, Any
import struct
from ctypes import c_int32, c_uint32
from ..core.warp import Warp
from ..core.memory import Memory, MemorySpace
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
        # Check predicate
        if instruction.predicate_reg is not None:
            pred_value = self._evaluate_predicate(instruction)
            should_execute = (pred_value == instruction.predicate_condition)
        else:
            should_execute = True

        # Update execution mask based on predicates and active lanes
        self.warp.update_execution_mask()

        # If no lanes should execute, skip but still advance PC
        if not should_execute or not self.warp.any_executing():
            self.warp.advance_pc(1)
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
            self.warp.advance_pc(1)
            return True

        # Advance PC (unless instruction modified it)
        if opcode not in {Opcode.BRA, Opcode.BRX, Opcode.CALL, Opcode.CAL, Opcode.RET, Opcode.EXIT}:
            self.warp.advance_pc(1)

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

        if src.type == OperandType.REGISTER:
            src_val = src.value
            for lane_id in self.warp.get_executing_lane_ids():
                val = self.warp.read_lane_reg(lane_id, src_val)
                self.warp.write_lane_reg(lane_id, dst, val)
        elif src.type == OperandType.IMMEDIATE:
            for lane_id in self.warp.get_executing_lane_ids():
                self.warp.write_lane_reg(lane_id, dst, src.value)

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

    # ==================== New Integer Arithmetic Functions ====================

    def _exec_iadd(self, instr: Instruction) -> None:
        """Execute IADD: Rd = Ra + Rb"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            result = (val1 + val2) & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_isub(self, instr: Instruction) -> None:
        """Execute ISUB: Rd = Ra - Rb"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            result = (val1 - val2) & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_imul(self, instr: Instruction) -> None:
        """Execute IMUL: Rd = Ra * Rb"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            # Sign-extend and multiply
            prod = c_int32(val1).value * c_int32(val2).value
            result = prod & 0xFFFFFFFF
            self.warp.write_lane_reg(lane_id, dst, result)

    def _exec_iminmax(self, instr: Instruction) -> None:
        """Execute IMIN/IMAX: Rd = min(Ra, Rb) or max(Ra, Rb)"""
        dst = instr.operands[0].value
        src1 = instr.operands[1].value
        src2 = instr.operands[2].value
        is_max = (instr.opcode == Opcode.IMAX)

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = c_int32(self.warp.read_lane_reg(lane_id, src1)).value
            val2 = c_int32(self.warp.read_lane_reg(lane_id, src2)).value
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
        src = instr.operands[1].value

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
            value = self.warp.read_lane_reg(lane_id, src)
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
        # Simplified: set all predicates based on comparison
        pred_dst = instr.operands[0].value
        src1 = instr.operands[1].value if len(instr.operands) > 1 else 0
        src2 = instr.operands[2].value if len(instr.operands) > 2 else 0

        for lane_id in self.warp.get_executing_lane_ids():
            val1 = self.warp.read_lane_reg(lane_id, src1)
            val2 = self.warp.read_lane_reg(lane_id, src2)
            # Simple comparison: less than
            result = c_int32(val1).value < c_int32(val2).value
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
                    self.memory.write_u32(MemorySpace.SHARED, shared_addr, value)

                # Signal mbarrier completion if mbarrier_addr is set
                if hasattr(op, 'mbarrier_addr') and op.mbarrier_addr is not None:
                    self.mbarrier_manager.complete_tx(op.mbarrier_addr)

            # Create and submit async operation
            tma_op = self.async_queue.create_tma_load(
                dst_addr=shared_base,
                src_addr=global_base,
                size=tile_size,
                warp_id=self.warp.warp_id,
                cycles=50  # Simulation: 50 cycles to complete
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
                    self.memory.write_u32(MemorySpace.GLOBAL, global_addr, value)

                # Signal mbarrier completion if mbarrier_addr is set
                if hasattr(op, 'mbarrier_addr') and op.mbarrier_addr is not None:
                    self.mbarrier_manager.complete_tx(op.mbarrier_addr)

            tma_op = self.async_queue.create_tma_store(
                dst_addr=global_base,
                src_addr=shared_base,
                size=tile_size,
                warp_id=self.warp.warp_id,
                cycles=50
            )
            tma_op.callback = tma_store_complete
            # Link to active mbarrier if set
            if self.active_mbarrier_addr is not None:
                tma_op.mbarrier_addr = self.active_mbarrier_addr
            self.async_queue.submit(tma_op)

        elif instr.opcode == Opcode.TMA_WAIT:
            # TMA.WAIT: Wait for TMA operations to complete
            # In real hardware, this waits for the TMA unit to finish
            # In simulation, we spin until async ops complete
            # The main simulation loop will handle ticking the async queue
            pass  # Actual waiting happens in simulation loop

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
        lane_id = next(iter(self.warp.get_executing_lane_ids()), 0)

        if instr.opcode == Opcode.MBARRIER_INIT:
            # MBARRIER_INIT: Initialize mbarrier
            # Format: MBARRIER_INIT [mbarrier_addr], count
            mbarrier_addr_str = instr.operands[0].value
            count = instr.operands[1].value if len(instr.operands) > 1 else 1

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            self.mbarrier_manager.init_barrier(addr, count, self.warp.warp_id)

            # Also write to shared memory for debugging/visibility
            self.memory.write_u32(MemorySpace.SHARED, addr, count)

        elif instr.opcode == Opcode.MBARRIER_INVAL:
            # MBARRIER_INVAL: Invalidate mbarrier
            mbarrier_addr_str = instr.operands[0].value

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            self.mbarrier_manager.invalidate(addr)
            self.memory.write_u32(MemorySpace.SHARED, addr, 0)

        elif instr.opcode == Opcode.MBARRIER_ARRIVE:
            # MBARRIER_ARRIVE: Arrive at mbarrier (decrement counter)
            mbarrier_addr_str = instr.operands[0].value

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            current = self.memory.read_u32(MemorySpace.SHARED, addr)
            if current > 0:
                self.memory.write_u32(MemorySpace.SHARED, addr, current - 1)

        elif instr.opcode == Opcode.MBARRIER_TEST_WAIT:
            # MBARRIER_TEST_WAIT: Test and wait for mbarrier
            # Spins until mbarrier is ready
            mbarrier_addr_str = instr.operands[0].value
            addr = self._compute_address(mbarrier_addr_str, lane_id)

            # Set predicate to true if ready, false if not
            ready = self.mbarrier_manager.try_wait(addr)
            # Set predicate register (R0) to result
            self.warp.write_lane_reg(lane_id, 0, 1 if ready else 0)

        elif instr.opcode == Opcode.MBARRIER_EXPECT_TX:
            # MBARRIER_EXPECT_TX: Expect transaction (producer)
            # Tells mbarrier how many async operations to expect
            # Also links subsequent TMA operations to this mbarrier
            mbarrier_addr_str = instr.operands[0].value
            count = instr.operands[1].value if len(instr.operands) > 1 else 1

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            self.mbarrier_manager.expect_tx(addr, count)
            self.memory.write_u32(MemorySpace.SHARED, addr, count)

            # Track this mbarrier for subsequent TMA operations
            self.active_mbarrier_addr = addr

        elif instr.opcode == Opcode.MBARRIER_COMPLETE_TX:
            # MBARRIER_COMPLETE_TX: Complete transaction
            # Called manually or by async operation callback
            mbarrier_addr_str = instr.operands[0].value

            addr = self._compute_address(mbarrier_addr_str, lane_id)
            is_complete = self.mbarrier_manager.complete_tx(addr)
            current = self.mbarrier_manager.get_barrier(addr).current_count if self.mbarrier_manager.get_barrier(addr) else 0
            self.memory.write_u32(MemorySpace.SHARED, addr, current)

    def _compute_address(self, addr_str: str, lane_id: int) -> int:
        """
        Compute memory address from address string.

        Supports formats:
        - "[Rn]" - register indirect
        - "[Rn + offset]" - register + immediate offset

        Args:
            addr_str: Address string (e.g., "[R1+16]")
            lane_id: Lane ID for reading register

        Returns:
            Computed address
        """
        import re
        # Remove brackets
        inner = addr_str.strip().strip('[]')

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
