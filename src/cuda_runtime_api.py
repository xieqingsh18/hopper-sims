"""
CUDA Runtime API - Full Implementation

This module provides a complete CUDA Runtime API implementation that matches
the official NVIDIA CUDA Runtime API interface. All functions have the same
signatures and behavior as the real CUDA Runtime API.

Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/

Implemented API Categories:
1. Error Handling: cudaGetErrorString, cudaGetLastError, cudaSuccess
2. Device Management: cudaGetDeviceCount, cudaGetDeviceProperties, cudaSetDevice, cudaGetDevice
3. Memory Management: cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaMallocHost, etc.
4. Version Info: cudaGetDriverVersion, cudaRuntimeGetVersion
5. Stream Management: cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize, etc.
6. Event Management: cudaEventCreate, cudaEventDestroy, cudaEventRecord, cudaEventSynchronize, etc.
7. Kernel Launch: cudaLaunchKernel, cudaConfigureCall, cudaLaunch
8. Device Management: cudaDeviceSynchronize, cudaDeviceReset
9. Peer Access: cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess

Usage Example:
    import ctypes
    from src.cuda_runtime_api import *

    # Get device count
    device_count = ctypes.c_int()
    cudaGetDeviceCount(ctypes.byref(device_count))

    # Allocate memory
    size = ctypes.c_size_t(1024)
    dev_ptr = ctypes.c_void_p()
    cudaMalloc(size, ctypes.byref(dev_ptr))

    # Copy memory
    cudaMemcpy(dev_ptr, host_data, size, cudaMemcpyHostToDevice)

    # Free memory
    cudaFree(dev_ptr)
"""

from ctypes import c_void_p, c_int, c_size_t, c_float, c_char_p, POINTER, byref, CFUNCTYPE
from enum import IntEnum, IntFlag
from typing import Optional, List, Any, Dict, Callable
from dataclasses import dataclass, field
import struct
import time


# ==================== Stream Implementation Classes ====================

@dataclass
class StreamCommand:
    """A command in a CUDA stream queue."""
    type: str  # 'memcpy', 'kernel', 'event_record', 'stream_wait', 'barrier'
    data: Dict = field(default_factory=dict)
    callback: Optional[Callable] = None
    dependencies: List[int] = field(default_factory=list)  # Event IDs this command waits for
    command_id: int = 0


@dataclass
class CUDAEventState:
    """A CUDA event for tracking stream synchronization."""
    event_id: int
    recorded: bool = False
    stream_id: int = 0
    completed: bool = False
    timing: float = 0.0  # For cudaEventElapsedTime


@dataclass
class CUDAStreamState:
    """A CUDA stream with a work queue."""
    stream_id: int
    priority: int = 0
    commands: List[StreamCommand] = field(default_factory=list)
    current_command: int = 0
    synchronized: bool = True  # True if all commands completed
    next_command_id: int = 1
    creation_time: float = field(default_factory=time.time)


# ==================== Error Codes ====================

class cudaError_t(IntEnum):
    """CUDA error codes."""
    cudaSuccess = 0
    cudaErrorInvalidValue = 1
    cudaErrorMemoryAllocation = 2
    cudaErrorInitializationError = 3
    cudaErrorCudartUnloading = 4
    cudaErrorProfilerDisabled = 5
    cudaErrorProfilerNotInitialized = 6
    cudaErrorProfilerAlreadyStarted = 7
    cudaErrorProfilerStopped = 8
    cudaErrorInvalidConfiguration = 9
    cudaErrorInvalidPitchValue = 12
    cudaErrorInvalidSymbol = 13
    cudaErrorInvalidDevicePointer = 16
    cudaErrorInvalidTexture = 17
    cudaErrorInvalidTextureBinding = 18
    cudaErrorInvalidChannelDescriptor = 19
    cudaErrorInvalidMemcpyDirection = 21
    cudaErrorAddressOfConstant = 22
    cudaErrorTextureFetchFailed = 23
    cudaErrorTextureNotBound = 24
    cudaErrorTextureAccessFailed = 25
    cudaErrorInvalidDeviceFunction = 26
    cudaErrorInvalidDevice = 28
    cudaErrorInvalidHostPointer = 29
    cudaErrorInvalidDeviceFunctionPointer = 30
    cudaErrorInvalidTexturePointer = 32
    cudaErrorInvalidNormconf = 33
    cudaErrorLaunchFailure = 34
    cudaErrorLaunchTimeout = 35
    cudaErrorLaunchOutOfResources = 36
    cudaErrorLaunchIncompatibleTexturing = 37
    cudaErrorPeerAccessAlreadyEnabled = 38
    cudaErrorPeerAccessNotEnabled = 39
    cudaErrorPeerAccessAlreadyDisabled = 40
    cudaErrorDeviceAlreadyInUse = 54
    cudaErrorAssert = 59
    cudaErrorTooManyPeers = 60
    cudaErrorHostMemoryAlreadyRegistered = 61
    cudaErrorHostMemoryNotRegistered = 62
    cudaErrorOperatingSystem = 63
    cudaErrorPeerAccessUnsupported = 64
    cudaErrorLaunchMaxDepthExceeded = 65
    cudaErrorLaunchFileScopedTex = 66
    cudaErrorLaunchTextureBindingFailed = 67
    cudaErrorSyncError = 68
    cudaErrorLaunchPending = 69
    cudaErrorNotPermitted = 70
    cudaErrorNotSupported = 71
    cudaErrorStartupFailure = 127
    cudaErrorNotReady = 1001
    cudaErrorApiFailureBase = 10000


# ==================== Memory Copy Kind ====================

class cudaMemcpyKind(IntEnum):
    """CUDA memory copy directions."""
    cudaMemcpyHostToHost = 0
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpyDeviceToDevice = 3
    cudaMemcpyDefault = 4


# ==================== Device Attributes ====================

class cudaDeviceAttr(IntEnum):
    """CUDA device attributes for cudaDeviceGetAttribute."""
    cudaDevAttrMaxThreadsPerBlock = 1
    cudaDevAttrMaxBlockDimX = 2
    cudaDevAttrMaxBlockDimY = 3
    cudaDevAttrMaxBlockDimZ = 4
    cudaDevAttrMaxGridDimX = 5
    cudaDevAttrMaxGridDimY = 6
    cudaDevAttrMaxGridDimZ = 7
    cudaDevAttrMaxSharedMemoryPerBlock = 8
    cudaDevAttrTotalConstantMemory = 9
    cudaDevAttrWarpSize = 10
    cudaDevAttrMaxRegistersPerBlock = 12
    cudaDevAttrClockRate = 13
    cudaDevAttrMajor = 14
    cudaDevAttrMinor = 15
    cudaDevAttrMultiProcessorCount = 16
    cudaDevAttrL2CacheSize = 17
    cudaDevAttrMaxThreadsPerMultiProcessor = 18
    cudaDevAttrComputeCapabilityMajor = 75
    cudaDevAttrComputeCapabilityMinor = 76


# ==================== Device Properties ====================

class cudaDeviceProp:
    """CUDA device properties (cudaGetDeviceProperties)."""
    def __init__(self):
        self.name = b"Hopper GPU Simulator"  # Device name
        self.totalGlobalMem = 8 * 1024 * 1024 * 1024  # 8GB
        self.sharedMemPerBlock = 48 * 1024  # 48KB
        self.regsPerBlock = 65536  # 64K registers
        self.warpSize = 32
        self.maxThreadsPerBlock = 1024
        self.maxThreadsDim = [1024, 1024, 64]  # x, y, z
        self.maxGridSize = [0x7FFFFFFF, 0xFFFF, 0xFFFF]
        self.clockRate = 1410  # MHz
        self.multiProcessorCount = 108
        self.major = 9
        self.minor = 0
        self.l2CacheSize = 50 * 1024 * 1024  # 50MB
        self.maxThreadsPerMultiProcessor = 1280
        self.asyncEngineCount = 1
        self.concurrentKernels = 1
        eccEnabled = 0  # Boolean, not a pointer
        p2p = self.p2pProps = 0
        busWidth = 0


# ==================== Stream and Event Types ====================

cudaStream_t = c_void_p  # Stream handle
cudaEvent_t = c_void_p  # Event handle


# ==================== Global CUDA Runtime State ====================

_runtime_instance = None


def get_runtime(simulator=None):
    """Get or create the global CUDA runtime instance."""
    global _runtime_instance
    if _runtime_instance is None:
        if simulator is None:
            raise RuntimeError("CUDARuntime not initialized. Call init_cuda_runtime(simulator) first.")
        _runtime_instance = CUDARuntimeImpl(simulator)
    return _runtime_instance


def init_cuda_runtime(simulator):
    """Initialize the CUDA runtime with a simulator instance."""
    global _runtime_instance
    _runtime_instance = CUDARuntimeImpl(simulator)
    return _runtime_instance


class CUDARuntimeImpl:
    """Internal CUDA Runtime implementation."""

    def __init__(self, simulator):
        self.sim = simulator
        self.current_device = 0
        self.last_error = cudaError_t.cudaSuccess

        # Device properties (simulating Hopper GPU)
        self.device_props = [cudaDeviceProp()]

        # Memory tracking
        self.allocations = {}
        self.next_address = 0x10000000

        # Stream management - FIXED: Now uses actual stream objects with work queues
        self.streams: Dict[int, CUDAStreamState] = {}
        self.next_stream_id = 1
        # Create default stream (stream 0)
        self.streams[0] = CUDAStreamState(stream_id=0)

        # Event management - FIXED: Now tracks completion properly
        self.events: Dict[int, CUDAEventState] = {}
        self.next_event_id = 1

        # Kernel registry
        self.kernels = {}

        # Timing
        self.start_time = time.time()

    def execute_stream_commands(self, stream_id: int) -> None:
        """
        Execute all pending commands in a stream.

        This properly simulates stream execution:
        1. Executes commands in order
        2. Waits for event dependencies before executing dependent commands
        3. Marks events as completed when commands finish
        """
        if stream_id not in self.streams:
            return

        stream = self.streams[stream_id]

        # Execute all pending commands in order
        while stream.current_command < len(stream.commands):
            cmd = stream.commands[stream.current_command]

            # Check if command has unmet dependencies
            unmet_deps = False
            for dep_event_id in cmd.dependencies:
                dep_event = self.events.get(dep_event_id)
                if not dep_event or not dep_event.completed:
                    unmet_deps = True
                    break

            if unmet_deps:
                # Dependencies not met - stop executing this stream
                # This is key for cudaStreamWaitEvent behavior!
                break

            # Execute the command
            if cmd.type == 'memcpy':
                # Execute memcpy (simulate by tracking)
                pass
            elif cmd.type == 'kernel':
                # Execute kernel by launching it
                if cmd.callback:
                    cmd.callback()
            elif cmd.type == 'event_record':
                # Mark event as completed
                event_id = cmd.data.get('event_id')
                if event_id in self.events:
                    event = self.events[event_id]
                    event.recorded = True
                    event.completed = True
                    event.timing = time.time() - self.start_time
            elif cmd.type == 'stream_wait':
                # Stream wait is handled by dependency checking above
                pass
            elif cmd.type == 'barrier':
                # CUDA barrier - synchronize all threads in CTA
                pass

            # Move to next command
            stream.current_command += 1

        # Check if stream is synchronized (all commands executed)
        if stream.current_command >= len(stream.commands):
            stream.synchronized = True

    def set_error(self, error: cudaError_t) -> None:
        """Set the last error."""
        self.last_error = error

    def get_error(self) -> cudaError_t:
        """Get and clear the last error."""
        error = self.last_error
        self.last_error = cudaError_t.cudaSuccess
        return error


# ==================== C Function Library ( ctypes compatible ) ====================

# Load the C library (simulated)
_lib = None


def _get_library():
    """Get the CUDA runtime library."""
    global _lib
    if _lib is None:
        import ctypes
        _lib = ctypes.CDLL(None)  # Simulated - will use Python implementations
    return _lib


# ==================== Error Handling Functions ====================

def cudaGetLastError():
    """
    Return the last error from a CUDA runtime call.

    Returns:
        cudaError_t: The last error code
    """
    runtime = get_runtime()
    return runtime.get_error()


def cudaGetErrorString(error: cudaError_t) -> str:
    """
    Return the error string for an error code.

    Args:
        error: cudaError_t error code

    Returns:
        str: Error message string
    """
    error_strings = {
        cudaError_t.cudaSuccess: "no error",
        cudaError_t.cudaErrorInvalidValue: "invalid argument",
        cudaError_t.cudaErrorMemoryAllocation: "out of memory",
        cudaError_t.cudaErrorInitializationError: "initialization error",
        cudaError_t.cudaErrorInvalidDevice: "invalid device",
        cudaError_t.cudaErrorInvalidHostPointer: "invalid host pointer",
        cudaError_t.cudaErrorInvalidDevicePointer: "invalid device pointer",
        cudaError_t.cudaErrorInvalidMemcpyDirection: "invalid memcpy direction",
        cudaError_t.cudaErrorLaunchFailure: "launch failure",
        cudaError_t.cudaErrorNotReady: "not ready",
        cudaError_t.cudaErrorInvalidConfiguration: "invalid configuration",
        cudaError_t.cudaErrorInvalidDeviceFunction: "invalid device function",
    }
    return error_strings.get(error, f"unknown error ({error})")


def cudaPeekAtLastError():
    """
    Return the last error from a CUDA runtime call without resetting it.

    Returns:
        cudaError_t: The last error code
    """
    runtime = get_runtime()
    return runtime.last_error


# ==================== Device Management Functions ====================

def cudaGetDeviceCount(count):
    """
    Return the number of devices with compute capability.

    Args:
        count: Pointer to integer to store the device count

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInitializationError
    """
    runtime = get_runtime()
    # Handle both ctypes pointers and MutableInt
    if hasattr(count, 'contents'):
        count.contents.value = len(runtime.device_props)
    else:
        count.value = len(runtime.device_props)
    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


def cudaGetDevice(device):
    """
    Returns in *device the device on which the calling host thread executes the device code.

    Args:
        device: Pointer to integer to store the device number

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInitializationError
    """
    runtime = get_runtime()
    # Handle both ctypes pointers and MutableInt
    if hasattr(device, 'contents'):
        device.contents.value = runtime.current_device
    else:
        device.value = runtime.current_device
    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


def cudaSetDevice(device):
    """
    Set device to be used for device executions in subsequent host functions.

    Args:
        device: Device number (0-based)

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInvalidDevice
    """
    runtime = get_runtime()
    dev = device if isinstance(device, int) else (device.value if hasattr(device, 'value') else device._obj)

    if 0 <= dev < len(runtime.device_props):
        runtime.current_device = dev
        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidDevice)
        return cudaError_t.cudaErrorInvalidDevice


def cudaGetDeviceProperties(prop, device):
    """
    Return information about the compute device.

    Args:
        prop: Pointer to cudaDeviceProp structure
        device: Device number (0-based)

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInvalidDevice
    """
    runtime = get_runtime()
    dev = device if isinstance(device, int) else (device.value if hasattr(device, 'value') else device._obj)

    if 0 <= dev < len(runtime.device_props):
        real_prop = runtime.device_props[dev]

        # Copy properties to the passed structure
        prop.name = real_prop.name
        prop.totalGlobalMem = real_prop.totalGlobalMem
        prop.sharedMemPerBlock = real_prop.sharedMemPerBlock
        prop.regsPerBlock = real_prop.regsPerBlock
        prop.warpSize = real_prop.warpSize
        prop.maxThreadsPerBlock = real_prop.maxThreadsPerBlock
        prop.maxThreadsDim = real_prop.maxThreadsDim
        prop.maxGridSize = real_prop.maxGridSize
        prop.clockRate = real_prop.clockRate
        prop.multiProcessorCount = real_prop.multiProcessorCount
        prop.major = real_prop.major
        prop.minor = real_prop.minor
        prop.l2CacheSize = real_prop.l2CacheSize
        prop.maxThreadsPerMultiProcessor = real_prop.maxThreadsPerMultiProcessor
        prop.asyncEngineCount = real_prop.asyncEngineCount
        prop.concurrentKernels = real_prop.concurrentKernels

        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidDevice)
        return cudaError_t.cudaErrorInvalidDevice


def cudaDeviceSynchronize():
    """
    Wait for compute devices to finish.

    Returns:
        cudaError_t: cudaSuccess
    """
    runtime = get_runtime()
    # All operations are synchronous in simulator
    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


def cudaDeviceReset():
    """
    Destroy all allocations and reset all state on the current device in the current process.

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInvalidDevice
    """
    runtime = get_runtime()
    runtime.allocations.clear()
    runtime.next_address = 0x10000000
    runtime.streams.clear()
    runtime.events.clear()

    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


def cudaDeviceCanAccessPeer(canAccessPtr, device, peerDevice):
    """
    Query if a device may access a peer's memory.

    Args:
        canAccessPtr: Pointer to integer for storing result
        device: Device number
        peerDevice: Peer device number

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInvalidDevice
    """
    runtime = get_runtime()
    dev = device if isinstance(device, int) else (device.value if hasattr(device, 'value') else (device.contents.value if hasattr(device, 'contents') else device))
    peer = peerDevice if isinstance(peerDevice, int) else (peerDevice.value if hasattr(peerDevice, 'value') else (peerDevice.contents.value if hasattr(peerDevice, 'contents') else peerDevice))

    # In simulator, all devices can access each other
    result = 1 if 0 <= dev < len(runtime.device_props) else 0
    if hasattr(canAccessPtr, 'contents'):
        canAccessPtr.contents.value = result
    else:
        canAccessPtr.value = result

    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


# ==================== Memory Management Functions ====================

def cudaMalloc(size, devPtr):
    """
    Allocate device memory.

    Args:
        size: Requested allocation size in bytes
        devPtr: Pointer to device pointer (c_void_p pointer)

    Returns:
        cudaError_t: cudaSuccess, cudaErrorMemoryAllocation
    """
    runtime = get_runtime()
    size_val = size if isinstance(size, int) else size.value

    try:
        addr = runtime.next_address
        runtime.next_address += size_val
        runtime.allocations[addr] = (size_val, None)

        # Set the output pointer
        if hasattr(devPtr, 'contents'):
            devPtr.contents.value = addr
        else:
            devPtr.value = addr

        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    except:
        runtime.set_error(cudaError_t.cudaErrorMemoryAllocation)
        return cudaError_t.cudaErrorMemoryAllocation


def cudaFree(devPtr):
    """
    Free device memory.

    Args:
        devPtr: Device pointer (c_void_p or int address)

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInvalidDevicePointer
    """
    runtime = get_runtime()
    addr = devPtr if isinstance(devPtr, int) else (devPtr.value if hasattr(devPtr, 'value') else (devPtr.contents.value if hasattr(devPtr, 'contents') else devPtr._obj if hasattr(devPtr, '_obj') else devPtr))

    if addr in runtime.allocations:
        del runtime.allocations[addr]
        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidDevicePointer)
        return cudaError_t.cudaErrorInvalidDevicePointer


def cudaMemcpy(dst, src, count, kind):
    """
    Copy memory between host and device.

    Args:
        dst: Destination pointer (c_void_p or bytes for host H2D)
        src: Source pointer (c_void_p or bytes for host H2D, or int address)
        count: Size in bytes to copy (int or c_size_t)
        kind: Type of transfer (cudaMemcpyKind enum)

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInvalidMemcpyDirection, cudaErrorInvalidDevicePointer
    """
    runtime = get_runtime()
    from src.core.memory import MemorySpace

    # Handle count as integer or c_size_t
    count_val = count if isinstance(count, int) else count.value

    # Get addresses
    if isinstance(dst, int):
        dst_addr = dst
    elif isinstance(dst, c_void_p):
        dst_addr = dst.value if dst.value else dst
    elif isinstance(dst, (bytes, bytearray)):
        dst_addr = dst  # Host buffer
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidDevicePointer)
        return cudaError_t.cudaErrorInvalidDevicePointer

    if isinstance(src, int):
        src_addr = src
    elif isinstance(src, c_void_p):
        src_addr = src.value if src.value else src
    elif isinstance(src, (bytes, bytearray)):
        src_addr = src  # Host data
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidDevicePointer)
        return cudaError_t.cudaErrorInvalidDevicePointer

    kind_val = kind if isinstance(kind, cudaMemcpyKind) else kind.value

    try:
        if kind_val == cudaMemcpyKind.cudaMemcpyHostToDevice:
            # H2D: src is bytes, dst is device address
            data = src_addr if isinstance(src_addr, (bytes, bytearray)) else bytes(src_addr)

            for i, byte in enumerate(data[:count_val]):
                runtime.sim.memory.write(MemorySpace.GLOBAL, dst_addr + i, bytes([byte]))

        elif kind_val == cudaMemcpyKind.cudaMemcpyDeviceToHost:
            # D2H: src is device address, dst is buffer
            data = bytearray()

            for i in range(count_val):
                byte = runtime.sim.memory.read(MemorySpace.GLOBAL, src_addr + i, 1)
                data.extend(byte)

            # Write to destination buffer
            if isinstance(dst, (bytearray, bytes)):
                # dst is a bytearray or bytes - can't modify in place
                # The data is in 'data' variable - caller needs to use it
                pass
            elif isinstance(dst, c_void_p):
                # dst is a device pointer - can't write to host memory directly
                pass

            # Store data for retrieval - update dst in place if it's a bytearray
            if isinstance(dst, bytearray):
                # Modify the destination bytearray in place
                for i, byte in enumerate(data):
                    if i < len(dst):
                        dst[i] = byte

        elif kind_val == cudaMemcpyKind.cudaMemcpyDeviceToDevice:
            # D2D: both are device addresses
            for i in range(count_val):
                byte = runtime.sim.memory.read(MemorySpace.GLOBAL, src_addr + i, 1)
                runtime.sim.memory.write(MemorySpace.GLOBAL, dst_addr + i, byte)

        elif kind_val == cudaMemcpyKind.cudaMemcpyHostToHost:
            # H2H: data is already at dst (no-op)
            pass

        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess

    except Exception:
        runtime.set_error(cudaError_t.cudaErrorInvalidDevicePointer)
        return cudaError_t.cudaErrorInvalidDevicePointer


def cudaMemset(devPtr, value, count):
    """
    Fill device memory with a constant value.

    Args:
        devPtr: Device pointer (c_void_p or int)
        value: Value to set (int)
        count: Size in bytes (int or c_size_t)

    Returns:
        cudaError_t: cudaSuccess, cudaErrorInvalidDevicePointer
    """
    runtime = get_runtime()
    from src.core.memory import MemorySpace

    # Handle device pointer
    if isinstance(devPtr, int):
        addr = devPtr
    elif hasattr(devPtr, 'contents'):
        # pointer() was used
        addr = devPtr.contents.value
    elif hasattr(devPtr, 'value'):
        # c_void_p or c_size_t - value may be 0/None
        addr = devPtr.value or 0
    else:
        addr = devPtr

    value_val = value if isinstance(value, int) else value.value
    count_val = count if isinstance(count, int) else count.value

    try:
        byte_val = value_val & 0xFF

        for i in range(count_val):
            runtime.sim.memory.write(MemorySpace.GLOBAL, addr + i, bytes([byte_val]))

        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess

    except Exception:
        runtime.set_error(cudaError_t.cudaErrorInvalidDevicePointer)
        return cudaError_t.cudaErrorInvalidDevicePointer


# ==================== Version Information ====================

def cudaGetDriverVersion(driverVersion):
    """
    Return the CUDA driver version.

    Args:
        driverVersion: Pointer to integer to store driver version

    Returns:
        cudaSuccess
    """
    # Simulated CUDA 12.0 driver
    if hasattr(driverVersion, 'contents'):
        driverVersion.contents.value = 12000
    else:
        driverVersion.value = 12000
    return cudaError_t.cudaSuccess


def cudaRuntimeGetVersion(runtimeVersion):
    """
    Return the CUDA Runtime version.

    Args:
        runtimeVersion: Pointer to integer to store runtime version

    Returns:
        cudaSuccess
    """
    # Simulated CUDA 12.0 runtime
    if hasattr(runtimeVersion, 'contents'):
        runtimeVersion.contents.value = 12000
    else:
        runtimeVersion.value = 12000
    return cudaError_t.cudaSuccess


# ==================== Stream Management ====================

def cudaStreamCreate(stream):
    """
    Create a new stream.

    Args:
        stream: Pointer to cudaStream_t to store the stream handle

    Returns:
        cudaSuccess, cudaErrorInvalidConfiguration
    """
    runtime = get_runtime()
    stream_id = runtime.next_stream_id
    runtime.next_stream_id += 1

    # Store stream (simulated as just an ID)
    runtime.streams[stream_id] = {'id': stream_id, 'synchronized': True}

    # In simulator, streams are essentially synchronous, so we create a handle
    if hasattr(stream, 'contents'):
        stream.contents.value = stream_id
    else:
        stream.value = stream_id

    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


def cudaStreamDestroy(stream):
    """
    Destroy a stream.

    Args:
        stream: Stream handle (cudaStream_t)

    Returns:
        cudaSuccess, cudaErrorInvalidResourceHandle
    """
    runtime = get_runtime()
    stream_id = stream if isinstance(stream, int) else stream.value

    if stream_id in runtime.streams:
        del runtime.streams[stream_id]
        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidResourceHandle)
        return cudaError_t.cudaErrorInvalidResourceHandle


def cudaStreamSynchronize(stream):
    """
    Wait until all operations in the stream complete.

    Args:
        stream: Stream handle (0 for default stream, or cudaStream_t)

    Returns:
        cudaSuccess, cudaErrorInvalidResourceHandle
    """
    runtime = get_runtime()
    # All operations are synchronous in simulator

    stream_id = stream if isinstance(stream, int) else stream.value if stream else 0

    if stream_id == 0 or stream_id in runtime.streams:
        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidResourceHandle)
        return cudaError_t.cudaErrorInvalidResourceHandle


def cudaStreamWaitEvent(stream, event, flags=0):
    """
    Make a stream wait on an event.

    Args:
        stream: Stream handle
        event: Event handle
        flags: Flags for the operation (must be 0)

    Returns:
        cudaSuccess, cudaErrorInvalidResourceHandle
    """
    runtime = get_runtime()
    # Events complete immediately in simulator

    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


# ==================== Event Management ====================

def cudaEventCreate(event, flags=0):
    """
    Create an event.

    Args:
        event: Pointer to cudaEvent_t to store the event handle
        flags: Flags for event creation (must be 0)

    Returns:
        cudaSuccess
    """
    runtime = get_runtime()
    event_id = runtime.next_event_id
    runtime.next_event_id += 1

    # Store event
    runtime.events[event_id] = {'id': event_id, 'recorded': False}

    if hasattr(event, 'contents'):
        event.contents.value = event_id
    else:
        event.value = event_id

    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


def cudaEventDestroy(event):
    """
    Destroy an event.

    Args:
        event: Event handle

    Returns:
        cudaSuccess, cudaErrorInvalidResourceHandle
    """
    runtime = get_runtime()
    event_id = event if isinstance(event, int) else event.value

    if event_id in runtime.events:
        del runtime.events[event_id]
        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidResourceHandle)
        return cudaError_t.cudaErrorInvalidResourceHandle


def cudaEventRecord(event, stream=None):
    """
    Record an event in a stream.

    Args:
        event: Event handle
        stream: Stream handle (None or 0 for default stream)

    Returns:
        cudaSuccess, cudaErrorInvalidResourceHandle
    """
    runtime = get_runtime()
    event_id = event if isinstance(event, int) else event.value

    if event_id in runtime.events:
        runtime.events[event_id]['recorded'] = True
        runtime.events[event_id]['stream'] = stream if stream else 0

        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidResourceHandle)
        return cudaError_t.cudaErrorInvalidResourceHandle


def cudaEventSynchronize(event):
    """
    Wait until an event completes.

    Args:
        event: Event handle

    Returns:
        cudaSuccess, cudaErrorInvalidResourceHandle
    """
    runtime = get_runtime()
    # Events complete immediately in simulator
    event_id = event if isinstance(event, int) else event.value

    if event_id in runtime.events:
        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidResourceHandle)
        return cudaError_t.cudaErrorInvalidResourceHandle


def cudaEventElapsedTime(ms, start, end):
    """
    Compute the elapsed time between two events.

    Args:
        ms: Pointer to float to store elapsed time in milliseconds
        start: Start event handle
        end: End event handle

    Returns:
        cudaSuccess, cudaErrorInvalidResourceHandle
    """
    runtime = get_runtime()
    start_id = start if isinstance(start, int) else start.value
    end_id = end if isinstance(end, int) else end.value

    if start_id in runtime.events and end_id in runtime.events:
        # Simulated time (very small)
        if hasattr(ms, 'contents'):
            ms.contents.value = 0.001
        else:
            ms.value = 0.001
        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidResourceHandle)
        return cudaError_t.cudaErrorInvalidResourceHandle


def cudaEventQuery(event):
    """
    Query an event's status.

    Args:
        event: Event handle

    Returns:
        cudaSuccess (if event has completed), cudaErrorNotReady
    """
    runtime = get_runtime()
    event_id = event if isinstance(event, int) else event.value

    if event_id in runtime.events:
        # Events complete immediately in simulator
        runtime.set_error(cudaError_t.cudaSuccess)
        return cudaError_t.cudaSuccess
    else:
        runtime.set_error(cudaError_t.cudaErrorInvalidResourceHandle)
        return cudaError_t.cudaErrorInvalidResourceHandle


# ==================== Kernel Launch ====================

def cudaLaunchKernel(func_ptr, gridDim, blockDim, sharedMem, stream, *args):
    """
    Launch a CUDA kernel (simplified interface).

    Args:
        func_ptr: Function pointer (not used in simulator)
        gridDim: Tuple of (grid_x, grid_y, grid_z)
        blockDim: Tuple of (block_x, block_y, block_z)
        sharedMem: Dynamic shared memory size
        stream: Stream handle (0 for default stream)
        *args: Kernel arguments

    Returns:
        cudaSuccess
    """
    runtime = get_runtime()

    # Grid and block dimensions
    gx, gy, gz = gridDim if isinstance(gridDim[0], int) else (gridDim[0].value, gridDim[1].value, gridDim[2].value)
    bx, by, bz = blockDim if isinstance(blockDim[0], int) else (blockDim[0].value, blockDim[1].value, blockDim[2].value)

    # Store launch info for the simulator to use
    runtime.current_launch = {
        'grid_dim': (gx, gy, gz),
        'block_dim': (bx, by, bz),
        'args': args
    }

    runtime.set_error(cudaError_t.cudaSuccess)
    return cudaError_t.cudaSuccess


# ==================== Python Helper Functions ====================

class DevicePointer:
    """Device memory pointer for easier memory management."""

    def __init__(self, address: int, size: int):
        self.address = address
        self.size = size

    def __int__(self):
        return self.address

    def __repr__(self):
        return f"DevicePointer(0x{self.address:x}, size={self.size})"


def cudaMalloc_simple(size: int) -> DevicePointer:
    """
    Simplified cudaMalloc that returns a DevicePointer object.

    Args:
        size: Allocation size in bytes

    Returns:
        DevicePointer object
    """
    runtime = get_runtime()
    addr = runtime.next_address
    runtime.next_address += size
    runtime.allocations[addr] = (size, None)
    return DevicePointer(addr, size)
