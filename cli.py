#!/usr/bin/env python3
"""
CLI Interface for Hopper GPU Simulator

Command-line interface for running GPU simulations.
"""

import argparse
import sys
from pathlib import Path

from src.simulator import HopperSimulator, SimulatorConfig, SimulationResult
from src.isa.decoder import parse_program


def print_result(result: SimulationResult, verbose: bool = False) -> None:
    """Print simulation results."""
    if result.success:
        print(f"\n{'='*50}")
        print("Simulation completed successfully")
        print(f"{'='*50}")
        print(f"  Cycles:            {result.cycles}")
        print(f"  Instructions:      {result.instructions_executed}")
        print(f"  IPC:               {result.instructions_executed / max(result.cycles, 1):.2f}")

        if verbose and result.warp_stats:
            print(f"\nWarp Statistics:")
            for warp_id, stats in result.warp_stats.items():
                print(f"  Warp {warp_id}:")
                for key, val in stats.items():
                    print(f"    {key}: {val}")
    else:
        print(f"\nSimulation failed: {result.error}")


def run_file(filename: str, args: argparse.Namespace) -> None:
    """Run simulation from file."""
    print(f"Loading program from: {filename}")

    # Create simulator with config
    config = SimulatorConfig(
        num_sms=args.sms,
        warps_per_sm=args.warps,
        threads_per_warp=32,  # Always 32 for NVIDIA
        global_mem_size=args.mem * 1024 * 1024,
        max_cycles=args.max_cycles,
    )

    sim = HopperSimulator(config)

    # Load program
    try:
        sim.load_program_from_file(filename, warp_id=0)
    except Exception as e:
        print(f"Error loading program: {e}", file=sys.stderr)
        sys.exit(1)

    # Run simulation
    print(f"\nRunning simulation...")
    result = sim.run()

    # Print results
    print_result(result, verbose=args.verbose)

    # Print register dump if requested
    if args.dump_regs:
        print("\nRegister Dump (Warp 0, Lane 0):")
        for reg_idx in range(16):
            val = sim.read_register(0, 0, reg_idx)
            if val != 0:
                print(f"  R{reg_idx}: {val:#x} ({val})")


def run_interactive(args: argparse.Namespace) -> None:
    """Run interactive simulator shell."""
    print("Hopper GPU Simulator - Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit")

    config = SimulatorConfig(
        num_sms=args.sms,
        warps_per_sm=args.warps,
        max_cycles=args.max_cycles,
    )

    sim = HopperSimulator(config)

    while True:
        try:
            line = input("\nhopper> ").strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd == 'quit' or cmd == 'exit':
                break
            elif cmd == 'help':
                print_help()
            elif cmd == 'load':
                if len(parts) < 2:
                    print("Usage: load <filename>")
                else:
                    sim.load_program_from_file(parts[1])
                    print(f"Loaded: {parts[1]}")
            elif cmd == 'run':
                result = sim.run()
                print_result(result, verbose=True)
            elif cmd == 'rd':
                if len(parts) < 3:
                    print("Usage: rd <warp> <lane> <reg>")
                else:
                    val = sim.read_register(int(parts[1]), int(parts[2]), int(parts[3]))
                    print(f"R{parts[3]} = {val:#x}")
            elif cmd == 'wd':
                if len(parts) < 4:
                    print("Usage: wd <warp> <lane> <reg> <value>")
                else:
                    sim.write_register(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            elif cmd == 'rm':
                if len(parts) < 2:
                    print("Usage: rm <address> [size]")
                else:
                    addr = int(parts[1], 0)
                    size = int(parts[2]) if len(parts) > 2 else 4
                    data = sim.read_memory(addr, size)
                    print(f"Memory@{addr:#x}: {data.hex()}")
            elif cmd == 'wm':
                if len(parts) < 3:
                    print("Usage: wm <address> <value>")
                else:
                    addr = int(parts[1], 0)
                    val = int(parts[2], 0)
                    sim.write_memory(addr, val.to_bytes(4, byteorder='little'))
            elif cmd == 'reset':
                sim.reset()
                print("Simulator reset")
            elif cmd == 'state':
                state = sim.get_state()
                print(f"State: {state}")
            else:
                print(f"Unknown command: {cmd}")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")


def print_help() -> None:
    """Print interactive help."""
    print("""
Commands:
  load <file>     Load program from file
  run             Run the loaded program
  rd <w> <l> <r>  Read register (warp, lane, register)
  wd <w> <l> <r> <v>  Write register
  rm <addr> [n]   Read memory (address, bytes)
  wm <addr> <v>   Write memory (32-bit value)
  reset           Reset simulator
  state           Show current state
  help            Show this help
  quit/exit       Exit simulator
    """)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hopper GPU Simulator - Instruction-level GPU simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s program.sass              Run program from file
  %(prog)s program.sass -v           Run with verbose output
  %(prog)s program.sass --dump-regs  Dump registers after execution
  %(prog)s -i                        Interactive mode

Assembly Format:
  Instructions are in Hopper SASS format:
  IADD R1, R2, R3          # Integer add
  FADD R1, R2, R3          # Float add
  FFMA R1, R2, R3, R4      # Fused multiply-add
  LDG R1, [R2+0x10]        # Load from global memory
  STG [R1], R2             # Store to global memory
  HMMA R1, R2, R3, R4      # Tensor Core operation
  BRA 0x100                # Branch
  EXIT                     # Exit kernel
        """
    )

    parser.add_argument('file', nargs='?', help='Assembly file to execute')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--dump-regs', action='store_true',
                        help='Dump registers after execution')
    parser.add_argument('--sms', type=int, default=1,
                        help='Number of SMs (default: 1)')
    parser.add_argument('--warps', type=int, default=1,
                        help='Warps per SM (default: 1)')
    parser.add_argument('--mem', type=int, default=1024,
                        help='Global memory size in MB (default: 1024)')
    parser.add_argument('--max-cycles', type=int, default=10000,
                        help='Maximum execution cycles (default: 10000)')

    args = parser.parse_args()

    if args.interactive:
        run_interactive(args)
    elif args.file:
        run_file(args.file, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
