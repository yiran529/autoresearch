"""Single-node diagnostic torchrun with an IPv4 rendezvous and automatic port selection.

Usable as a module or as /run/coral-ipv4-launch.py inside diagnostic sandboxes.
This helper does not select/reserve GPUs or alter the formal grading recipes.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys


def _port(value):
    number = int(value)
    if not 0 <= number <= 65535:
        raise argparse.ArgumentTypeError('port must be 0 (automatic) or 1..65535')
    return number


def _processes(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('nproc-per-node must be positive')
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--nproc-per-node', type=_processes, default=2)
    result.add_argument('--port', type=_port, default=0, help='0 chooses an available rendezvous port')
    result.add_argument('script', nargs=argparse.REMAINDER, help='script and its arguments, optionally after --')
    return result


def launch_command(args):
    script = args.script[1:] if args.script[:1] == ['--'] else args.script
    if not script:
        raise ValueError('a diagnostic script is required')
    port = args.port
    if port == 0:
        # torchrun's static rendezvous requires the numeric port before launch.
        # Releasing the probe socket leaves a small bind-to-launch race; explicit
        # --port is available when a caller manages concurrent local launches.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
    return [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
            '--nproc-per-node', str(args.nproc_per_node), '--rdzv-backend=static',
            '--master-addr', '127.0.0.1', '--master-port', str(port), *script]


def main():
    arguments = parser()
    try:
        command = launch_command(arguments.parse_args())
    except ValueError as exc:
        arguments.error(str(exc))
    port = command[command.index('--master-port') + 1]
    print(f'IPv4 rendezvous: 127.0.0.1:{port}. If the port is taken, retry with a new '
          'managed diagnostic request or a host-managed --port.', file=sys.stderr, flush=True)
    # exec preserves torchrun exit status and direct signal delivery.
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
