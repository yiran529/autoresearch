#!/usr/bin/python3
"""Dependency-free Analyzer CLI shim; runs without the private grader environment."""

import argparse
import json
import os
import socket
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser(prog='autoresearch')
    parser.add_argument('command', choices=['diagnostic'])
    parser.add_argument('action', choices=['run'])
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--request-id', default=None)
    arguments = sys.argv[1:]
    if '--' not in arguments:
        parser.error('use -- before the diagnostic command argv')
    separator = arguments.index('--')
    args = parser.parse_args(arguments[:separator])
    payload = {'request_id': args.request_id or uuid.uuid4().hex,
               'gpus': args.gpus, 'timeout': args.timeout,
               'argv': arguments[separator + 1:]}
    # Print before waiting, so a disconnected caller can repeat the same request ID.
    print(f"diagnostic request: {payload['request_id']}", file=sys.stderr, flush=True)
    message = json.dumps({'version': 1, 'operation': 'run_diagnostic', 'payload': payload}).encode() + b'\n'
    if len(message) > 64 * 1024:
        raise SystemExit('diagnostic request exceeds 64 KiB')
    deadline = time.monotonic() + 2
    while True:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(4800)
            client.connect(os.path.join(os.environ['CORAL_DIAGNOSTIC_RUN'], 'public/trusted.sock'))
            client.sendall(message)
            response = bytearray()
            while b'\n' not in response:
                block = client.recv(4096)
                if not block:
                    break
                response.extend(block)
                if len(response) > 64 * 1024:
                    raise SystemExit('diagnostic response exceeds 64 KiB')
        result = json.loads(response)
        if result.get('ok'):
            print(json.dumps(result['result']))
            return
        error = result.get('error', 'diagnostic rejected')
        if 'active Analyzer diagnostic task' not in error or time.monotonic() >= deadline:
            raise SystemExit(error)
        time.sleep(0.05)


if __name__ == '__main__':
    main()
