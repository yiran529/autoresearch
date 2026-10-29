"""Dependency-free limits and newline framing for trusted-write clients."""

import json

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
PROTOCOL_VERSION = 1


class FrameError(ValueError):
    """A trusted channel frame is missing, incomplete, or too large."""


def read_frame(connection, limit, label):
    raw = bytearray()
    while True:
        chunk = connection.recv(min(4096, limit + 1 - len(raw)))
        if not chunk:
            raise FrameError(f'empty trusted {label}' if not raw else f'truncated trusted {label}')
        raw.extend(chunk)
        newline = raw.find(b'\n')
        if newline >= 0:
            if newline + 1 > limit:
                break
            return bytes(raw[:newline])
        if len(raw) > limit:
            break
    unit = '64 KiB' if limit == MAX_REQUEST_BYTES else '4 MiB'
    raise FrameError(f'trusted {label} exceeds {unit}')


def parse_response(raw):
    """Validate a response envelope before treating it as success or rejection."""
    try:
        response = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise FrameError('invalid trusted response') from exc
    if not isinstance(response, dict) or type(response.get('ok')) is not bool:
        raise FrameError('invalid trusted response')
    if response['ok']:
        if not isinstance(response.get('result'), dict):
            raise FrameError('invalid trusted response')
    elif not isinstance(response.get('error'), str):
        raise FrameError('invalid trusted response')
    return response
