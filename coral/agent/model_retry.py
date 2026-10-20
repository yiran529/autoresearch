"""Bounded connection recovery for standalone model subprocesses."""

import logging
import time

logger = logging.getLogger(__name__)


class ModelConnectionError(RuntimeError):
    """A model subprocess exited with a transient connection failure."""


def check_connection_failure(result) -> None:
    if not result.returncode:
        return
    text = (result.stderr or '').strip().lower()
    # Auth/config errors take precedence over incidental transport warnings.
    if any(value in text for value in (
        'unauthorized', 'invalid api key', 'authentication failed', 'forbidden',
        'invalid configuration', 'error loading config',
    )):
        return
    # Do not mistake an earlier MCP/tool warning for the model's terminal error.
    errors = [line for line in text.splitlines() if line.startswith('error:')]
    terminal = errors[-1] if errors else (text.splitlines()[-1] if text else '')
    if any(value in terminal for value in (
        'workspace routing discovery failed', 'request timed out',
        'connection reset', 'connection refused', 'error sending request',
        'stream disconnected before completion',
    )):
        raise ModelConnectionError((result.stderr or '').strip()[-2000:])


def retry_model_connection(invoke, *, timeout):
    """At most three launches, sharing one deadline; caller owns process cleanup."""
    deadline = time.monotonic() + timeout
    for attempt in range(1, 4):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('model connection recovery exceeded overall timeout')
        try:
            return invoke(remaining, attempt)
        except ModelConnectionError:
            delay = 15 * attempt
            if attempt == 3 or deadline - time.monotonic() <= delay:
                raise
            logger.warning('Model connection failed; retrying launch %s/3 in %ss', attempt + 1, delay)
            time.sleep(delay)
