"""Host-owned, durable identity for one Engineer's temporary scripts."""

import re
import secrets
from pathlib import Path


def scratch_dir(worker, agent_id):
    identity = worker.get('scratch_identity')
    if not isinstance(identity, str) or not re.fullmatch(r'[0-9a-f]{32}', identity):
        raise ValueError('Engineer scratch identity is invalid; reclaim the Engineering assignment')
    if not re.fullmatch(r'engineer-E[0-9]{4,}', agent_id):
        raise ValueError('invalid Engineer identity')
    return Path('/tmp/coral-engineer') / identity / agent_id


def ensure_scratch_identity(worker):
    if 'scratch_identity' not in worker:
        worker['scratch_identity'] = secrets.token_hex(16)
        return True
    return False
