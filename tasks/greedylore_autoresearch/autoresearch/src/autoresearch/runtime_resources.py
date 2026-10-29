"""Bounded host-only resource declarations shared by prompts and sandbox mounts.

Only trusted run configuration may supply declarations. Target roots authorize
whole dedicated read-only trees; discovering a link selects its declared tree,
which is also inspected before mounting. No candidate-controlled mount requests.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from .runtime_artifacts import safe_path


def _root(value):
    if not isinstance(value, (str, Path)):
        raise ValueError('resource root must be a path')
    path = Path(value)
    if (not path.is_absolute() or '..' in path.parts or len(path.parts) < 3
            or path in {Path('/dev/shm'), Path.home()}
            or path.parts[:2] == ('/', 'home') and len(path.parts) <= 3
            or any(part in {'private', '.ssh', '.aws', '.codex'} for part in path.parts)):
        raise ValueError('resource root is private or too broad')
    safe_path(path)
    if not path.exists() or not (path.is_dir() or path.is_file()):
        raise ValueError(f'resource root is missing or not regular: {path}')
    return path


def resolve_resources(declarations, *, runtime_roots=(), max_entries=4096, max_depth=64):
    """Return a JSON catalog once; its mounts and descriptions travel together.

    Entry/depth limits cover the whole catalog, including selected target trees.
    Existing runtime-library roots remain direct mounts and are not traversed.
    """
    if not isinstance(declarations, list) or len(declarations) > 64:
        raise ValueError('resource declarations must be a bounded list')
    mounts, resources, identities = [], [], set()
    entries = 0
    for declaration in declarations:
        if (not isinstance(declaration, dict) or not {'id', 'path'} <= declaration.keys()
                or declaration.keys() - {'id', 'path', 'target_roots', 'view'}):
            raise ValueError('invalid resource declaration')
        identity = declaration['id']
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identity) or identity in identities:
            raise ValueError('invalid or duplicate resource id')
        identities.add(identity)
        source = _root(declaration['path'])
        targets = declaration.get('target_roots', [])
        if not isinstance(targets, list) or len(targets) > 64:
            raise ValueError('target_roots must be a bounded list')
        allowed = [source, *(_root(value) for value in targets)]
        selected, visited = {source}, set()

        def authorize(path):
            matches = [root for root in allowed if path.is_relative_to(root)]
            if not matches:
                raise ValueError(f'resource link target is not explicitly authorized: {path}')
            selected.add(max(matches, key=lambda root: len(root.parts)))

        def resolve_link(path):
            # Inspect every hop, including intermediate directory links; never
            # use resolve() to jump over an unauthorized intermediate target.
            hops = 0
            while True:
                link = next((part for part in (*reversed(path.parents), path) if part.is_symlink()), None)
                if link is None:
                    return path
                hops += 1
                if hops > 40:
                    raise ValueError('resource link loop or hop budget exceeded')
                target = Path(os.readlink(link))
                # Collapsing '..' after a directory link changes POSIX meaning.
                # Reject that ambiguous spelling instead of authorizing its
                # lexical decoy while the kernel follows a different target.
                prefix = Path('/') if target.is_absolute() else link.parent
                linked_prefix = False
                for component in target.parts:
                    if component == '..':
                        if linked_prefix:
                            raise ValueError('resource parent traversal follows a directory link')
                        # POSIX must enter this directory before processing '..'.
                        # Lexical normalization cannot erase a missing/file component.
                        if not prefix.is_dir():
                            raise ValueError('resource parent traversal requires an existing directory')
                        prefix = prefix.parent
                    else:
                        prefix /= component
                        linked_prefix |= prefix.is_symlink()
                if (declaration.get('view') and link.is_relative_to(source) and not target.is_absolute()
                        and not Path(os.path.normpath(link.parent / target)).is_relative_to(source)):
                    raise ValueError('stable resource view cannot preserve an escaping relative link; '
                                     'declare its shared parent root or omit view')
                target = target if target.is_absolute() else link.parent / target
                target = Path(os.path.normpath(target))
                authorize(target)
                path = target / path.relative_to(link)
                authorize(path)

        def walk(path, ancestors=(), depth=0):
            nonlocal entries
            entries += 1
            if entries > max_entries or depth > max_depth:
                raise ValueError('resource discovery budget exceeded')
            path = resolve_link(path)
            if path in ancestors:
                raise ValueError('resource directory link loop')
            if path in visited:
                return
            visited.add(path)
            try:
                info = path.stat()
                if stat.S_ISDIR(info.st_mode):
                    with os.scandir(path) as children:
                        for child in children:
                            walk(Path(child.path), (*ancestors, path), depth + 1)
                elif not stat.S_ISREG(info.st_mode):
                    raise ValueError('resource contains a non-regular entry')
            except OSError as exc:
                raise ValueError(f'resource target missing or unreadable: {path}') from exc

        walked = set()
        while selected - walked:
            root = min(selected - walked)
            walk(root)
            walked.add(root)
        # A view may never shadow system/runtime paths or the workspace.
        view = declaration.get('view')
        if view is not None and view != f'/resources/{identity}':
            raise ValueError('resource view must be /resources/<id>')
        mounts.extend((str(root), str(root)) for root in sorted(selected))
        if view:
            mounts.append((str(source), view))
        resources.append({'id': identity, 'path': str(source), 'view': view or str(source),
                          'target_roots': [str(root) for root in sorted(selected - {source})],
                          'access': 'managed_diagnostic_readonly'})
    mounts.extend((str(Path(root).absolute()), str(Path(root).absolute())) for root in runtime_roots)
    return {'version': 1, 'resources': resources,
            'checkpoint_access': {'metadata': 'agent_readonly', 'researcher': 'host_selected_iteration',
                                  'analyzer': 'host_selected_attempt', 'engineer': 'no_selected_tensors'},
            'mounts': [list(pair) for pair in dict.fromkeys(mounts)]}


def configured_resources(options):
    catalog = resolve_resources(options.get('diagnostic_resources', []),
                                runtime_roots=options.get('diagnostic_runtime_roots', []))
    dynamic = Path(options.get('dynamic_artifact_root', '/dev/shm/wyr_tmp')).expanduser()
    for source, _ in catalog['mounts']:
        source = Path(source)
        for owned in (dynamic / 'coral', dynamic / 'tmp'):
            if source.is_relative_to(owned) or owned.is_relative_to(source):
                raise ValueError('resource declaration exposes a managed artifact namespace')
    return catalog


def describe_resources(catalog):
    lines = ['Declared resources (read-only in managed diagnostics; original stream paths preserved):']
    for row in catalog['resources']:
        lines.append(f"- {row['id']}: {row['view']} (original: {row['path']}; {row['access']})")
    return '\n'.join(lines)


def publish_catalog(toolkit, options):
    """Publish at session preparation, never on status/evidence reads."""
    import json

    from .atomic import atomic_write_text

    target = safe_path(Path(toolkit) / 'runtime_resources.json')
    try:
        catalog = configured_resources(options)
    except (ValueError, OSError) as exc:
        # This is public metadata only. Execution independently resolves the
        # declarations and still fails closed when a resource is unavailable.
        catalog = {'version': 1, 'status': 'unavailable', 'error': str(exc)}
    return atomic_write_text(target, json.dumps(catalog, indent=2) + '\n', mode=0o444)
