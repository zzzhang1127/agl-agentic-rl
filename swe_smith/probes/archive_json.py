"""Read immutable JSON archives across atomic plain-to-gzip replacement."""
import gzip
import json
from pathlib import Path


def iter_json_archives(root):
    root = Path(root)
    paths = {p.name: p for p in root.glob('*.json')}
    for path in root.glob('*.json.gz'):
        paths.setdefault(path.name[:-3], path)
    return [paths[name] for name in sorted(paths)]


def read_json_archive(path):
    path = Path(path)
    try:
        stream = gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8')
    except FileNotFoundError:
        if path.suffix == '.gz':
            raise
        # Compressor publishes verified gzip before unlinking the plain file.
        stream = gzip.open(path.with_name(path.name + '.gz'), 'rt', encoding='utf-8')
    with stream:
        return json.load(stream)
