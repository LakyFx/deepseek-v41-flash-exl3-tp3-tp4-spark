"""Validate the balanced cache's receipt, index and native key order at startup."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .repack_weights import Cache, LAYOUT

HERE = Path(__file__).resolve().parent


def validate(directory, rank, deep=False):
    found = list(directory.glob(f'rank{rank}of4-*.bin'))
    if len(found) != 1 or found[0].is_symlink():
        raise ValueError('Require exactly one balanced native-discovery cache')
    path = found[0]
    if not path.name.startswith(f'rank{rank}of4-{LAYOUT}-'):
        raise ValueError('An old TP4 layout cannot be used with 4.0.2')
    receipt = json.loads(path.with_suffix('.receipt.json').read_bytes())
    keys = gzip.decompress((HERE/f'plan/rank{rank}.keys.gz').read_bytes())
    spec = json.loads((HERE/'plan/KEYS.json').read_bytes())[str(rank)]
    if hashlib.sha256(keys).hexdigest() != spec['sha256']:
        raise ValueError('Native keys manifest changed')
    if (receipt.get('layout') != LAYOUT or receipt.get('rank') != rank
            or receipt.get('world') != 4
            or receipt.get('all_written_tensors_readback_verified') is not True
            or receipt.get('native_key_plan_verified') is not True
            or receipt.get('entries') != spec['entries']
            or receipt.get('changed_entries') != spec['changed_entries']
            or receipt.get('file_bytes') != path.stat().st_size):
        raise ValueError('Require the verified, complete balanced weight export')
    cache = Cache(path, receipt['index_sha256'])
    try:
        if (cache.end != receipt['payload_bytes']
                or [r[0] for r in cache.rows] != keys.decode().splitlines()):
            raise ValueError('Rank cache does not match the native key plan')
    finally:
        cache.close()
    if deep:
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != receipt['sha256']:
                raise ValueError('Weight payload checksum mismatch')
    return path


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=range(4),required=True)
    p.add_argument('--deep',action='store_true')
    a=p.parse_args()
    print(json.dumps(dict(rank=a.rank,path=str(validate(a.directory,a.rank,a.deep)),passed=True,deep=a.deep)))
