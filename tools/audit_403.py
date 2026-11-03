"""Verify production 4.0.3 source, payload and numeric exports offline."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tensorfold.variants import patch


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b):
    require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), f'{a} != {b}')


def audit():
    spec, _ = patch.recipe('v403')
    archive = ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz'
    require(patch.digest(archive.read_bytes()) == spec['base_archive_sha256'], 'Base archive')
    with tarfile.open(archive) as stream:
        source = {m.name: patch.digest(stream.extractfile(m).read()) for m in stream if m.isfile()}
    for name in spec['remove']:
        source.pop(name)
    for name, record in spec['overlays'].items():
        patch.relative(name)
        raw = (ROOT/'tensorfold/variants'/record['object']).read_bytes()
        require(patch.digest(raw) == record['sha256'], 'Source object: '+name)
        source[name] = record['sha256']
    require(source == spec['source_files'], 'Complete source reconstruction')
    folder = ROOT/'tensorfold/tp4/v403'
    profile = load(folder/'profile.json')
    payload = load(folder/'PAYLOAD.json')
    require(payload['source_tree_sha256'] == spec['source_tree_sha256'], 'Source binding')
    require(patch.digest((folder/'native-linux-arm64.tar.gz').read_bytes()) ==
            profile['native_archive_sha256'] == payload['archive_sha256'], 'Native archive')
    with tarfile.open(folder/'native-linux-arm64.tar.gz') as stream:
        files = {m.name:patch.digest(stream.extractfile(m).read()) for m in stream if m.isfile()}
    require(files == payload['files'] == {'native/'+k:v for k,v in {
        **profile['historical_native_binaries'],**profile['runtime_kernel_hashes']}.items()}, 'Native files')
    evidence = ROOT/'results/20261102-4.0.3'
    for name, digest in load(evidence/'FILES.json')['files'].items():
        require(patch.digest((evidence/name).read_bytes()) == digest, 'Evidence: '+name)
    parity = load(evidence/'PRODUCTION-PARITY.json')
    require(parity['source_commit'] == spec['source_commit'] == payload['source_commit'], 'Source revision')
    require(parity['native_binaries'] == profile['historical_native_binaries'], 'Production binaries')
    gateway = load(ROOT/'gateway/source-manifest.json')
    historical = None
    if gateway['release'] != '4.0.3':
        # A newer installable gateway must not invalidate frozen 4.0.3 evidence.
        history = ROOT/'gateway/history/4.0.3.tar.gz'
        pin = load(ROOT/'gateway/history/FILES.json')['files']['4.0.3.tar.gz']
        require(patch.digest(history.read_bytes()) == pin, 'Historical gateway archive')
        with tarfile.open(history) as stream:
            historical = {m.name:stream.extractfile(m).read() for m in stream if m.isfile()}
        gateway = json.loads(historical['source-manifest.json'])
    require(gateway['source_revision'] == parity['source_commit'], 'Gateway source revision')
    names = ({p.name for p in (ROOT/'gateway/laguna_gateway').glob('*.py')} if historical is None
             else {Path(n).name for n in historical if n.startswith('laguna_gateway/')})
    require(names == set(gateway['modules']), 'Gateway completeness')
    for name, record in gateway['modules'].items():
        raw = ((ROOT/'gateway/laguna_gateway'/name).read_bytes() if historical is None
               else historical['laguna_gateway/'+name])
        require(patch.digest(raw) == record['public_sha256'], 'Gateway bytes: '+name)
        if not record['adaptation']:
            require(record['production_sha256'] == record['public_sha256'], 'Gateway parity: '+name)
    cells = load(evidence/'SHORT-DECODE.json')['cells']
    expected = {(run,c,kind) for run in (1,2) for c in (1,3,6,12,16) for kind in ('code','prose')}
    require(len(cells) == 20 and {(r['run'],r['concurrency'],r['kind']) for r in cells} == expected, 'Short cells')
    for cell in cells:
        require(len(cell['requests']) == cell['concurrency'], 'Request count')
        rates = [q['native']['tokens_per_second'] for q in cell['requests']]
        require(all(q['usage']['completion_tokens'] == 512 and not q['native']['enable_thinking']
                    and q['native']['sampling'] == 'greedy' for q in cell['requests']), 'Sampling/output length')
        equal(cell['native_decode_sum_tps'], sum(rates))
        equal(cell['native_decode_median_tps'], statistics.median(rates))
        equal(cell['wall_tps'], cell['completion_tokens']/cell['elapsed_seconds'])
    comparison = load(evidence/'SHORT-COMPARISON.json')
    old = load(ROOT/'results/20261101-4.0.2/SHORT-DECODE.json')['cells']
    for row in comparison['records']:
        current = next(c for c in cells if (c['run'],c['concurrency'],c['kind']) == (2,row['concurrency'],row['kind']))
        previous = next(c for c in old if (c['run'],c['concurrency'],c['kind']) == (2,row['concurrency'],row['kind']))
        equal(row['new_request_median_tps'], current['native_decode_median_tps'])
        equal(row['old_request_median_tps'], previous['native_decode_median_tps'])
        equal(row['new_wall_tps'], current['wall_tps'])
        equal(row['old_wall_tps'], previous['wall_tps'])
        equal(row['wall_change_pct'], (row['new_wall_tps']/row['old_wall_tps']-1)*100)
    live = load(evidence/'LIVE-SUMMARY.json')
    workflow = live['workflow']; graphs = live['graphs']
    equal(workflow['prefix_hit_percent'], workflow['cached']/workflow['input']*100)
    equal(graphs['reuse_percent'], graphs['hits']/(graphs['hits']+graphs['misses'])*100)
    return dict(passed=True, source_files=len(source), native_files=len(files),
                gateway_modules=len(gateway['modules']), short_cells=len(cells),
                model_requests=0, services_touched=False)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
