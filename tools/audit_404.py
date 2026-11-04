"""Recalculate 4.0.4 exports and verify exact source/payload/gateway offline."""
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tensorfold.variants import patch


def load(p): return json.loads(p.read_bytes())
def require(ok, message):
    if not ok: raise ValueError(message)
def equal(a,b): require(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),f'{a} != {b}')


def audit():
    spec,_=patch.recipe('v404')
    base=ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz'
    require(patch.digest(base.read_bytes())==spec['base_archive_sha256'],'Base source archive')
    with tarfile.open(base) as t:
        source={m.name:patch.digest(t.extractfile(m).read()) for m in t if m.isfile()}
    for name in spec['remove']: source.pop(name)
    for name,record in spec['overlays'].items():
        patch.relative(name);patch.relative(record['object'])
        raw=(ROOT/'tensorfold/variants'/record['object']).read_bytes()
        require(patch.digest(raw)==record['sha256'],'Source object '+name)
        source[name]=record['sha256']
    require(source==spec['source_files'],'Complete source reconstruction')
    folder=ROOT/'tensorfold/tp4/v404';profile=load(folder/'profile.json');payload=load(folder/'PAYLOAD.json')
    require(payload['source_tree_sha256']==spec['source_tree_sha256'],'Payload/source binding')
    require(patch.digest((folder/'native-linux-arm64.tar.gz').read_bytes())==profile['native_archive_sha256']==payload['archive_sha256'],'Payload archive')
    with tarfile.open(folder/'native-linux-arm64.tar.gz') as t:
        files={m.name:patch.digest(t.extractfile(m).read()) for m in t if m.isfile()}
    expected={'native/'+n:h for n,h in {**profile['historical_native_binaries'],**profile['runtime_kernel_hashes']}.items()}
    expected.update({'kit/cubins/'+n:h for n,h in profile['additive_kit_kernel_hashes'].items()})
    require(files==payload['files']==expected,'Native and additive kernel files')
    for name,h in profile['additive_kit_kernel_hashes'].items():
        require(patch.digest((folder/'kernels'/name.removeprefix('404-')).read_bytes())==h,'Rebuild kernel '+name)
    evidence=ROOT/'results/20261103-4.0.4'
    for name,h in load(evidence/'FILES.json')['files'].items():
        require(patch.digest((evidence/name).read_bytes())==h,'Evidence '+name)
    parity=load(evidence/'PRODUCTION-PARITY.json')
    require(parity['native_source_commit']==spec['source_commit']==payload['source_commit'],'Native revision')
    require(parity['native_binaries']==profile['historical_native_binaries'],'Production binaries')
    gm=load(ROOT/'gateway/source-manifest.json')
    historical = None
    if gm['release'] != '4.0.4':
        archive = ROOT/'gateway/history/4.0.4.tar.gz'
        require(patch.digest(archive.read_bytes()) == load(ROOT/'gateway/history/FILES.json')['files']['4.0.4.tar.gz'], 'Historical gateway archive')
        with tarfile.open(archive) as stream:
            historical = {m.name:stream.extractfile(m).read() for m in stream if m.isfile()}
        gm = json.loads(historical['source-manifest.json'])
    require(gm['source_revision']==parity['gateway_source_commit'],'Gateway revision')
    names = ({p.name for p in (ROOT/'gateway/laguna_gateway').glob('*.py')} if historical is None else {Path(n).name for n in historical if n.startswith('laguna_gateway/')})
    require(set(gm['modules'])==names,'Gateway completeness')
    for name,r in gm['modules'].items():
        require(patch.digest((ROOT/'gateway/laguna_gateway'/name).read_bytes() if historical is None else historical['laguna_gateway/'+name])==r['public_sha256'],'Gateway '+name)
        if not r['adaptation']:require(r['production_sha256']==r['public_sha256'],'Gateway production parity '+name)
    short=load(evidence/'SHORT-DECODE.json');cells=short['cells'];cs=(1,3,6,12,16)
    require(len(cells)==140,'Short cell count')
    signatures={}
    for label in short['versions']:
        actual={(c['run'],c['concurrency'],c['kind']) for c in cells if c['label']==label}
        require(actual=={(r,c,k) for r in (1,2) for c in cs for k in ('code','prose')},'Missing/duplicate short cells '+label)
    for c in cells:
        qs=c['requests'];require(len(qs)==c['concurrency'],'Request count')
        key=(c['run'],c['concurrency'],c['kind']);sig=[q['body_sha256'] for q in qs]
        require(signatures.setdefault(key,sig)==sig,'Matched fixture bodies')
        require(all(q['usage']['completion_tokens']==512 and not q['native']['enable_thinking'] and q['native']['sampling']=='greedy' and q['finish_reason']=='length' for q in qs),'Sampling/output length')
        rates=[q['native']['tokens_per_second'] for q in qs]
        equal(c['native_decode_sum_tps'],sum(rates));equal(c['native_decode_median_tps'],statistics.median(rates))
        require(sum(q['usage']['completion_tokens'] for q in qs)==c['completion_tokens'],'Completion count')
        equal(c['wall_tps'],c['completion_tokens']/c['elapsed_seconds'])
    for label,c in load(evidence/'COLD-PREFILL.json').items():
        require(len(c['samples'])==3,'Cold count')
        for q in c['samples']:
            require(q['prompt_tokens']==65536 and q['cached_tokens']==0,'Cold zero-hit count')
            equal(q['prefill_tps'],q['prompt_tokens']/q['prefill_seconds'])
        equal(c['median_prefill_tps'],statistics.median(q['prefill_tps'] for q in c['samples']))
    flyer=load(evidence/'FLYER.json')
    for row in flyer['short']:
        for kind in ('code','prose'):
            cell=next(c for c in cells if (c['label'],c['run'],c['concurrency'],c['kind'])==('4.0.4-B',2,row['concurrency'],kind))
            equal(row[kind],cell['native_decode_sum_tps'])
    for r in flyer['heavy']:
        require(len(r['selected'])==r['samples'] and r['samples']>0,'Full-load samples')
        for q in r['selected']:equal(q['per_task'],q['total']/r['concurrency'])
        equal(r['total'],statistics.median(q['total'] for q in r['selected']))
        equal(r['per_task'],statistics.median(q['per_task'] for q in r['selected']))
    heavy=load(evidence/'HEAVY.json')
    require(heavy['complete'] and [r['concurrency'] for r in heavy['cells']]==list(range(1,17)),'Complete C1-C16 workflow')
    require(sum(r['correct'] for r in heavy['cells'])==sum(r['total'] for r in heavy['cells'])==136,'Workflow correctness')
    return dict(passed=True,source_files=len(source),native_files=len(files),short_cells=len(cells),heavy_tasks=136,gateway_modules=len(gm['modules']),model_requests=0,services_touched=False)


if __name__=='__main__': print(json.dumps(audit(),indent=2))
