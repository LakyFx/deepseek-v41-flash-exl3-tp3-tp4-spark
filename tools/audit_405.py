"""Offline4.0.5 source, binary, profile, gateway and numerical evidence audit."""
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tensorfold.variants import patch

def load(p):return json.loads(p.read_bytes())
def same(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),(a,b)
def audit():
    spec,_=patch.recipe('v405');parent,_=patch.recipe('v404')
    base=ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz'
    assert patch.digest(base.read_bytes())==spec['base_archive_sha256']
    with tarfile.open(base) as t:
        files={m.name:patch.digest(t.extractfile(m).read()) for m in t if m.isfile()}
    for name in spec['remove']:files.pop(name)
    for name,record in spec['overlays'].items():
        patch.relative(name);patch.relative(record['object'])
        raw=(ROOT/'tensorfold/variants'/record['object']).read_bytes()
        assert patch.digest(raw)==record['sha256'];files[name]=record['sha256']
    assert files==spec['source_files']
    assert patch.tree_digest(files)==spec['source_tree_sha256']=='1ebb92311aa9198c9ca69db01b3a40d0b561fe70bae02b9c909b7c9df3c9cc85'
    changed={n for n,h in files.items() if parent['source_files'].get(n)!=h}
    assert len(changed)==8
    folder=ROOT/'tensorfold/tp4/v405';profile=load(folder/'profile.json');payload=load(folder/'PAYLOAD.json')
    assert payload['source_tree_sha256']==spec['source_tree_sha256']
    assert payload['source_commit']==spec['source_commit']
    assert patch.digest((folder/'native-linux-arm64.tar.gz').read_bytes())==payload['archive_sha256']==profile['native_archive_sha256']
    with tarfile.open(folder/'native-linux-arm64.tar.gz') as t:
        native={m.name:patch.digest(t.extractfile(m).read()) for m in t if m.isfile()}
    expected={'native/'+n:h for n,h in {**profile['historical_native_binaries'],**profile['runtime_kernel_hashes']}.items()}
    expected.update({'kit/cubins/'+n:h for n,h in profile['additive_kit_kernel_hashes'].items()})
    assert native==payload['files']==expected
    old=load(ROOT/'tensorfold/tp4/v404/profile.json')
    for key in ['parallel','pool_tokens','prompt_snapshot_gib_head','round_rows','round_ms',
                'context_tokens','prefill_chunk_tokens','physical_kv_quantum_tokens','weight_layout',
                'runtime_kernel_hashes','additive_kit_kernel_hashes','prefill_assets_sha256']:
        assert profile[key]==old[key],key
    for key,value in old['runtime_environment'].items():
        if key not in {'DS41F_RECIPE_VERSION','TF_DS_GRAPH_PROFILE_ID'}:
            assert profile['runtime_environment'][key]==value,key
    gm=load(ROOT/'gateway/source-manifest.json')
    assert gm['release']=='4.0.5'
    assert set(gm['modules'])=={p.name for p in (ROOT/'gateway/laguna_gateway').glob('*.py')}
    for name,r in gm['modules'].items():
        assert patch.digest((ROOT/'gateway/laguna_gateway'/name).read_bytes())==r['public_sha256']
        if not r['adaptation']:assert r['production_sha256']==r['public_sha256']
    gp=load(folder/'gateway-profile.json')
    assert gp==load(ROOT/'gateway/laguna_gateway/production-settings.json')
    assert gp['max_concurrency']==16 and gp['max_admitted_tokens']==8000000
    assert gp['tensorfold_client_json'] and gp['tensorfold_internal_json']
    assert patch.digest((folder/'imgcache-1.2.8.diff.txt').read_bytes())=='c0af0ad93728295751081d671951f997715b21e7ff789f74de5a1f01f42a2247'
    result=ROOT/'results/20261104-4.0.5'
    for name,h in load(result/'FILES.json')['files'].items():assert patch.digest((result/name).read_bytes())==h,name
    parity=load(result/'PRODUCTION-PARITY.json')
    assert parity['native_source_tree_sha256']==spec['source_tree_sha256']
    assert parity['native_binaries']==profile['historical_native_binaries']
    short=load(result/'SHORT-DECODE.json');cells=short['cells']
    assert {(r['run'],r['concurrency'],r['kind']) for r in cells}=={(n,c,k) for n in [1,2,3] for c in [1,3,6,12,16] for k in ['code','prose']}
    reference=load(ROOT/'results/20261103-4.0.4/SHORT-DECODE.json')['cells']
    for row in cells:
        requests=row['requests'];assert len(requests)==row['concurrency']
        previous=next(x for x in reference if (x['label'],x['run'],x['concurrency'],x['kind'])==('4.0.4-B',min(row['run'],2),row['concurrency'],row['kind']))
        assert [r['body_sha256'] for r in requests]==[r['body_sha256'] for r in previous['requests']]
        assert all(r['finish_reason']=='length' and r['usage']['completion_tokens']==512 and not r['native']['enable_thinking'] and r['native']['sampling']=='greedy' for r in requests)
        rates=[r['native']['tokens_per_second'] for r in requests]
        same(row['native_decode_sum_tps'],sum(rates));same(row['native_request_median_tps'],statistics.median(rates))
        same(row['aggregate_wall_tps'],row['completion_tokens']/row['elapsed_seconds'])
    cold=load(result/'COLD-PREFILL.json');assert len(cold['samples'])==3
    for r in cold['samples']:
        assert r['prompt_tokens']==65536 and r['cached_tokens']==0
        same(r['prefill_tps'],r['prompt_tokens']/r['prefill_seconds'])
    same(cold['median_prefill_tps'],statistics.median(r['prefill_tps'] for r in cold['samples']))
    full=load(result/'FULL-CONCURRENCY.json')
    for rows in [full['historical_404'],*full['runs'].values()]:
        assert [r['concurrency'] for r in rows]==list(range(1,17))
        for r in rows:
            assert len(r['selected'])==r['samples']>0
            for q in r['selected']:same(q['per_task'],q['total']/r['concurrency'])
            same(r['total'],statistics.median(q['total'] for q in r['selected']))
            same(r['per_task'],statistics.median(q['per_task'] for q in r['selected']))
    for run in [1,2]:
        heavy=load(result/f'HEAVY-PASS{run}.json')
        assert heavy['failures_retained'] and [r['concurrency'] for r in heavy['cells']]==list(range(1,17))
        assert sum(r['summary']['correct'] for r in heavy['cells'])==135
        assert sum(r['summary']['total'] for r in heavy['cells'])==136
    flyer=load(result/'FLYER.json')
    assert flyer['tp3']['heavy_source_version']=='1.2.5 / K2a'
    for name,h in flyer['artwork'].items():assert patch.digest((ROOT/'assets'/name).read_bytes())==h
    for row in flyer['tp4']['short']:
        for kind in ['code','prose']:
            r=next(x for x in cells if (x['run'],x['concurrency'],x['kind'])==(2,row['concurrency'],kind))
            same(row[kind],r['native_decode_sum_tps'])
    for row in flyer['tp4']['heavy']:
        r=next(x for x in full['runs']['pass2'] if x['concurrency']==row['concurrency'])
        same(row['total'],r['total']);same(row['per_task'],r['per_task'])
    q=load(result/'QUALIFICATION.json')
    assert q['json_passed'] and q['json_completed']==11 and all(r['passed'] for r in q['json_cases'])
    assert len(q['image_cases'])==6 and len(q['multi_image_cases'])==4
    assert q['owned_cancel_and_followup_passed']
    return dict(passed=True,source_files=len(files),changed_native_source_files=len(changed),payload_files=len(native),
        short_cells=len(cells),heavy_tasks_per_pass=136,heavy_correct_per_pass=135,
        json_checks=11,image_checks=10,gateway_modules=len(gm['modules']),model_requests=0,services_touched=False)

if __name__=='__main__':print(json.dumps(audit(),indent=2))
