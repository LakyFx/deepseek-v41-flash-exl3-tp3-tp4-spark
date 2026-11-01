"""Recompute the published 4.0.2 evidence and verify its payload, entirely offline."""
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
from tensorfold.tp4.v402.repack_weights import changed
from benchmarks.short_decode.run import body_bytes, fixture
from benchmarks.workflow.run import inputs


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b):
    require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), f'{a} != {b}')


def audit():
    evidence = ROOT/'results/20261101-4.0.2'
    for name, digest in load(evidence/'FILES.json')['files'].items():
        require(patch.digest((evidence/name).read_bytes()) == digest, 'Evidence hash: '+name)
    short = load(evidence/'SHORT-DECODE.json')['cells']
    require(len(short) == 20, 'Both complete code/prose passes required')
    require({(r['run'], r['concurrency'], r['kind']) for r in short} ==
            {(run, c, kind) for run in (1, 2) for c in (1, 3, 6, 12, 16)
             for kind in ('code', 'prose')}, 'Missing short-screen cell')
    for row in short:
        require(len(row['requests']) == row['concurrency'], 'Short-screen request count')
        rates = []
        for i, q in enumerate(row['requests']):
            require(q['body_sha256'] == patch.digest(body_bytes(fixture(row['kind'], i))),
                    'Short authored fixture changed')
            require(q['usage']['completion_tokens'] == 512, 'Short output length')
            require(q['native']['enable_thinking'] is False and
                    q['native']['sampling'] == 'greedy', 'Short sampling changed')
            rates.append(q['native']['tokens_per_second'])
        equal(row['native_decode_sum_tps'], sum(rates))
        equal(row['native_decode_median_tps'], statistics.median(rates))
        equal(row['wall_tps'], row['completion_tokens']/row['elapsed_seconds'])
    full = load(evidence/'HEAVY-FULL-CONCURRENCY.json')
    require([r['concurrency'] for r in full['rows']] == list(range(1, 17)), 'Workflow C1-C16')
    sample_count = 0
    for row in full['rows']:
        c = row['concurrency']; samples = row['selected_samples']; sample_count += len(samples)
        require(len(samples) == row['samples'] and samples, 'Workflow sample count')
        for s in samples:
            tasks = s['active_tasks']; t = s['t']
            require(len(tasks) == c and len({a['replica'] for a in tasks}) == c,
                    'Actual full concurrency required')
            require(all(a['first_token_epoch'] + full['boundary_margin_seconds'] <=
                        t - full['native_rate_window_seconds'] and
                        a['request_end_epoch'] > t + full['boundary_margin_seconds']
                        for a in tasks), 'Rate window crosses a task boundary')
            equal(s['per_task_average_decode_tps'], s['aggregate_decode_tps']/c)
        equal(row['median_total_tps'], statistics.median(s['aggregate_decode_tps'] for s in samples))
        equal(row['median_per_task_tps'], row['median_total_tps']/c)
    heavy = load(evidence/'HEAVY-WORKFLOW.json')['cells']
    input_hashes = {(r['concurrency'], r['replica']): r['sha256'] for r in inputs(16)}
    requests = []; errors = 0; tasks = 0
    for row in heavy:
        c = row['concurrency']; require(len(row['tasks']) == c, 'Workflow task count')
        for task in row['tasks']:
            require(task['correct'] and task['finished'], 'Unsuccessful workflow task')
            require(task['requests'][0]['matched_input_sha256'] == input_hashes[c, task['replica']],
                    'Historical workflow input changed')
            tasks += 1; errors += task['tool_errors']; requests.extend(task['requests'])
        qs = [q for task in row['tasks'] for q in task['requests']]
        require(row['summary']['requests'] == len(qs), 'Request coverage')
        equal(row['summary']['aggregate_workflow_completion_tps'],
              sum(q['usage']['completion_tokens'] for q in qs)/row['summary']['wave_elapsed_seconds'])
    require((tasks, len(requests), errors) == (136, 606, 51), 'Workflow totals')
    require(all(q['native']['enable_thinking'] and q['native']['reasoning_effort'] == 'max'
                and q['native']['sampling'] == 'exact' for q in requests), 'MAX exact sampling')
    cold = load(evidence/'COLD-PREFILL.json')['results']
    require(len(cold) == 7 and sum(len(r['samples']) for r in cold) == 21, 'Cold-prefill scope')
    for row in cold:
        for s in row['samples']:
            require(s['cached_tokens'] == 0 and s['computed_prompt_tokens'] == s['prompt_tokens'],
                    'Cold-prefill sample has a cache hit')
            equal(s['native_prefill_tps'], s['computed_prompt_tokens']/s['native_prefill_seconds'])
        equal(row['median_tps'], statistics.median(s['native_prefill_tps'] for s in row['samples']))
    equal(next(r['median_tps'] for r in cold if r['target_tokens'] == 65536), 4263.530475825136)

    spec, _ = patch.recipe('v402')
    archive = ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz'
    require(patch.digest(archive.read_bytes()) == spec['base_archive_sha256'], 'Source base archive')
    with tarfile.open(archive) as stream:
        source = {m.name: patch.digest(stream.extractfile(m).read()) for m in stream if m.isfile()}
    for name in spec['remove']:
        source.pop(name)
    for name, rec in spec['overlays'].items():
        data = (ROOT/'tensorfold/variants'/rec['object']).read_bytes()
        require(patch.digest(data) == rec['sha256'], 'Source overlay: '+name)
        source[name] = rec['sha256']
    require(source == spec['source_files'], 'Reconstructed complete source tree')
    folder = ROOT/'tensorfold/tp4/v402'
    profile = load(folder/'profile.json'); payload = load(folder/'PAYLOAD.json')
    require(payload['source_tree_sha256'] == spec['source_tree_sha256'], 'Payload/source binding')
    require(patch.digest((folder/'native-linux-arm64.tar.gz').read_bytes()) ==
            profile['native_archive_sha256'] == payload['archive_sha256'], 'Native archive')
    with tarfile.open(folder/'native-linux-arm64.tar.gz') as stream:
        files = {m.name: patch.digest(stream.extractfile(m).read()) for m in stream if m.isfile()}
    require(files == payload['files'] == {'native/'+k: v for k, v in {
        **profile['historical_native_binaries'], **profile['runtime_kernel_hashes']}.items()}, 'Native payload files')
    parity = load(evidence/'PRODUCTION-PARITY.json')
    require(parity['native_binaries'] == profile['historical_native_binaries'], 'Production binary parity')
    require(parity['source_commit'] == spec['source_commit'], 'Source commit parity')
    keys = load(folder/'plan/KEYS.json')
    for rank in range(4):
        raw = gzip.decompress((folder/f'plan/rank{rank}.keys.gz').read_bytes())
        base = gzip.decompress((ROOT/f'tensorfold/tp4/plan/rank{rank}.keys.gz').read_bytes())
        expected = []; count = 0
        for key in base.decode().splitlines():
            change = changed(key, rank)
            if change:
                fields, _, new = change
                key = '|'.join([fields[0], str(new), fields[2], fields[3]]); count += 1
            expected.append(key)
        require(raw.decode().splitlines() == expected and count == 46200, 'Balanced native key order')
        require(patch.digest(raw) == keys[str(rank)]['sha256'] and len(expected) == 143530,
                'Balanced native key manifest')
    require(load(evidence/'QUALIFICATION.json')['passed'], 'Qualification')
    return dict(passed=True, services_touched=False, model_requests=0,
                short_cells=len(short), workflow_tasks=tasks, workflow_requests=len(requests),
                recovered_tool_errors=errors, full_concurrency_samples=sample_count,
                cold_prefill_samples=21, source_files=len(source), native_files=len(files))


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
