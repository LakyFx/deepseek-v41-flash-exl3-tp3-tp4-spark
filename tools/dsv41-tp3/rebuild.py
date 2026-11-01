"""CPU-only rebuild of the published TP3 AOT launch sets (no kernel execution).

Read-only inputs; exclusively new output directories. Rebuilt binaries are never
installed. Historical PTX hashes and shipped cubin hashes are separate checks.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def archive_json(path, member):
    with tarfile.open(path) as stream:
        return json.load(stream.extractfile(member))


@lru_cache(maxsize=1)
def debug_metadata():
    from debug_paths import source_paths
    result = {}
    for archive in ('tensorfold/artifacts/tensorfold-kernels.tar.gz', 'tensorfold/prefill-128/payload.tar.gz'):
        with tarfile.open(ROOT / archive) as stream:
            for member in stream:
                if member.isfile() and member.name.startswith('aot/cubins/') and member.name.endswith('.cubin'):
                    result[Path(member.name).stem] = source_paths(stream.extractfile(member).read())
    return result


def restore_debug_paths(expected_hash):
    """Reproduce debug metadata only; never read code from an embedded path.

    Historical build roots already occur in the shipped ELF's DWARF. Changing a
    checkout path changes PTX hashes even when source/arithmetic are identical.
    Read that metadata statically and set Triton's filename labels, without
    creating those paths or loading the ELF through the CUDA driver.
    """
    import triton
    paths = debug_metadata().get(expected_hash)
    if paths is None:
        raise ValueError('Missing static debug metadata: ' + expected_hash)
    filenames = {}
    for path in paths:
        basename = path.rsplit('/', 1)[-1]
        if basename in filenames and filenames[basename] != path:
            raise ValueError('Ambiguous historical debug filename: ' + basename)
        filenames[basename] = path
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, '__name__', '').startswith(('tensorfold.', 'triton.', 'tp3_triton_', 'prefill127k_')):
            continue
        for fn in vars(module).values():
            if isinstance(fn, triton.runtime.jit.JITFunction):
                basename = fn.file_name.replace('\\', '/').rsplit('/', 1)[-1]
                if basename in filenames:
                    fn.file_name = filenames[basename]


def case_key(case):
    return json.dumps({k: case[k] for k in ('module', 'function', 'consts', 'params', 'options')}, sort_keys=True)


def compile_inputs(case):
    """Lower the one retired, now-unconditional selector flag explicitly.

    The pinned native tri_index.zig documents that Python 8b1fe210 always uses
    the tree sum. Preserve TREE=1 in the native manifest, but do not invent a
    missing function parameter in the compiler. TREE=0 is not supported.
    """
    if case['function'] == '_index_score_rows' and 'TREE' in case['consts']:
        if case['consts']['TREE'] != 1:
            raise ValueError('The pinned source only supports TREE=1')
        return dict(case, consts={k: v for k, v in case['consts'].items() if k != 'TREE'})
    return case


def inputs():
    lock = read_json(HERE / 'SOURCES.json')
    for name, wanted in lock['files'].items():
        if sha((ROOT / name).read_bytes()) != wanted:
            raise ValueError('Source/input hash mismatch: ' + name)
    pins = read_json(ROOT / 'tensorfold/inputs.json')['files']
    for name in ('artifacts/tensorfold-kernels.tar.gz', 'artifacts/rope-reference-files.json'):
        if sha((ROOT / 'tensorfold' / name).read_bytes()) != pins[name]['sha256']:
            raise ValueError('Pinned input mismatch: ' + name)
    base = archive_json(ROOT / 'tensorfold/artifacts/tensorfold-kernels.tar.gz', 'aot/aot.json')
    af = archive_json(ROOT / 'tensorfold/prefill-128/payload.tar.gz', 'aot/aot.json')
    # Bind the *bytes*, not a JSON reserialization, to the qualified manifests.
    for archive, wanted in [('tensorfold/artifacts/tensorfold-kernels.tar.gz', lock['base_aot_sha256']),
                            ('tensorfold/prefill-128/payload.tar.gz', lock['v128_aot_sha256'])]:
        with tarfile.open(ROOT / archive) as stream:
            if sha(stream.extractfile('aot/aot.json').read()) != wanted:
                raise ValueError('AOT manifest mismatch: ' + archive)
    if af['kernels'][:len(base['kernels'])] != base['kernels']:
        raise ValueError('1.2.8 does not preserve the base launch entries')
    cases = read_json(HERE / 'base-cases.json')
    extra = read_json(HERE / 'v128-cases.json')
    def verify_case(c, k):
        consts = {n: v['int'] if 'int' in v else struct.unpack('<f', struct.pack('<I', v['f32']))[0]
                  for n, v in k['consts'].items()}
        params = {p['name']: {n: p[n] for n in ('type', 'div16')} for p in k['params']}
        return (c['function'] == k['fn'] and c['consts'] == consts and c['params'] == params
                and c['options']['num_warps'] == k['num_warps'] and c['options']['launch_pdl'] == k['pdl'])
    if len(cases) != 542 or len(extra) != 24 or len(af['kernels']) != 566:
        raise ValueError('Incomplete launch set')
    if not all(verify_case(c, k) for c, k in zip(cases + extra, af['kernels'])):
        raise ValueError('Compilation cases disagree with the published ABI')
    return lock, base, af, cases, extra


def expected_ptx():
    """Recorded receipts exist for these additions, not for every base launch."""
    expected = {}
    with tarfile.open(ROOT / 'tensorfold/artifacts/tensorfold-kernels.tar.gz') as stream:
        for prefix in ('integer-fallback', 'router-rows'):
            cases = json.load(stream.extractfile('aot/' + prefix + '-CASES.json'))
            receipts = json.load(stream.extractfile('aot/' + prefix + '-provenance.json'))['sources']
            if len(cases) != len(receipts):
                raise ValueError('Historical receipt length mismatch')
            for c, p in zip(cases, receipts):
                expected[case_key(c)] = p['ptx_sha256']
    cases = read_json(ROOT / 'tensorfold/hc-prefill/COMPILE-CASES.json')
    receipts = read_json(ROOT / 'tensorfold/hc-prefill/COMPILE-PROVENANCE.json')['sources']
    for c, p in zip(cases, receipts):
        expected[case_key(c)] = p['ptx_sha256']
    return expected


def ptx_reference(case, entry, references):
    # Addition receipts describe the explicit-case compiler. A captured upstream
    # entry with the same shape can have different source/debug provenance.
    if not all(p['nospec'] for p in entry['params']):
        return None
    return references.get(case_key(case))


def check_reference(reference):
    files = read_json(ROOT / 'tensorfold/artifacts/rope-reference-files.json')
    for name, pin in files.items():
        p = reference / name
        if p.is_symlink() or sha(p.read_bytes()) != pin['sha256']:
            raise ValueError('Extracted reference differs: ' + name)
    # Import the entire pinned package, not the partial browsing copy at repo root.
    sys.path.insert(0, str(reference / 'src'))
    sys.path.insert(0, str(HERE))


def summarize(records, recipe, historical_debug_paths):
    mismatches = sum(not r['cubin_match'] or r['ptx_match'] is False for r in records)
    return dict(recipe=recipe, target='cuda:121:32', triton='3.8.0',
                entries=len(records), unique_rebuilt_cubins=len({r['cubin_sha256'] for r in records}),
                cubin_matches=sum(r['cubin_match'] for r in records),
                recorded_ptx_checks=sum(r['ptx_match'] is not None for r in records),
                recorded_ptx_matches=sum(r['ptx_match'] is True for r in records),
                mismatched_entries=mismatches, gpu_execution=False,
                historical_debug_paths=historical_debug_paths,
                status='EXACT_BINARY_REBUILD_NOT_GPU_QUALIFIED' if not mismatches else 'REBUILD_DIFFERS_NOT_GPU_QUALIFIED')


def compare(args):
    """Recheck generated files without recompiling; preserve the build evidence."""
    _, base, af, cases, extra = inputs()
    build = read_json(args.built / 'BUILD.json')
    if build['recipe'] not in ('base', 'v128') or build['triton'] != '3.8.0':
        raise ValueError('Expected a complete base/v128 source build')
    if build['recipe'] == 'v128':
        cases += extra
    expected = base['kernels'] if build['recipe'] == 'base' else af['kernels']
    actual = read_json(args.built / 'aot.json')['kernels']
    receipts = read_json(args.built / 'comparison.json')
    if len(actual) != len(expected) or len(receipts) != len(expected):
        raise ValueError('Incomplete source build')
    references, records = expected_ptx(), []
    for i, (case, got, wanted, receipt) in enumerate(zip(cases, actual, expected, receipts)):
        if any(got[k] != wanted[k] for k in ('fn', 'num_warps', 'num_ctas', 'global_scratch', 'profile_scratch', 'pdl', 'params', 'consts')):
            raise ValueError('Built native ABI differs at entry ' + str(i))
        if sha((args.built / 'cubins' / (got['hash'] + '.cubin')).read_bytes()) != got['hash']:
            raise ValueError('Changed rebuilt cubin at entry ' + str(i))
        if sha((args.built / 'ptx' / (receipt['ptx_sha256'] + '.ptx')).read_bytes()) != receipt['ptx_sha256']:
            raise ValueError('Changed generated PTX at entry ' + str(i))
        ptx = ptx_reference(case, wanted, references)
        records.append(dict(receipt, expected_ptx_sha256=ptx,
                            ptx_match=None if ptx is None else ptx == receipt['ptx_sha256'],
                            expected_cubin_sha256=wanted['hash'], cubin_sha256=got['hash'],
                            cubin_match=wanted['hash'] == got['hash']))
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'comparison.json', records)
    report = summarize(records, build['recipe'], build['historical_debug_paths'])
    report['method'] = 'Rehash and recompare existing source-build artifacts; no recompilation or GPU execution'
    write_json(args.out / 'BUILD.json', report)
    print(json.dumps(report), flush=True)
    return int(bool(any(r['ptx_match'] is False for r in records) or (args.require_exact_cubins and report['mismatched_entries'])))


def build(args):
    lock, base, af, cases, extra = inputs()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Set CUDA_VISIBLE_DEVICES= (empty): this is an offline build')
    if os.environ.get('TRITON_ALWAYS_COMPILE') != '1':
        raise ValueError('Set TRITON_ALWAYS_COMPILE=1 to prevent reuse of cached binaries')
    import triton
    if triton.__version__ != lock['triton']:
        raise ValueError('Use exactly Triton ' + lock['triton'])
    check_reference(args.reference.resolve())
    from compile_triton import compile_case
    import importlib
    if args.recipe == 'hc-prefill':
        cases = read_json(ROOT / 'tensorfold/hc-prefill/COMPILE-CASES.json')
        selected = {k['hash'] for k in read_json(ROOT / 'tensorfold/hc-prefill/PATCH.json')['new_kernels']}
        kernels = [k for k in base['kernels'] if k['hash'] in selected]
        # Historical case order is authoritative, not its position in the merged kit.
        kernels = [next(k for k in kernels if k['consts'] == {n: {'int': int(v)} for n, v in c['consts'].items()}) for c in cases]
    else:
        kernels = base['kernels']
        if args.recipe == 'v128':
            cases += extra
            kernels = af['kernels']
            spec = importlib.util.spec_from_file_location('prefill127k_kernels', ROOT / 'tensorfold/prefill-128/kernels.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'cubins').mkdir()
    (args.out / 'ptx').mkdir()
    wanted_ptx = expected_ptx()
    built, records = [], []
    # Record PTX without changing the historical compile helper or its source hash.
    original_compile = triton.compile
    latest = {}
    def capture(*a, **kw):
        compiled = original_compile(*a, **kw)
        latest['ptx'] = compiled.asm['ptx']
        return compiled
    triton.compile = capture
    try:
        for i, (case, expected) in enumerate(zip(cases, kernels)):
            # Ensure the source module and its JIT helpers exist before applying labels.
            importlib.import_module(case['module'])
            if args.historical_debug_paths:
                restore_debug_paths(expected['hash'])
            lowered = compile_inputs(case)
            actual, receipt = compile_case(lowered, args.out)
            if lowered is not case:
                actual['consts']['TREE'] = {'int': 1}
            # nospec is runtime matching metadata, not an additional compile option.
            # Retain it exactly instead of broadening the native selector.
            actual['params'] = expected['params']
            abi_fields = ('fn', 'num_warps', 'num_ctas', 'global_scratch', 'profile_scratch', 'pdl', 'params', 'consts')
            if any(actual[k] != expected[k] for k in abi_fields):
                raise ValueError('Native ABI/scratch contract differs at entry ' + str(i))
            ptx_hash = receipt['ptx_sha256']
            (args.out / 'ptx' / (ptx_hash + '.ptx')).write_bytes(latest['ptx'].encode())
            wanted = ptx_reference(case, expected, wanted_ptx)
            record = dict(index=i, function=case['function'], expected_cubin_sha256=expected['hash'],
                          cubin_sha256=actual['hash'], cubin_match=actual['hash'] == expected['hash'],
                          ptx_sha256=ptx_hash, expected_ptx_sha256=wanted,
                          ptx_match=None if wanted is None else ptx_hash == wanted,
                          compiler_hash=receipt['compiler_hash'], module_sha256=receipt['module_sha256'])
            record['compatibility'] = 'retired TREE=1; pinned source unconditionally sums as a tree' if lowered is not case else None
            records.append(record)
            built.append(actual)
            # Each entry is durable even if a later compile fails.
            write_json(args.out / 'comparison.json', records)
            print(json.dumps(record), flush=True)
    finally:
        triton.compile = original_compile
    mismatches = sum(not r['cubin_match'] or r['ptx_match'] is False for r in records)
    write_json(args.out / 'aot.json', dict(generator='source-rebuild: ' + args.recipe, kernels=built))
    report = summarize(records, args.recipe, args.historical_debug_paths)
    write_json(args.out / 'BUILD.json', report)
    print(json.dumps(report), flush=True)
    # A source rebuild can be valid with different assembler-produced bytes.
    # Recorded PTX failures always fail; exact binary enforcement is opt-in.
    return 1 if any(r['ptx_match'] is False for r in records) or (args.require_exact_cubins and mismatches) else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('audit', help='Verify sources and complete launch coverage without importing Triton')
    b = sub.add_parser('build', help='Compile all selected variants, compare hashes, never install')
    b.add_argument('--recipe', choices=('base', 'v128', 'hc-prefill'), required=True)
    b.add_argument('--reference', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--historical-debug-paths', action='store_true', help='Reproduce filename labels from static ELF DWARF; does not load/execute binaries')
    b.add_argument('--require-exact-cubins', action='store_true', help='Fail if assembler output differs, even when recorded PTX matches')
    c = sub.add_parser('compare', help='Rehash an existing base/v128 source build without recompiling or overwriting it')
    c.add_argument('--built', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--require-exact-cubins', action='store_true')
    a = p.parse_args()
    if a.command == 'build':
        return build(a)
    if a.command == 'compare':
        return compare(a)
    _, base, af, _, _ = inputs()
    print(json.dumps(dict(source_hashes='PASS', base_entries=len(base['kernels']),
                          base_unique_cubins=len({k['hash'] for k in base['kernels']}),
                          v128_entries=len(af['kernels']),
                          v128_unique_cubins=len({k['hash'] for k in af['kernels']}), gpu_execution=False), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
