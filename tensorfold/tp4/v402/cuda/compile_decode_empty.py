"""Bounded CPU-only control/early-empty MoE kernel experiment, SM121."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.output.mkdir(exist_ok=False)
    original = Path(__file__).with_name('decode_cp.cu')
    source = original.read_text(encoding='utf-8')
    anchor = '    const int mat = bz / MT / SK;\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + '''#if TF_CP_EARLY_EMPTY
    // Every thread observes the same immutable first member. Reject an empty
    // tile before trellis-pointer loads, shared-row setup and block barriers.
    // Grouping guarantees live members form a prefix; arithmetic is untouched.
    if (members[u * maxm + mtile * 16] < 0) return;
#endif
''')
    emitted = args.output / 'decode_empty.cu'
    emitted.write_text(source, encoding='utf-8')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    records = []
    for early in [0, 1]:
        output = args.output / ('early.cubin' if early else 'control.cubin')
        command = ['/usr/local/cuda/bin/nvcc', '--cubin', '-arch=sm_121',
                   '-O3', '-std=c++17', '-Xptxas=-v',
                   '-DTF_CP_STAGES=3', '-DTF_CP_MIN_BLOCKS=4',
                   '-DTF_CP_EARLY_EMPTY=' + str(early), str(emitted), '-o', str(output)]
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, timeout=480)
        (args.output / (output.stem + '-compile.log')).write_text(
            result.stdout + result.stderr, encoding='utf-8')
        assert result.returncode == 0, result.stderr[-4000:]
        resources = subprocess.check_output(
            ['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(output)], text=True)
        (args.output / (output.stem + '-resources.txt')).write_text(resources, encoding='utf-8')
        records.append(dict(early_empty=bool(early), seconds=time.monotonic()-started,
                            command=command, sha256=sha(output)))
    proof = dict(status='PASS', gpu_access=False, arithmetic_changed=False,
                 original_source_sha256=sha(original), emitted_source_sha256=sha(emitted),
                 records=records)
    (args.output / 'COMPILED.json').write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
