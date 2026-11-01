"""Offline source coverage and native compatibility checks; no Triton/GPU import."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools/dsv41-tp3'
spec = importlib.util.spec_from_file_location('aot_source_rebuild', TOOLS / 'rebuild.py')
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)


class AotSourceRebuild(unittest.TestCase):
    def test_every_manifest_entry_has_a_complete_source_signature(self):
        _, _, _, base, extra = rebuild.inputs()
        sources = {
            'tp3_triton_kernels': TOOLS / 'tp3_triton_kernels.py',
            'tensorfold.families.deepseek_v41.cuda.kernels': ROOT / 'python-reference/src/tensorfold/families/deepseek_v41/cuda/kernels.py',
            'tensorfold.families.deepseek_v41.cuda.markov': ROOT / 'python-reference/src/tensorfold/families/deepseek_v41/cuda/markov.py',
            'tensorfold.cuda.exl3.prefill': ROOT / 'python-reference/src/tensorfold/cuda/exl3/prefill.py',
            'prefill127k_kernels': ROOT / 'tensorfold/prefill-128/kernels.py',
        }
        definitions = {name: {f.name: {a.arg for a in f.args.args}
                             for f in ast.parse(path.read_bytes()).body if isinstance(f, ast.FunctionDef)}
                       for name, path in sources.items()}
        for c in base + extra:
            lowered = rebuild.compile_inputs(c)
            self.assertEqual(definitions[c['module']][c['function']],
                             set(lowered['params']) | set(lowered['consts']), c['function'])
        self.assertNotIn('triton', sys.modules)

    def test_retired_flag_cannot_select_a_different_reduction(self):
        case = dict(function='_index_score_rows', consts={'TREE': 0, 'RB': 1})
        with self.assertRaisesRegex(ValueError, 'TREE=1'):
            rebuild.compile_inputs(case)
        case['consts']['TREE'] = 1
        original = copy.deepcopy(case)
        self.assertNotIn('TREE', rebuild.compile_inputs(case)['consts'])
        self.assertEqual(case, original)
        unknown = dict(function='_index_score', consts={'UNKNOWN': 1})
        self.assertIs(rebuild.compile_inputs(unknown), unknown)

    def test_browsable_sources_are_exact_copies_not_truncated_stubs(self):
        with tarfile.open(ROOT / 'tensorfold/artifacts/tensorfold-rope-reference.tar.gz') as stream:
            for path in (ROOT / 'python-reference/src').rglob('*.py'):
                name = path.relative_to(ROOT / 'python-reference').as_posix()
                self.assertEqual(path.read_bytes(), stream.extractfile(name).read(), name)

    def test_static_elf_reader_recovers_the_requested_dependency_filenames(self):
        spec = importlib.util.spec_from_file_location('aot_debug_paths', TOOLS / 'debug_paths.py')
        debug = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(debug)
        with tarfile.open(ROOT / 'tensorfold/artifacts/tensorfold-kernels.tar.gz') as stream:
            patch = json.loads((ROOT / 'tensorfold/hc-prefill/PATCH.json').read_bytes())
            for entry in patch['new_kernels']:
                data = stream.extractfile('aot/cubins/' + entry['hash'] + '.cubin').read()
                names = {p.rsplit('/', 1)[-1] for p in debug.source_paths(data)}
                self.assertTrue({'tp3_triton_kernels.py', 'kernels.py', 'standard.py'} <= names)
        with self.assertRaises(ValueError):
            debug.source_paths(b'not an ELF')

    def test_same_shape_does_not_attach_a_later_receipt_to_a_captured_kernel(self):
        _, base, _, cases, _ = rebuild.inputs()
        references = rebuild.expected_ptx()
        # Both _pick entries have the same shapes/options, but only the later
        # explicit-case addition has this recorded PTX provenance.
        self.assertEqual(rebuild.case_key(cases[87]), rebuild.case_key(cases[367]))
        self.assertIsNone(rebuild.ptx_reference(cases[87], base['kernels'][87], references))
        self.assertIsNotNone(rebuild.ptx_reference(cases[367], base['kernels'][367], references))


if __name__ == '__main__':
    unittest.main()
