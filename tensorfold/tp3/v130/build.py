"""Verify/stage exact qualified ARM64 files; install additively, never start services."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from tensorfold.variants import patch

HERE = Path(__file__).resolve().parent


def payload():
    record = json.loads((HERE/'PAYLOAD.json').read_bytes())
    spec, _ = patch.recipe('v130')
    if record['source_tree_sha256'] != spec['source_tree_sha256']:
        raise ValueError('Native source binding differs')
    archive = HERE/'native-linux-arm64.tar.gz'
    if patch.digest(archive.read_bytes()) != record['archive_sha256']:
        raise ValueError('Native archive differs')
    files = {}
    with tarfile.open(archive) as stream:
        for member in stream:
            patch.relative(member.name)
            if not member.isfile() or member.name in files or member.name not in record['files']:
                raise ValueError('Unexpected payload member')
            raw = stream.extractfile(member).read()
            if patch.digest(raw) != record['files'][member.name]:
                raise ValueError('Payload member differs: '+member.name)
            files[member.name] = raw
    if files.keys() != record['files'].keys():
        raise ValueError('Incomplete payload')
    return record, files


def prepare(out):
    record, files = payload()
    out = Path(out)
    if out.exists():
        raise FileExistsError('Choose a new output; preserve existing installations')
    out.mkdir(parents=True)
    for name, raw in files.items():
        target = out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        if name.startswith('bin-'):
            target.chmod(0o755)
    (out/'PAYLOAD.json').write_bytes((HERE/'PAYLOAD.json').read_bytes())
    return dict(version='1.3.0', files=len(files), services_touched=False, model_requests=0)


def verify_installed(root, bundle=HERE):
    root = Path(root)
    record = json.loads((Path(bundle)/'PAYLOAD.json').read_bytes())
    spec, _ = patch.recipe('v130')
    if record['source_tree_sha256'] != spec['source_tree_sha256']:
        raise ValueError('Wrong source identity')
    for name, digest in record['files'].items():
        patch.relative(name)
        target = root/name if not name.startswith('kernel/') else root/'kit-1.3.0/cubins'/Path(name).name
        if target.is_symlink() or patch.digest(target.read_bytes()) != digest:
            raise ValueError('Installed file differs: '+name)


def install(staged, root):
    staged, root = Path(staged), Path(root)
    record, _ = payload()
    for name, digest in record['files'].items():
        patch.relative(name)
        if patch.digest((staged/name).read_bytes()) != digest:
            raise ValueError('Staged file differs: '+name)
    targets = [root/'bin-1.3.0', root/'prefill-1.3.0', root/'kit-1.3.0']
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Version already installed; preserve it')
    base = root/'kit'
    if not (base/'engram-token-map.json').is_file() or not (base/'cubins').is_dir():
        raise ValueError('Require the preserved verified TP3 base kit')
    # Aliases preserve base-kit bytes; this release owns only its B-kernel overlay.
    for name in ['bin-1.3.0', 'prefill-1.3.0']:
        shutil.copytree(staged/name, root/name)
    kit = root/'kit-1.3.0'
    kit.mkdir()
    for path in base.iterdir():
        if path.name == 'cubins':
            continue
        (kit/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (kit/'cubins').mkdir()
    for path in (base/'cubins').iterdir():
        if path.name != '404-midm-l10.cubin':
            (kit/'cubins'/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    shutil.copy2(staged/'kernel/404-midm-l10.cubin', kit/'cubins/404-midm-l10.cubin')
    verify_installed(root)
    return dict(installed='1.3.0', services_touched=False, model_requests=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('stage'); s.add_argument('--out', type=Path, required=True)
    i = sub.add_parser('install'); i.add_argument('--staged', type=Path, required=True); i.add_argument('--runtime', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.out) if a.action == 'stage' else install(a.staged, a.runtime), indent=2))
