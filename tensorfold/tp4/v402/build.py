"""Stage the exact measured 4.0.2 payload into a NEW directory, offline."""
import argparse
import json
from pathlib import Path
import tarfile

from tensorfold.tp4 import assets
from tensorfold.variants import patch

HERE = Path(__file__).resolve().parent


def prepare(kit, out):
    profile = json.loads((HERE/'profile.json').read_bytes())
    spec, identity = patch.recipe('v402')
    payload = json.loads((HERE/'PAYLOAD.json').read_bytes())
    archive = HERE/'native-linux-arm64.tar.gz'
    if patch.digest(archive.read_bytes()) != profile['native_archive_sha256']:
        raise ValueError('Wrong 4.0.2 native archive')
    if payload['source_tree_sha256'] != spec['source_tree_sha256']:
        raise ValueError('Wrong native source binding')
    expected = {'native/'+k:v for k,v in {
        **profile['historical_native_binaries'], **profile['runtime_kernel_hashes']}.items()}
    if payload['files'] != expected or payload['archive_sha256'] != profile['native_archive_sha256']:
        raise ValueError('Wrong payload manifest')
    assets.inputs(kit, profile['prefill_assets_sha256'])
    if out.exists():
        raise FileExistsError('Preserve old installations; choose a new output')
    files = {}
    with tarfile.open(archive) as stream:
        for member in stream:
            patch.relative(member.name)
            if not member.isfile() or member.name not in expected or member.name in files:
                raise ValueError('Unexpected archive member')
            data = stream.extractfile(member).read()
            if patch.digest(data) != expected[member.name]:
                raise ValueError('Payload checksum mismatch')
            files[member.name] = data
    if files.keys() != expected.keys():
        raise ValueError('Incomplete payload')
    out.mkdir(parents=True)
    for name, data in files.items():
        target = out/name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        if target.name in profile['historical_native_binaries']:
            target.chmod(0o755)
    assets.prepare(kit, out/'prefill', profile['prefill_assets_sha256'])
    receipt = dict(schema='ds41f.tp4-build.v1', version='4.0.2', recipe='v402',
        recipe_manifest_sha256=identity, source_tree_sha256=spec['source_tree_sha256'],
        source_archive_sha256=spec['base_archive_sha256'],
        prefill_assets_sha256=profile['prefill_assets_sha256'],
        provenance='Exact measured 4.0.1a production payload, public name 4.0.2',
        binaries={name:dict(path='native/'+name,sha256=h)
                  for name,h in profile['historical_native_binaries'].items()},
        kernels=profile['runtime_kernel_hashes'],
        gpu_qualified_on_this_host=False, services_touched=False)
    (out/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.kit.resolve(),a.out.resolve()),indent=2))
