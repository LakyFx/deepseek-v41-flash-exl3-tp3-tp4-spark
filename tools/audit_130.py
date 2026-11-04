"""Offline exact-source/payload/gateway/evidence audit; no model or network calls."""
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tensorfold.tp3.v130 import build
from tensorfold.variants import patch


def audit():
    spec,_=patch.recipe('v130')
    with tarfile.open(ROOT/'tensorfold/artifacts/tensorfold-source.tar.gz') as stream:
        source={m.name:patch.digest(stream.extractfile(m).read()) for m in stream if m.isfile()}
    for name in spec['remove']:source.pop(name)
    for name,record in spec['overlays'].items():
        patch.relative(name);patch.relative(record['object'])
        raw=(ROOT/'tensorfold/variants'/record['object']).read_bytes()
        assert patch.digest(raw)==record['sha256'],name
        source[name]=record['sha256']
    assert source==spec['source_files']
    receipt,files=build.payload()
    gm=json.loads((ROOT/'gateway/source-manifest.json').read_bytes())
    historical = None
    if gm['release'] != '1.3.0':
        archive=ROOT/'gateway/history/1.3.0.tar.gz'
        assert patch.digest(archive.read_bytes())==json.loads((ROOT/'gateway/history/FILES.json').read_bytes())['files']['1.3.0.tar.gz']
        with tarfile.open(archive) as stream:
            historical={m.name:stream.extractfile(m).read() for m in stream if m.isfile()}
        gm=json.loads(historical['source-manifest.json'])
    names=({p.name for p in (ROOT/'gateway/laguna_gateway').glob('*.py')} if historical is None else {Path(n).name for n in historical if n.startswith('laguna_gateway/') and n.endswith('.py')})
    assert gm['release']=='1.3.0' and set(gm['modules'])==names
    for name,record in gm['modules'].items():
        raw=((ROOT/'gateway/laguna_gateway'/name).read_bytes() if historical is None else historical['laguna_gateway/'+name])
        assert patch.digest(raw)==record['public_sha256']
        if not record['adaptation']:assert record['production_sha256']==record['public_sha256']
    result=ROOT/'results/20261104-1.3.0'
    for name,digest in json.loads((result/'FILES.json').read_bytes())['files'].items():
        assert patch.digest((result/name).read_bytes())==digest
    return dict(passed=True,source_files=len(source),payload_files=len(files),gateway_modules=len(gm['modules']),model_requests=0,services_touched=False)


if __name__=='__main__':print(json.dumps(audit(),indent=2))
