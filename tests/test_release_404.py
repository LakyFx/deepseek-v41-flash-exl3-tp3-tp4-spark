"""Portable 4.0.4 release integrity and all-rank compatibility boundaries."""
import json
from pathlib import Path
import pytest
from tensorfold.tp4.v404 import build, launch
from tools.audit_404 import audit

ROOT=Path(__file__).resolve().parents[1]


def test_release_source_payload_metrics_and_gateway_match():
    assert audit()['passed']


def test_all_ranks_select_qualified_b_and_keep_controllers():
    config=json.loads((ROOT/'tensorfold/tp4/cluster.example.json').read_bytes())
    for rank in range(4):
        argv,env=launch.command(config,rank)
        assert env['TF404_VARIANT']=='b'
        assert env['TF404_POLICY_CONTROL']==env['TF404_DROP_WEIGHT_FILE_CACHE']=='0'
        assert env['TF_DS_SMART_GRAPH']==env['TF_CACHE_ADAPTIVE_PREFIX']==env['TF_DS_ADAPTIVE_LEARNING']=='1'
        assert env['TF_DS_POOL_TOKENS']=='9000000'
        assert env['TF_DS_STREAMS']=='16' and env['TF_DS_ROUND_ROWS']=='48'
        assert env['DS41F_RECIPE_VERSION']=='4.0.4'
        assert Path(env['TF_DS_GRAPH_STATS_PATH']).name=='4.0.4-B.stats'
    config['graph_statistics_path']='/srv/private/custom.stats'
    assert Path(launch.command(config,0)[1]['TF_DS_GRAPH_STATS_PATH'])==Path(config['graph_statistics_path'])


def test_corrupt_payload_cannot_create_runtime(tmp_path,monkeypatch):
    package=tmp_path/'package';package.mkdir()
    for n in ('profile.json','PAYLOAD.json'):(package/n).write_bytes((ROOT/'tensorfold/tp4/v404'/n).read_bytes())
    (package/'native-linux-arm64.tar.gz').write_bytes(b'corrupt')
    monkeypatch.setattr(build,'HERE',package)
    out=tmp_path/'runtime'
    with pytest.raises(ValueError,match='native archive'):build.prepare(tmp_path/'kit',out)
    assert not out.exists()
