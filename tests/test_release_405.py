"""Portable release boundaries; no GPU or external services."""
import json
from pathlib import Path
import pytest
from tensorfold.tp4.v405 import build,launch
from tools.audit_405 import audit

ROOT=Path(__file__).resolve().parents[1]

def test_exact_source_payload_evidence_and_history():
    assert audit()['passed']
    from tools.audit_130 import audit as historical
    assert historical()['passed']

def test_all_four_commands_keep_production_capacity_and_controllers():
    config=json.loads((ROOT/'tensorfold/tp4/cluster.example.json').read_bytes())
    for rank in range(4):
        argv,env=launch.command(config,rank)
        assert env['TF_TP_WORLD']=='4'
        assert env['TF_DS_STREAMS']=='16' and env['TF_DS_ROUND_ROWS']=='48'
        assert env['TF_DS_POOL_TOKENS']=='9000000'
        assert env['TF_DS_SMART_GRAPH']==env['TF_CACHE_ADAPTIVE_PREFIX']==env['TF_DS_ADAPTIVE_LEARNING']=='1'
        assert env['TF404_VARIANT']=='b' and env['TF404_POLICY_CONTROL']==env['TF404_DROP_WEIGHT_FILE_CACHE']=='0'
        assert env['DS41F_RECIPE_VERSION']=='4.0.5'
        assert Path(env['TF_DS_GRAPH_STATS_PATH']).name=='4.0.5.stats'
        assert '--benchmark-control' not in argv
    config['graph_statistics_path']='/srv/private/custom.stats'
    assert Path(launch.command(config,0)[1]['TF_DS_GRAPH_STATS_PATH'])==Path(config['graph_statistics_path'])

def test_corrupt_payload_is_rejected_before_destination_creation(tmp_path,monkeypatch):
    folder=tmp_path/'package';folder.mkdir()
    for n in ['profile.json','PAYLOAD.json']:(folder/n).write_bytes((ROOT/'tensorfold/tp4/v405'/n).read_bytes())
    (folder/'native-linux-arm64.tar.gz').write_bytes(b'corrupt')
    monkeypatch.setattr(build,'HERE',folder)
    out=tmp_path/'runtime'
    with pytest.raises(ValueError,match='native archive'):build.prepare(tmp_path/'kit',out)
    assert not out.exists()

def test_tp3_requires_its_explicit_profile_and_preserves_capacity():
    tp3=json.loads((ROOT/'tensorfold/tp3/v130/gateway-profile.json').read_bytes())
    tp4=json.loads((ROOT/'tensorfold/tp4/v405/gateway-profile.json').read_bytes())
    assert tp3['max_concurrency']==12 and tp4['max_concurrency']==16
    assert tp3['tensorfold_client_json'] and tp4['tensorfold_client_json']
    assert (ROOT/'gateway/history/1.3.0.tar.gz').is_file()
