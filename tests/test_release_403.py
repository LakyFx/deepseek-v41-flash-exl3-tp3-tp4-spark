"""Portable release boundaries: exact payload and all-rank controller policy."""
import json
from pathlib import Path

import pytest

from tensorfold.tp4.v403 import build, launch
from tools.audit_403 import audit

ROOT = Path(__file__).resolve().parents[1]


def test_complete_release_matches_production_and_numeric_evidence():
    assert audit()['passed']


def test_all_rank_commands_enable_same_graph_protocol_and_preserve_capacity():
    config = json.loads((ROOT/'tensorfold/tp4/cluster.example.json').read_bytes())
    for rank in range(4):
        argv, env = launch.command(config, rank)
        assert env['TF_DS_SMART_GRAPH'] == '1'
        assert env['TF_DS_GRAPH_RESERVE_MIB'] == '1024'
        assert env['TF_DS_GRAPH_PROFILE_ID'] == '4.0.3-DeepSeek-V4.1-Flash-1017-EXL3-TP4-C16-R48-9M'
        assert env['TF_DS_POOL_TOKENS'] == '9000000'
        assert env['TF_DS_STREAMS'] == '16' and env['TF_DS_ROUND_ROWS'] == '48'
        assert env['TF_CACHE_ADAPTIVE_PREFIX'] == env['TF_DS_ADAPTIVE_LEARNING'] == '1'
        assert env['DS41F_RECIPE_VERSION'] == '4.0.3'
        assert Path(env['TF_DS_GRAPH_STATS_PATH']).name == '4.0.3.stats'
    config['graph_statistics_path'] = '/srv/operator-private/stats/current.stats'
    assert Path(launch.command(config, 0)[1]['TF_DS_GRAPH_STATS_PATH']) == Path(config['graph_statistics_path'])


def test_tampered_native_payload_is_rejected_before_output_creation(tmp_path, monkeypatch):
    package = tmp_path/'package'
    package.mkdir()
    for name in ('profile.json','PAYLOAD.json'):
        (package/name).write_bytes((ROOT/'tensorfold/tp4/v403'/name).read_bytes())
    (package/'native-linux-arm64.tar.gz').write_bytes(b'tampered')
    monkeypatch.setattr(build, 'HERE', package)
    out = tmp_path/'not-created'
    with pytest.raises(ValueError, match='native archive'):
        build.prepare(tmp_path/'missing-kit', out)
    assert not out.exists()
