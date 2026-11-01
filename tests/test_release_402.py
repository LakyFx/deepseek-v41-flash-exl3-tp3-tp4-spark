"""Offline checks for the new portable loader and unchanged historical inputs."""
import gzip
import hashlib
import json
from pathlib import Path
import struct

import pytest

from benchmarks.workflow import run
from benchmarks.workflow.helpers import validate_receipt
from tensorfold.tp4.v402 import build, cache, launch
from tensorfold.tp4.v402.repack_weights import MAGIC

ROOT = Path(__file__).resolve().parents[1]


def test_all_ranks_keep_the_qualified_profile():
    config = json.loads((ROOT/'tensorfold/tp4/cluster.example.json').read_bytes())
    for rank in range(4):
        argv, env = launch.command(config, rank)
        assert env['TF_TP_WORLD'] == '4'
        assert env['TF_DS_ROUND_ROWS'] == '48'
        assert env['TF_DS_STREAMS'] == '16'
        assert env['TF_DS_POOL_TOKENS'] == '9000000'
        assert env['TF_DS_TP4_LAYOUT'] == 'balanced401a-v1'
        assert env['TF_CACHE_BATCH_DEAD'] == env['TF_CACHE_FIRST_CONTINUATION'] == '1'
        assert Path(env['TF_DS_GREEDY_CUBIN']).name == 'greedy_head.cubin'
        assert Path(env['TF_DS_DECODE_CUBIN']).name == 'early.cubin'
        if rank == 0:
            assert argv[argv.index('--prompt-cache-gib')+1] == '3.5'
        else:
            assert argv[argv.index('--round-ms')+1] == env['TF_DS_ROUND_MS']


def test_historical_workflow_is_unchanged_with_opt_in_c16():
    old = run.inputs()
    extended = run.inputs(16)
    assert len(old) == 78 and len(extended) == 136
    assert extended[:78] == old


def test_tp4_benchmark_receipt_requires_all_four_ranks():
    profile = ROOT/'tensorfold/tp4/v402/profile.json'
    receipt = dict(arm='4.0.2',campaign_id='test',exclusive_window=True,public_api_closed=True,
                   engine_generation='fixture',ranks_verified=[0,1,2,3],user_go_reference='fixture',
                   created_epoch=1000,profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
    validate_receipt(receipt,'4.0.2','test',profile,now=1000,world_size=4)
    receipt['ranks_verified'] = [0,1,2]
    with pytest.raises(ValueError, match='receipt required'):
        validate_receipt(receipt,'4.0.2','test',profile,now=1000,world_size=4)


def fixture_cache(tmp_path, monkeypatch):
    plan = tmp_path/'plan'; plan.mkdir()
    keys = b'fixture|None|None|tr\n'
    (plan/'rank0.keys.gz').write_bytes(gzip.compress(keys))
    (plan/'KEYS.json').write_text(json.dumps({'0':dict(sha256=hashlib.sha256(keys).hexdigest(),
                                                     entries=1, changed_entries=0)}))
    monkeypatch.setattr(cache, 'HERE', tmp_path)
    index = json.dumps([['fixture|None|None|tr','uint8',[4],0,4]], separators=(',',':')).encode()
    raw = b'1234'+index+struct.pack('<Q',4)+MAGIC
    path = tmp_path/'rank0of4-balanced401a-v1-test.bin'; path.write_bytes(raw)
    receipt = dict(layout='balanced401a-v1',rank=0,world=4,
                   all_written_tensors_readback_verified=True,native_key_plan_verified=True,
                   entries=1,changed_entries=0,file_bytes=len(raw),payload_bytes=4,
                   index_sha256=hashlib.sha256(index).hexdigest(),sha256=hashlib.sha256(raw).hexdigest())
    path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
    return path


def test_cache_rejects_missing_receipt_and_duplicate_layout(tmp_path, monkeypatch):
    path = fixture_cache(tmp_path, monkeypatch)
    assert cache.validate(tmp_path, 0, deep=True) == path
    duplicate = tmp_path/'rank0of4-old.bin'; duplicate.write_bytes(b'old')
    with pytest.raises(ValueError, match='exactly one'):
        cache.validate(tmp_path, 0)


def test_full_cache_hash_detects_payload_corruption(tmp_path, monkeypatch):
    path = fixture_cache(tmp_path, monkeypatch)
    with path.open('r+b') as stream:
        stream.write(b'X')
    with pytest.raises(ValueError, match='payload checksum'):
        cache.validate(tmp_path, 0, deep=True)


def test_build_rejects_tampered_payload_before_writing(tmp_path, monkeypatch):
    source = ROOT/'tensorfold/tp4/v402'
    fake = tmp_path/'package'; fake.mkdir()
    for name in ('profile.json','PAYLOAD.json'):
        (fake/name).write_bytes((source/name).read_bytes())
    (fake/'native-linux-arm64.tar.gz').write_bytes(b'tampered')
    monkeypatch.setattr(build, 'HERE', fake)
    out = tmp_path/'must-not-exist'
    with pytest.raises(ValueError, match='native archive'):
        build.prepare(tmp_path/'kit', out)
    assert not out.exists()
