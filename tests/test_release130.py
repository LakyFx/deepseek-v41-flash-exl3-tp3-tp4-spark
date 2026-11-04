import json
from pathlib import Path

import pytest

from tensorfold.tp3.v130 import build, launch
from tensorfold.variants import patch

ROOT=Path(__file__).resolve().parents[1]


def test_source_and_image_fix_match_qualified_native():
    a,_=patch.recipe('v129');b,_=patch.recipe('v130')
    assert a['source_files']==b['source_files']
    assert b['source_tree_sha256']=='e909cd073cc19f6fb9fda98882e72bad8911aee7b1336c1fa2ebc0fc8bf99a30'
    for name in ['core/image_key.zig','core/lane_host_image_test.zig','server/prompt.zig']:
        assert 'zig/src/'+name in b['overlays']
    record,files=build.payload()
    assert record['source_tree_sha256']==b['source_tree_sha256']
    assert patch.digest(files['bin-1.3.0/tensorfold-native'])=='1ac317853b5e73f1c80b048f9a4b4dc3e7f75466af9798ea88f7f39392649f58'


def test_printed_rank_commands_preserve_tp3_and_clear_tp4():
    config=json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
    for rank in range(3):
        argv,env=launch.command(config,'1.3.0',rank)
        assert env['TF_TP_WORLD']=='3' and env['TF_DS_NEIGHBOR_RDMA']=='0'
        assert env['TF_DS_STREAMS']=='12' and env['TF_DS_POOL_TOKENS']=='7000000'
        assert env['TF_DS_KIT'].endswith('kit-1.3.0')
        assert env['TF_DS_SMART_GRAPH']=='1' and env['TF_CACHE_ADAPTIVE_PREFIX']=='1'
        assert env['TF404_VARIANT']=='b'
    clean=launch.clean_environment({'TF_DS_TP4_LAYOUT':'balanced','TF_DS_STREAMS':'16','PATH':'keep'},env)
    assert clean['TF_DS_TP4_LAYOUT']=='legacy' and clean['TF_DS_STREAMS']=='12' and clean['PATH']=='keep'


def test_reject_wrong_masks_and_rank():
    config=json.loads((ROOT/'tensorfold/cluster.example.json').read_bytes())
    config['ranks'][0]['peer_masks']=[0,3,12]
    with pytest.raises(ValueError,match='triangle'):launch.command(config,'1.3.0',0)
    with pytest.raises(ValueError):launch.command(config,'1.3.0',-1)


def test_gateway_profiles_do_not_cross_capacity():
    folder=ROOT/'tensorfold/tp3/v130'
    g=json.loads((folder/'gateway-profile.json').read_bytes())
    t=json.loads((ROOT/'tensorfold/tp4/v404/gateway-profile.json').read_bytes())
    assert g['max_concurrency']==12 and t['max_concurrency']==16
    assert g['tensorfold_client_json'] and g['tensorfold_internal_json']


def test_original_contribution_and_failures_preserved():
    assert patch.digest((ROOT/'tensorfold/tp3/v130/imgcache-1.2.8.diff.txt').read_bytes())=='c0af0ad93728295751081d671951f997715b21e7ff789f74de5a1f01f42a2247'
    q=json.loads((ROOT/'results/20261104-1.3.0/QUALIFICATION.json').read_bytes())
    assert q['native_heavy']['correct']==77 and q['native_heavy']['tasks']==78
    assert (ROOT/'gateway/history/4.0.4.tar.gz').is_file()
