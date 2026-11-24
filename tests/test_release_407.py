"""Offline release contract for the measured four-rank runtime."""
import hashlib
import json
from pathlib import Path
import tarfile

from tensorfold.tp4.v407 import launch
from tensorfold.variants import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tensorfold/tp4/v407"


def read_json(path):
    return json.loads(path.read_bytes())


def test_407_native_payload_and_source():
    spec, _ = patch.recipe("v407")
    profile = read_json(PACKAGE / "profile.json")
    payload = read_json(PACKAGE / "PAYLOAD.json")
    assert spec["release_version"] == profile["version"] == payload["version"] == "4.0.7"
    files = dict(read_json(ROOT / "tensorfold/variants/base.json")["files"])
    for name in spec["remove"]:
        del files[name]
    for name, obj in spec["overlays"].items():
        data = (ROOT / "tensorfold/variants" / obj["object"]).read_bytes()
        files[name] = hashlib.sha256(data).hexdigest()
        assert files[name] == obj["sha256"]
    assert files == spec["source_files"]
    assert patch.tree_digest(files) == payload["source_tree_sha256"] == profile["native_source_tree_sha256"]
    binary = PACKAGE / "native-linux-arm64.tar.gz"
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == profile["native_archive_sha256"] == payload["archive_sha256"]
    with tarfile.open(binary) as stream:
        hashes = {m.name: hashlib.sha256(stream.extractfile(m).read()).hexdigest() for m in stream}
    assert hashes == payload["files"]
    assert hashes["native/tensorfold-native"] == "ebd70a289e080839d5510a7588e2b5a1e559d5b271c97fee2ca70f60d4210a8c"


def test_each_rank_selects_the_measured_kernels_and_controller():
    cfg = read_json(ROOT / "tensorfold/tp4/cluster.example.json")
    payload = read_json(PACKAGE / "PAYLOAD.json")
    for rank, node in enumerate(cfg["ranks"]):
        _, env = launch.command(cfg, rank)
        assert env["DS41F_RECIPE_VERSION"] == "4.0.7"
        assert env["TF_DS_406F_CONTROLLER"] == env["TF_DS_406G_PREFILL"] == env["TF_DS_406G_SPEC"] == "1"
        assert env["TF_DS_PREFILL_PIPELINE"] == env["TF_DS_PREFILL_ENGRAM_AUDIT"] == "0"
        assert env["TF_DS_STREAMS"] == "16" and env["TF_DS_POOL_TOKENS"] == "9000000"
        for var, expected in {
            "TF_DS_SAMPLE_CUBIN": "134bace16bf755fcd36abf2d315a578e5bb2a38179e6f39569ec79e441466610",
            "TF_DS_PREFILL_RAGGED_CUBIN": "9e69540ba5240208cc80abddb3a6ac7b4887ff20cbc5645cd81fd2531f923e9b",
        }.items():
            member = Path(env[var]).relative_to(node["root"]).as_posix()
            assert payload["files"][member] == expected


def test_results_and_gateway_use_release_version():
    results = read_json(ROOT / "results/20261124-4.0.7/PRODUCTION.json")
    assert results["version"] == read_json(ROOT / "gateway/source-manifest.json")["release"] == "4.0.7"
    assert read_json(PACKAGE / "gateway-profile.json") == read_json(ROOT / "gateway/laguna_gateway/production-settings.json")
    assert [row["concurrency"] for row in results["decode"]] == list(range(1, 17))
    c16 = results["decode"][-1]
    assert c16["median_aggregate_decode_tok_s"] == {"4.0.5": None, "4.0.7": 280.5}
    assert c16["delta_percent"] is None
