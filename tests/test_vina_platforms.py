import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chem_repro.common import vina_binaries
from chem_repro.heracleum import redocking


def test_exact_assets_exclude_ahead_of_tag_mac_arm_build():
    assert len(vina_binaries.catalog()["accepted"]) == 3
    with pytest.raises(ValueError, match="1.2.5-17-gda92a68"):
        vina_binaries.identity("4999b3133f162b66d8b21bafdb54aa53c507ed6347489844e6778dac7871ff5f")
    with pytest.raises(ValueError, match="SHA256"):
        vina_binaries.identity("0" * 64)


def test_cached_binary_validation_uses_recorded_runtime_and_tracks_emulation():
    asset = vina_binaries.identity("fa0126a28a9ea9162d1b161dfa92bc76e632416db28ca246278ea4b2dc6860cb")
    execution = {"system": "Linux", "machine": "aarch64", "version_output": "AutoDock Vina v1.2.5",
                 "native_architecture_match": False}
    vina_binaries.validate_recorded(asset, execution)
    with pytest.raises(ValueError, match="classification"):
        vina_binaries.validate_recorded(asset, {**execution, "native_architecture_match": True})
    with pytest.raises(ValueError, match="platform/version"):
        vina_binaries.validate_recorded(asset, {**execution, "version_output": "AutoDock Vina v1.2.5-17"})


def test_named_fake_binary_is_rejected_before_execution(tmp_path, monkeypatch):
    binary = tmp_path / "vina_1.2.5_linux_x86_64"
    binary.write_text("#!/bin/sh\necho 'AutoDock Vina v1.2.5'\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unrecognized executable must not be run")

    monkeypatch.setattr(vina_binaries.subprocess, "check_output", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        vina_binaries.probe(binary)


def test_legacy_preparation_worker_is_unchanged():
    assert redocking.sha(redocking.WORKER) == "a1feea5ee9930301f48781fcf5ce8b00d1fe5b3e37ab25a22251a71de5722ff5"


def test_portable_cached_spec_keeps_original_preparation_and_binary_identity(tmp_path):
    # Reuse the full42target fixture, preserving the original v1 acceptance tests.
    from test_redocking import failed_panel

    spec, _ = failed_panel(tmp_path)
    asset = vina_binaries.identity("5bf4eba87f133b5c2d35e5e6c21c062221e2a35a91ee7fa1dbd69d341efb3516")
    spec.update(protocol="public_crystal_redocking_reconstruction.v2",
                worker_sha256=redocking.sha(redocking.PORTABLE_WORKER),
                preparation_worker_sha256=redocking.sha(redocking.WORKER),
                binary_policy_sha256=redocking.sha(Path(vina_binaries.__file__)),
                binary_catalog_sha256=vina_binaries.CATALOG_SHA256,
                binary_sha256=asset["sha256"], binary_identity=asset,
                execution_platform={"system": "Linux", "machine": "aarch64",
                                    "version_output": "AutoDock Vina v1.2.5", "native_architecture_match": True})
    (tmp_path / "run_spec.json").write_text(json.dumps(spec))
    _, result = redocking.inspect_run(tmp_path)
    assert result["finished"] == 42 and result["computed"] == 0
    spec["preparation_worker_sha256"] = "0" * 64
    (tmp_path / "run_spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="protocol"):
        redocking.inspect_run(tmp_path)


def test_portable_entrypoint_records_platform_and_refuses_relabeling(tmp_path, monkeypatch):
    # Optional chemical dependencies are replaced only in this control-flow test.
    # Actual binaries and archived chemistry are verified separately by integration checks.
    core_file = tmp_path / "frozen_core.py"
    core_file.write_text("unit fixture")
    core = SimpleNamespace(PINS={}, FIDELITY="unit fixture", __file__=str(core_file),
                           digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                           write_json=lambda p, v: Path(p).write_text(json.dumps(v)),
                           run_target=lambda row, *args: {"target": row["target"], "status": "failed",
                                                         "failure_phase": "unit_fixture", "elapsed_seconds": 0})
    from chem_repro import vendor

    monkeypatch.setattr(vendor, "heracleum_redocking_worker", core, raising=False)
    monkeypatch.setitem(sys.modules, "chem_repro.vendor.heracleum_redocking_worker", core)
    spec = importlib.util.spec_from_file_location("portable_entrypoint_test", redocking.PORTABLE_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    asset = vina_binaries.identity("5bf4eba87f133b5c2d35e5e6c21c062221e2a35a91ee7fa1dbd69d341efb3516")
    execution = {"system": "Linux", "machine": "aarch64", "version_output": "AutoDock Vina v1.2.5",
                 "native_architecture_match": True}
    monkeypatch.setattr(vina_binaries, "probe", lambda path: (asset, execution))
    table = tmp_path / "table.csv"
    table.write_text("target,evaluated\nTEST1,True\nTEST2,False\n")
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["worker", "--table", str(table), "--output", str(output),
                                     "--sources", str(tmp_path), "--vina", "fixture"])
    module.main()
    record = json.loads((output / "run_spec.json").read_text())
    assert record["binary_identity"] == asset
    assert record["execution_platform"] == execution
    assert record["preparation_worker_sha256"] == core.digest(core_file)
    (output / "run_spec.json").unlink()
    with pytest.raises(ValueError, match="refuse to relabel"):
        module.main()


def test_relocated_structures_require_bytes_and_receipt_not_historical_absolute_path(tmp_path, monkeypatch):
    root = tmp_path / "relocated"
    directory = root / "2ydo"
    directory.mkdir(parents=True)
    path = directory / "2ydo.cif"
    path.write_bytes(b"immutable structural input")
    expected = redocking.sha(path)
    path.with_suffix(".cif.source.json").write_text(json.dumps({"sha256": expected, "url": "https://files.rcsb.org/download/2YDO.cif"}))
    monkeypatch.setenv("CHEM_REPRO_REDOCKING_SOURCES_DIR", str(root))
    old_locator = "/old-machine/source/2ydo/2ydo.cif"
    resolved, _, receipt = redocking.structural_source(old_locator, expected, "2YDO")
    assert resolved == path and receipt["sha256"] == expected
    path.write_bytes(b"different structure at the same filename")
    with pytest.raises(ValueError, match="changed"):
        redocking.structural_source(old_locator, expected, "2YDO")
