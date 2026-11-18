import json
import os
import subprocess
import sys
from pathlib import Path

from chem_repro.common.worker_environment import worker_environment


def test_installed_package_path_does_not_leak_sibling_dependencies(tmp_path):
    site = tmp_path / "main_site_packages"
    package = site / "chem_repro"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("marker='application code'\n")
    (site / "worker_dependency.py").write_text("marker='wrong caller version'\n")
    own = tmp_path / "worker_packages"
    own.mkdir()
    (own / "worker_dependency.py").write_text("marker='pinned worker version'\n")
    code = ("import sys,json;sys.path.append(" + repr(str(own)) + ");"
            "import chem_repro,worker_dependency;"
            "print(json.dumps([chem_repro.marker,worker_dependency.marker]))")
    inherited = {**os.environ, "PYTHONPATH": str(site), "PYTHONHOME": "/invalid/caller/python"}
    with worker_environment(package, base_environment=inherited) as environment:
        root = Path(environment["PYTHONPATH"])
        assert "PYTHONHOME" not in environment
        result = subprocess.check_output([sys.executable, "-c", code], text=True, env=environment,
                                         cwd=tmp_path, timeout=30)
        assert json.loads(result) == ["application code", "pinned worker version"]
        assert root.exists()
    assert not root.exists()
    assert (package / "__init__.py").is_file()


def test_declared_addon_is_available_without_caller_pythonpath(tmp_path):
    package = tmp_path / "chem_repro"
    package.mkdir()
    addon = tmp_path / "addon"
    addon.mkdir()
    with worker_environment(package, extra_import_roots=[addon], base_environment={"PYTHONPATH": "/unrelated"}) as environment:
        paths = environment["PYTHONPATH"].split(os.pathsep)
        assert paths[1:] == [str(addon)]
        assert "/unrelated" not in paths
