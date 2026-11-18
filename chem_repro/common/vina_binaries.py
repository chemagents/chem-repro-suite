"""Exact, version-probed Vina assets; release filenames alone are not identity."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "data/heracleum/vina_binary_catalog.json"
CATALOG_SHA256 = "4341c03754d17d91faa45d8d860c5c1a751a66e79599396348cce37b80b0dfd4"


def catalog():
    data = CATALOG.read_bytes()
    if hashlib.sha256(data).hexdigest() != CATALOG_SHA256:
        raise ValueError("Vina binary catalog changed")
    return json.loads(data)


def identity(digest):
    records = catalog()
    for row in records["accepted"]:
        if row["sha256"] == digest:
            return row
    for row in records["excluded"]:
        if row["sha256"] == digest:
            raise ValueError("Official asset rejected: " + row["reported_version"] + "; " + row["reason"])
    raise ValueError("Vina binary SHA256 is not in the pinned exact-version catalog")


def architecture(name):
    names = {"aarch64": "aarch64", "arm64": "aarch64", "x86_64": "x86_64", "amd64": "x86_64"}
    if name.lower() not in names:
        raise ValueError("Unsupported recorded CPU architecture")
    return names[name.lower()]


def validate_recorded(asset, execution):
    """Validate the source execution, not the machine currently replaying its poses."""
    if identity(asset["sha256"]) != asset:
        raise ValueError("Recorded binary identity differs from the pinned release asset")
    host_arch = architecture(execution["machine"])
    if execution["system"] != asset["system"] or execution["version_output"] != catalog()["version_output"]:
        raise ValueError("Recorded runtime platform/version is inconsistent with the binary")
    if (type(execution["native_architecture_match"]) is not bool
            or execution["native_architecture_match"] != (host_arch == asset["architecture"])):
        raise ValueError("Recorded native/emulated execution classification is inconsistent")


def probe(binary):
    path = Path(binary)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    asset = identity(digest)
    system, machine = platform.system(), platform.machine()
    if system != asset["system"]:
        raise ValueError("Selected Vina binary targets a different operating system")
    output = subprocess.check_output([str(path.resolve()), "--version"], text=True, timeout=30).strip()
    execution = {"system": system, "machine": machine, "version_output": output,
                 "native_architecture_match": architecture(machine) == asset["architecture"]}
    validate_recorded(asset, execution)
    return asset, execution
