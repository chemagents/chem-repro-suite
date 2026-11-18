"""Expose application code without leaking the caller's installed dependencies."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def worker_environment(package, *, extra_import_roots=(), base_environment=None):
    package = Path(package).resolve()
    if not package.is_dir() or not package.name.isidentifier():
        raise ValueError("Require a real Python package directory")
    with tempfile.TemporaryDirectory(prefix="chem-worker-code-") as temporary:
        root = Path(temporary)
        (root / package.name).symlink_to(package, target_is_directory=True)
        extra = [str(Path(p).resolve()) for p in extra_import_roots]
        if any(not Path(p).is_dir() for p in extra):
            raise ValueError("Declared worker import extension is unavailable")
        environment = dict(os.environ if base_environment is None else base_environment)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONPATH"] = os.pathsep.join([str(root), *extra])
        yield environment
