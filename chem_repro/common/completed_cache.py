"""Bounded reuse of completed deterministic calls; never hides missing artifacts."""
from collections import OrderedDict
from hashlib import sha256
import json
import os
from urllib.parse import urlsplit

from .artifacts import _FILE, artifact_root
from .contracts import AnalysisResult, EvidenceNote
from .datasets import implementation_sha256


class CompletedResultCache:
    def __init__(self, maxsize=4):
        if maxsize < 1:
            raise ValueError("Cache capacity must be positive")
        self.maxsize = maxsize
        self._values = OrderedDict()

    def _intact(self, result):
        for artifact in result.artifacts:
            name = urlsplit(artifact.download_url).path.rsplit("/", 1)[-1]
            if not _FILE.fullmatch(name) or not name.startswith(artifact.sha256 + "."):
                return False
            path = artifact_root() / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact.size_bytes:
                return False
            if sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                return False
        return True

    def call(self, operation, parameters, compute, *, force_recompute=False):
        key = json.dumps({"operation": operation, "parameters": parameters,
                          "implementation": implementation_sha256(), "artifact_root": str(artifact_root()),
                          "public_base": os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "")}, sort_keys=True, allow_nan=False)
        prior = self._values.get(key)
        if not force_recompute and prior is not None and self._intact(prior):
            self._values.move_to_end(key)
            result = prior.model_copy(deep=True)
            result.metadata.reused_from_cache = True
            result.answer.finding = "Reused the identical completed computation; no fresh generation on this call. " + result.answer.finding
            result.answer.limitations.append(EvidenceNote(code="COMPLETED_RESULT_REUSED", severity="info",
                message="The same implementation and parameters completed earlier in this server process. "
                        "Artifact bytes were rechecked. This call reuses that result, not an independent rerun."))
            return result
        self._values.pop(key, None)
        result = compute()
        if not isinstance(result, AnalysisResult):
            raise TypeError("Completed scientific cache requires a validated AnalysisResult")
        if result.status != "unsupported" and self._intact(result):
            self._values[key] = result.model_copy(deep=True)
            while len(self._values) > self.maxsize:
                self._values.popitem(last=False)
        return result
