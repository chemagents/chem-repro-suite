"""Install only the cataloged exact-version Linux asset for the current image."""

import hashlib
import json
import platform
import sys
import urllib.request
from pathlib import Path

from chem_repro.common.vina_binaries import architecture, catalog, probe

destination = Path(sys.argv[1])
asset = next(row for row in catalog()["accepted"]
             if row["system"] == platform.system() and row["architecture"] == architecture(platform.machine()))
payload = urllib.request.urlopen(asset["url"], timeout=60).read()
if hashlib.sha256(payload).hexdigest() != asset["sha256"]:
    raise ValueError("Downloaded Vina asset differs from the pinned hash")
destination.write_bytes(payload)
destination.chmod(0o755)
identity, execution = probe(destination)
print(json.dumps({"binary": identity, "execution": execution}), flush=True)
