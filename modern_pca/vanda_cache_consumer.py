
"""Read an authenticated Stage 2G cache on Vanda.

This does not regenerate preprocessing data or claim that
Vanda matches the original WSL preprocessing environment.
"""

from pathlib import Path
import hashlib

from . import full_cache as f
from . import reimplementation as r


def open_verified_cache(root, *, verify_all=False):
    root = Path(root).resolve()

    if not root.is_dir() or root.is_symlink():
        raise ValueError("Invalid cache directory")

    verification = f.read(root / "verification.json")
    if verification.get("status") != "PASS":
        raise ValueError("Original Stage 2G verification did not pass")

    actual_sha = r.sha256_file(root / "cache.json")

    if verification.get("cache_catalog_sha256") != actual_sha:
        raise ValueError("Cache catalog fingerprint mismatch")

    # Preserve the original cache provenance.
    # Do not invoke environment_contract(), which verifies
    # the environment used to GENERATE the cache.
    reader = f.FullCacheReader(root, verify_all=verify_all)

    if len(reader.items) != 145819:
        raise ValueError("Unexpected cache sample count")

    if len(reader.indices("train", 1)) != 116655:
        raise ValueError("Unexpected training count")

    if len(reader.indices("internal_validation")) != 29164:
        raise ValueError("Unexpected validation count")

    return reader
