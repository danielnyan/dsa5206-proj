"""Fast read-only PyTorch adapter for the already verified Stage-2G VAL1 cache.

No preprocessing, image hashing, cache generation or TensorFlow execution occurs
in __getitem__. The canonical full-cache reader authenticates the frozen sample
mapping and shard metadata *once* during construction (verify_all=False).
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import full_cache
from . import reimplementation as r

PREPROCESSING = "Stage2G-v1-Lanczos350-brightness-Macenko-uint8-div255"


class FastStage2GPatches(Dataset):
    """Return (CHW uint8, integer label) using lazy shard mmap handles.

    Each DataLoader worker owns its own LRU mmap handles. The OS page cache is
    shared; a memory map is not a copy of the entire shard in physical RAM.
    """

    def __init__(self, root, rows, by_id, *, split, max_shards=256):
        self.root = Path(root)
        self.records = []
        self.max_shards = int(max_shards)
        if self.max_shards < 1:
            raise ValueError("max_shards must be positive")
        self._maps = OrderedDict()

        for row in rows:
            sample_id = row["sample_id"]
            try:
                item = by_id[sample_id]
            except KeyError as e:
                raise ValueError(f"Missing Stage-2G image: {sample_id}") from e
            label = int(row["ground_truth_code"])
            if (item["split"] != split
                    or int(item["ground_truth_code"]) != label
                    or item["class_name"] != row["class_name"]
                    or item["relative_path"] != row["relative_path"]
                    or item["source_jpeg_sha256"] != row["file_sha256"]):
                raise ValueError(f"Stage-2G metadata mismatch: {sample_id}")
            shard = item["shard"]
            if Path(shard).name != shard or not shard.endswith(".npy"):
                raise ValueError(f"Unsafe shard filename: {shard}")
            self.records.append((shard, int(item["index"]), label))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name, offset, label = self.records[index]
        if name not in self._maps:
            path = self.root / name
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 4 or array.dtype != np.uint8 or array.shape[1:] != (350, 350, 3):
                raise ValueError(f"Unexpected Stage-2G shard shape/dtype: {name}")
            self._maps[name] = array
            if len(self._maps) > self.max_shards:
                self._maps.popitem(last=False)
        self._maps.move_to_end(name)

        pixels = self._maps[name][offset]
        # Materialize only one uint8 image. CHW data must be contiguous for
        # efficient pinning and CPU -> GPU transfers.
        chw = np.array(pixels.transpose(2, 0, 1), copy=True, order="C")
        return torch.from_numpy(chw), label


def load_stage2g_datasets(cache_root, train_rows, validation_rows):
    """Validate catalog/manifest once, return fast datasets + catalog digest.

    This deliberately skips verify_all and per-image hash checks. Do NOT apply
    it to unverified/untrusted caches or silently change their content.
    """
    root = Path(cache_root).resolve(strict=True)
    verification = full_cache.read(root / "verification.json")
    catalog_sha = r.sha256_file(root / "cache.json")
    if (verification.get("status") != "PASS"
            or verification.get("cache_catalog_sha256") != catalog_sha):
        raise ValueError("Unverified Stage-2G cache or catalog mismatch")

    reader = full_cache.FullCacheReader(root, verify_all=False)
    by_id = {item["sample_id"]: item for item in reader.items}
    if len(by_id) != len(reader.items):
        raise ValueError("Duplicate Stage-2G sample ID")
    train = FastStage2GPatches(root, train_rows, by_id, split="train")
    val = FastStage2GPatches(root, validation_rows, by_id, split="internal_validation")
    if len(train) != 116655 or len(val) != 29164:
        raise ValueError("Unexpected Stage-2G train/validation counts")
    return train, val, catalog_sha
