"""VAL2 ENS routing and authenticated ZIP-source tests; no models or real data."""
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pytest

from ens_model_training.ens_val2 import Val2ZipSource, route_ens_scores
from modern_pca import reimplementation as r


def val2_row(relative, label, payload):
    return dict(
        cohort="VAL2",
        class_name=label,
        ground_truth_code=r.CLASSES[label],
        relative_path=f"{label}/{relative}",
        sample_id=f"VAL2:{label}/{relative}",
        file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_route_ens_scores_uses_model2_only_inside_inclusive_band():
    native = [0.19, 0.2, 0.45, 0.8, 0.81]
    model2 = [0.99, 0.9, 0.1, 0.1, 0.01]

    final, triggered = route_ens_scores(native, model2)

    np.testing.assert_array_equal(triggered, [False, True, True, True, False])
    np.testing.assert_allclose(final, [0.19, 0.9, 0.1, 0.1, 0.81])


@pytest.fixture
def val2_zip_source(tmp_path):
    payloads = {"benign": b"benign-image", "tumour": b"tumour-image"}
    paths, expected = {}, {}
    rows = {}
    for label, payload in payloads.items():
        filename = f"{label}.zip"
        path = tmp_path / filename
        relative = "norm/norm.0.jpg" if label == "benign" else "tu/tu.0.jpg"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(relative, payload)
        paths[label] = path
        expected[label] = hashlib.md5(path.read_bytes()).hexdigest()
        rows[label] = val2_row(relative, label, payload)
    return Val2ZipSource(paths, expected), rows, payloads


def test_val2_zip_source_checks_members_and_image_hashes(val2_zip_source):
    source, rows, payloads = val2_zip_source
    try:
        ordered = [rows["benign"], rows["tumour"]]
        source.check(ordered)
        assert source.read(rows["benign"]) == payloads["benign"]
        assert source.read(rows["tumour"]) == payloads["tumour"]
        with pytest.raises(ValueError, match="SHA-256"):
            source.read(dict(rows["benign"], file_sha256="0" * 64))
        with pytest.raises(ValueError, match="absent"):
            source.check([val2_row("norm/missing.jpg", "benign", b"missing")])
        with pytest.raises(ValueError, match="VAL2"):
            source.member(dict(rows["benign"], cohort="VAL1"))
    finally:
        source.close()