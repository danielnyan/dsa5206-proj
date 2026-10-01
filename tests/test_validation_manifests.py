"""Small offline fixtures; no full datasets or TensorFlow required."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from tools import build_validation_manifests as tool


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    base = tmp_path / "dataset"
    counts = {c: {label: 2 for label in tool.CLASSES} for c in tool.COUNTS}
    monkeypatch.setattr(tool, "COUNTS", counts)
    archive_records = []
    index = 0
    for cohort in counts:
        for label in tool.CLASSES:
            root = base / "extracted" / tool.ARCHIVES[cohort, label]
            (root / "nested").mkdir(parents=True)
            # Distinct pixel content, dimension variation, deliberately unsorted creation.
            for name, size in [("z.png", (12, 10)), ("nested/a.png", (11, 10))]:
                index += 1
                Image.new("RGB", size, (index*10, 40, 60)).save(root / name)
            archive_records.append(dict(cohort=cohort, class_name=label, image_files=2,
                                         dimensions={"12x10": 1, "11x10": 1}, modes={"RGB": 2}))
    report = tmp_path / "stage2a.json"
    report.write_text(json.dumps(dict(status="verification_complete", all_archive_md5_passed=True, archives=archive_records)))
    return base, report


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_generation_portable_deterministic_complete(dataset, tmp_path):
    base, report = dataset
    originals = {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*.png")}
    first = tmp_path / "first"
    summary = tool.generate(base, first, report, workers=1)
    relocated = tmp_path / "relocated"
    shutil.copytree(base, relocated)
    second = tmp_path / "second"
    tool.generate(relocated, second, report, workers=4)
    assert len(list(first.iterdir())) == 8
    all_ids = []
    for cohort in tool.COUNTS:
        combined = read_csv(first / f"{cohort}_manifest.csv")
        assert len(combined) == 4
        assert [int(r["row_index"]) for r in combined] == list(range(4))
        assert [r["class_name"] for r in combined] == ["benign", "benign", "tumour", "tumour"]
        assert [r["ground_truth_code"] for r in combined] == ["0", "0", "1", "1"]
        assert combined[0]["relative_path"] == "benign/nested/a.png"
        for row in combined:
            assert row["sample_id"] == cohort + ":" + row["relative_path"]
            relative = row["relative_path"].split("/", 1)[1]
            path = base / "extracted" / tool.ARCHIVES[cohort, row["class_name"]] / relative
            assert row["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert row["mode"] == "RGB" and row["width"] in ("11", "12")
        all_ids += [r["sample_id"] for r in combined]
        for suffix in ("benign", "tumour", "manifest"):
            name = f"{cohort}_{suffix}.csv"
            payload = (first / name).read_bytes()
            assert payload == (second / name).read_bytes()
            assert b"\r" not in payload and not payload.startswith(b"\xef\xbb\xbf")
            assert str(base).encode() not in payload
            rows = read_csv(first / name)
            assert list(rows[0]) == tool.FIELDS
            assert [int(r["row_index"]) for r in rows] == list(range(len(rows)))
        assert summary["cohorts"][cohort]["manifest_sha256"] == hashlib.sha256((first / f"{cohort}_manifest.csv").read_bytes()).hexdigest()
        for label in tool.CLASSES:
            assert {r["sample_id"] for r in read_csv(first / f"{cohort}_{label}.csv")} == {r["sample_id"] for r in combined if r["class_name"] == label}
    assert len(set(all_ids)) == 8
    assert originals == {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*.png")}
    assert summary["source_images_modified"] is False


def test_normalized_paths():
    assert tool.normalized_path("benign\\nested\\a.png") == "benign/nested/a.png"
    assert tool.normalized_path("benign/e\u0301.png") == "benign/\u00e9.png"
    for path in ("../x", "/x", "C:/x", "a//b"):
        with pytest.raises(ValueError):
            tool.normalized_path(path)


@pytest.mark.parametrize("field,value", [("sample_id", "bad"), ("row_index", 7), ("cohort", "VAL2"),
    ("ground_truth_code", 1), ("class_name", "other"), ("file_sha256", ""), ("width", 0), ("height", None),
    ("mode", ""), ("relative_path", "../x")])
def test_row_validation_rejects_bad_fields(field, value):
    row = dict(row_index=0, sample_id="VAL1:benign/a.png", cohort="VAL1", class_name="benign",
               ground_truth_code=0, relative_path="benign/a.png", file_sha256="a"*64,
               file_size_bytes=10, width=11, height=10, mode="RGB")
    row[field] = value
    with pytest.raises(ValueError):
        tool.validate_rows([row], "VAL1", {"benign": 1})


def test_duplicates_rejected():
    row = dict(sample_id="VAL1:benign/a.png", cohort="VAL1", class_name="benign", ground_truth_code=0,
               relative_path="benign/a.png", file_sha256="a"*64, file_size_bytes=10, width=2, height=2, mode="RGB")
    with pytest.raises(ValueError, match="Duplicate"):
        tool.validate_rows(tool.ordered_rows([row, copy.deepcopy(row)]), "VAL1", {"benign": 2})


@pytest.mark.parametrize("problem", ["missing", "extra", "corrupt", "duplicate_content", "stage2a_failed"])
def test_generation_fails_before_publication(dataset, tmp_path, problem):
    base, report = dataset
    root = base / "extracted" / tool.ARCHIVES["VAL1", "benign"]
    if problem == "missing":
        (root / "z.png").unlink()
    elif problem == "extra":
        (root / "unexpected.txt").write_text("unexpected")
    elif problem == "corrupt":
        (root / "z.png").write_bytes(b"invalid image")
    elif problem == "duplicate_content":
        shutil.copyfile(root / "z.png", root / "nested/a.png")
    else:
        doc = json.loads(report.read_text()); doc["all_archive_md5_passed"] = False
        report.write_text(json.dumps(doc))
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        tool.generate(base, output, report)
    assert not output.exists()


def test_existing_output_preserved(dataset, tmp_path):
    base, report = dataset
    output = tmp_path / "output"; output.mkdir()
    sentinel = output / "keep"; sentinel.write_text("unchanged")
    with pytest.raises(ValueError, match="refusing overwrite"):
        tool.generate(base, output, report)
    assert sentinel.read_text() == "unchanged"


def test_case_collision_inventory(tmp_path, monkeypatch):
    # Simulates a case-sensitive filesystem without relying on Windows creating aliases.
    monkeypatch.setattr(tool.os, "walk", lambda *args, **kwargs: [(str(tmp_path), [], ["A.png", "a.png"])])
    with pytest.raises(ValueError, match="collision"):
        tool.inventory(tmp_path)
