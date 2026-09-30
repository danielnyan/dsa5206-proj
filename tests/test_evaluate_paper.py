"""Offline, lightweight C1 contract tests. No author weights, GPU or downloads."""
from __future__ import annotations

from contextlib import nullcontext
import ast
import copy
import csv
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from modern_pca import evaluate_paper as ep


def image_file(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color).save(path)
    return path


@pytest.fixture
def cohort(tmp_path):
    benign, tumour = tmp_path / "benign", tmp_path / "tumour"
    image_file(benign / "z.png", (10, 20, 30))
    image_file(benign / "nested" / "a.png", (20, 30, 40))
    image_file(tumour / "a.png", (100, 120, 130))
    return benign, tumour


@pytest.fixture
def manifest(cohort):
    return ep.build_manifest(*cohort, "FIXTURE")


def relu(value):
    return value


def softmax(value):
    return value


class Dense:
    def __init__(self, units, activation):
        self.units, self.activation = units, activation


class Flatten:
    pass


def model_contract_fixture(width=3, shape=(None, 350, 350, 3)):
    """Structural double only; not a real or authenticated NASNet checkpoint."""
    backbone = SimpleNamespace(output_shape=(None, 11, 11, 4032), layers=[
        SimpleNamespace(name=name) for name in ["stem_conv1", *[f"normal_conv_1_{i}" for i in range(1, 19)]]])
    return SimpleNamespace(inputs=[SimpleNamespace(shape=shape)], outputs=[SimpleNamespace(shape=(None, width))],
                           layers=[backbone, Flatten(), Dense(256, relu), Dense(3, softmax)],
                           trainable_weights=[], non_trainable_weights=[], count_params=lambda: 123,
                           to_json=lambda: '{"fixture":true}')


def cli(tmp_path, **overrides):
    values = {"purpose": "sanity", "model": str(tmp_path / "model.h5"),
              "class-order": "gland,nongland,tumour", "cohort": "FIXTURE",
              "benign-dir": str(tmp_path / "benign"), "tumour-dir": str(tmp_path / "tumour"),
              "stain-reference": str(tmp_path / "reference.png"), "device": "cpu",
              "output": str(tmp_path / "output")}
    values.update(overrides)
    return [item for key, value in values.items() for item in ["--" + key, str(value)]]


def metadata(model_hash):
    return {"model_sha256": model_hash, "source": "controlled test fixture; not author weights",
            "acquisition_date": "2026-01-01", "identity": "native", "architecture": "NASNetLarge",
            "class_order": ep.CLASS_ORDER, "class_order_evidence": "fixture specification",
            "conversion_history": [], "input_shape": [None, 350, 350, 3], "output_shape": [None, 3]}


class HostOutput:
    def __init__(self, array, events=None):
        self.array, self.events = array, events

    def numpy(self):
        if self.events is not None:
            self.events.append("materialized")
        return self.array


class Predictor:
    def __init__(self):
        self.calls = 0
        self.presentations = 0
        self.weights = np.array([1.0, 2.0])

    def __call__(self, batch, training):
        assert training is False
        self.calls += 1
        self.presentations += len(batch)
        return HostOutput(np.tile(np.array([0.30, 0.25, 0.45], dtype=np.float32), (len(batch), 1)))


def fake_preprocess(row, path):
    return np.zeros((2, 2, 3), dtype=np.float32)


def test_threshold_and_argmax_regression():
    assert ep.binary_decision([0.499999, 0.5, 0.500001]).tolist() == [0, 0, 1]
    probabilities = np.array([[0.30, 0.25, 0.45]])
    assert probabilities.argmax(axis=1).tolist() == [2]
    assert ep.binary_decision(probabilities[:, 2]).tolist() == [0]


def test_hand_metrics():
    result = ep.calculate_binary_metrics([0, 0, 0, 1, 1, 1], [0.1, 0.6, 0.4, 0.9, 0.8, 0.3])
    assert result["confusion_matrix"] == [[2, 1], [1, 2]]
    assert [result[key] for key in ["TN", "FP", "FN", "TP"]] == [2, 1, 1, 2]
    for key in ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1"]:
        assert result[key] == pytest.approx(2 / 3)
    assert result["roc_auc"] == pytest.approx(7 / 9)


def test_auc_ties_and_undefined_precision():
    result = ep.calculate_binary_metrics([0, 1], [0.5, 0.5])
    assert result["roc_auc"] == 0.5
    assert result["precision"] is None
    assert "precision" in result["undefined_metrics"]
    assert result["exact_threshold_ties"] == 2


def test_fake_123_parameter_model_rejected():
    with pytest.raises(ep.EvaluationError, match="parameter count"):
        ep.validate_model_contract(model_contract_fixture(), {"class_order": ep.CLASS_ORDER})


@pytest.mark.parametrize("width", [1, 2, 4])
def test_output_width_rejected(width):
    with pytest.raises(ep.EvaluationError, match="exactly 3"):
        ep.validate_model_contract(model_contract_fixture(width), {"class_order": ep.CLASS_ORDER})


@pytest.mark.parametrize("shape", [(None, 64, 64, 3), (None, 3, 350, 350), (None, 350, 350, 1)])
def test_input_shape_rejected(shape):
    with pytest.raises(ep.EvaluationError, match="input shape"):
        ep.validate_model_contract(model_contract_fixture(shape=shape), {"class_order": ep.CLASS_ORDER})


@pytest.mark.parametrize("mapping", [None, [], ["tumour", "gland", "nongland"]])
def test_mapping_rejected(mapping):
    with pytest.raises(ep.EvaluationError, match="class mapping"):
        ep.validate_model_contract(model_contract_fixture(), {"class_order": mapping})


def test_nonpaper_backbone_and_head_rejected():
    model = model_contract_fixture()
    model.layers[0].layers = []
    with pytest.raises(ep.EvaluationError, match="NASNetLarge"):
        ep.validate_model_contract(model, {"class_order": ep.CLASS_ORDER})
    model = model_contract_fixture()
    model.layers[2].units = 128
    with pytest.raises(ep.EvaluationError, match="original"):
        ep.validate_model_contract(model, {"class_order": ep.CLASS_ORDER})


def test_mixed_precision_model_rejected():
    model = model_contract_fixture()
    model.outputs[0].dtype = "float16"
    with pytest.raises(ep.EvaluationError, match="float32"):
        ep.validate_model_contract(model, {"class_order": ep.CLASS_ORDER})


def test_sidecar_required_and_checksum_bound(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata("a" * 64)))
    assert ep.load_checkpoint_metadata(path, "a" * 64, "paper", ep.CLASS_ORDER)["identity"] == "native"
    with pytest.raises(ep.EvaluationError, match="SHA-256"):
        ep.load_checkpoint_metadata(path, "b" * 64, "paper", ep.CLASS_ORDER)
    with pytest.raises(ep.EvaluationError, match="requires"):
        ep.load_checkpoint_metadata(None, "a" * 64, "paper", ep.CLASS_ORDER)
    doc = metadata("a" * 64)
    doc["identity"] = "post-trained"
    path.write_text(json.dumps(doc))
    with pytest.raises(ep.EvaluationError, match="conflicts"):
        ep.load_checkpoint_metadata(path, "a" * 64, "paper", ep.CLASS_ORDER)


def test_manifest_complete_no_split_and_deterministic(manifest, cohort):
    assert manifest.counts == {"total": 3, "benign": 2, "tumour": 1}
    assert [row["relative_path"] for row in manifest.rows] == ["benign/nested/a.png", "benign/z.png", "tumour/a.png"]
    assert manifest.sha256 == ep.build_manifest(*cohort, "FIXTURE").sha256
    assert [row["row_index"] for row in manifest.rows] == [0, 1, 2]


def test_manifest_hash_root_independent(manifest, cohort, tmp_path):
    relocated = tmp_path / "relocated"
    shutil.copytree(cohort[0], relocated / "benign")
    shutil.copytree(cohort[1], relocated / "tumour")
    other = ep.build_manifest(relocated / "benign", relocated / "tumour", "FIXTURE")
    assert manifest.sha256 == other.sha256
    assert ep.csv_bytes(manifest.rows, ep.MANIFEST_FIELDS) == ep.csv_bytes(other.rows, ep.MANIFEST_FIELDS)


@pytest.mark.parametrize("paths", [["benign/a.jpg", "benign/a.jpg"],
                                   ["benign/A.jpg", "benign/a.jpg"],
                                   ["benign/\u00e9.jpg", "benign/e\u0301.jpg"]])
def test_logical_path_collisions(paths):
    with pytest.raises(ep.EvaluationError, match="collision"):
        ep.check_logical_paths(paths)


def test_same_basename_distinct_paths_allowed(cohort):
    image_file(cohort[0] / "a.png", (45, 55, 65))
    manifest = ep.build_manifest(*cohort, "FIXTURE")
    assert len(manifest.rows) == 4
    assert len({row["sample_id"] for row in manifest.rows}) == 4


def test_duplicate_content_retained_and_reported(cohort):
    shutil.copyfile(cohort[0] / "z.png", cohort[0] / "copy.png")
    manifest = ep.build_manifest(*cohort, "FIXTURE")
    assert len(manifest.rows) == 4
    assert manifest.duplicate_content == [["benign/copy.png", "benign/z.png"]]


def test_conflicting_label_content_rejected(cohort):
    shutil.copyfile(cohort[0] / "z.png", cohort[1] / "conflict.png")
    with pytest.raises(ep.EvaluationError, match="Conflicting-label"):
        ep.build_manifest(*cohort, "FIXTURE")


def test_duplicate_resolved_path_rejected(cohort, monkeypatch):
    original = Path.resolve
    target = (cohort[0] / "z.png").resolve()
    def alias(path, *args, **kwargs):
        if path.name == "a.png" and path.parent.name == "nested":
            return target
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", alias)
    with pytest.raises(ep.EvaluationError, match="physical path"):
        ep.build_manifest(*cohort, "FIXTURE")


def test_empty_directory_rejected(cohort, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ep.EvaluationError, match="Empty"):
        ep.build_manifest(empty, cohort[1], "FIXTURE")


def test_unreadable_and_ambiguous_images_rejected(cohort):
    bad = cohort[0] / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ep.EvaluationError, match="Unreadable"):
        ep.build_manifest(*cohort, "FIXTURE")
    bad.unlink()
    (cohort[0] / "ambiguous.gif").write_bytes(b"anything")
    with pytest.raises(ep.EvaluationError, match="ambiguous"):
        ep.build_manifest(*cohort, "FIXTURE")


def test_manifest_image_mutation_detected(manifest, cohort):
    image_file(cohort[0] / "z.png", (1, 1, 1))
    with pytest.raises(ep.EvaluationError, match="changed"):
        ep.validate_manifest(manifest)


def test_manifest_addition_detected(manifest, cohort):
    image_file(cohort[0] / "new.png", (4, 4, 4))
    with pytest.raises(ep.EvaluationError, match="changed"):
        ep.validate_manifest(manifest)


@pytest.mark.parametrize("artifact", ["reference.png", "model.h5"])
def test_artifact_mutation_detected(tmp_path, artifact):
    path = tmp_path / artifact
    path.write_bytes(b"original")
    digest = ep.sha256_file(path)
    path.write_bytes(b"mutated")
    with pytest.raises(ep.EvaluationError, match="SHA-256 mismatch"):
        ep.sha256_file(path, digest)


@pytest.mark.parametrize("values", [[np.nan, 0, 1], [np.inf, 0, 1], [0.1, 0.2, 0.3], [-0.1, 0.5, 0.6], [0, 0, 1.1]])
def test_invalid_probabilities_rejected_with_identity(values):
    with pytest.raises(ep.EvaluationError, match="sample-42"):
        ep.validate_probabilities([values], ["sample-42"])


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_c1_exactly_once_order_no_updates_and_benign_sum(manifest, batch_size):
    model = Predictor()
    before = model.weights.copy()
    predictions, accounting = ep.infer_c1(model, manifest, fake_preprocess, batch_size,
                                           ep.Timings(), logging.getLogger("test"))
    assert model.presentations == len(manifest.rows)
    assert model.calls == (len(manifest.rows) + batch_size - 1) // batch_size
    assert accounting["cohort_image_presentations"] == len(manifest.rows)
    assert np.array_equal(before, model.weights)
    assert [row["sample_id"] for row in predictions] == [row["sample_id"] for row in manifest.rows]
    for row in predictions:
        assert row["p_benign"] == row["p_gland"] + row["p_nongland"]
        assert row["predicted_binary_label"] == "benign"


def test_full_precision_roundtrip(manifest, tmp_path):
    predictions, _ = ep.infer_c1(Predictor(), manifest, fake_preprocess, 1, ep.Timings(), logging.getLogger("test"))
    predictions[0].update(p_gland=float(np.float32(0.123456789)), p_nongland=float(np.float32(0.37654319)),
                          p_tumour=float(np.nextafter(np.float32(0.5), np.float32(1.0))),
                          predicted_binary_label="tumour", predicted_binary_code=1, correct=False)
    predictions[0]["p_benign"] = predictions[0]["p_gland"] + predictions[0]["p_nongland"]
    metrics = ep.calculate_binary_metrics([row["ground_truth_code"] for row in predictions], [row["p_tumour"] for row in predictions])
    output = tmp_path / "predictions.csv"
    ep.atomic_write(output, ep.csv_bytes(predictions, ep.PREDICTION_FIELDS))
    ep.verify_prediction_export(output, predictions, metrics)
    assert "0.50000005960464478" in output.read_text()


def test_timing_derivations():
    times = iter([1.0, 3.0])
    timings = ep.Timings(clock=lambda: next(times))
    with timings.measure("inference"):
        pass
    timings.seconds["preprocessing"] = 4.0
    report = timings.report(10, 10.0)
    assert report["average_preprocessing_ms_per_image"] == 400
    assert report["average_inference_ms_per_image"] == 200
    assert report["end_to_end_ms_per_image"] == 1000
    assert report["inference_images_per_second"] == 5
    assert report["end_to_end_images_per_second"] == 1


def test_host_materialized_before_inference_timer_stops(manifest):
    events = []
    timings = ep.Timings(clock=lambda: (events.append("timer") or float(len(events))))
    class Model:
        def __call__(self, batch, training):
            events.append("forward")
            return HostOutput(np.tile([0.25, 0.25, 0.5], (len(batch), 1)), events)
    ep.infer_c1(Model(), manifest, fake_preprocess, 3, timings, logging.getLogger("test"))
    index = events.index("forward")
    assert events[index:index + 3] == ["forward", "materialized", "timer"]


def test_progress_intervals_and_no_prediction_changes(manifest, caplog):
    logger = logging.getLogger("paper-progress-test")
    caplog.set_level(logging.INFO, logger=logger.name)
    times = iter([0.0, 1.0, 31.0, 32.0])
    progress = ep.Progress(100, 50, 30, logger, lambda: next(times))
    progress.update(1)
    assert not caplog.records
    progress.update(2)
    progress.update(100)
    assert len(caplog.records) == 2
    assert "ETA=" in caplog.records[0].message
    first, _ = ep.infer_c1(Predictor(), manifest, fake_preprocess, 1, ep.Timings(), logger, 1)
    second, _ = ep.infer_c1(Predictor(), manifest, fake_preprocess, 1, ep.Timings(), logger, 1000)
    assert first == second


def test_missing_staintools_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "staintools", None)
    with pytest.raises(ep.EvaluationError, match="No NumPy fallback"):
        ep.create_legacy_normalizer(tmp_path / "reference.png")


def test_legacy_brightness_standardizer_prefers_original_api():
    sentinel = object()
    fake = SimpleNamespace(
        BrightnessStandardizer=lambda: sentinel,
        LuminosityStandardizer=object,
    )

    result = ep.create_legacy_brightness_standardizer(fake)

    assert result is sentinel


def test_legacy_brightness_standardizer_supports_luminosity_api():
    calls = []

    class LuminosityStandardizer:
        @staticmethod
        def standardize(image):
            calls.append(image)
            return image + 1

    fake = SimpleNamespace(
        LuminosityStandardizer=LuminosityStandardizer,
    )

    standardizer = ep.create_legacy_brightness_standardizer(fake)

    image = np.array([1, 2, 3], dtype=np.uint8)
    result = standardizer.transform(image)

    np.testing.assert_array_equal(
        result,
        np.array([2, 3, 4], dtype=np.uint8),
    )
    assert calls == [image]


def test_legacy_brightness_standardizer_rejects_unknown_api():
    fake = SimpleNamespace()

    with pytest.raises(
        ep.EvaluationError,
        match="neither BrightnessStandardizer nor LuminosityStandardizer",
    ):
        ep.create_legacy_brightness_standardizer(fake)

def test_preprocessing_order_and_float_scaling(manifest):
    events = []
    class Brightness:
        def transform(self, pixels):
            assert pixels.shape == (350, 350, 3)
            assert pixels.dtype == np.uint8
            events.append("brightness")
            return pixels
    class Stain:
        def transform(self, pixels):
            assert events == ["brightness"]
            events.append("stain")
            return pixels
    row = manifest.rows[0]
    result = ep.preprocess_legacy(row, manifest.paths[row["sample_id"]], (Brightness(), Stain()))
    assert events == ["brightness", "stain"]
    assert result.dtype == np.float32
    assert result[0, 0, 0] == np.float32(20 / 255)


def test_reference_not_brightness_standardized(monkeypatch, tmp_path):
    events = []
    target = object()
    class Normalizer:
        def __init__(self, method):
            assert method == "macenko"
        def fit(self, image):
            assert image is target
            events.append("fit_raw_reference")
    fake = SimpleNamespace(read_image=lambda path: target, BrightnessStandardizer=lambda: object(), StainNormalizer=Normalizer)
    monkeypatch.setitem(sys.modules, "staintools", fake)
    ep.create_legacy_normalizer(tmp_path / "reference.png")
    assert events == ["fit_raw_reference"]


def test_normalization_failure_has_sample_id(manifest):
    class Broken:
        def transform(self, image):
            raise ValueError("not enough tissue")
    row = manifest.rows[0]
    with pytest.raises(ep.EvaluationError, match=row["sample_id"]):
        ep.preprocess_legacy(row, manifest.paths[row["sample_id"]], (Broken(), Broken()))


def test_real_staintools_adapter_parity(record_property):
    """Current installed legacy-call parity only, not historical-version parity."""
    pytest.importorskip("staintools", reason="Original staintools/native dependencies unavailable; no Macenko parity claimed")
    root = Path(__file__).resolve().parents[1]
    reference = root / "4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg"
    normalizers = ep.create_legacy_normalizer(reference)
    path = root / "2_validation_Tumor_vs_Benign/Examples/validation dataset 1/tumor/tum.16285.jpg"
    row = {"sample_id": "legacy-comparison", "file_sha256": ep.sha256_file(path)}
    actual = ep.preprocess_legacy(row, path, normalizers)
    with Image.open(path) as source:
        pixels = np.array(source.convert("RGB").resize((350, 350), Image.Resampling.LANCZOS))
    expected = np.float32(normalizers[1].transform(normalizers[0].transform(pixels))) / 255.0
    np.testing.assert_array_equal(actual, expected)
    # Diagnostic comparison only: the modern replacement is NOT a parity oracle.
    from modern_pca.stain import MacenkoNormalizer
    modern = MacenkoNormalizer(reference).transform(pixels).astype(np.float32) / 255.0
    record_property("legacy_vs_modern_max_abs_difference", float(np.max(np.abs(actual - modern))))
    record_property("historical_preprocessing_parity_verified", False)


def test_gpu_unavailable_fails():
    tf = SimpleNamespace(config=SimpleNamespace(get_visible_devices=lambda kind: []))
    with pytest.raises(ep.EvaluationError, match="GPU explicitly requested"):
        ep.select_device(tf, "gpu")


def test_scientific_fingerprint_runtime_independent(tmp_path):
    args = ep.parse_args(cli(tmp_path))
    original = ep.scientific_config(args, "model", "manifest", "reference", {"numpy": "test"})
    digest = ep.fingerprint(original)
    args.device, args.log_every, args.log_seconds, args.log_level = "gpu", 1, 2, "DEBUG"
    assert ep.fingerprint(ep.scientific_config(args, "model", "manifest", "reference", {"numpy": "test"})) == digest
    for key, value in original.items():
        changed = dict(original)
        changed[key] = "changed" if value != "changed" else "other"
        assert ep.fingerprint(changed) != digest


@pytest.mark.parametrize("override", [{"threshold": "0.4"}, {"image-size": "64"}, {"preprocessing": "numpy"}, {"warmup-batches": "1"}])
def test_paper_scientific_guards(tmp_path, override):
    with pytest.raises(SystemExit):
        ep.parse_args(cli(tmp_path, purpose="paper", cohort="VAL2", **{"checkpoint-metadata": "sidecar.json", **override}))


def test_existing_output_not_overwritten(tmp_path):
    args = ep.parse_args(cli(tmp_path))
    args.output.mkdir()
    sentinel = args.output / "metrics.json"
    sentinel.write_text("preserve")
    with pytest.raises(ep.EvaluationError, match="will not overwrite"):
        ep.run(args)
    assert sentinel.read_text() == "preserve"


def test_output_dataset_overlap_rejected(tmp_path):
    args = ep.parse_args(cli(tmp_path, output=tmp_path / "benign" / "output"))
    with pytest.raises(ep.EvaluationError, match="must not overlap"):
        ep.run(args)
    assert not args.output.exists()


def test_inference_error_identifies_batch(manifest):
    def broken(batch, training):
        raise RuntimeError("test device error")
    with pytest.raises(ep.EvaluationError, match=manifest.rows[0]["sample_id"]):
        ep.infer_c1(broken, manifest, fake_preprocess, 1, ep.Timings(), logging.getLogger("test"))


def test_failed_run_logs_without_completed_metrics(tmp_path, monkeypatch):
    args = ep.parse_args(cli(tmp_path))
    def broken(*args):
        raise ep.EvaluationError("fixture sample failed")
    monkeypatch.setattr(ep, "_execute", broken)
    assert ep.run(args) == 1
    assert "fixture sample failed" in (args.output / "evaluation.log").read_text()
    assert not (args.output / "metrics.json").exists()
    assert not (args.output / "timing_summary.json").exists()
    assert json.loads((args.output / "environment.json").read_text())["run"]["status"] == "failed"


def test_weights_only_and_unknown_container_rejected(tmp_path):
    path = tmp_path / "model.weights"
    path.write_bytes(b"weights, not a whole model")
    with pytest.raises(ep.EvaluationError, match="weights-only"):
        ep.load_paper_model(path, "auto", {}, None)


def test_hdf5_weights_only_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "weights.h5"
    with h5py.File(path, "w") as stream:
        stream.create_dataset("weights", data=[1.0])
    with pytest.raises(ep.EvaluationError, match="Weights-only"):
        ep.load_paper_model(path, "auto", {}, None)


def test_loading_is_compile_false_and_no_fallback(tmp_path):
    import zipfile
    path = tmp_path / "model.keras"
    with zipfile.ZipFile(path, "w") as archive:
        for name in ["config.json", "metadata.json", "model.weights.h5"]:
            archive.writestr(name, "fixture")
    calls = []
    def load(path, **kwargs):
        calls.append(kwargs)
        raise ValueError("incompatible fixture")
    tf = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=load)))
    with pytest.raises(ep.EvaluationError, match="no conversion/fallback"):
        ep.load_paper_model(path, "auto", {}, tf)
    assert calls == [{"compile": False, "safe_mode": True}]


def test_complete_run_with_offline_doubles(cohort, tmp_path, monkeypatch):
    """Integration of artifacts/timing/integrity, explicitly using no real checkpoint."""
    args = ep.parse_args(cli(tmp_path, **{"benign-dir": cohort[0], "tumour-dir": cohort[1]}))
    args.model.write_bytes(b"controlled fixture")
    image_file(args.stain_reference, (150, 120, 200))
    tf = SimpleNamespace(keras=SimpleNamespace(mixed_precision=SimpleNamespace(set_global_policy=lambda policy: None)),
                         device=lambda device: nullcontext())
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    monkeypatch.setattr(ep, "select_device", lambda tf, requested: ("/CPU:0", {"tensorflow_visible_gpus": []}))
    monkeypatch.setattr(ep, "load_paper_model", lambda *args: (Predictor(), {"format": "fixture"}))
    monkeypatch.setattr(ep, "create_legacy_normalizer", lambda path: (None, None))
    monkeypatch.setattr(ep, "preprocess_legacy", lambda row, path, normalizers: fake_preprocess(row, path))
    assert ep.run(args) == 0
    assert {path.name for path in args.output.iterdir()} == {"manifest.csv", "predictions.csv", "metrics.json", "environment.json", "timing_summary.json", "evaluation.log", "run_complete.json"}
    metrics = json.loads((args.output / "metrics.json").read_text())
    timing = json.loads((args.output / "timing_summary.json").read_text())
    environment = json.loads((args.output / "environment.json").read_text())
    assert metrics["status"] == timing["status"] == environment["run"]["status"] == "complete"
    assert metrics["performance"]["timing"] == timing["timing"] == environment["timing"]
    assert timing["samples"] == 3
    assert timing["benign_samples"] == 2
    assert timing["tumour_samples"] == 1
    assert environment["dataset"]["cohort_completeness_verified"] is False
    assert environment["evaluation"]["cohort_forward_calls"] == 3
    # Repeat with changed logging/device-independent configuration and locked fingerprint.
    baseline = args.output / "timing_summary.json"
    args.compare_run = baseline
    args.output = tmp_path / "repeat"
    args.log_every = 1
    assert ep.run(args) == 0
    # A changed scientific artifact must abort before any cohort inference.
    args.model.write_bytes(b"different fixture")
    args.output = tmp_path / "mismatched"
    assert ep.run(args) == 1
    assert not (args.output / "metrics.json").exists()
    assert "differs from complete baseline" in (args.output / "evaluation.log").read_text()


@pytest.fixture
def nasnet_source(monkeypatch):
    """Read installed Keras builder definitions without importing TF or allocating weights.

    This tests the reference derivation, not deserialization of an author checkpoint.
    Only function definitions are compiled; no imports, decorators or top-level code run.
    """
    try:
        distribution = importlib.metadata.distribution("keras")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Installed Keras source required for weight-free architecture derivation")
    candidates = [path for path in distribution.files or [] if str(path).replace("\\", "/").endswith("/applications/nasnet.py")]
    if not candidates:
        pytest.skip("Installed Keras NASNet source unavailable")
    tree = ast.parse(Path(distribution.locate_file(candidates[0])).read_text(encoding="utf-8"))
    functions = []
    names = {"NASNet", "NASNetLarge", "_separable_conv_block", "_adjust_block", "_normal_a_cell", "_reduction_a_cell"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "installed-nasnet-contract", "exec"), namespace)
    monkeypatch.setattr(ep, "_nasnet_namespace", lambda: namespace)
    return ep.reference_architecture()


def test_weight_free_nasnet_exact_count(nasnet_source):
    backbone, full = nasnet_source["backbone"], nasnet_source["full"]
    assert backbone["nodes"][backbone["output"]]["shape"] == [None, 11, 11, 4032]
    assert backbone["parameters"] == 84_916_818
    assert full["parameters"] == 84_916_818 + (11 * 11 * 4032 + 1) * 256 + (256 + 1) * 3
    assert full["parameters"] == 209_813_077
    assert ep.validate_architecture_document(full, copy.deepcopy(full)) == ep.fingerprint(full)


def test_real_keras_tiny_graph_adapter():
    """Tiny random model only; no NASNet allocation, downloads or forward calls.

    Exercises real Keras nested-model tensor histories against an independently
    specified 4x4 graph. This is NOT an author-checkpoint integration test.
    """
    tf = pytest.importorskip("tensorflow")
    inputs = tf.keras.Input(shape=(4, 4, 3))
    conv = tf.keras.layers.Conv2D(2, (1, 1), use_bias=False)(inputs)
    backbone = tf.keras.Model(inputs, conv)
    model = tf.keras.Sequential([tf.keras.Input(shape=(4, 4, 3)), backbone,
                                 tf.keras.layers.Flatten(), tf.keras.layers.Dense(3, activation="softmax")])
    layers = ep.GraphLayers()
    expected = layers.Dense(3, activation="softmax")(
        layers.Flatten()(layers.Conv2D(2, (1, 1), use_bias=False)(layers.Input((4, 4, 3)))))
    assert ep.loaded_architecture(model) == ep.graph_document(expected)
    assert model.count_params() == 3*2 + (4*4*2 + 1)*3


@pytest.mark.parametrize("mutation", ["operation", "connectivity", "kernel", "epsilon", "head_width", "head_activation"])
def test_graph_contract_rejects_same_shape_spoofs(nasnet_source, mutation):
    expected = nasnet_source["full"]
    actual = copy.deepcopy(expected)
    nodes = actual["nodes"]
    if mutation == "operation":
        nodes[1]["operation"] = "UnrelatedConv"
    elif mutation == "connectivity":
        nodes[-1]["inputs"] = [0]
    elif mutation == "kernel":
        nodes[1]["config"]["kernel_size"] = [1, 1]
    elif mutation == "epsilon":
        next(node for node in nodes if node["operation"] == "BatchNormalization")["config"]["epsilon"] = 0.1
    elif mutation == "head_width":
        nodes[-2]["config"]["units"] = 128
    else:
        nodes[-1]["config"]["activation"] = "sigmoid"
    with pytest.raises(ep.EvaluationError, match="connectivity contract mismatch"):
        ep.validate_architecture_document(actual, expected)


def graph_model_double(document):
    """Keras tensor-history/type doubles: full graph traversal remains real.

    Configs are contract fixtures, not evidence that Keras can load the author file.
    """
    classes = {name: type(name, (), {}) for name in ep.GRAPH_DEFAULTS}
    tensors, layers = [], []
    for node in document["nodes"]:
        kind = node["operation"]
        layer = classes[kind]()
        cfg = copy.deepcopy(node["config"])
        layer.get_config = lambda cfg=cfg: cfg
        layer.count_params = lambda count=node["parameters"]: count
        layer._inbound_nodes = [SimpleNamespace(input_tensors=[tensors[index] for index in node["inputs"]])]
        for key, value in cfg.items():
            setattr(layer, key, value)
        if "activation" in cfg:
            layer.activation = SimpleNamespace(__name__=cfg["activation"])
        layer.dtype_policy = SimpleNamespace(compute_dtype="float32")
        tensor = SimpleNamespace(shape=tuple(node["shape"]), dtype="float32", _keras_history=(layer, 0, 0))
        tensors.append(tensor)
        layers.append(layer)
    model = Predictor()
    model.inputs, model.outputs = [tensors[0]], [tensors[document["output"]]]
    model.layers = [SimpleNamespace(output_shape=(None, 11, 11, 4032), layers=layers[:-3]), *layers[-3:]]
    model.trainable_weights, model.non_trainable_weights = [], []
    model.count_params = lambda: document["parameters"]
    return model, SimpleNamespace(**classes)


@pytest.fixture
def paper_runtime(cohort, tmp_path, monkeypatch, nasnet_source):
    """Paper orchestration with only TF runtime/deserialization and stain engine doubled.

    Metadata, architecture comparison AND tensor traversal, manifest, RGB/resize/
    scaling, C1, metrics, fingerprint, writes and marker validation are real code.
    Identity stain transforms intentionally do NOT establish Macenko parity.
    """
    model, layer_classes = graph_model_double(nasnet_source["full"])
    sidecar = tmp_path / "checkpoint.json"
    args = ep.parse_args(cli(tmp_path, purpose="paper", cohort="VAL2", model=tmp_path / "model.keras",
                             **{"checkpoint-metadata": sidecar, "benign-dir": cohort[0], "tumour-dir": cohort[1]}))
    with zipfile.ZipFile(args.model, "w") as archive:
        for name in ["config.json", "metadata.json", "model.weights.h5"]:
            archive.writestr(name, "controlled fixture; not a Keras checkpoint")
    sidecar.write_text(json.dumps(metadata(ep.sha256_file(args.model))))
    image_file(args.stain_reference, (150, 120, 200))
    def deserialize(path, **kwargs):
        assert kwargs == {"compile": False, "safe_mode": True}
        return model
    tf = SimpleNamespace(keras=SimpleNamespace(
        mixed_precision=SimpleNamespace(set_global_policy=lambda policy: None),
        models=SimpleNamespace(load_model=deserialize), Model=type("Model", (), {}), layers=layer_classes),
        device=lambda device: nullcontext())
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    monkeypatch.setattr(ep, "select_device", lambda tf, requested: ("/CPU:0", {"tensorflow_visible_gpus": []}))
    identity = SimpleNamespace(transform=lambda pixels: pixels)
    monkeypatch.setattr(ep, "create_legacy_normalizer", lambda path: (identity, identity))
    return args, model


def test_paper_mode_orchestration(paper_runtime):
    args, model = paper_runtime
    assert ep.run(args) == 0
    marker = ep.validate_completed_run(args.output)
    environment = json.loads((args.output / "environment.json").read_text())
    metrics = json.loads((args.output / "metrics.json").read_text())
    assert marker["artifacts"].keys() == set(ep.FINAL_ARTIFACTS)
    assert marker["scientific_config_sha256"] == ep.fingerprint(environment["scientific_configuration"])
    assert environment["model"]["parameter_count"] == 209_813_077
    assert environment["model"]["class_mapping"] == ["gland", "nongland", "tumour"]
    assert metrics["scientific"]["confusion_matrix"] == [[2, 0], [1, 0]]
    assert model.calls == model.presentations == 3
    assert environment["preprocessing"]["historical_numerical_parity_verified"] is False


@pytest.mark.parametrize("mutation", ["unrelated", "wrong_head_width", "wrong_activation", "wrong_backbone", "forced_training"])
def test_paper_validator_rejects_graph_model(paper_runtime, mutation):
    args, model = paper_runtime
    if mutation == "unrelated":
        model.layers[0].layers[1].__class__ = type("UnrelatedCNN", (), {})
    elif mutation == "wrong_head_width":
        model.layers[2].units = 128
    elif mutation == "wrong_activation":
        model.layers[3].activation = relu
    elif mutation == "forced_training":
        model.layers[0].layers[2]._inbound_nodes[0].call_kwargs = {"training": True}
    else:
        layer = model.layers[0].layers[1]
        original = layer.get_config()
        layer.get_config = lambda: {**original, "kernel_size": [1, 1]}
    with pytest.raises(ep.EvaluationError):
        ep.validate_model_contract(model, metadata(ep.sha256_file(args.model)), "paper")


@pytest.mark.parametrize("mutation", ["missing_marker", "run_id", "fingerprint", "missing_artifact", "modified_artifact"])
def test_completed_baseline_integrity(paper_runtime, mutation):
    args, _ = paper_runtime
    assert ep.run(args) == 0
    marker_path = args.output / "run_complete.json"
    marker = ep.validate_completed_run(args.output)
    if mutation == "missing_marker":
        marker_path.unlink()
    elif mutation in ("run_id", "fingerprint"):
        marker["run_id" if mutation == "run_id" else "scientific_config_sha256"] = "0" * 64
        ep.write_json(marker_path, marker)
    elif mutation == "missing_artifact":
        (args.output / "predictions.csv").unlink()
    else:
        with (args.output / "predictions.csv").open("a") as stream:
            stream.write("corrupted")
    with pytest.raises(ep.EvaluationError):
        ep.validate_completed_run(args.output / "timing_summary.json")
    args.compare_run = args.output / "timing_summary.json"
    args.output = args.output.parent / "rejected-comparison"
    assert ep.run(args) == 1
    assert not (args.output / "run_complete.json").exists()


@pytest.mark.parametrize("failed_artifact", ["timing_summary.json", "run_complete.json"])
def test_late_write_failure_incomplete(paper_runtime, monkeypatch, failed_artifact):
    args, _ = paper_runtime
    original = ep.write_json
    observed = []
    def failing_write(path, document):
        if path.name == failed_artifact:
            assert (args.output / "environment.json").exists()
            if failed_artifact == "run_complete.json":
                assert (args.output / "metrics.json").exists()
            observed.append(path.name)
            raise OSError("simulated final write failure")
        return original(path, document)
    monkeypatch.setattr(ep, "write_json", failing_write)
    assert ep.run(args) == 1
    assert observed == [failed_artifact]
    assert not (args.output / "run_complete.json").exists()
    assert "simulated final write failure" in (args.output / "evaluation.log").read_text()
    failed = json.loads((args.output / "environment.json").read_text())
    assert failed["run"]["status"] == "failed"
    assert failed["partial_timing"]["status"] == "partial"
    with pytest.raises(ep.EvaluationError):
        ep.validate_completed_run(args.output)


def test_abrupt_interruption_before_marker(paper_runtime, monkeypatch):
    """SystemExit bypasses caught-failure cleanup, like a terminated finalization."""
    args, _ = paper_runtime
    original = ep.write_json
    def interrupted(path, document):
        if path.name == "run_complete.json":
            raise SystemExit("interrupted before final marker")
        return original(path, document)
    monkeypatch.setattr(ep, "write_json", interrupted)
    with pytest.raises(SystemExit):
        ep.run(args)
    assert json.loads((args.output / "metrics.json").read_text())["status"] == "complete"
    assert json.loads((args.output / "timing_summary.json").read_text())["status"] == "complete"
    assert not (args.output / "run_complete.json").exists()
    with pytest.raises(ep.EvaluationError):
        ep.validate_completed_run(args.output)


def test_asymmetric_metrics_and_independent_auc():
    # TN=4, FP=1, FN=2, TP=3. Of 5*5 positive/negative pairs, positives
    # .15/.25/.7/.8/.9 outrank 1/2/5/5/5 negatives: AUC=18/25.
    result = ep.calculate_binary_metrics([0]*5 + [1]*5,
                                         [.1, .2, .3, .4, .6, .15, .25, .7, .8, .9])
    assert result["confusion_matrix"] == [[4, 1], [2, 3]]
    for key, value in dict(accuracy=7/10, precision=3/4, recall=3/5,
                           sensitivity=3/5, specificity=4/5, f1=6/9, roc_auc=18/25).items():
        assert result[key] == pytest.approx(value)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_distinct_image_outputs_remain_ordered(cohort, batch_size):
    image_file(cohort[0] / "extra.png", (40, 50, 60))
    image_file(cohort[1] / "extra.png", (140, 150, 160))
    manifest = ep.build_manifest(*cohort, "VAL2")
    vectors = np.array([[.6, .3, .1], [.1, .1, .8], [.3, .200001, .499999],
                        [.2, .3, .5], [.2, .299999, .500001]], dtype=np.float64)
    seen = []
    def preprocess(row, path):
        return np.array([row["row_index"]], dtype=np.float32)
    def model(batch, training):
        assert training is False
        indices = batch[:, 0].astype(int)
        seen.extend(indices.tolist())
        return HostOutput(vectors[indices])
    predictions, counts = ep.infer_c1(model, manifest, preprocess, batch_size, ep.Timings(), logging.getLogger("distinct"))
    assert seen == list(range(5))
    assert counts["cohort_forward_calls"] == (5 + batch_size - 1) // batch_size
    for row, original, vector, decision in zip(predictions, manifest.rows, vectors, [0, 1, 0, 0, 1]):
        for field in ["row_index", "sample_id", "cohort", "relative_path", "ground_truth", "ground_truth_code"]:
            assert row[field] == original[field]
        assert [row["p_gland"], row["p_nongland"], row["p_tumour"]] == vector.tolist()
        assert row["predicted_binary_code"] == decision


@pytest.mark.parametrize("field", ep.PREDICTION_FIELDS)
def test_export_checks_every_field(manifest, tmp_path, field):
    rows, _ = ep.infer_c1(Predictor(), manifest, fake_preprocess, 2, ep.Timings(), logging.getLogger("export"))
    metrics = ep.calculate_binary_metrics([row["ground_truth_code"] for row in rows], [row["p_tumour"] for row in rows])
    altered = copy.deepcopy(rows)
    altered[0][field] = "corrupted"
    path = tmp_path / "predictions.csv"
    ep.atomic_write(path, ep.csv_bytes(altered, ep.PREDICTION_FIELDS))
    with pytest.raises(ep.EvaluationError, match="mismatch"):
        ep.verify_prediction_export(path, rows, metrics)


def test_swapped_ground_truth_preserves_metrics_but_fails_export(manifest, tmp_path):
    rows, _ = ep.infer_c1(Predictor(), manifest, fake_preprocess, 1, ep.Timings(), logging.getLogger("export"))
    metrics = ep.calculate_binary_metrics([row["ground_truth_code"] for row in rows], [row["p_tumour"] for row in rows])
    altered = copy.deepcopy(rows)
    for field in ("ground_truth", "ground_truth_code"):
        altered[0][field], altered[2][field] = altered[2][field], altered[0][field]
    assert ep.calculate_binary_metrics([row["ground_truth_code"] for row in altered], [row["p_tumour"] for row in altered]) == metrics
    path = tmp_path / "predictions.csv"
    ep.atomic_write(path, ep.csv_bytes(altered, ep.PREDICTION_FIELDS))
    with pytest.raises(ep.EvaluationError, match="ground_truth"):
        ep.verify_prediction_export(path, rows, metrics)


@pytest.mark.parametrize("document, message", [
    ([], "top level"), (None, "top level"), ({}, "SHA-256"),
    ({"model_sha256": None}, "SHA-256"), ({"model_sha256": 123}, "SHA-256"),
    ({"model_sha256": "abc"}, "SHA-256"), ({"model_sha256": "g"*64}, "SHA-256")])
def test_metadata_invalid_json_schema(tmp_path, document, message):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ep.EvaluationError, match=message):
        ep.load_checkpoint_metadata(path, "a"*64, "paper", None)


@pytest.mark.parametrize("key,value", [("class_order", None), ("class_order", "gland,nongland,tumour"),
    ("input_shape", "350,350,3"), ("input_shape", [None, True, 350, 3]), ("output_shape", None),
    ("source", None), ("architecture", None), ("identity", None), ("conversion_history", None)])
def test_metadata_invalid_required_fields(tmp_path, key, value):
    doc = metadata("a"*64)
    doc[key] = value
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ep.EvaluationError):
        ep.load_checkpoint_metadata(path, "a"*64, "paper", None)


def test_metadata_missing_field_and_malformed_json(tmp_path):
    path = tmp_path / "metadata.json"
    doc = metadata("a"*64)
    del doc["source"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ep.EvaluationError, match="Incomplete"):
        ep.load_checkpoint_metadata(path, "a"*64, "paper", None)
    path.write_text("{bad json")
    with pytest.raises(ep.EvaluationError, match="valid UTF-8 JSON"):
        ep.load_checkpoint_metadata(path, "a"*64, "paper", None)


def test_fingerprint_scientific_sources_and_runtime_exclusions(tmp_path, monkeypatch):
    args = ep.parse_args(cli(tmp_path))
    def config():
        return ep.scientific_config(args, "model", "manifest", "reference", {})
    original = config()
    digest = ep.fingerprint(original)
    args.output, args.device = tmp_path / "other", "gpu"
    args.log_level, args.log_every, args.log_seconds = "DEBUG", 2, 5
    args.runtime_seconds = 1234
    assert ep.fingerprint(config()) == digest
    args.threshold = .4
    assert ep.fingerprint(config()) != digest
    args.threshold = .5
    for index in range(3):
        hashes = ["model", "manifest", "reference"]
        hashes[index] = "different"
        assert ep.fingerprint(ep.scientific_config(args, *hashes, {})) != digest
    with monkeypatch.context() as patch:
        patch.setattr(ep, "CLASS_ORDER", ["tumour", "nongland", "gland"])
        assert ep.fingerprint(config()) != digest
    with monkeypatch.context() as patch:
        patch.setattr(ep, "preprocessing_settings", lambda: {"brightness_percentile": 91})
        assert ep.fingerprint(config()) != digest
    for key in ["evaluator_source_sha256", "preprocessing_source_sha256"]:
        with monkeypatch.context() as patch:
            identity = {**original["implementation"], key: "changed source"}
            patch.setattr(ep, "implementation_identity", lambda: identity)
            assert ep.fingerprint(config()) != digest
