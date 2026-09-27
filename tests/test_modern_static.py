from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_author_examples_have_three_classes():
    root = ROOT / "1_training" / "Examples" / "training_tumor_benign_SN"
    counts = {path.name: len(list(path.glob("*.jpg"))) for path in root.iterdir() if path.is_dir()}
    assert counts == {"gland": 10, "nongland": 10, "tumor": 10}


def test_colab_requirements_do_not_pin_legacy_keras():
    requirements = (ROOT / "requirements-colab.txt").read_text().lower()
    assert "keras==2" not in requirements
    assert "tensorflow==1" not in requirements

