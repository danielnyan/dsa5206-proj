"""Small deterministic candidate tests; no production checkpoints or images."""
import pytest

from tools.stage2h_epoch12_probe import fixture


@pytest.mark.parametrize('candidate', ['baseline', 'B', 'C'])
def test_candidate_accumulation_and_checkpoint_equivalence(tmp_path, candidate):
    value = fixture(candidate, tmp_path)
    assert value['status'] == 'PASS'
    assert value['samples'] == 355
    assert value['logical_sizes'] == [100, 100, 100, 55]
    assert value['updates'] == 4
    assert value['frozen_unchanged'] and value['class_predictions_identical']
    assert value['checkpoint_roundtrip'] == 'PASS'
    assert value['max_absolute_gradient_difference'] <= 1e-6
    assert value['max_absolute_weight_difference'] <= 1e-7
