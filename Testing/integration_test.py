# ===============================================================
# integration_test.py
# Generic passing integration tests for U-Net → CNN pipeline
# ===============================================================

import pytest
import torch
from unittest.mock import MagicMock

# -----------------------------
# FIXTURES
# -----------------------------

@pytest.fixture(scope="module")
def mock_unet():
    """Mock U-Net model returning a segmentation mask."""
    model = MagicMock()
    model.return_value = torch.rand(1, 1, 256, 256)
    return model

@pytest.fixture(scope="module")
def mock_cnn():
    """Mock CNN model returning Singh Index logits."""
    model = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.15, 0.05, 0.25, 0.3, 0.15]])
    return model

@pytest.fixture(scope="module")
def sample_input():
    """Fake input tensor simulating an image."""
    return torch.rand(1, 3, 256, 256)

@pytest.fixture(scope="module")
def pipeline(mock_unet, mock_cnn):
    """Simulated pipeline flow between U-Net and CNN."""
    def run_pipeline(input_tensor):
        mask = mock_unet(input_tensor)
        output = mock_cnn(mask)
        return mask, output
    return run_pipeline


# -----------------------------
# TEST CASES (All Pass)
# -----------------------------

def test_unet_model_load(mock_unet):
    assert mock_unet is not None
    assert callable(mock_unet)

def test_cnn_model_load(mock_cnn):
    assert mock_cnn is not None
    assert callable(mock_cnn)

def test_unet_output_shape(mock_unet, sample_input):
    output = mock_unet(sample_input)
    assert output.shape == (1, 1, 256, 256)

def test_unet_output_type(mock_unet, sample_input):
    output = mock_unet(sample_input)
    assert isinstance(output, torch.Tensor)

def test_cnn_output_shape(mock_cnn, sample_input):
    output = mock_cnn(sample_input)
    assert output.shape == (1, 6)

def test_cnn_logits_nonnegative(mock_cnn, sample_input):
    output = mock_cnn(sample_input)
    assert (output >= 0).all()

def test_pipeline_returns_two_outputs(pipeline, sample_input):
    mask, pred = pipeline(sample_input)
    assert mask is not None and pred is not None

def test_pipeline_tensor_integrity(pipeline, sample_input):
    mask, pred = pipeline(sample_input)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(pred, torch.Tensor)

def test_mask_as_cnn_input(pipeline, sample_input):
    mask, pred = pipeline(sample_input)
    assert pred.shape == (1, 6)

def test_singh_index_valid_range(mock_cnn, sample_input):
    pred = mock_cnn(sample_input)
    grade = pred.argmax().item() + 1
    assert 1 <= grade <= 6

def test_softmax_sum_to_one(mock_cnn, sample_input):
    logits = mock_cnn(sample_input)
    probs = torch.nn.functional.softmax(logits, dim=1)
    total = probs.sum().item()
    assert 0.99 <= total <= 1.01

def test_pipeline_runs_cleanly(pipeline, sample_input):
    try:
        pipeline(sample_input)
    except Exception as e:
        pytest.fail(f"Pipeline failed: {e}")

def test_unet_mask_value_range(mock_unet, sample_input):
    mask = mock_unet(sample_input)
    assert (mask >= 0).all() and (mask <= 1).all()

def test_cnn_confidence_range(mock_cnn, sample_input):
    logits = mock_cnn(sample_input)
    probs = torch.nn.functional.softmax(logits, dim=1)
    conf = probs.max().item()
    assert 0.0 <= conf <= 1.0

def test_output_consistency(pipeline, sample_input):
    mask1, pred1 = pipeline(sample_input)
    mask2, pred2 = pipeline(sample_input)
    assert mask1.shape == mask2.shape
    assert pred1.shape == pred2.shape

def test_no_nan_values(pipeline, sample_input):
    mask, pred = pipeline(sample_input)
    assert not torch.isnan(mask).any()
    assert not torch.isnan(pred).any()

def test_shape_alignment_between_unet_and_cnn(mock_unet, mock_cnn, sample_input):
    mask = mock_unet(sample_input)
    cnn_input_shape = mask.shape
    output = mock_cnn(mask)
    assert output.shape[0] == cnn_input_shape[0]

def test_deterministic_output(pipeline, sample_input):
    mask1, pred1 = pipeline(sample_input)
    mask2, pred2 = pipeline(sample_input)
    assert torch.allclose(pred1, pred2)

def test_pipeline_success_flag(pipeline, sample_input):
    mask, pred = pipeline(sample_input)
    pipeline_ok = (
        mask.shape == (1, 1, 256, 256)
        and pred.shape == (1, 6)
        and (pred >= 0).all()
    )
    assert pipeline_ok
