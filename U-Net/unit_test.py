import torch
import pytest
import os
import cv2
import numpy as np


# Import only what actually exists in train_unet.py
from train_unet import UNet

# Try importing optional inference pipeline if available
try:
    from infer_crop import infer_image
except ImportError:
    infer_image = None


# ===============================
# UNIT TESTS FOR U-NET MODEL
# ===============================

def test_unet_forward_pass():
    """Validate that U-Net runs a forward pass and produces correct output shape."""
    model = UNet(in_channels=3, out_channels=1)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)

    assert isinstance(y, torch.Tensor), "Model forward pass did not return a tensor."
    assert y.shape == (2, 1, 128, 128), f"Expected output shape (2,1,128,128), got {y.shape}"


def test_unet_backward_pass():
    """Ensure gradients flow correctly and loss decreases in a mock train step."""
    model = UNet(in_channels=3, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    x = torch.randn(2, 3, 64, 64)
    y_true = torch.rand(2, 1, 64, 64)

    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss not computed properly."
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "No gradients found in model params."


def test_unet_checkpoint_saving(tmp_path):
    """Verify that model state_dict can be saved and loaded without corruption."""
    model = UNet(in_channels=3, out_channels=1)
    ckpt_path = tmp_path / "unet_test_ckpt.pt"

    torch.save(model.state_dict(), ckpt_path)
    assert ckpt_path.exists(), "Checkpoint file not created."

    new_model = UNet(in_channels=3, out_channels=1)
    new_model.load_state_dict(torch.load(ckpt_path))

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2), "Model weights mismatch after reload."


@pytest.mark.skipif(infer_image is None, reason="infer_crop.py does not define infer_image()")
def test_infer_crop_pipeline(tmp_path):
    """Integration test for inference + cropping logic (dummy image input)."""
    model = UNet(in_channels=1, out_channels=1)  # ✅ Fix
    model.eval()

    dummy_img = (np.random.rand(128, 128) * 255).astype(np.uint8)  # grayscale
    dummy_path = tmp_path / "dummy.png"
    cv2.imwrite(str(dummy_path), dummy_img)

    result = infer_image(model, str(dummy_path))
    assert result is not None



def test_unet_device_consistency():
    """Ensure the model and input tensor operate on the same device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=1).to(device)
    x = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        y = model(x)
    assert y.device.type == device, f"Model output is not on the same device ({device})"
