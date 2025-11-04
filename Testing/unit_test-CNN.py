import os
import torch
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from CNN.data.inference import YOLOv7Classifier, preprocess_image, load_model, predict

# ==============================
# CONFIGURATION
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DUMMY_PATH = "dummy_model.pt"


@pytest.fixture(scope="session", autouse=True)
def create_dummy_model():
    """Create and save a dummy YOLOv7Classifier checkpoint for testing."""
    model = YOLOv7Classifier(num_classes=6)
    torch.save(model.state_dict(), MODEL_DUMMY_PATH)
    yield
    if os.path.exists(MODEL_DUMMY_PATH):
        os.remove(MODEL_DUMMY_PATH)


# ==============================
# 1️⃣ MODEL FORWARD PASS
# ==============================
def test_yolov7_forward_pass():
    model = YOLOv7Classifier(num_classes=6).to(DEVICE)
    model.eval()  # FIX: switch to eval mode to avoid BatchNorm error on batch size = 1
    dummy_input = torch.randn(1, 3, 640, 640).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 6), "Model output shape should be (1, 6)"
    assert torch.isfinite(output).all(), "Model output contains NaN or Inf values"


# ==============================
# 2️⃣ CHECK MODEL LOADING
# ==============================
def test_load_model_from_checkpoint():
    model = load_model(MODEL_DUMMY_PATH, DEVICE)
    assert isinstance(model, YOLOv7Classifier), "Loaded model must be YOLOv7Classifier"
    assert not model.training, "Model should be in eval mode after loading"


# ==============================
# 3️⃣ IMAGE PREPROCESS PIPELINE
# ==============================
def test_preprocess_image(tmp_path):
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img_path = tmp_path / "dummy.jpg"
    Image.fromarray(dummy_img).save(img_path)

    tensor = preprocess_image(str(img_path))
    assert tensor.shape == (1, 3, 640, 640), "Preprocessed tensor should be (1, 3, 640, 640)"
    assert abs(float(tensor.mean())) < 2, "Normalized tensor mean should be near zero"


# ==============================
# 4️⃣ INFERENCE PIPELINE (NO CRASH)
# ==============================
def test_predict_pipeline(tmp_path):
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "dummy_infer.jpg"
    Image.fromarray(dummy_img).save(img_path)

    grade, confidence = predict(str(img_path), MODEL_DUMMY_PATH, DEVICE, img_size=640)
    assert isinstance(grade, int), "Predicted grade must be int"
    assert 1 <= grade <= 6, "Predicted grade must be between 1 and 6"
    assert 0 <= confidence <= 100, "Confidence must be between 0 and 100"


# ==============================
# 5️⃣ MODEL DEVICE CONSISTENCY
# ==============================
def test_device_consistency():
    model = YOLOv7Classifier(num_classes=6)
    cpu_model = model.to("cpu")
    device_model = model.to(DEVICE)
    assert next(cpu_model.parameters()).device.type in ["cpu"], "Model must be on CPU"
    assert next(device_model.parameters()).device.type in [DEVICE], "Model must be on target device"
