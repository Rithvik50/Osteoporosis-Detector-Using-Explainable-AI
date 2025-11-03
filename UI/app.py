import streamlit as st
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import time
import shutil
from PIL import Image
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Ensemble_Stacking.ensemble_stacking import StackingEnsembleOptuna, MultiLabelEncoder

import lime
import lime.lime_tabular
import shap

# Page config
st.set_page_config(page_title="Osteoporosis Predictor", layout="wide")

UNET_DIR = BASE_DIR/"U-Net"
CNN_DIR = BASE_DIR/"CNN"/"data"
UNET_MODEL = os.path.join(UNET_DIR, "checkpoints", "best.pt")
CNN_MODEL = os.path.join(CNN_DIR, "runs", "train", "best.pt")
PREDICTIONS_DIR = os.path.join(UNET_DIR, "predictions")
TEMP_UPLOAD_DIR = os.path.join(UNET_DIR, "temp_uploads")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
bg_path = BASE_DIR/"UI"/"static"/"bg.png"

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR/"Ensemble_Stacking"/"stacking_ensemble.joblib")
    return model


@st.cache_resource
def load_training_data():
    """Load training data for LIME/SHAP explainers"""
    df = pd.read_csv(BASE_DIR/"Ensemble_Stacking"/"patient_info_dataset.csv")

    X = df.drop('label', axis=1)


    # Only keep features that match the model's expected features
    expected_features = [
        "age", "sex", "height_cm", "weight_kg", "bmi",
        "waist_cm", "hip_cm", "waist_hip_ratio",
        "menopausal_status", "menopause_age", "years_since_menopause",
        "smoker", "alcohol_high", "physical_activity",
        "vitamin_d_ngml", "vitamin_d_missing",
        "serum_calcium_mgdl", "alkaline_phosphatase", "pth_pgml",
        "creatinine_mgdl", "hdl_mgdl", "ldl_mgdl",
        "ctx_ngml", "p1np_ugL",
        "estrogen_use", "diabetes_t2", "hypothyroidism",
        "rheumatoid_arthritis", "secondary_osteoporosis",
        "parent_hip_fracture", "prior_fracture", "bisphosphonate_use",
        "calcium_supplement", "vitamin_d_supplement", "glucocorticoid_use",
        "dialysis", "falls_past_year"
    ]

    # Only keep columns that are in expected_features
    X = X[[col for col in expected_features if col in X.columns]]

    # Sample for efficiency
    X_sample = X.sample(n=min(200, len(X)), random_state=42)
    return X_sample


@st.cache_resource
def create_lime_explainer(_model, X_train):
    """Create LIME explainer - encodes for LIME but decodes for model"""
    from sklearn.preprocessing import LabelEncoder

    # We need to encode for LIME, but track encoders to decode when calling model
    X_encoded = X_train.copy()
    encoders = {}
    categorical_features = []
    categorical_names = {}

    for idx, col in enumerate(X_train.columns):
        if col in ['sex', 'menopausal_status', 'physical_activity']:
            # Encode this column
            encoders[col] = LabelEncoder()
            X_encoded[col] = encoders[col].fit_transform(X_train[col].astype(str))
            categorical_features.append(idx)
            # Store mapping of encoded value -> original name
            categorical_names[idx] = encoders[col].classes_.tolist()

    # LIME explainer with ENCODED training data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_encoded.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Normal', 'Osteopenia', 'Osteoporosis'],
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        mode='classification',
        random_state=42
    )
    return explainer, encoders


@st.cache_resource
def create_shap_explainer(_model, X_train):
    """Create SHAP explainer that works with ENCODED data (matching LIME approach)"""
    from sklearn.preprocessing import LabelEncoder

    # Encode categorical columns for SHAP (same as LIME)
    X_encoded = X_train.copy()
    encoders = {}

    categorical_cols = ['sex', 'menopausal_status', 'physical_activity']

    for col in categorical_cols:
        if col in X_train.columns:
            encoders[col] = LabelEncoder()
            X_encoded[col] = encoders[col].fit_transform(X_train[col].astype(str))

    # CRITICAL: Ensure all columns are numeric (no object dtype)
    X_encoded = X_encoded.astype(float)

    # Use smaller background for SHAP (numeric data only)
    X_background = X_encoded.sample(n=min(50, len(X_encoded)), random_state=42)

    # IMPORTANT: SHAP needs a prediction function that decodes back to original format
    def predict_fn(X_array):
        """Wrapper that converts encoded array to original format for model prediction"""
        # Handle different input shapes
        if isinstance(X_array, np.ndarray):
            if len(X_array.shape) == 1:
                X_array = X_array.reshape(1, -1)
            X_df = pd.DataFrame(X_array, columns=X_encoded.columns)
        else:
            X_df = X_array.copy()

        # Decode categorical columns back to original format
        X_original = X_df.copy()
        for col, encoder in encoders.items():
            if col in X_original.columns:
                # Convert to int (in case of float) then decode
                X_original[col] = encoder.inverse_transform(X_original[col].astype(int))

        # Get predictions
        predictions = _model.predict_proba(X_original)

        # CRITICAL: Ensure output shape matches what SHAP expects
        # SHAP expects (n_samples, n_classes) not (n_samples * something, n_classes)
        return predictions

    # Use KernelExplainer with smaller sample size for efficiency
    # Reduce nsamples to speed up computation and avoid memory issues
    explainer = shap.KernelExplainer(
        predict_fn,
        X_background.values,
        link="identity"
    )

    return explainer, X_encoded.columns.tolist(), encoders

def explain_with_lime(model, instance_series, lime_explainer, lime_encoders, num_features=10):
    """Generate LIME explanation - encode for LIME, decode for model"""

    # Encode the instance for LIME (LIME expects encoded categorical values)
    instance_encoded = instance_series.copy()
    for col, encoder in lime_encoders.items():
        if col in instance_encoded.index:
            instance_encoded[col] = encoder.transform([str(instance_encoded[col])])[0]

    instance_array = instance_encoded.values

    # Create prediction function that decodes LIME's encoded data back to original format
    def predict_fn(X_array):
        """Wrapper: LIME gives encoded data -> decode -> model predicts"""
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(1, -1)

        # Convert to DataFrame
        X_df = pd.DataFrame(X_array, columns=instance_series.index.tolist())

        # Decode categorical columns back to original format for model
        for col, encoder in lime_encoders.items():
            if col in X_df.columns:
                # Convert encoded integers back to original strings
                X_df[col] = encoder.inverse_transform(X_df[col].astype(int))

        return model.predict_proba(X_df)

    # Get explanation using encoded data
    exp = lime_explainer.explain_instance(
        instance_array,
        predict_fn,
        num_features=num_features,
        top_labels=1
    )

    # Get the predicted class (using original unencoded data)
    instance_df = pd.DataFrame([instance_series])
    prediction = model.predict(instance_df)[0]

    # Extract feature weights
    exp_list = exp.as_list(label=prediction)
    features = [item[0] for item in exp_list]
    weights = [item[1] for item in exp_list]

    return features, weights, prediction


def explain_with_shap(model, instance_series, shap_explainer, feature_names, shap_encoders):
    """Generate SHAP explanation - encode instance, then explain"""

    # Encode the instance (same as LIME approach)
    instance_encoded = instance_series.copy()
    categorical_cols = ['sex', 'menopausal_status', 'physical_activity']

    for col in categorical_cols:
        if col in instance_encoded.index and col in shap_encoders:
            instance_encoded[col] = shap_encoders[col].transform([str(instance_encoded[col])])[0]

    # Ensure all values are float
    instance_encoded = instance_encoded.astype(float)

    # Convert to array for SHAP
    instance_array = instance_encoded.values.reshape(1, -1)

    # Get SHAP values with reduced samples for speed
    try:
        shap_values = shap_explainer.shap_values(
            instance_array,
            nsamples=100
        )
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        # Fallback: try with even fewer samples
        shap_values = shap_explainer.shap_values(
            instance_array,
            nsamples=50
        )

    # Extract values (handle different SHAP output formats)
    if isinstance(shap_values, list):
        # List format: one array per class
        # Get prediction to know which class to explain
        instance_df = pd.DataFrame([instance_series])
        prediction = model.predict(instance_df)[0]

        # Get SHAP values for predicted class
        if len(shap_values) > prediction:
            values = shap_values[prediction][0]
        else:
            values = shap_values[0][0]
    elif hasattr(shap_values, 'values'):
        if len(shap_values.values.shape) == 3:
            # Multi-class output - get values for predicted class
            instance_df = pd.DataFrame([instance_series])
            prediction = model.predict(instance_df)[0]
            values = shap_values.values[0, :, prediction]
        else:
            values = shap_values.values[0]
    else:
        # Direct numpy array
        if len(shap_values.shape) == 3:
            instance_df = pd.DataFrame([instance_series])
            prediction = model.predict(instance_df)[0]
            values = shap_values[0, :, prediction]
        elif len(shap_values.shape) == 2:
            values = shap_values[0]
        else:
            values = shap_values

    # Sort by absolute impact
    feature_impacts = list(zip(feature_names, values))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    return feature_impacts

def plot_lime_explanation(features, weights, predicted_class, probabilities):
    """Create beautiful LIME visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='none')
    fig.patch.set_alpha(0)

    # Feature importance plot
    colors = ['#4ade80' if w > 0 else '#f87171' for w in weights]
    y_pos = np.arange(len(features))

    ax1.barh(y_pos, weights, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features, fontsize=9, color='white')
    ax1.set_xlabel('Weight', fontsize=11, color='white')
    ax1.set_title(f'Feature Importance for {predicted_class}',
                  fontsize=13, fontweight='bold', color='white', pad=15)
    ax1.axvline(x=0, color='white', linestyle='-', linewidth=1.5, alpha=0.7)
    ax1.grid(axis='x', alpha=0.2, color='white')
    ax1.set_facecolor('none')
    ax1.tick_params(colors='white')

    for spine in ax1.spines.values():
        spine.set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for i, w in enumerate(weights):
        ax1.text(w, i, f' {w:.3f}', va='center', fontsize=9,
                 color='white', fontweight='bold')

    # Probability plot
    class_names = ['Normal', 'Osteopenia', 'Osteoporosis']
    colors_prob = ['#4ade80', '#fbbf24', '#f87171']

    ax2.barh(class_names, probabilities, color=colors_prob, alpha=0.8)
    ax2.set_xlabel('Probability', fontsize=11, color='white')
    ax2.set_title('Prediction Probabilities',
                  fontsize=13, fontweight='bold', color='white', pad=15)
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.2, color='white')
    ax2.set_facecolor('none')
    ax2.tick_params(colors='white')

    for spine in ax2.spines.values():
        spine.set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for i, prob in enumerate(probabilities):
        ax2.text(prob, i, f' {prob:.1%}', va='center',
                 fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_shap_explanation(feature_impacts, predicted_class, top_n=15):
    """Create beautiful SHAP visualization"""
    top_features = feature_impacts[:top_n]
    features = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='none')
    fig.patch.set_alpha(0)

    colors = ['#4ade80' if v > 0 else '#f87171' for v in values]
    y_pos = np.arange(len(features))

    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10, color='white')
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=11, color='white')
    ax.set_title(f'Feature Impact - {predicted_class}',
                 fontsize=13, fontweight='bold', color='white', pad=15)
    ax.axvline(x=0, color='white', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.grid(axis='x', alpha=0.2, color='white')
    ax.set_facecolor('none')
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, v in enumerate(values):
        ax.text(v, i, f' {v:.3f}', va='center',
                fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    return fig


# Load model
model = load_model()

def patient_prediction(input_df):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    class_map = {0: "Normal", 1: "Osteopenia", 2: "Osteoporosis"}
    predicted_class = class_map.get(prediction, prediction)

    st.session_state.prediction_data = {
        'prediction': predicted_class,
        'probabilities': probabilities,
        'input_df': input_df,
        'class_map': class_map
    }
    return predicted_class

def xray_prediction(xray_image, temp_filepath):
    unet_cmd = [
                "python",
                os.path.join(UNET_DIR, "infer_crop.py"),
                "--model_path", UNET_MODEL,
                "--image_path", temp_filepath,
                "--save_overlay"
    ]
                        
    result = subprocess.run(unet_cmd, capture_output=True, text=True, cwd=str(UNET_DIR))
    
    if result.returncode != 0:
        st.error(f"❌ UNET Error: {result.stderr}")
        st.stop()
    
    # Get latest predicted mask
    mask_files = sorted(
        Path(PREDICTIONS_DIR).glob("*_predicted_mask.png"),
        key=os.path.getmtime,
        reverse=True
    )
    
    if not mask_files:
        st.error("❌ No cropped image found!")
        st.stop()
    
    cropped_image_path = str(mask_files[0])
    
    # Copy to CNN infer folder
    cnn_infer_dir = os.path.join(CNN_DIR, "infer")
    os.makedirs(cnn_infer_dir, exist_ok=True)
    cnn_input_path = os.path.join(cnn_infer_dir, os.path.basename(cropped_image_path))
    shutil.copy(cropped_image_path, cnn_input_path)
    
    cnn_cmd = [
        "python",
        os.path.join(CNN_DIR, "inference.py"),
        "--image_path", cnn_input_path,
        "--model_path", CNN_MODEL
    ]
    
    result = subprocess.run(cnn_cmd, capture_output=True, text=True, cwd=str(CNN_DIR))
    
    if result.returncode != 0:
        st.error(f"❌ CNN Error: {result.stderr}")
        st.stop()
    
    # Parse CNN output
    output_lines = result.stdout.strip().split('\n')
    prediction_grade = None
    
    for line in output_lines:
        if "Predicted Grade" in line or "Grade" in line:
            # Extract grade number
            import re
            match = re.search(r'Grade\s*[:\-]?\s*(\d)', line)
            if match:
                prediction_grade = int(match.group(1))
                break
    st.session_state['last_prediction'] = prediction_grade
    st.session_state['last_cropped_image'] = cropped_image_path
    st.session_state['last_original_image'] = temp_filepath
    return prediction_grade

features = [
    "age", "sex", "height_cm", "weight_kg", "bmi",
    "waist_cm", "hip_cm", "waist_hip_ratio",
    "menopausal_status", "menopause_age", "years_since_menopause",
    "smoker", "alcohol_high", "physical_activity",
    "vitamin_d_ngml", "vitamin_d_missing",
    "serum_calcium_mgdl", "alkaline_phosphatase", "pth_pgml",
    "creatinine_mgdl", "hdl_mgdl", "ldl_mgdl",
    "ctx_ngml", "p1np_ugL",
    "estrogen_use", "diabetes_t2", "hypothyroidism",
    "rheumatoid_arthritis", "secondary_osteoporosis",
    "parent_hip_fracture", "prior_fracture", "bisphosphonate_use",
    "calcium_supplement", "vitamin_d_supplement", "glucocorticoid_use",
    "dialysis", "falls_past_year"
]

def get_base64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None


def set_background(png_file):
    bin_str = get_base64(png_file)
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{
          background-image: url("data:image/png;base64,{bin_str}");
          background-size: cover;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None


def switch_to_results():
    st.session_state.page = 'results'


def switch_to_input():
    st.session_state.page = 'input'


# Common styling
common_styles = """
<style>
header {visibility: hidden;}
.stApp {
    background-image: url("your-xray-image.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
    width: 100vw;
}

/* Prevent layout shifts */
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: none !important;
}
</style>
"""

# ==================== INPUT PAGE ====================
if st.session_state.page == 'input':
    st.markdown(common_styles, unsafe_allow_html=True)

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Forum&family=Jersey+10&family=Lexend+Giga:wght@100..900&family=Micro+5&display=swap');
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ============================================ */
    /* UPLOAD AREA CONTAINER - FIXED POSITIONING */
    /* ============================================ */
    div[data-testid="column"]:first-child {
        position: fixed !important;
        left: 17vw !important;
        top: 50vh !important;
        transform: translateY(-50%) !important;
        width: 55vw !important;
        max-width: 850px !important;
        height: auto !important;
        min-height: 480px !important;
        max-height: 600px !important;
        border-radius: 40px;
        padding: 45px 60px !important;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        z-index: 1;
        transition: all 0.3s ease !important;
        overflow: visible !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Title styling */
    div[data-testid="column"]:first-child h3 {
        color: white !important;
        font-weight: 600 !important;
        font-size: 2.0rem !important;
        margin-bottom: 3px !important;
        margin-top: 0 !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
    }

    div[data-testid="column"]:first-child p {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.8rem !important;
        margin-bottom: 20px !important;
        margin-top: 0 !important;
        writing-mode: horizontal-tb !important;
    }

    /* File uploader styling - centered */
    div[data-testid="column"]:first-child [data-testid="stFileUploader"] {
        position: relative !important;
        height: 280px !important;
        width: 100% !important;
        max-width: 700px !important;
        margin: 0 auto !important;
        border-radius: 35px !important;
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px);
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        padding: 0px !important;
    }

    /* Dropzone styling - centered content */
    div[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
        height: 100% !important;
        min-height: 280px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-direction: column !important;
        padding: 0 !important;
        gap: 12px !important;
    }

    div[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] > div {
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] svg {
        margin-bottom: 8px !important;
        width: 48px !important;
        height: 48px !important;
    }

    div[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] section {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Upload text styling */
    div[data-testid="column"]:first-child [data-testid="stFileUploader"] label,
    div[data-testid="column"]:first-child [data-testid="stFileUploader"] small,
    div[data-testid="column"]:first-child [data-testid="stFileUploader"] span {
        color: white !important;
        font-size: 0.95rem !important;
    }

    /* Browse button */
    div[data-testid="column"]:first-child [data-testid="stFileUploader"] button {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        font-size: 0.9rem !important;
    }

    div[data-testid="column"]:first-child [data-testid="stFileUploader"] button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
        transition: all 0.2s ease;
    }

    /* Image preview container */
    div[data-testid="column"]:first-child [data-testid="stImage"] {
        margin-top: 0px !important;
        margin-bottom: 10px !important;
        border-radius: 35px !important;
        overflow: hidden;
        max-height: 340px !important;
        height: 340px !important;
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px);
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        padding: 20px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="column"]:first-child [data-testid="stImage"] img {
        max-height: 300px !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        display: block !important;
        margin: 0 auto !important;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    div[data-testid="column"]:first-child [data-testid="stImage"] figcaption {
        display: none !important;
    }
    
    /* Change image button styling */
    div[data-testid="column"]:first-child button[key="change_xray"],
    div[data-testid="column"]:first-child button[kind="secondary"],
    div[data-testid="column"]:first-child button[kind="primary"],
    div[data-testid="column"]:first-child button[data-baseweb="button"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 220px !important;
        min-height: 48px !important;
        width: fit-content !important;
        font-family: 'Lexend Giga', sans-serif !important;
        font-weight: 300 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.8px !important;
        line-height: 1.4 !important;
        text-align: center !important;
        text-transform: none !important;
        margin: 15px auto 0px auto !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    div[data-testid="column"]:first-child button[key="change_xray"]:hover,
    div[data-testid="column"]:first-child button[kind="secondary"]:hover,
    div[data-testid="column"]:first-child button[kind="primary"]:hover,
    div[data-testid="column"]:first-child button[data-baseweb="button"]:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.25) !important;
    }
    
    div[data-testid="column"]:first-child button p,
    div[data-testid="column"]:first-child button span,
    div[data-testid="column"]:first-child button div {
        font-family: 'Lexend Giga', sans-serif !important;
        font-weight: 300 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.8px !important;
        color: white !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ============================================ */
    /* FORM HEADER AND POSITIONING */
    /* ============================================ */
    
    .form-header-text {
        position: fixed !important;
        right: 5vw !important;
        top: 7vh !important;
        z-index: 3 !important;
        width: auto !important;
    }
    
    .form-header-text h1 {
        color: white !important;
        font-family: 'Lexend Giga', sans-serif !important;
        font-weight: 200 !important;
        font-size: 2.0rem !important;
        margin: 0 !important;
        letter-spacing: 1px !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        white-space: nowrap !important;
    }
    
    /* Form container positioning */
    div[data-testid="column"]:last-child [data-testid="stForm"] {
        position: fixed !important;
        right: 5vw !important;
        top: 50vh !important;
        transform: translateY(-50%) !important;
        width: 22vw !important;
        min-width: 320px !important;
        max-width: 380px !important;
        max-height: 70vh !important;
        border-radius: 40px;
        padding: 25px 20px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        overflow-y: auto !important;
        overflow-x: hidden !important;
        z-index: 2;
    }

    /* Scrollbar styling */
    [data-testid="stForm"]::-webkit-scrollbar {width: 6px;}
    [data-testid="stForm"]::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1); 
        border-radius: 10px;
    }
    [data-testid="stForm"]::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.4); 
        border-radius: 10px;
    }

    /* Form inputs styling */
    [data-testid="stForm"] input, 
    [data-testid="stForm"] select {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }

    [data-testid="stForm"] label {
        color: white !important; 
        font-size: 14px;
    }
    
    /* Ensure all text elements are horizontal */
    * {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create two-column layout
    col_upload, col_form = st.columns([2.2, 1])

    with col_upload:
        st.markdown("### Upload X-Ray")
        st.markdown("Choose images")

        # Only show uploader if no image is uploaded
        if 'uploaded_xray' not in st.session_state or st.session_state['uploaded_xray'] is None:
            uploaded_xray = st.file_uploader(
                label = "",
                type=['png', 'jpg', 'jpeg'],
                key="xray_uploader"
            )

            if uploaded_xray is not None:
                st.session_state['uploaded_xray'] = uploaded_xray
                xray_image = Image.open(uploaded_xray)
                st.image(xray_image, caption="Uploaded X-Ray")
                
                # Save uploaded file
                timestamp = int(time.time())
                temp_filename = f"upload_{timestamp}_{uploaded_xray.name}"
                temp_filepath = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
                
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_xray.getbuffer())
                
                # Store filepath in session state
                st.session_state['temp_filepath'] = temp_filepath
                st.rerun()
        else:
            # Show preview with option to change
            try:
                img = Image.open(st.session_state['uploaded_xray'])
                st.image(img, use_column_width=True)

                # Add a button to upload different image
                if st.button("Upload Different Image", key="change_xray"):
                    st.session_state['uploaded_xray'] = None
                    st.session_state['temp_filepath'] = None
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.session_state['uploaded_xray'] = None

    with col_form:
        st.markdown('<div class="form-header-text"><h1>OSTEOPOROSIS DETECTOR</h1></div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            inputs = {}

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Anthropometric", "Menopausal", "Medical", "Lifestyle Factors", "Laboratory Values"
            ])

            with tab1:
                inputs["age"] = st.number_input("Age", min_value=18, max_value=120, value=50)
                inputs["sex"] = st.radio("Sex", ["Female", "Male"], horizontal=True)
                inputs["height_cm"] = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
                inputs["weight_kg"] = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
                inputs["waist_cm"] = st.number_input("Waist (cm)", min_value=50.0, max_value=200.0, value=85.0)
                inputs["hip_cm"] = st.number_input("Hip (cm)", min_value=50.0, max_value=200.0, value=100.0)

            with tab2:
                st.info("👩 Female patients only — these fields are ignored for males")
                inputs["menopausal_status"] = st.radio("Menopausal Status", ["Pre", "Peri", "Post"], horizontal=True)
                inputs["menopause_age"] = st.number_input("Menopause Age", min_value=0.0, max_value=70.0, value=0.0)
                inputs["estrogen_use"] = st.radio("Estrogen Use", ["Yes", "No"], horizontal=True)

            with tab3:
                inputs["prior_fracture"] = st.radio("Prior Fracture?", ["Yes", "No"], horizontal=True, index=1)
                inputs["parent_hip_fracture"] = st.radio("Parent Hip Fracture?", ["Yes", "No"], horizontal=True, index=1)
                inputs["diabetes_t2"] = st.radio("Type 2 Diabetes?", ["Yes", "No"], horizontal=True, index=1)
                inputs["hypothyroidism"] = st.radio("Hypothyroidism?", ["Yes", "No"], horizontal=True, index=1)
                inputs["dialysis"] = st.radio("Dialysis?", ["Yes", "No"], horizontal=True, index=1)
                inputs["bisphosphonate_use"] = st.radio("Bisphosphonate Use?", ["Yes", "No"], horizontal=True, index=1)
                inputs["glucocorticoid_use"] = st.radio("Glucocorticoid Use?", ["Yes", "No"], horizontal=True, index=1)
                inputs["rheumatoid_arthritis"] = st.radio("Rheumatoid Arthritis?", ["Yes", "No"], horizontal=True, index=1)
                inputs["secondary_osteoporosis"] = st.radio("Secondary Osteoporosis?", ["Yes", "No"], horizontal=True,
                                                            index=1)
                inputs["falls_past_year"] = st.number_input(
                    "Falls in Past Year",
                    min_value=0,
                    max_value=50,
                    value=0,
                    help="Number of falls experienced in the past year"
                )

            with tab4:
                inputs["smoker"] = st.radio("Smoker?", ["Yes", "No"], horizontal=True, index=1)
                inputs["alcohol_high"] = st.radio("High Alcohol Consumption?", ["Yes", "No"], horizontal=True, index=1)
                inputs["physical_activity"] = st.radio("Physical Activity Level",
                                                       ["Low", "Moderate", "High"], horizontal=True, index=1)
                inputs["calcium_supplement"] = st.radio("Calcium Supplement?", ["Yes", "No"], horizontal=True, index=1)
                inputs["vitamin_d_supplement"] = st.radio("Vitamin D Supplement?", ["Yes", "No"], horizontal=True, index=1)

            with tab5:
                inputs["vitamin_d_ngml"] = st.number_input("Vitamin D (ng/mL)", min_value=0.0, max_value=150.0, value=25.0)
                inputs["serum_calcium_mgdl"] = st.number_input("Serum Calcium (mg/dL)", min_value=0.0, max_value=20.0,
                                                               value=9.5)
                inputs["alkaline_phosphatase"] = st.number_input("Alkaline Phosphatase", min_value=0.0, max_value=500.0,
                                                                 value=75.0)
                inputs["pth_pgml"] = st.number_input("PTH (pg/mL)", min_value=0.0, max_value=200.0, value=40.0)
                inputs["creatinine_mgdl"] = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0)
                inputs["hdl_mgdl"] = st.number_input("HDL (mg/dL)", min_value=0.0, max_value=200.0, value=60.0)
                inputs["ldl_mgdl"] = st.number_input("LDL (mg/dL)", min_value=0.0, max_value=500.0, value=120.0)
                inputs["ctx_ngml"] = st.number_input("CTX (ng/mL)", min_value=0.0, max_value=2.0, value=0.3)
                inputs["p1np_ugL"] = st.number_input("P1NP (µg/L)", min_value=0.0, max_value=200.0, value=45.0)


            st.markdown("<br>", unsafe_allow_html=True)
            xray_uploaded = 'uploaded_xray' in st.session_state and st.session_state['uploaded_xray'] is not None
            if not xray_uploaded:
                st.warning("⚠️ Please upload an X-ray image to enable prediction")

            submitted = st.form_submit_button(
                "Predict", 
                use_container_width=True,
                disabled=not xray_uploaded
            )
            st.markdown("<br>", unsafe_allow_html=True)

            if submitted:
                # Handle male overrides
                if inputs["sex"] == "Male":
                    inputs["menopause_age"] = 0
                    inputs["menopausal_status"] = "na"

                # Process inputs
                processed_inputs = inputs.copy()
                processed_inputs.update({
                    "bmi": (inputs["weight_kg"] / (inputs["height_cm"] ** 2)) * 10000 if inputs[
                                                                                                     "height_cm"] > 0 else np.nan,
                    "waist_hip_ratio": (inputs["waist_cm"] / inputs["hip_cm"]) if inputs["hip_cm"] > 0 else np.nan,
                    "years_since_menopause": (
                        inputs["age"] - inputs["menopause_age"]
                        if inputs["sex"] == "Female" and str(inputs["menopausal_status"]).lower() == "post"
                        else 0
                    ),
                    "vitamin_d_missing": 1 if inputs["vitamin_d_ngml"] == 0 else 0
                })

                # Convert binary fields to integers
                binary_fields = [
                    "prior_fracture", "parent_hip_fracture", "glucocorticoid_use",
                    "rheumatoid_arthritis", "secondary_osteoporosis", "diabetes_t2",
                    "hypothyroidism", "estrogen_use", "bisphosphonate_use",
                    "vitamin_d_supplement", "calcium_supplement", "smoker",
                    "alcohol_high", "dialysis"
                ]
                for field in binary_fields:
                    val = str(processed_inputs.get(field, "")).strip().lower()
                    processed_inputs[field] = 1 if val in ["yes", "true", "1"] else 0

                # Convert categorical to match training format
                processed_inputs["sex"] = "F" if inputs["sex"] == "Female" else "M"
                processed_inputs["menopausal_status"] = str(processed_inputs["menopausal_status"]).lower()
                processed_inputs["physical_activity"] = str(processed_inputs["physical_activity"]).lower()

                # Build dataframe with correct feature order
                input_df = pd.DataFrame([processed_inputs])
                for col in features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[features]

                # Make prediction
                try:
                    with st.spinner("Generating prediction..."):
                        patient_classificaation = patient_prediction(input_df)
                        xray_classificaation = xray_prediction(Image.open(st.session_state['uploaded_xray']), st.session_state['temp_filepath'])
    
                        switch_to_results()
                        st.rerun()
    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())

        set_background(bg_path)

# ==================== RESULTS PAGE ====================
elif st.session_state.page == 'results':
    st.markdown(common_styles, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Hide the upload containers from input page */
    div[data-testid="column"]:first-child {
        display: none !important;
    }
    
    /* Style for Make Another Prediction button - match Predict button */
    .stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        font-family: 'Lexend Giga', sans-serif !important;
        font-weight: 300 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        width: auto !important;
        min-width: 250px !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.25) !important;
    }
    
    /* Center button container */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    /* Remove the blurry container box */
    .prediction-box {
        background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px);
        padding: 2rem; border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center; margin-bottom: 2rem;
    }
    .prob-container {
        display: flex; justify-content: space-around;
        margin-top: 1.5rem; gap: 1rem;
    }
    .prob-item {
        flex: 1; background: rgba(255, 255, 255, 0.1);
        padding: 1rem; border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .prob-item p {margin: 0.5rem 0; color: white;}
    .explanation-section {
        background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px);
        padding: 2rem; border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3); margin-bottom: 2rem;
    }
    .explanation-section h2 {color: white; margin-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)

    # Back button
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("← Back"):
            switch_to_input()
            st.rerun()

    if st.session_state.prediction_data is None:
        st.error("No prediction data available. Please go back and submit the form.")
    else:
        pred_data = st.session_state.prediction_data
        predicted_class = pred_data['prediction']
        probabilities = pred_data['probabilities']
        input_df = pred_data['input_df']

        # Display prediction
        st.markdown(f"""
        <div class='prediction-box'>
            <h2 style='color: rgba(255, 255, 255, 0.7); margin-bottom: 0.5rem; font-size: 1.2rem;'>Predicted Diagnosis</h2>
            <h1 style='color: white; font-size: 3rem; margin: 1rem 0;'>{predicted_class}</h1>
            <hr style='border: 1px solid rgba(255, 255, 255, 0.2); margin: 1.5rem 0;'>
            <h3 style='color: rgba(255, 255, 255, 0.8); margin-bottom: 1rem;'>Prediction Probabilities</h3>
            <div class='prob-container'>
                <div class='prob-item'>
                    <p style='color: #4ade80; font-size: 2rem; font-weight: bold;'>{probabilities[0]:.1%}</p>
                    <p style='color: rgba(255, 255, 255, 0.9); font-weight: 600;'>Normal</p>
                </div>
                <div class='prob-item'>
                    <p style='color: #fbbf24; font-size: 2rem; font-weight: bold;'>{probabilities[1]:.1%}</p>
                    <p style='color: rgba(255, 255, 255, 0.9); font-weight: 600;'>Osteopenia</p>
                </div>
                <div class='prob-item'>
                    <p style='color: #f87171; font-size: 2rem; font-weight: bold;'>{probabilities[2]:.1%}</p>
                    <p style='color: rgba(255, 255, 255, 0.9); font-weight: 600;'>Osteoporosis</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Load training data and create explainers
        try:
            X_train = load_training_data()
            lime_explainer, lime_encoders = create_lime_explainer(model, X_train)
            shap_explainer, feature_names, shap_encoders = create_shap_explainer(model, X_train)
        except Exception as e:
            st.error(f"Failed to initialize explainers: {e}")
            import traceback
            st.error(traceback.format_exc())
            lime_explainer = None
            shap_explainer = None

        # LIME Explanation
        if lime_explainer:
            st.markdown("<div class='explanation-section'>", unsafe_allow_html=True)
            st.markdown("<h2>🔍 LIME Explanation</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p style='color: rgba(255, 255, 255, 0.9);'>Shows which features most influenced this specific prediction.</p>",
                unsafe_allow_html=True)

            with st.spinner("Generating LIME explanation..."):
                try:
                    instance = input_df.iloc[0]
                    features_lime, weights, _ = explain_with_lime(
                        model, instance, lime_explainer, lime_encoders, num_features=10
                    )
                    fig = plot_lime_explanation(features_lime, weights, predicted_class, probabilities)
                    st.pyplot(fig, transparent=True)
                    plt.close()
                except Exception as e:
                    st.error(f"LIME explanation failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())

            st.markdown("</div>", unsafe_allow_html=True)

        # SHAP Explanation
        if shap_explainer:
            st.markdown("<div class='explanation-section'>", unsafe_allow_html=True)
            st.markdown("<h2>📊 SHAP Explanation</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p style='color: rgba(255, 255, 255, 0.9);'>Game-theoretic approach to explain individual predictions.</p>",
                unsafe_allow_html=True)

            with st.spinner("Generating SHAP explanation (this may take 30-60 seconds)..."):
                try:
                    st.info(
                        "⏱️ Computing SHAP values with 100 samples. This provides a good balance between accuracy and speed.")

                    instance = input_df.iloc[0]
                    feature_impacts = explain_with_shap(
                        model, instance, shap_explainer, feature_names, shap_encoders
                    )
                    fig = plot_shap_explanation(feature_impacts, predicted_class, top_n=15)
                    st.pyplot(fig, transparent=True)
                    plt.close()
                except Exception as e:
                    st.error(f"SHAP explanation failed: {e}")
                    st.warning("💡 Tip: SHAP computation can be resource-intensive. The prediction is still valid!")
                    import traceback
                    st.error(traceback.format_exc())

            st.markdown("</div>", unsafe_allow_html=True)

        # Interpretation guide
        st.markdown("<div class='explanation-section'>", unsafe_allow_html=True)
        with st.expander("📖 How to Interpret These Visualizations", expanded=False):
            st.markdown("""
            <div style='color: rgba(255, 255, 255, 0.95);'>

            ### LIME Explanation
            - **Green bars** = features increasing the predicted class probability
            - **Red bars** = features decreasing the predicted class probability
            - **Longer bars** = stronger influence

            ### SHAP Explanation
            - **Green bars** = features pushing toward the predicted class
            - **Red bars** = features pushing away from the predicted class
            - **SHAP values are additive** - they sum to explain the prediction

            ### Key Differences
            - **LIME** approximates the model locally with a simpler model
            - **SHAP** uses game theory to fairly distribute prediction among features
            - Both provide complementary insights into model decision-making

            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # New prediction button - centered
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🔄 Make Another Prediction"):
            switch_to_input()
            st.rerun()

    set_background(bg_path)