
'''
*****************************************************        **********************************************
'''

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all necessary libraries for the StackingEnsembleOptuna class
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin, clone

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix

# Define the MultiLabelEncoder class
class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            if col in self.label_encoders:
                X_encoded[col] = self.label_encoders[col].transform(X[col].astype(str))
        return X_encoded.values

# Define the StackingEnsembleOptuna class
class StackingEnsembleOptuna:
    def __init__(self, cv_folds=5, random_state=42, n_classes=3):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_classes = n_classes
        self.best_base_models = {}
        self.best_meta_model = None
        self.trained_pipelines = {}
        self.study = None

        # Define column types based on dataset info
        self.numerical_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'waist_cm', 'hip_cm',
                              'waist_hip_ratio', 'menopause_age', 'years_since_menopause',
                              'vitamin_d_ngml', 'serum_calcium_mgdl', 'alkaline_phosphatase',
                              'pth_pgml', 'creatinine_mgdl', 'hdl_mgdl', 'ldl_mgdl', 'ctx_ngml', 'p1np_ugL']

        self.binary_cols = ['estrogen_use', 'diabetes_t2', 'hypothyroidism', 'dialysis',
                           'bisphosphonate_use', 'prior_fracture', 'parent_hip_fracture',
                           'smoker', 'alcohol_high', 'glucocorticoid_use', 'rheumatoid_arthritis',
                           'secondary_osteoporosis', 'calcium_supplement', 'vitamin_d_supplement',
                           'vitamin_d_missing']

        self.categorical_cols = ['sex', 'menopausal_status', 'physical_activity']

    def _get_preprocessor_for_model(self, model_name):
        """Create model-specific preprocessor based on model requirements"""
        # Tree-based models
        if model_name in ['decision_tree', 'random_forest', 'extra_trees', 'xgboost', 'lightgbm']:
            return ColumnTransformer([
                ("num", "passthrough", self.numerical_cols),
                ("bin", "passthrough", self.binary_cols),
                ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.categorical_cols)
            ], remainder='drop')

        # CatBoost can handle categorical natively
        elif model_name == 'catboost':
            return ColumnTransformer([
                ("num", StandardScaler(), self.numerical_cols),
                ("bin", "passthrough", self.binary_cols),
                ("cat", "passthrough", self.categorical_cols)
            ], remainder='drop')

        # Linear models and others: StandardScaler + OneHotEncoder needed
        else:  # logistic_regression, svm, polynomial, naive_bayes, knn, adaboost
            return ColumnTransformer([
                ("num", StandardScaler(), self.numerical_cols),
                ("bin", StandardScaler(), self.binary_cols),  # Scale binary for linear models
                ("cat", OneHotEncoder(drop='first', sparse_output=False), self.categorical_cols)
            ], remainder='drop')

    def _create_pipeline_for_model(self, model, model_name):
        """Create a pipeline with model-specific preprocessing"""
        # Special handling for tree-based models that need label encoding
        if model_name in ['decision_tree', 'random_forest', 'extra_trees', 'xgboost', 'lightgbm']:
            preprocessor = ColumnTransformer([
                ("num", "passthrough", self.numerical_cols),
                ("bin", "passthrough", self.binary_cols),
                ("cat", MultiLabelEncoder(), self.categorical_cols)
            ], remainder='drop')
        else:
            preprocessor = self._get_preprocessor_for_model(model_name)

        if model_name == "polynomial":
            return Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
        else:
            return Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

    def predict(self, X):
        """Make predictions using the trained stacking ensemble"""
        if not self.trained_pipelines or self.best_meta_model is None:
            raise ValueError("Must fit the model first!")

        n_samples = len(X)
        n_models = len(self.trained_pipelines)

        # Get base model predictions - all class probabilities
        base_predictions = np.zeros((n_samples, n_models * self.n_classes))

        for model_idx, (model_name, pipeline) in enumerate(self.trained_pipelines.items()):
            base_probs = pipeline.predict_proba(X)

            # Store all class probabilities
            start_col = model_idx * self.n_classes
            end_col = start_col + self.n_classes
            base_predictions[:, start_col:end_col] = base_probs

        # Meta-model final predictions
        final_predictions = self.best_meta_model.predict(base_predictions)
        return final_predictions

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.trained_pipelines or self.best_meta_model is None:
            raise ValueError("Must fit the model first!")

        n_samples = len(X)
        n_models = len(self.trained_pipelines)

        # Get all class probabilities from base models
        base_predictions = np.zeros((n_samples, n_models * self.n_classes))

        for model_idx, (model_name, pipeline) in enumerate(self.trained_pipelines.items()):
            base_probs = pipeline.predict_proba(X)

            # Store all class probabilities
            start_col = model_idx * self.n_classes
            end_col = start_col + self.n_classes
            base_predictions[:, start_col:end_col] = base_probs

        final_probabilities = self.best_meta_model.predict_proba(base_predictions)
        return final_probabilities

    @property
    def classes_(self):
        """Return the classes for compatibility"""
        if self.best_meta_model is not None:
            return self.best_meta_model.classes_
        else:
            return np.array([0, 1, 2])  # Default classes

# Load trained ensemble stacking model
MODEL_PATH = "models/ensemble_stacking_model.joblib"  # Updated path to match your saved file
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Define features (order matters!)
features = [
    "age","sex","height_cm","weight_kg","bmi","waist_cm","hip_cm","waist_hip_ratio","menopause_age","menopausal_status",
    "years_since_menopause","estrogen_use","diabetes_t2","hypothyroidism","dialysis","bisphosphonate_use","prior_fracture",
    "parent_hip_fracture","smoker","alcohol_high","glucocorticoid_use","rheumatoid_arthritis","secondary_osteoporosis",
    "physical_activity","calcium_supplement","vitamin_d_supplement","falls_past_year","vitamin_d_ngml","serum_calcium_mgdl",
    "alkaline_phosphatase","pth_pgml","creatinine_mgdl","hdl_mgdl","ldl_mgdl","ctx_ngml","p1np_ugL","vitamin_d_missing"
]

# Mandatory fields (choose most clinically relevant)
mandatory = [
    "age","sex","bmi","menopausal_status","prior_fracture",
    "parent_hip_fracture","smoker","vitamin_d_ngml"
]

# Default values (could be updated from training dataset)
defaults = {
    "age": 60,
    "sex": "F",
    "height_cm": 160,
    "weight_kg": 65,
    "bmi": 25,
    "waist_cm": 85,
    "hip_cm": 95,
    "waist_hip_ratio": 0.89,
    "menopause_age": 50,
    "menopausal_status": "post",
    "years_since_menopause": 10,
    "estrogen_use": 0,
    "diabetes_t2": 0,
    "hypothyroidism": 0,
    "dialysis": 0,
    "bisphosphonate_use": 0,
    "prior_fracture": 0,
    "parent_hip_fracture": 0,
    "smoker": 0,
    "alcohol_high": 0,
    "glucocorticoid_use": 0,
    "rheumatoid_arthritis": 0,
    "secondary_osteoporosis": 0,
    "physical_activity": "moderate",
    "calcium_supplement": 0,
    "vitamin_d_supplement": 0,
    "falls_past_year": 0,
    "vitamin_d_ngml": 25,
    "serum_calcium_mgdl": 9.5,
    "alkaline_phosphatase": 70,
    "pth_pgml": 40,
    "creatinine_mgdl": 1.0,
    "hdl_mgdl": 50,
    "ldl_mgdl": 100,
    "ctx_ngml": 0.3,
    "p1np_ugL": 40,
    "vitamin_d_missing": 0
}

# Class labels mapping
class_labels = {0: 'Normal', 1: 'Osteopenia', 2: 'Osteoporosis'}

# ----------------- UI -----------------
st.set_page_config(page_title="Osteoporosis Risk Prediction", layout="wide")

st.markdown("""
<style>
.header {background-color:#0b78a0;padding:20px;border-radius:10px;margin-bottom:20px}
.big-title {font-size:28px;color:white;font-weight:700}
.mand {color:#d9534f;font-weight:700}
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}
.normal {background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;}
.osteopenia {background-color: #fff3cd; color: #856404; border: 2px solid #ffeaa7;}
.osteoporosis {background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><div class="big-title">🦴 Osteoporosis Clinical Prediction Tool</div></div>', unsafe_allow_html=True)

st.write("Fill in patient details below. Fields marked with ***** are mandatory. Defaults are prefilled based on clinical data averages.")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📋 Patient Input", "📊 Results", "ℹ️ Model Info"])

with tab1:
    with st.form("input_form"):
        # Organize inputs into sections
        st.subheader("👤 Basic Demographics")
        col1, col2, col3 = st.columns(3)
        
        inputs = {}
        
        # Demographics section
        with col1:
            inputs["age"] = st.number_input("Age *", value=float(defaults["age"]), min_value=18.0, max_value=120.0)
            inputs["height_cm"] = st.number_input("Height (cm)", value=float(defaults["height_cm"]), min_value=100.0, max_value=250.0)
            inputs["sex"] = st.selectbox("Sex *", options=["F", "M"], index=0)
        
        with col2:
            inputs["weight_kg"] = st.number_input("Weight (kg)", value=float(defaults["weight_kg"]), min_value=30.0, max_value=200.0)
            inputs["bmi"] = st.number_input("BMI *", value=float(defaults["bmi"]), min_value=10.0, max_value=60.0)
            inputs["waist_cm"] = st.number_input("Waist (cm)", value=float(defaults["waist_cm"]), min_value=50.0, max_value=200.0)
        
        with col3:
            inputs["hip_cm"] = st.number_input("Hip (cm)", value=float(defaults["hip_cm"]), min_value=50.0, max_value=200.0)
            inputs["waist_hip_ratio"] = st.number_input("Waist-Hip Ratio", value=float(defaults["waist_hip_ratio"]), min_value=0.5, max_value=2.0)
        
        st.subheader("🩺 Menopause & Hormones")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            inputs["menopausal_status"] = st.selectbox("Menopausal Status *", options=["pre", "peri", "post","na"], index=2)
            inputs["menopause_age"] = st.number_input("Menopause Age", value=float(defaults["menopause_age"]), min_value=0.0, max_value=70.0)
        
        with col5:
            inputs["years_since_menopause"] = st.number_input("Years Since Menopause", value=float(defaults["years_since_menopause"]), min_value=0.0, max_value=50.0)
            inputs["estrogen_use"] = st.selectbox("Estrogen Use", options=[0, 1], index=0)
        
        st.subheader("🏥 Medical History")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            inputs["prior_fracture"] = st.selectbox("Prior Fracture *", options=[0, 1], index=0, help="0=No, 1=Yes")
            inputs["parent_hip_fracture"] = st.selectbox("Parent Hip Fracture *", options=[0, 1], index=0, help="0=No, 1=Yes")
            inputs["diabetes_t2"] = st.selectbox("Type 2 Diabetes", options=[0, 1], index=0)
            inputs["hypothyroidism"] = st.selectbox("Hypothyroidism", options=[0, 1], index=0)
        
        with col8:
            inputs["dialysis"] = st.selectbox("Dialysis", options=[0, 1], index=0)
            inputs["bisphosphonate_use"] = st.selectbox("Bisphosphonate Use", options=[0, 1], index=0)
            inputs["glucocorticoid_use"] = st.selectbox("Glucocorticoid Use", options=[0, 1], index=0)
            inputs["rheumatoid_arthritis"] = st.selectbox("Rheumatoid Arthritis", options=[0, 1], index=0)
        
        with col9:
            inputs["secondary_osteoporosis"] = st.selectbox("Secondary Osteoporosis", options=[0, 1], index=0)
            inputs["falls_past_year"] = st.number_input("Falls Past Year", value=float(defaults["falls_past_year"]), min_value=0.0, max_value=50.0)
        
        st.subheader("🚬 Lifestyle")
        col10, col11, col12 = st.columns(3)
        
        with col10:
            inputs["smoker"] = st.selectbox("Smoker *", options=[0, 1], index=0, help="0=No, 1=Yes")
            inputs["alcohol_high"] = st.selectbox("High Alcohol Consumption", options=[0, 1], index=0)
        
        with col11:
            inputs["physical_activity"] = st.selectbox("Physical Activity Level", 
                                                     options=["low", "moderate", "high"], 
                                                     index=1, 
                                                     help="Physical activity level")
            inputs["calcium_supplement"] = st.selectbox("Calcium Supplement", options=[0, 1], index=1)
        
        with col12:
            inputs["vitamin_d_supplement"] = st.selectbox("Vitamin D Supplement", options=[0, 1], index=1)
        
        st.subheader("🧪 Laboratory Values")
        col13, col14, col15 = st.columns(3)
        
        with col13:
            inputs["vitamin_d_ngml"] = st.number_input("Vitamin D (ng/mL) *", value=float(defaults["vitamin_d_ngml"]), min_value=0.0, max_value=150.0)
            inputs["serum_calcium_mgdl"] = st.number_input("Serum Calcium (mg/dL)", value=float(defaults["serum_calcium_mgdl"]), min_value=0.0, max_value=20.0)
            inputs["alkaline_phosphatase"] = st.number_input("Alkaline Phosphatase", value=float(defaults["alkaline_phosphatase"]), min_value=0.0, max_value=500.0)
        
        with col14:
            inputs["pth_pgml"] = st.number_input("PTH (pg/mL)", value=float(defaults["pth_pgml"]), min_value=0.0, max_value=200.0)
            inputs["creatinine_mgdl"] = st.number_input("Creatinine (mg/dL)", value=float(defaults["creatinine_mgdl"]), min_value=0.0, max_value=10.0)
            inputs["hdl_mgdl"] = st.number_input("HDL (mg/dL)", value=float(defaults["hdl_mgdl"]), min_value=0.0, max_value=200.0)
        
        with col15:
            inputs["ldl_mgdl"] = st.number_input("LDL (mg/dL)", value=float(defaults["ldl_mgdl"]), min_value=0.0, max_value=500.0)
            inputs["ctx_ngml"] = st.number_input("CTX (ng/mL)", value=float(defaults["ctx_ngml"]), min_value=0.0, max_value=2.0)
            inputs["p1np_ugL"] = st.number_input("P1NP (ug/L)", value=float(defaults["p1np_ugL"]), min_value=0.0, max_value=200.0)
            inputs["vitamin_d_missing"] = st.selectbox("Vitamin D Missing", options=[0, 1], index=0)

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

with tab2:
    if submitted:
        # Validate mandatory fields
        missing = [f for f in mandatory if str(inputs.get(f, "")).strip() == ""]
        if missing:
            st.error(f"❌ Please fill mandatory fields: {', '.join(missing)}")
        else:
            try:
                # Create DataFrame with correct column order
                input_df = pd.DataFrame([inputs])
                
                # Ensure all expected columns are present
                for feat in features:
                    if feat not in input_df.columns:
                        input_df[feat] = defaults.get(feat, 0)
                
                # Reorder columns to match training data
                input_df = input_df[features]
                
                # CRITICAL: Proper data type conversion and encoding
                # Convert all numeric fields to proper numeric types
                numeric_fields = [
                    'age', 'height_cm', 'weight_kg', 'bmi', 'waist_cm', 'hip_cm',
                    'waist_hip_ratio', 'menopause_age', 'years_since_menopause',
                    'estrogen_use', 'diabetes_t2', 'hypothyroidism', 'dialysis',
                    'bisphosphonate_use', 'prior_fracture', 'parent_hip_fracture',
                    'smoker', 'alcohol_high', 'glucocorticoid_use', 'rheumatoid_arthritis',
                    'secondary_osteoporosis','calcium_supplement', 
                    'vitamin_d_supplement', 'falls_past_year', 'vitamin_d_ngml',
                    'serum_calcium_mgdl', 'alkaline_phosphatase', 'pth_pgml',
                    'creatinine_mgdl', 'hdl_mgdl', 'ldl_mgdl', 'ctx_ngml', 'p1np_ugL',
                    'vitamin_d_missing'
                ]
                
                for col in numeric_fields:
                    if col in input_df.columns:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
                # Handle categorical variables - keep them as strings since the model's 
                # preprocessing pipelines will handle the encoding internally
                # DON'T manually encode - let the trained pipelines handle it
                
                # Just ensure categorical variables are strings
                categorical_cols = ['sex', 'menopausal_status', 'physical_activity']
                for col in categorical_cols:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(str)
                
                # Ensure no NaN values (but keep strings as strings)
                categorical_cols = ['sex', 'menopausal_status', 'physical_activity']
                for col in input_df.columns:
                    if col in categorical_cols:
                        # Keep categorical as strings
                        input_df[col] = input_df[col].fillna('unknown')
                    else:
                        # Fill numeric with 0 and convert to float
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
                
                # Debug information
                # st.write("🔍 **Debug Information:**")
                # for i in range(37):
                #     st.write(f"  {i+1}. {input_df.columns[i]}: {input_df[input_df.columns[i]].dtype} = {input_df[input_df.columns[i]].iloc[0]}")
                # # st.write(f"Data by column:{input_df.columns}")
                # # st.write(f"Data by column:{input_df.values}")
                # st.write(f"Input dataframe shape: {input_df.shape}")
                # st.write(f"Expected features: {len(features)}")
                # st.write(f"Actual features: {len(input_df.columns)}")
                
                # st.write(f"Categorical columns: sex={input_df['sex'].iloc[0]}, menopausal_status={input_df['menopausal_status'].iloc[0]}, physical_activity={input_df['physical_activity'].iloc[0]}")
                
                # Try prediction with more detailed error handling
                st.write("🔄 Attempting predictions...")
                
                # Test each pipeline individually to find the problematic one
                for model_name, pipeline in model.trained_pipelines.items():
                    try:
                        st.write(f"Testing {model_name}...")
                        test_pred = pipeline.predict_proba(input_df)
                        st.write(f"✅ {model_name}: OK (shape: {test_pred.shape})")
                    except Exception as e:
                        st.write(f"❌ {model_name}: FAILED - {str(e)}")
                        # Let's examine this pipeline more closely
                        st.write(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")
                        
                        # Try just the preprocessor
                        if hasattr(pipeline, 'steps') and len(pipeline.steps) > 0:
                            preprocessor = pipeline.steps[0][1]  # Usually the first step
                            try:
                                transformed_data = preprocessor.transform(input_df)
                                st.write(f"  ✅ Preprocessing OK: shape {transformed_data.shape}")
                                st.write(f"  Data types in transformed: {transformed_data.dtype if hasattr(transformed_data, 'dtype') else 'mixed'}")
                            except Exception as preprocess_error:
                                st.write(f"  ❌ Preprocessing failed: {str(preprocess_error)}")
                
                # If we get here without individual pipeline errors, try the full prediction
                pred = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0]
                
                # Map prediction to label
                pred_label = class_labels.get(pred, f"Class {pred}")
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                # Main prediction with colored box
                if pred == 0:  # Normal
                    st.markdown(f'<div class="prediction-box normal">🟢 PREDICTION: {pred_label}</div>', unsafe_allow_html=True)
                elif pred == 1:  # Osteopenia
                    st.markdown(f'<div class="prediction-box osteopenia">🟡 PREDICTION: {pred_label}</div>', unsafe_allow_html=True)
                else:  # Osteoporosis
                    st.markdown(f'<div class="prediction-box osteoporosis">🔴 PREDICTION: {pred_label}</div>', unsafe_allow_html=True)

                # Probabilities
                st.subheader("📊 Class Probabilities")
                prob_df = pd.DataFrame({
                    'Condition': ['Normal', 'Osteopenia', 'Osteoporosis'],
                    'Probability': pred_proba,
                    'Percentage': [f"{p*100:.1f}%" for p in pred_proba]
                })
                
                # Create columns for better display
                col_norm, col_osteo, col_osteoporosis = st.columns(3)
                
                with col_norm:
                    st.metric("Normal", f"{pred_proba[0]*100:.1f}%", 
                             delta=None, delta_color="normal")
                
                with col_osteo:
                    st.metric("Osteopenia", f"{pred_proba[1]*100:.1f}%",
                             delta=None, delta_color="normal")
                
                with col_osteoporosis:
                    st.metric("Osteoporosis", f"{pred_proba[2]*100:.1f}%",
                             delta=None, delta_color="normal")
                
                # Bar chart of probabilities
                st.bar_chart(prob_df.set_index('Condition')['Probability'])
                
                # Clinical interpretation
                st.subheader("🩺 Clinical Interpretation")
                max_prob = max(pred_proba)
                if max_prob > 0.7:
                    confidence = "High"
                    confidence_color = "🟢"
                elif max_prob > 0.5:
                    confidence = "Moderate"
                    confidence_color = "🟡"
                else:
                    confidence = "Low"
                    confidence_color = "🔴"
                
                st.write(f"{confidence_color} **Model Confidence:** {confidence} ({max_prob*100:.1f}%)")
                
                if pred == 2:  # Osteoporosis
                    st.warning("⚠️ **High Risk:** Consider immediate clinical evaluation and bone density testing.")
                elif pred == 1:  # Osteopenia
                    st.info("ℹ️ **Moderate Risk:** Consider lifestyle modifications and regular monitoring.")
                else:  # Normal
                    st.success("✅ **Low Risk:** Continue preventive measures and routine screening.")
                
                st.info("⚠️ **Disclaimer:** This tool is for decision-support only. Always confirm with clinical judgment and appropriate diagnostic tests.")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.error("Please check your input values and try again.")

with tab3:
    st.subheader("📋 Model Information")
    st.write("**Model Type:** Ensemble Stacking with Optuna Optimization")
    
    if hasattr(model, 'trained_pipelines'):
        st.write(f"**Base Models Used:** {len(model.trained_pipelines)}")
        for i, model_name in enumerate(model.trained_pipelines.keys(), 1):
            st.write(f"  {i}. {model_name.replace('_', ' ').title()}")
    
    st.subheader("📊 Feature Importance")
    st.write("**Total Features:** 37")
    st.write("**Mandatory Fields:** 8")
    
    mandatory_display = [f.replace('_', ' ').title() for f in mandatory]
    st.write("**Required Fields:**")
    for field in mandatory_display:
        st.write(f"  • {field}")
    
    st.subheader("🎯 Model Performance")
    st.write("- **Cross-validation:** 5-fold stratified")
    st.write("- **Optimization:** Optuna TPE sampler")
    st.write("- **Classes:** Normal, Osteopenia, Osteoporosis")
    
    st.subheader("⚠️ Important Notes")
    st.write("- This model is trained on clinical data and should be used as a decision support tool")
    st.write("- Always validate predictions with clinical expertise")
    st.write("- Regular model updates recommended with new data")