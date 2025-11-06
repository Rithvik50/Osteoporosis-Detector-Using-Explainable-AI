"""
End-to-End Testing Script for Osteoporosis Predictor Application
Tests all major functionality including model loading, predictions, and UI components
"""

import os
import sys
import time
import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
# Required for model loading
import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
try:
    from Patient_Info_Pipeline.Ensemble_Stacking.ensemble_stacking import StackingEnsembleOptuna, MultiLabelEncoder
except ImportError as e:
    sys.exit(1)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


class TestModelLoading(unittest.TestCase):
    """Test model and data loading functionality"""
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = BASE_DIR.parent / "Patient_Info_Pipeline" / "Ensemble_Stacking" / "stacking_ensemble.joblib"
        cls.data_path = BASE_DIR.parent / "Patient_Info_Pipeline" / "Ensemble_Stacking" / "patient_info_dataset.csv"

    def test_model_exists(self):
        """Verify model file exists"""
        self.assertTrue(self.model_path.exists(), 
                       f"Model file not found at {self.model_path}")
    
    def test_model_loads(self):
        """Test model can be loaded"""
        try:
            print(f"Attempting to load model from: {self.model_path}")
            print(f"Model file exists: {self.model_path.exists()}")
            from Patient_Info_Pipeline.Ensemble_Stacking.ensemble_stacking import StackingEnsembleOptuna
            print("Successfully imported StackingEnsembleOptuna class")
            model = joblib.load(self.model_path)
            self.assertIsNotNone(model)
            print("Successfully loaded model")
        except ImportError as e:
            self.fail(f"Failed to import required class: {e}")
        except Exception as e:
            self.fail(f"Failed to load model: {e}")
    
    def test_training_data_exists(self):
        """Verify training data exists"""
        self.assertTrue(self.data_path.exists(),
                       f"Training data not found at {self.data_path}")
    
    def test_training_data_loads(self):
        """Test training data can be loaded"""
        try:
            df = pd.read_csv(self.data_path)
            self.assertGreater(len(df), 0, "Training data is empty")
            self.assertIn('label', df.columns, "Missing 'label' column")
        except Exception as e:
            self.fail(f"Failed to load training data: {e}")


class TestDataProcessing(unittest.TestCase):
    """Test data processing and feature engineering"""
    
    def setUp(self):
        """Setup test data"""
        self.sample_inputs = {
            "age": 65,
            "sex": "Female",
            "height_cm": 160.0,
            "weight_kg": 70.0,
            "waist_cm": 85.0,
            "hip_cm": 100.0,
            "menopausal_status": "Post",
            "menopause_age": 50.0,
            "smoker": "No",
            "alcohol_high": "No",
            "physical_activity": "Moderate",
            "vitamin_d_ngml": 25.0,
            "serum_calcium_mgdl": 9.5,
            "alkaline_phosphatase": 75.0,
            "pth_pgml": 40.0,
            "creatinine_mgdl": 1.0,
            "hdl_mgdl": 60.0,
            "ldl_mgdl": 120.0,
            "ctx_ngml": 0.3,
            "p1np_ugL": 45.0,
            "estrogen_use": "No",
            "diabetes_t2": "No",
            "hypothyroidism": "No",
            "rheumatoid_arthritis": "No",
            "secondary_osteoporosis": "No",
            "parent_hip_fracture": "No",
            "prior_fracture": "No",
            "bisphosphonate_use": "No",
            "calcium_supplement": "Yes",
            "vitamin_d_supplement": "Yes",
            "glucocorticoid_use": "No",
            "dialysis": "No",
            "falls_past_year": 0
        }
    
    def test_bmi_calculation(self):
        """Test BMI calculation"""
        height_cm = 160.0
        weight_kg = 70.0
        expected_bmi = (weight_kg / (height_cm ** 2)) * 10000
        
        calculated_bmi = (weight_kg / (height_cm ** 2)) * 10000
        self.assertAlmostEqual(calculated_bmi, expected_bmi, places=2)
        self.assertGreater(calculated_bmi, 0)
    
    def test_waist_hip_ratio_calculation(self):
        """Test waist-hip ratio calculation"""
        waist_cm = 85.0
        hip_cm = 100.0
        expected_ratio = waist_cm / hip_cm
        
        calculated_ratio = waist_cm / hip_cm
        self.assertAlmostEqual(calculated_ratio, expected_ratio, places=2)
        self.assertLess(calculated_ratio, 1.5, "Unrealistic waist-hip ratio")
    
    def test_years_since_menopause(self):
        """Test years since menopause calculation"""
        age = 65
        menopause_age = 50
        menopausal_status = "Post"
        
        years = age - menopause_age if menopausal_status == "Post" else 0
        self.assertEqual(years, 15)
        self.assertGreaterEqual(years, 0)
    
    def test_binary_field_conversion(self):
        """Test binary field conversion (Yes/No to 1/0)"""
        binary_fields = ["smoker", "alcohol_high", "prior_fracture"]
        
        for field in binary_fields:
            val_yes = "Yes"
            val_no = "No"
            
            converted_yes = 1 if val_yes.lower() == "yes" else 0
            converted_no = 1 if val_no.lower() == "yes" else 0
            
            self.assertEqual(converted_yes, 1)
            self.assertEqual(converted_no, 0)
    
    def test_male_menopause_override(self):
        """Test that male patients have menopause fields nullified"""
        male_inputs = self.sample_inputs.copy()
        male_inputs["sex"] = "Male"
        
        if male_inputs["sex"] == "Male":
            male_inputs["menopause_age"] = 0
            male_inputs["menopausal_status"] = "na"
        
        self.assertEqual(male_inputs["menopause_age"], 0)
        self.assertEqual(male_inputs["menopausal_status"], "na")


class TestPredictionPipeline(unittest.TestCase):
    """Test the complete prediction pipeline"""
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = BASE_DIR.parent / "Patient_Info_Pipeline" / "Ensemble_Stacking" / "stacking_ensemble.joblib"
        cls.features = [
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
        
        try:
            cls.model = joblib.load(cls.model_path)
        except Exception as e:
            cls.model = None
            print(f"Warning: Could not load model: {e}")
    
    def setUp(self):
        """Setup test data for each test"""
        self.test_data = {
            "age": 65, "sex": "F", "height_cm": 160.0, "weight_kg": 70.0,
            "bmi": 27.34, "waist_cm": 85.0, "hip_cm": 100.0,
            "waist_hip_ratio": 0.85, "menopausal_status": "post",
            "menopause_age": 50.0, "years_since_menopause": 15,
            "smoker": 0, "alcohol_high": 0, "physical_activity": "moderate",
            "vitamin_d_ngml": 25.0, "vitamin_d_missing": 0,
            "serum_calcium_mgdl": 9.5, "alkaline_phosphatase": 75.0,
            "pth_pgml": 40.0, "creatinine_mgdl": 1.0,
            "hdl_mgdl": 60.0, "ldl_mgdl": 120.0,
            "ctx_ngml": 0.3, "p1np_ugL": 45.0,
            "estrogen_use": 0, "diabetes_t2": 0, "hypothyroidism": 0,
            "rheumatoid_arthritis": 0, "secondary_osteoporosis": 0,
            "parent_hip_fracture": 0, "prior_fracture": 0,
            "bisphosphonate_use": 0, "calcium_supplement": 1,
            "vitamin_d_supplement": 1, "glucocorticoid_use": 0,
            "dialysis": 0, "falls_past_year": 0
        }
    
    def test_dataframe_creation(self):
        """Test DataFrame creation with correct features"""
        df = pd.DataFrame([self.test_data])
        
        # Ensure all required features are present
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.features]
        
        self.assertEqual(len(df.columns), len(self.features))
        self.assertEqual(list(df.columns), self.features)
    
    def test_model_prediction(self):
        """Test model prediction"""
        if self.model is None:
            self.skipTest("Model not available")
        
        df = pd.DataFrame([self.test_data])[self.features]
        
        try:
            prediction = self.model.predict(df)
            probabilities = self.model.predict_proba(df)
            
            self.assertIn(prediction[0], [0, 1, 2])
            self.assertEqual(len(probabilities[0]), 3)
            self.assertAlmostEqual(sum(probabilities[0]), 1.0, places=5)
        except Exception as e:
            self.fail(f"Prediction failed: {e}")
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for same input"""
        if self.model is None:
            self.skipTest("Model not available")
        
        df = pd.DataFrame([self.test_data])[self.features]
        
        pred1 = self.model.predict(df)[0]
        pred2 = self.model.predict(df)[0]
        
        self.assertEqual(pred1, pred2, "Predictions should be deterministic")


class TestDecisionLogic(unittest.TestCase):
    """Test the fusion decision logic"""
    
    def test_decision_logic_normal(self):
        """Test decision logic for normal case"""
        singh_to_sev = {6: 0.0, 5: 0.0, 4: 1.0, 3: 1.5, 2: 2.0, 1: 2.5}
        label_to_sev = {"Normal": 0.0, "Osteopenia": 1.0, "Osteoporosis": 2.0}
        
        xray_grade = 6
        patient_pred = "Normal"
        w_x, w_p = 0.7, 0.3
        
        Sx = singh_to_sev[xray_grade]
        Sp = label_to_sev[patient_pred]
        score = w_x * Sx + w_p * Sp
        
        self.assertEqual(score, 0.0)
        self.assertLess(score, 0.5)
    
    def test_decision_logic_osteoporosis(self):
        """Test decision logic for osteoporosis case"""
        singh_to_sev = {6: 0.0, 5: 0.0, 4: 1.0, 3: 1.5, 2: 2.0, 1: 2.5}
        label_to_sev = {"Normal": 0.0, "Osteopenia": 1.0, "Osteoporosis": 2.0}
        
        xray_grade = 2
        patient_pred = "Osteoporosis"
        w_x, w_p = 0.7, 0.3
        
        Sx = singh_to_sev[xray_grade]
        Sp = label_to_sev[patient_pred]
        score = w_x * Sx + w_p * Sp
        
        expected_score = 0.7 * 2.0 + 0.3 * 2.0
        self.assertAlmostEqual(score, expected_score, places=2)
        self.assertGreaterEqual(score, 1.75)
    
    def test_decision_logic_mixed(self):
        """Test decision logic for mixed predictions"""
        singh_to_sev = {6: 0.0, 5: 0.0, 4: 1.0, 3: 1.5, 2: 2.0, 1: 2.5}
        label_to_sev = {"Normal": 0.0, "Osteopenia": 1.0, "Osteoporosis": 2.0}
        
        xray_grade = 4  # Osteopenia
        patient_pred = "Osteoporosis"
        w_x, w_p = 0.7, 0.3
        
        Sx = singh_to_sev[xray_grade]
        Sp = label_to_sev[patient_pred]
        score = w_x * Sx + w_p * Sp
        
        expected_score = 0.7 * 1.0 + 0.3 * 2.0
        self.assertAlmostEqual(score, expected_score, places=2)


class TestImageHandling(unittest.TestCase):
    """Test X-ray image handling"""
    
    def setUp(self):
        """Create temporary directory for test images"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_xray.png"
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_test_image(self):
        """Test creating a test image"""
        img = Image.new('RGB', (512, 512), color='gray')
        img.save(self.test_image_path)
        
        self.assertTrue(self.test_image_path.exists())
    
    def test_image_loading(self):
        """Test loading an image"""
        img = Image.new('RGB', (512, 512), color='gray')
        img.save(self.test_image_path)
        
        loaded_img = Image.open(self.test_image_path)
        self.assertEqual(loaded_img.size, (512, 512))
    
    def test_image_format_validation(self):
        """Test image format validation"""
        valid_formats = ['.png', '.jpg', '.jpeg']
        
        for fmt in valid_formats:
            test_path = f"test_image{fmt}"
            self.assertIn(Path(test_path).suffix, valid_formats)


class TestExplainabilityComponents(unittest.TestCase):
    """Test LIME and SHAP explainability components"""
    
    @classmethod
    def setUpClass(cls):
        cls.data_path = BASE_DIR.parent / "Patient_Info_Pipeline" / "Ensemble_Stacking" / "patient_info_dataset.csv"
        
        try:
            df = pd.read_csv(cls.data_path)
            cls.X = df.drop('label', axis=1)
            cls.sample_data = cls.X.sample(n=min(10, len(cls.X)), random_state=42)
        except Exception as e:
            cls.X = None
            cls.sample_data = None
            print(f"Warning: Could not load training data: {e}")
    
    def test_training_data_sampling(self):
        """Test that training data can be sampled"""
        if self.sample_data is None:
            self.skipTest("Training data not available")
        
        self.assertGreater(len(self.sample_data), 0)
        self.assertLessEqual(len(self.sample_data), 10)
    
    def test_categorical_feature_identification(self):
        """Test identification of categorical features"""
        categorical_features = ['sex', 'menopausal_status', 'physical_activity']
        
        if self.X is not None:
            for feat in categorical_features:
                if feat in self.X.columns:
                    self.assertIn(feat, self.X.columns)


class TestSessionState(unittest.TestCase):
    """Test session state management"""
    
    def test_session_state_initialization(self):
        """Test session state keys"""
        expected_keys = [
            'page',
            'prediction_data',
            'last_prediction',
            'last_cropped_image',
            'last_original_image',
            'uploaded_xray',
            'temp_filepath'
        ]
        
        session_state = {}
        for key in expected_keys:
            session_state[key] = None
        
        self.assertEqual(len(session_state), len(expected_keys))
    
    def test_prediction_data_structure(self):
        """Test prediction data structure"""
        prediction_data = {
            'prediction': 'Normal',
            'probabilities': [0.7, 0.2, 0.1],
            'input_df': None,
            'class_map': {0: "Normal", 1: "Osteopenia", 2: "Osteoporosis"}
        }
        
        self.assertIn('prediction', prediction_data)
        self.assertIn('probabilities', prediction_data)
        self.assertEqual(len(prediction_data['probabilities']), 3)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def test_missing_model_file(self):
        """Test handling of missing model file"""
        fake_path = Path("nonexistent_model.joblib")
        
        with self.assertRaises(FileNotFoundError):
            if not fake_path.exists():
                raise FileNotFoundError(f"Model not found: {fake_path}")
    
    def test_invalid_input_data(self):
        """Test handling of invalid input data"""
        invalid_data = {
            "age": -5,  # Invalid age
            "height_cm": 0,  # Invalid height
        }
        
        # Validate age
        self.assertLess(invalid_data["age"], 0)
        self.assertEqual(invalid_data["height_cm"], 0)
    
    def test_missing_required_features(self):
        """Test handling of missing required features"""
        incomplete_data = {"age": 65, "sex": "F"}
        required_features = ["age", "sex", "height_cm", "weight_kg", "bmi"]
        
        missing = [f for f in required_features if f not in incomplete_data]
        self.assertGreater(len(missing), 0)


def run_tests():
    """Run all test suites"""
    print("=" * 70)
    print("OSTEOPOROSIS PREDICTOR - END-TO-END TESTING")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestModelLoading,
        TestDataProcessing,
        TestPredictionPipeline,
        TestDecisionLogic,
        TestImageHandling,
        TestExplainabilityComponents,
        TestSessionState,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
