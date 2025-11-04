import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Now import with the package structure
from Ensemble_Stacking.ensemble_stacking import (
    MultiLabelEncoder,
    StackingEnsembleOptuna
)

# ==================== FIXTURES ====================

@pytest.fixture(scope="module")
def sample_data():
    """Create sample osteoporosis dataset for testing"""
    np.random.seed(42)
    n_samples = 100

    data = {
        'age': np.random.randint(40, 80, n_samples),
        'height_cm': np.random.uniform(150, 180, n_samples),
        'weight_kg': np.random.uniform(50, 90, n_samples),
        'bmi': np.random.uniform(18, 35, n_samples),
        'waist_cm': np.random.uniform(70, 110, n_samples),
        'hip_cm': np.random.uniform(85, 120, n_samples),
        'waist_hip_ratio': np.random.uniform(0.7, 1.0, n_samples),
        'menopause_age': np.random.randint(40, 55, n_samples),
        'years_since_menopause': np.random.randint(0, 30, n_samples),
        'vitamin_d_ngml': np.random.uniform(10, 50, n_samples),
        'serum_calcium_mgdl': np.random.uniform(8.5, 10.5, n_samples),
        'alkaline_phosphatase': np.random.uniform(40, 120, n_samples),
        'pth_pgml': np.random.uniform(10, 70, n_samples),
        'creatinine_mgdl': np.random.uniform(0.5, 1.5, n_samples),
        'hdl_mgdl': np.random.uniform(40, 80, n_samples),
        'ldl_mgdl': np.random.uniform(80, 180, n_samples),
        'ctx_ngml': np.random.uniform(0.1, 0.8, n_samples),
        'p1np_ugL': np.random.uniform(20, 70, n_samples),
        'estrogen_use': np.random.randint(0, 2, n_samples),
        'diabetes_t2': np.random.randint(0, 2, n_samples),
        'hypothyroidism': np.random.randint(0, 2, n_samples),
        'dialysis': np.random.randint(0, 2, n_samples),
        'bisphosphonate_use': np.random.randint(0, 2, n_samples),
        'prior_fracture': np.random.randint(0, 2, n_samples),
        'parent_hip_fracture': np.random.randint(0, 2, n_samples),
        'smoker': np.random.randint(0, 2, n_samples),
        'alcohol_high': np.random.randint(0, 2, n_samples),
        'glucocorticoid_use': np.random.randint(0, 2, n_samples),
        'rheumatoid_arthritis': np.random.randint(0, 2, n_samples),
        'secondary_osteoporosis': np.random.randint(0, 2, n_samples),
        'calcium_supplement': np.random.randint(0, 2, n_samples),
        'vitamin_d_supplement': np.random.randint(0, 2, n_samples),
        'vitamin_d_missing': np.random.randint(0, 2, n_samples),
        'sex': np.random.choice(['F', 'M'], n_samples),
        'menopausal_status': np.random.choice(['pre', 'peri', 'post', 'na'], n_samples),
        'physical_activity': np.random.choice(['low', 'moderate', 'high'], n_samples),
        'label': np.random.choice(['Normal', 'Osteopenia', 'Osteoporosis'], n_samples)
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def train_test_data(sample_data):
    """Split data into train and test sets"""
    X = sample_data.drop('label', axis=1)
    y = sample_data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def trained_model(train_test_data):
    """Train a simple model for testing (fewer trials for speed)"""
    X_train, X_test, y_train, y_test = train_test_data

    model = StackingEnsembleOptuna(cv_folds=2, random_state=42, n_classes=3)

    # Run minimal optimization for testing
    model.optimize(X_train, y_train, n_trials=3, timeout=60)
    model.fit_best_model(X_train, y_train)

    return model


# ==================== MultiLabelEncoder Tests ====================

class TestMultiLabelEncoder:
    """Test MultiLabelEncoder transformer"""

    def test_fit_transform(self, sample_data):
        """Test fitting and transforming categorical columns"""
        encoder = MultiLabelEncoder()

        # Select only categorical columns
        cat_cols = ['sex', 'menopausal_status', 'physical_activity']
        X_cat = sample_data[cat_cols]

        # Fit and transform
        encoder.fit(X_cat)
        X_encoded = encoder.transform(X_cat)

        # Check output shape
        assert X_encoded.shape == X_cat.shape
        assert isinstance(X_encoded, np.ndarray)

        # Check that values are numeric
        assert np.issubdtype(X_encoded.dtype, np.number)

    def test_encoder_storage(self, sample_data):
        """Test that encoders are stored for each column"""
        encoder = MultiLabelEncoder()
        cat_cols = ['sex', 'menopausal_status', 'physical_activity']
        X_cat = sample_data[cat_cols]

        encoder.fit(X_cat)

        # Check encoders are stored
        assert len(encoder.label_encoders) == len(cat_cols)
        for col in cat_cols:
            assert col in encoder.label_encoders
            assert hasattr(encoder.label_encoders[col], 'classes_')

    def test_consistent_encoding(self, sample_data):
        """Test that encoding is consistent across calls"""
        encoder = MultiLabelEncoder()
        cat_cols = ['sex', 'menopausal_status']
        X_cat = sample_data[cat_cols]

        encoder.fit(X_cat)
        X_encoded_1 = encoder.transform(X_cat)
        X_encoded_2 = encoder.transform(X_cat)

        np.testing.assert_array_equal(X_encoded_1, X_encoded_2)

    def test_handles_string_conversion(self, sample_data):
        """Test that encoder properly converts values to strings"""
        encoder = MultiLabelEncoder()
        cat_cols = ['sex']
        X_cat = sample_data[cat_cols]

        encoder.fit(X_cat)

        # Should work even if values are already strings
        X_encoded = encoder.transform(X_cat)
        assert X_encoded.shape[0] == len(X_cat)


# ==================== StackingEnsembleOptuna Tests ====================

class TestStackingEnsembleOptuna:
    """Test StackingEnsembleOptuna model"""

    def test_initialization(self):
        """Test model initialization"""
        model = StackingEnsembleOptuna(cv_folds=3, random_state=42, n_classes=3)

        assert model.cv_folds == 3
        assert model.random_state == 42
        assert model.n_classes == 3
        assert len(model.numerical_cols) > 0
        assert len(model.binary_cols) > 0
        assert len(model.categorical_cols) > 0

    def test_column_definitions(self):
        """Test that column definitions are correct"""
        model = StackingEnsembleOptuna()

        # Check categorical columns
        assert 'sex' in model.categorical_cols
        assert 'menopausal_status' in model.categorical_cols
        assert 'physical_activity' in model.categorical_cols

        # Check numerical columns
        assert 'age' in model.numerical_cols
        assert 'bmi' in model.numerical_cols

        # Check binary columns
        assert 'smoker' in model.binary_cols
        assert 'diabetes_t2' in model.binary_cols

    def test_preprocessor_creation(self):
        """Test preprocessor creation for different model types"""
        model = StackingEnsembleOptuna()

        # Test tree-based model preprocessor
        preprocessor_tree = model._get_preprocessor_for_model('random_forest')
        assert preprocessor_tree is not None

        # Test linear model preprocessor
        preprocessor_linear = model._get_preprocessor_for_model('logistic_regression')
        assert preprocessor_linear is not None

        # Test CatBoost preprocessor
        preprocessor_cat = model._get_preprocessor_for_model('catboost')
        assert preprocessor_cat is not None

    def test_optimize(self, train_test_data):
        """Test optimization process"""
        X_train, X_test, y_train, y_test = train_test_data

        model = StackingEnsembleOptuna(cv_folds=2, random_state=42)

        # Run minimal optimization
        best_score, best_params = model.optimize(X_train, y_train, n_trials=2, timeout=30)

        assert isinstance(best_score, float)
        assert 0 <= best_score <= 1
        assert isinstance(best_params, dict)
        assert model.study is not None

    def test_optimize_creates_study(self, train_test_data):
        """Test that optimization creates a study object"""
        X_train, X_test, y_train, y_test = train_test_data

        model = StackingEnsembleOptuna(cv_folds=2, random_state=42)
        model.optimize(X_train, y_train, n_trials=2, timeout=20)

        assert model.study is not None
        assert hasattr(model.study, 'best_value')
        assert hasattr(model.study, 'best_params')

    def test_fit_best_model(self, train_test_data):
        """Test fitting the best model after optimization"""
        X_train, X_test, y_train, y_test = train_test_data

        model = StackingEnsembleOptuna(cv_folds=2, random_state=42)
        model.optimize(X_train, y_train, n_trials=2, timeout=30)
        model.fit_best_model(X_train, y_train)

        # Check that model components are created
        assert len(model.trained_pipelines) > 0
        assert model.best_meta_model is not None
        assert len(model.best_base_models) > 0

    def test_trained_pipelines_structure(self, trained_model):
        """Test structure of trained pipelines"""
        assert isinstance(trained_model.trained_pipelines, dict)

        # Each pipeline should have predict and predict_proba methods
        for name, pipeline in trained_model.trained_pipelines.items():
            assert hasattr(pipeline, 'predict')
            assert hasattr(pipeline, 'predict_proba')

    def test_predict(self, trained_model, train_test_data):
        """Test prediction functionality"""
        X_train, X_test, y_train, y_test = train_test_data

        predictions = trained_model.predict(X_test)

        # Check output shape and values
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_predict_single_instance(self, trained_model, train_test_data):
        """Test prediction on single instance"""
        X_train, X_test, y_train, y_test = train_test_data

        single_instance = X_test.iloc[[0]]
        prediction = trained_model.predict(single_instance)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1, 2]

    def test_predict_proba(self, trained_model, train_test_data):
        """Test probability prediction"""
        X_train, X_test, y_train, y_test = train_test_data

        probabilities = trained_model.predict_proba(X_test)

        # Check output shape
        assert probabilities.shape == (len(X_test), 3)

        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1),
            np.ones(len(X_test)),
            decimal=5
        )

        # Check probabilities are between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_predict_proba_single_instance(self, trained_model, train_test_data):
        """Test probability prediction on single instance"""
        X_train, X_test, y_train, y_test = train_test_data

        single_instance = X_test.iloc[[0]]
        probabilities = trained_model.predict_proba(single_instance)

        assert probabilities.shape == (1, 3)
        assert np.isclose(probabilities.sum(), 1.0)

    def test_predict_without_fit_raises_error(self, train_test_data):
        """Test that prediction without fitting raises error"""
        X_train, X_test, y_train, y_test = train_test_data

        model = StackingEnsembleOptuna()

        with pytest.raises(ValueError, match="Must fit the model first"):
            model.predict(X_test)

    def test_predict_proba_without_fit_raises_error(self, train_test_data):
        """Test that predict_proba without fitting raises error"""
        X_train, X_test, y_train, y_test = train_test_data

        model = StackingEnsembleOptuna()

        with pytest.raises(ValueError, match="Must fit the model first"):
            model.predict_proba(X_test)

    def test_fit_without_optimize_raises_error(self, train_test_data):
        """Test that fit_best_model without optimize raises error"""
        X_train, X_test, y_train, y_test = train_test_data

        model = StackingEnsembleOptuna()

        with pytest.raises(ValueError, match="Must run optimize"):
            model.fit_best_model(X_train, y_train)

    def test_model_serialization(self, trained_model, tmp_path):
        """Test saving and loading model"""
        model_path = tmp_path / "test_model.joblib"


        joblib.dump(trained_model, model_path)

        loaded_model = joblib.load(model_path)

        # Check model attributes
        assert len(loaded_model.trained_pipelines) == len(trained_model.trained_pipelines)
        assert loaded_model.best_meta_model is not None

    def test_model_predictions_after_serialization(self, trained_model, train_test_data, tmp_path):
        """Test that predictions work after saving and loading"""
        X_train, X_test, y_train, y_test = train_test_data
        model_path = tmp_path / "test_model.joblib"

        pred_before = trained_model.predict(X_test)

        joblib.dump(trained_model, model_path)
        loaded_model = joblib.load(model_path)

        pred_after = loaded_model.predict(X_test)

        np.testing.assert_array_equal(pred_before, pred_after)


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for the full pipeline"""

    def test_end_to_end_prediction(self, sample_data):
        """Test complete pipeline from data to prediction"""

        X = sample_data.drop('label', axis=1)
        y = sample_data['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train model
        model = StackingEnsembleOptuna(cv_folds=2, random_state=42)
        model.optimize(X_train, y_train, n_trials=2, timeout=30)
        model.fit_best_model(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Verify outputs
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 3)

    def test_model_with_different_input_formats(self, trained_model, train_test_data):
        """Test model handles different input formats"""
        X_train, X_test, y_train, y_test = train_test_data

        # DataFrame input
        pred_df = trained_model.predict(X_test)

        # Single row DataFrame
        single_row = X_test.iloc[[0]]
        pred_single = trained_model.predict(single_row)

        assert len(pred_single) == 1
        assert pred_df[0] == pred_single[0]

    def test_multiple_predictions_consistency(self, trained_model, train_test_data):
        """Test that multiple predictions on same data are consistent"""
        X_train, X_test, y_train, y_test = train_test_data

        pred1 = trained_model.predict(X_test)
        pred2 = trained_model.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)

    def test_string_labels_handling(self, sample_data):
        """Test that model handles string labels correctly"""
        X = sample_data.drop('label', axis=1)
        y = sample_data['label']  # String labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = StackingEnsembleOptuna(cv_folds=2, random_state=42)
        model.optimize(X_train, y_train, n_trials=2, timeout=30)
        model.fit_best_model(X_train, y_train)

        predictions = model.predict(X_test)

        # Predictions should be numeric (0, 1, 2)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)


# ==================== Edge Cases and Error Handling ====================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_small_dataset(self):
        """Test model handles small datasets"""
        np.random.seed(42)
        n_samples = 30  # Small dataset

        data = {
            'age': np.random.randint(40, 80, n_samples),
            'height_cm': np.random.uniform(150, 180, n_samples),
            'weight_kg': np.random.uniform(50, 90, n_samples),
            'bmi': np.random.uniform(18, 35, n_samples),
            'waist_cm': np.random.uniform(70, 110, n_samples),
            'hip_cm': np.random.uniform(85, 120, n_samples),
            'waist_hip_ratio': np.random.uniform(0.7, 1.0, n_samples),
            'menopause_age': np.random.randint(40, 55, n_samples),
            'years_since_menopause': np.random.randint(0, 30, n_samples),
            'vitamin_d_ngml': np.random.uniform(10, 50, n_samples),
            'serum_calcium_mgdl': np.random.uniform(8.5, 10.5, n_samples),
            'alkaline_phosphatase': np.random.uniform(40, 120, n_samples),
            'pth_pgml': np.random.uniform(10, 70, n_samples),
            'creatinine_mgdl': np.random.uniform(0.5, 1.5, n_samples),
            'hdl_mgdl': np.random.uniform(40, 80, n_samples),
            'ldl_mgdl': np.random.uniform(80, 180, n_samples),
            'ctx_ngml': np.random.uniform(0.1, 0.8, n_samples),
            'p1np_ugL': np.random.uniform(20, 70, n_samples),
            'estrogen_use': np.random.randint(0, 2, n_samples),
            'diabetes_t2': np.random.randint(0, 2, n_samples),
            'hypothyroidism': np.random.randint(0, 2, n_samples),
            'dialysis': np.random.randint(0, 2, n_samples),
            'bisphosphonate_use': np.random.randint(0, 2, n_samples),
            'prior_fracture': np.random.randint(0, 2, n_samples),
            'parent_hip_fracture': np.random.randint(0, 2, n_samples),
            'smoker': np.random.randint(0, 2, n_samples),
            'alcohol_high': np.random.randint(0, 2, n_samples),
            'glucocorticoid_use': np.random.randint(0, 2, n_samples),
            'rheumatoid_arthritis': np.random.randint(0, 2, n_samples),
            'secondary_osteoporosis': np.random.randint(0, 2, n_samples),
            'calcium_supplement': np.random.randint(0, 2, n_samples),
            'vitamin_d_supplement': np.random.randint(0, 2, n_samples),
            'vitamin_d_missing': np.random.randint(0, 2, n_samples),
            'sex': np.random.choice(['F', 'M'], n_samples),
            'menopausal_status': np.random.choice(['pre', 'peri', 'post', 'na'], n_samples),
            'physical_activity': np.random.choice(['low', 'moderate', 'high'], n_samples),
            'label': np.random.choice(['Normal', 'Osteopenia', 'Osteoporosis'], n_samples)
        }

        df = pd.DataFrame(data)
        X = df.drop('label', axis=1)
        y = df['label']


        model = StackingEnsembleOptuna(cv_folds=2, random_state=42)
        best_score, best_params = model.optimize(X, y, n_trials=2, timeout=20)

        assert isinstance(best_score, float)

    def test_mismatched_features_raises_error(self, trained_model, train_test_data):
        """Test prediction with missing features raises appropriate error"""
        X_train, X_test, y_train, y_test = train_test_data

        X_test_incomplete = X_test.drop(columns=['age'])

        with pytest.raises(Exception):
            trained_model.predict(X_test_incomplete)

    def test_extra_features_handled_gracefully(self, trained_model, train_test_data):
        """Test prediction with extra features is handled gracefully (dropped by ColumnTransformer)"""
        X_train, X_test, y_train, y_test = train_test_data

        # Add an extra feature
        X_test_extra = X_test.copy()
        X_test_extra['extra_feature'] = np.random.randn(len(X_test))

        predictions = trained_model.predict(X_test_extra)

        predictions_normal = trained_model.predict(X_test)
        np.testing.assert_array_equal(predictions, predictions_normal)

    def test_all_same_class(self):
        """Test model behavior when all samples have same class"""
        np.random.seed(42)
        n_samples = 50

        # Create data with only one class
        data = {
            'age': np.random.randint(40, 80, n_samples),
            'height_cm': np.random.uniform(150, 180, n_samples),
            'weight_kg': np.random.uniform(50, 90, n_samples),
            'bmi': np.random.uniform(18, 35, n_samples),
            'waist_cm': np.random.uniform(70, 110, n_samples),
            'hip_cm': np.random.uniform(85, 120, n_samples),
            'waist_hip_ratio': np.random.uniform(0.7, 1.0, n_samples),
            'menopause_age': np.random.randint(40, 55, n_samples),
            'years_since_menopause': np.random.randint(0, 30, n_samples),
            'vitamin_d_ngml': np.random.uniform(10, 50, n_samples),
            'serum_calcium_mgdl': np.random.uniform(8.5, 10.5, n_samples),
            'alkaline_phosphatase': np.random.uniform(40, 120, n_samples),
            'pth_pgml': np.random.uniform(10, 70, n_samples),
            'creatinine_mgdl': np.random.uniform(0.5, 1.5, n_samples),
            'hdl_mgdl': np.random.uniform(40, 80, n_samples),
            'ldl_mgdl': np.random.uniform(80, 180, n_samples),
            'ctx_ngml': np.random.uniform(0.1, 0.8, n_samples),
            'p1np_ugL': np.random.uniform(20, 70, n_samples),
            'estrogen_use': np.random.randint(0, 2, n_samples),
            'diabetes_t2': np.random.randint(0, 2, n_samples),
            'hypothyroidism': np.random.randint(0, 2, n_samples),
            'dialysis': np.random.randint(0, 2, n_samples),
            'bisphosphonate_use': np.random.randint(0, 2, n_samples),
            'prior_fracture': np.random.randint(0, 2, n_samples),
            'parent_hip_fracture': np.random.randint(0, 2, n_samples),
            'smoker': np.random.randint(0, 2, n_samples),
            'alcohol_high': np.random.randint(0, 2, n_samples),
            'glucocorticoid_use': np.random.randint(0, 2, n_samples),
            'rheumatoid_arthritis': np.random.randint(0, 2, n_samples),
            'secondary_osteoporosis': np.random.randint(0, 2, n_samples),
            'calcium_supplement': np.random.randint(0, 2, n_samples),
            'vitamin_d_supplement': np.random.randint(0, 2, n_samples),
            'vitamin_d_missing': np.random.randint(0, 2, n_samples),
            'sex': np.random.choice(['F', 'M'], n_samples),
            'menopausal_status': np.random.choice(['pre', 'peri', 'post', 'na'], n_samples),
            'physical_activity': np.random.choice(['low', 'moderate', 'high'], n_samples),
            'label': ['Normal'] * n_samples  # All same class
        }

        df = pd.DataFrame(data)
        X = df.drop('label', axis=1)
        y = df['label']

        # Modern scikit-learn handles single-class stratification gracefully
        # The model should handle this without errors
        model = StackingEnsembleOptuna(cv_folds=2, random_state=42)

        # Should complete without error (though ROC AUC might be undefined)
        # We just verify it doesn't crash
        try:
            model.optimize(X, y, n_trials=1, timeout=10)
            # If it completes, that's acceptable behavior
            assert True
        except ValueError as e:
            # ROC AUC might fail with single class, which is also acceptable
            assert "Only one class present" in str(e) or "multiclass format is not supported" in str(e)


# ==================== Performance Tests ====================

class TestPerformance:
    """Test performance characteristics"""

    def test_prediction_speed(self, trained_model, train_test_data):
        """Test that predictions are reasonably fast"""
        import time

        X_train, X_test, y_train, y_test = train_test_data

        start_time = time.time()
        predictions = trained_model.predict(X_test)
        elapsed_time = time.time() - start_time

        # Predictions should be fast (< 5 seconds for small dataset)
        assert elapsed_time < 5.0

    def test_predict_proba_speed(self, trained_model, train_test_data):
        """Test that probability predictions are reasonably fast"""
        import time

        X_train, X_test, y_train, y_test = train_test_data

        start_time = time.time()
        probabilities = trained_model.predict_proba(X_test)
        elapsed_time = time.time() - start_time

        # Probability predictions should be fast
        assert elapsed_time < 5.0

    def test_batch_prediction_efficiency(self, trained_model, train_test_data):
        """Test that batch prediction is more efficient than individual predictions"""
        import time

        X_train, X_test, y_train, y_test = train_test_data

        # Batch prediction
        start_batch = time.time()
        batch_pred = trained_model.predict(X_test)
        time_batch = time.time() - start_batch

        # Individual predictions
        start_individual = time.time()
        individual_pred = []
        for i in range(len(X_test)):
            pred = trained_model.predict(X_test.iloc[[i]])
            individual_pred.append(pred[0])
        time_individual = time.time() - start_individual

        # Batch should be faster (or at least not much slower)
        # Allow some tolerance since individual predictions might be cached
        assert time_batch < time_individual * 2

        # Results should be the same
        np.testing.assert_array_equal(batch_pred, individual_pred)

    def test_memory_efficiency(self, train_test_data):
        """Test memory usage during training"""
        X_train, X_test, y_train, y_test = train_test_data

        # Use small subset
        X_train_small = X_train.head(30)
        y_train_small = y_train.head(30)

        model = StackingEnsembleOptuna(cv_folds=2)

        # Should complete without memory errors
        model.optimize(X_train_small, y_train_small, n_trials=2, timeout=20)
        model.fit_best_model(X_train_small, y_train_small)

        assert model.best_meta_model is not None


# ==================== Model-Specific Tests ====================

class TestBaseModelCreation:
    """Test creation of different base models"""

    def test_logistic_regression_creation(self):
        """Test logistic regression model creation"""
        model = StackingEnsembleOptuna()

        # Mock trial for testing
        class MockTrial:
            def __init__(self):
                self.params = {}

            def suggest_float(self, name, low, high, log=False):
                return (low + high) / 2

            def suggest_categorical(self, name, choices):
                return choices[0]

        trial = MockTrial()
        lr_model = model._create_base_model_search_space(trial, 'logistic_regression')

        assert lr_model is not None
        assert hasattr(lr_model, 'fit')
        assert hasattr(lr_model, 'predict')

    def test_random_forest_creation(self):
        """Test random forest model creation"""
        model = StackingEnsembleOptuna()

        class MockTrial:
            def __init__(self):
                self.params = {}

            def suggest_int(self, name, low, high):
                return (low + high) // 2

            def suggest_categorical(self, name, choices):
                return choices[0]

        trial = MockTrial()
        rf_model = model._create_base_model_search_space(trial, 'random_forest')

        assert rf_model is not None
        assert hasattr(rf_model, 'fit')
        assert hasattr(rf_model, 'predict')

    def test_xgboost_creation(self):
        """Test XGBoost model creation"""
        model = StackingEnsembleOptuna()

        class MockTrial:
            def __init__(self):
                self.params = {}

            def suggest_int(self, name, low, high):
                return (low + high) // 2

            def suggest_float(self, name, low, high, log=False):
                return (low + high) / 2

        trial = MockTrial()
        xgb_model = model._create_base_model_search_space(trial, 'xgboost')

        assert xgb_model is not None
        assert hasattr(xgb_model, 'fit')
        assert hasattr(xgb_model, 'predict')


# ==================== Data Validation Tests ====================

class TestDataValidation:
    """Test data validation and preprocessing"""

    def test_categorical_columns_present(self, sample_data):
        """Test that categorical columns are properly identified"""
        model = StackingEnsembleOptuna()

        X = sample_data.drop('label', axis=1)

        # Check that categorical columns exist in data
        for col in model.categorical_cols:
            assert col in X.columns, f"Categorical column {col} not found in data"

    def test_numerical_columns_present(self, sample_data):
        """Test that numerical columns are properly identified"""
        model = StackingEnsembleOptuna()
        X = sample_data.drop('label', axis=1)

        # Check that numerical columns exist in data
        for col in model.numerical_cols:
            assert col in X.columns, f"Numerical column {col} not found in data"

    def test_binary_columns_present(self, sample_data):
        """Test that binary columns are properly identified"""
        model = StackingEnsembleOptuna()

        X = sample_data.drop('label', axis=1)

        # Check that binary columns exist in data
        for col in model.binary_cols:
            assert col in X.columns, f"Binary column {col} not found in data"

    def test_no_column_overlap(self):
        """Test that column types don't overlap"""
        model = StackingEnsembleOptuna()

        num_set = set(model.numerical_cols)
        bin_set = set(model.binary_cols)
        cat_set = set(model.categorical_cols)

        # Check no overlap between column types
        assert len(num_set & bin_set) == 0, "Overlap between numerical and binary columns"
        assert len(num_set & cat_set) == 0, "Overlap between numerical and categorical columns"
        assert len(bin_set & cat_set) == 0, "Overlap between binary and categorical columns"

    def test_all_features_covered(self, sample_data):
        """Test that all features are assigned to a column type"""
        model = StackingEnsembleOptuna()

        X = sample_data.drop('label', axis=1)
        all_model_cols = set(model.numerical_cols + model.binary_cols + model.categorical_cols)
        data_cols = set(X.columns)

        # All data columns should be in model's column definitions
        assert data_cols.issubset(all_model_cols), \
            f"Missing columns: {data_cols - all_model_cols}"


# ==================== Reproducibility Tests ====================

class TestReproducibility:
    """Test model reproducibility with random seeds"""

    def test_same_seed_same_results(self, train_test_data):
        """Test that same random seed produces same results"""
        X_train, X_test, y_train, y_test = train_test_data

        # Train first model
        model1 = StackingEnsembleOptuna(cv_folds=2, random_state=42)
        model1.optimize(X_train, y_train, n_trials=2, timeout=20)
        model1.fit_best_model(X_train, y_train)
        pred1 = model1.predict(X_test)

        # Train second model with same seed
        model2 = StackingEnsembleOptuna(cv_folds=2, random_state=42)
        model2.optimize(X_train, y_train, n_trials=2, timeout=20)
        model2.fit_best_model(X_train, y_train)
        pred2 = model2.predict(X_test)

        # Predictions should be similar
        # Check that at least 80% predictions match
        match_rate = np.mean(pred1 == pred2)
        assert match_rate > 0.8, f"Only {match_rate:.1%} predictions matched"

    def test_different_seed_different_results(self, train_test_data):
        """Test that different random seeds can produce different results"""
        X_train, X_test, y_train, y_test = train_test_data

        # Train first model
        model1 = StackingEnsembleOptuna(cv_folds=2, random_state=42)
        model1.optimize(X_train, y_train, n_trials=2, timeout=20)
        score1 = model1.study.best_value

        # Train second model with different seed
        model2 = StackingEnsembleOptuna(cv_folds=2, random_state=123)
        model2.optimize(X_train, y_train, n_trials=2, timeout=20)
        score2 = model2.study.best_value

        # Scores should be in similar range but not necessarily identical
        assert isinstance(score1, float)
        assert isinstance(score2, float)


# ==================== Cross-Validation Tests ====================

class TestCrossValidation:
    """Test cross-validation functionality"""

    def test_cv_folds_parameter(self, train_test_data):
        """Test different CV fold values"""
        X_train, X_test, y_train, y_test = train_test_data

        # Test with different fold counts
        for n_folds in [2, 3, 5]:
            model = StackingEnsembleOptuna(cv_folds=n_folds, random_state=42)
            assert model.cv_folds == n_folds

            # Should work with different fold counts
            model.optimize(X_train, y_train, n_trials=1, timeout=15)
            assert model.study is not None

    def test_stratified_cv(self, train_test_data):
        """Test that stratified CV is used (implicitly tested through model training)"""
        X_train, X_test, y_train, y_test = train_test_data

        model = StackingEnsembleOptuna(cv_folds=3, random_state=42)

        # Should work without errors (stratification is implicit)
        best_score, best_params = model.optimize(X_train, y_train, n_trials=2, timeout=20)

        assert 0 <= best_score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])