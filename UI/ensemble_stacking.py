import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import shap

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("final_dataset.csv")

"""Implementation"""

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
            # Use the class-level MultiLabelEncoder
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

    def _create_base_model_search_space(self, trial, model_name):
        """Define hyperparameter search spaces for each base model"""

        if model_name == "logistic_regression":
            return LogisticRegression(
                C=trial.suggest_float('lr_C', 1e-4, 1e2, log=True),
                penalty=trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
                solver='liblinear' if trial.params.get('lr_penalty') in ['l1', 'l2'] else 'lbfgs',
                max_iter=1000,
                random_state=self.random_state
            )

        elif model_name == "svm":
            return SVC(
                C=trial.suggest_float('svm_C', 1e-3, 1e3, log=True),
                kernel=trial.suggest_categorical('svm_kernel', ['rbf', 'poly', 'sigmoid']),
                gamma=trial.suggest_categorical('svm_gamma', ['scale', 'auto']) if trial.params.get('svm_kernel') != 'linear' else 'scale',
                probability=True,
                decision_function_shape='ovr',
                random_state=self.random_state
            )

        elif model_name == "polynomial":
            degree = trial.suggest_int('poly_degree', 2, 4)
            C = trial.suggest_float('poly_C', 1e-4, 1e2, log=True)
            return Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('logistic', LogisticRegression(
                    C=C,
                    max_iter=1000,
                    multi_class='ovr',
                    random_state=self.random_state
                ))
            ])

        elif model_name == "naive_bayes":
            return GaussianNB(
                var_smoothing=trial.suggest_float('nb_var_smoothing', 1e-11, 1e-5, log=True)
            )

        elif model_name == "knn":
            return KNeighborsClassifier(
                n_neighbors=trial.suggest_int('knn_n_neighbors', 3, 20),
                weights=trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
                metric=trial.suggest_categorical('knn_metric', ['euclidean', 'manhattan', 'minkowski'])
            )

        elif model_name == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=trial.suggest_int('dt_max_depth', 3, 20),
                min_samples_split=trial.suggest_int('dt_min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('dt_min_samples_leaf', 1, 20),
                max_features=trial.suggest_categorical('dt_max_features', ['sqrt', 'log2', None]),
                random_state=self.random_state
            )

        elif model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('rf_n_estimators', 50, 300),
                max_depth=trial.suggest_int('rf_max_depth', 3, 20),
                min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('rf_max_features', ['sqrt', 'log2']),
                random_state=self.random_state
            )

        elif model_name == "adaboost":
            return AdaBoostClassifier(
                n_estimators=trial.suggest_int('ada_n_estimators', 50, 200),
                learning_rate=trial.suggest_float('ada_learning_rate', 0.01, 2.0, log=True),
                algorithm='SAMME',
                random_state=self.random_state
            )

        elif model_name == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=trial.suggest_int('et_n_estimators', 50, 300),
                max_depth=trial.suggest_int('et_max_depth', 3, 20),
                min_samples_split=trial.suggest_int('et_min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('et_min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('et_max_features', ['sqrt', 'log2']),
                random_state=self.random_state
            )

        elif model_name == "xgboost":
            return XGBClassifier(
                n_estimators=trial.suggest_int('xgb_n_estimators', 50, 300),
                max_depth=trial.suggest_int('xgb_max_depth', 3, 10),
                learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('xgb_subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('xgb_reg_alpha', 1e-8, 1.0, log=True),
                reg_lambda=trial.suggest_float('xgb_reg_lambda', 1e-8, 1.0, log=True),
                objective='multi:softprob',
                num_class=self.n_classes,
                random_state=self.random_state,
                eval_metric='mlogloss',
                enable_categorical=True
            )

        elif model_name == "lightgbm":
            return LGBMClassifier(
                n_estimators=trial.suggest_int('lgb_n_estimators', 50, 300),
                max_depth=trial.suggest_int('lgb_max_depth', 3, 10),
                learning_rate=trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('lgb_subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('lgb_colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('lgb_reg_alpha', 1e-8, 1.0, log=True),
                reg_lambda=trial.suggest_float('lgb_reg_lambda', 1e-8, 1.0, log=True),
                objective='multiclass',
                num_class=self.n_classes,
                random_state=self.random_state,
                verbose=-1,
            )

        elif model_name == "catboost":
            return CatBoostClassifier(
                iterations=trial.suggest_int('cat_iterations', 50, 300),
                depth=trial.suggest_int('cat_depth', 3, 10),
                learning_rate=trial.suggest_float('cat_learning_rate', 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float('cat_l2_leaf_reg', 1e-8, 10.0, log=True),
                loss_function='MultiClass',
                cat_features=list(range(len(self.numerical_cols) + len(self.binary_cols),
                                      len(self.numerical_cols) + len(self.binary_cols) + len(self.categorical_cols))),
                random_state=self.random_state,
                verbose=False
            )

    def _create_meta_model_search_space(self, trial):
        """Define hyperparameter search space for meta-classifier"""

        meta_model_type = trial.suggest_categorical('meta_model', ['logistic', 'random_forest', 'xgboost'])

        if meta_model_type == 'logistic':
            return LogisticRegression(
                C=trial.suggest_float('meta_C', 1e-4, 1e2, log=True),
                penalty=trial.suggest_categorical('meta_penalty', ['l1', 'l2']),
                solver='liblinear',
                multi_class='ovr',
                random_state=self.random_state
            )
        elif meta_model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('meta_rf_n_estimators', 50, 200),
                max_depth=trial.suggest_int('meta_rf_max_depth', 3, 10),
                random_state=self.random_state
            )
        else:
            return XGBClassifier(
                n_estimators=trial.suggest_int('meta_xgb_n_estimators', 50, 200),
                max_depth=trial.suggest_int('meta_xgb_max_depth', 3, 6),
                learning_rate=trial.suggest_float('meta_xgb_learning_rate', 0.01, 0.3),
                objective='multi:softprob',
                num_class=self.n_classes,
                random_state=self.random_state,
                eval_metric='mlogloss'
            )

    def _stacking_cross_validation(self, X, y, base_models, meta_model):
        """Perform stacking with cross-validation for multi-class"""

        # Ensure y is numeric for roc_auc_score
        if hasattr(y, 'dtype') and y.dtype == 'object':
            temp_le = LabelEncoder()
            y_numeric = temp_le.fit_transform(y)
        elif isinstance(y, (list, tuple)) and isinstance(y[0], str):
            temp_le = LabelEncoder()
            y_numeric = temp_le.fit_transform(y)
        else:
            y_numeric = y

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        n_samples = len(X)
        n_models = len(base_models)

        # OOF predictions matrix
        oof_predictions = np.zeros((n_samples, n_models * self.n_classes))

        # Generate OOF predictions for each base model
        for model_idx, (model_name, model) in enumerate(base_models.items()):
            for train_idx, val_idx in skf.split(X, y_numeric):
                # Create pipeline for this model with model-specific preprocessing
                pipeline = self._create_pipeline_for_model(clone(model), model_name)

                # Split data
                X_fold_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                X_fold_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]

                # Use numeric y for consistency
                y_fold_train = y_numeric[train_idx] if hasattr(y_numeric, '__getitem__') else y_numeric.iloc[train_idx]
                y_fold_val = y_numeric[val_idx] if hasattr(y_numeric, '__getitem__') else y_numeric.iloc[val_idx]

                # Train and predict
                pipeline.fit(X_fold_train, y_fold_train)

                # Get all class probabilities
                val_probs = pipeline.predict_proba(X_fold_val)

                # Store all class probabilities for this model
                start_col = model_idx * self.n_classes
                end_col = start_col + self.n_classes
                oof_predictions[val_idx, start_col:end_col] = val_probs

            # Train final pipeline on full data
            final_pipeline = self._create_pipeline_for_model(clone(model), model_name)
            final_pipeline.fit(X, y_numeric)
            self.trained_pipelines[model_name] = final_pipeline

        # Train meta-model
        self.best_meta_model = clone(meta_model)
        print(f"Shape of OOF predictions for meta-model training: {oof_predictions.shape}")
        self.best_meta_model.fit(oof_predictions, y_numeric)

        self.best_base_models = base_models

        # Evaluate the stacking ensemble
        meta_oof_predictions = self.best_meta_model.predict_proba(oof_predictions)

        # Calculate multi-class ROC AUC using numeric labels
        score = roc_auc_score(y_numeric, meta_oof_predictions, multi_class='ovr', average='macro')

        return score

    def _objective(self, trial, X, y):
        """Optuna objective function"""

        model_names = ['logistic_regression', 'svm', 'polynomial', 'naive_bayes', 'knn',
                      'decision_tree', 'random_forest', 'adaboost', 'extra_trees',
                      'xgboost', 'lightgbm', 'catboost']

        base_models = {}
        selected_model_names = []

        # Suggest whether to include each model
        for model_name in model_names:
            include_model = trial.suggest_categorical(f'include_{model_name}', ['include', 'exclude'])
            if include_model == 'include':
                try:
                    model = self._create_base_model_search_space(trial, model_name)
                    base_models[model_name] = model
                    selected_model_names.append(model_name)
                except Exception as e:
                    # Skip problematic model configurations
                    continue

        if len(base_models) < 2:
            return 0.0  # Need at least 2 base models

        # Store selected model names in trial params for later retrieval
        trial.set_user_attr('selected_models', tuple(sorted(selected_model_names)))


        # Create meta-model
        meta_model = self._create_meta_model_search_space(trial)

        try:
            # Evaluate stacking ensemble and return the score
            score = self._stacking_cross_validation(X, y, base_models, meta_model)
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0  # Return poor score for failed configurations

    def optimize(self, X, y, n_trials=100, timeout=None):
        """Optimize stacking ensemble using Optuna"""

        print(f"Starting Optuna optimization with {n_trials} trials---")

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )

        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        print(f"Optimization completed!")
        print(f"Best score: {self.study.best_value:.4f}")
        print(f"Best parameters: {self.study.best_params}")

        return self.study.best_value, self.study.best_params

    def fit_best_model(self, X, y):
        """Fit the best model found by optimization"""
        self.trained_pipelines = {}

        if self.study is None:
            raise ValueError("Must run optimize() first!")

        # Ensure y is numeric
        if hasattr(y, 'dtype') and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            temp_le = LabelEncoder()
            y_numeric = temp_le.fit_transform(y)
        elif isinstance(y, (list, tuple)) and isinstance(y[0], str):
            from sklearn.preprocessing import LabelEncoder
            temp_le = LabelEncoder()
            y_numeric = temp_le.fit_transform(y)
        else:
            y_numeric = y


        best_params = self.study.best_params

        # Reconstruct best base models
        base_models = {}

        # Need a mock trial to reconstruct models with best params
        class MockTrial:
            def __init__(self, params):
                self.params = params

            def suggest_float(self, name, low, high, log=False):
                return self.params.get(name, (low + high) / 2)

            def suggest_int(self, name, low, high):
                return self.params.get(name, (low + high) // 2)

            def suggest_categorical(self, name, choices):
                return self.params.get(name, choices[0])


        mock_trial = MockTrial(best_params)
        model_names = ['logistic_regression', 'svm', 'polynomial', 'naive_bayes', 'knn',
                      'decision_tree', 'random_forest', 'adaboost', 'extra_trees',
                      'xgboost', 'lightgbm', 'catboost']


        for model_name in model_names:
            if best_params.get(f'include_{model_name}') == 'include':
                try:
                    model = self._create_base_model_search_space(mock_trial, model_name)
                    base_models[model_name] = model
                except Exception as e:
                    print(f"Warning: Could not reconstruct best base model {model_name}: {e}")
                    continue

        # Reconstruct best meta-model
        meta_model = self._create_meta_model_search_space(mock_trial)

        # Train final stacking model
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        n_samples = len(X)
        n_models = len(base_models)

        # Multi-class predictions
        oof_predictions = np.zeros((n_samples, n_models * self.n_classes))

        # Generate OOF predictions and train final models
        for model_idx, (model_name, model) in enumerate(base_models.items()):
            for train_idx, val_idx in skf.split(X, y_numeric):
                # Create pipeline with model-specific preprocessing
                pipeline = self._create_pipeline_for_model(clone(model), model_name)

                # Split and train
                X_fold_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                X_fold_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]

                # Use numeric y
                y_fold_train = y_numeric[train_idx] if hasattr(y_numeric, '__getitem__') else y_numeric.iloc[train_idx]
                y_fold_val = y_numeric[val_idx] if hasattr(y_numeric, '__getitem__') else y_numeric.iloc[val_idx]


                pipeline.fit(X_fold_train, y_fold_train)


                # Get all class probabilities
                val_probs = pipeline.predict_proba(X_fold_val)

                # Store all class probabilities
                start_col = model_idx * self.n_classes
                end_col = start_col + self.n_classes
                oof_predictions[val_idx, start_col:end_col] = val_probs

            # Train final pipeline on full data
            final_pipeline = self._create_pipeline_for_model(clone(model), model_name)
            final_pipeline.fit(X, y_numeric)
            self.trained_pipelines[model_name] = final_pipeline


        # Train meta-model
        self.best_meta_model = clone(meta_model)
        self.best_meta_model.fit(oof_predictions, y_numeric)

        self.best_base_models = base_models


        print(f"Best stacking model trained successfully!")
        return self


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


        print(f"Shape of base probabilities for meta-model prediction_proba: {base_predictions.shape}")
        final_probabilities = self.best_meta_model.predict_proba(base_predictions)
        return final_probabilities

class LIMEExplainer:
    """LIME explainer wrapper for the stacking ensemble model"""

    def __init__(self, stacking_model, X_train, class_names, feature_names):
        """
        Initialize LIME explainer

        Args:
            stacking_model: Trained StackingEnsembleOptuna model
            X_train: Training data (pandas DataFrame)
            class_names: List of class names (e.g., ['Normal', 'Osteopenia', 'Osteoporosis'])
            feature_names: List of feature names
        """
        self.stacking_model = stacking_model
        self.class_names = class_names
        self.original_feature_names = feature_names

        # Store original data
        self.X_train_original = X_train.copy()

        # Encode categorical features for LIME
        self.categorical_cols = ['sex', 'menopausal_status', 'physical_activity']
        self.encoders = {}

        # Create encoded version of training data
        self.X_train_encoded = X_train.copy()

        for col in self.categorical_cols:
            if col in self.X_train_encoded.columns:
                le = LabelEncoder()
                self.X_train_encoded[col] = le.fit_transform(self.X_train_encoded[col].astype(str))
                self.encoders[col] = le

        # Get indices of categorical features in encoded data
        categorical_features_idx = []
        for i, feat in enumerate(self.X_train_encoded.columns):
            if feat in self.categorical_cols:
                categorical_features_idx.append(i)

        # Create categorical feature names mapping for better explanations
        self.categorical_names = {}
        for idx, col in enumerate(self.X_train_encoded.columns):
            if col in self.categorical_cols:
                # Map encoded values to original categories
                cat_idx = categorical_features_idx.index(idx) if idx in categorical_features_idx else idx
                unique_vals = self.encoders[col].classes_
                self.categorical_names[idx] = unique_vals

        # Create LIME explainer with encoded data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train_encoded.values,
            feature_names=list(self.X_train_encoded.columns),
            class_names=class_names,
            categorical_features=categorical_features_idx,
            categorical_names=self.categorical_names,
            mode='classification',
            random_state=42
        )

    def encode_instance(self, instance):
        """Encode categorical features in an instance"""
        if isinstance(instance, pd.Series):
            instance_encoded = instance.copy()
        elif isinstance(instance, pd.DataFrame):
            instance_encoded = instance.copy()
        else:
            # If numpy array, convert to Series first
            instance_encoded = pd.Series(instance, index=self.original_feature_names)

        # Encode categorical columns
        for col in self.categorical_cols:
            if col in instance_encoded.index:
                instance_encoded[col] = self.encoders[col].transform([str(instance_encoded[col])])[0]

        return instance_encoded

    def predict_proba_wrapper(self, X_encoded):
        """Wrapper for predict_proba to work with LIME - handles encoding/decoding"""
        # X_encoded is numpy array from LIME
        # Convert back to DataFrame with original categorical values
        X_df = pd.DataFrame(X_encoded, columns=self.X_train_encoded.columns)

        # Decode categorical features back to original values
        X_original = X_df.copy()
        for col in self.categorical_cols:
            if col in X_original.columns:
                # Convert to int first (LIME may pass floats)
                encoded_vals = X_original[col].astype(int)
                # Decode back to original categories
                X_original[col] = self.encoders[col].inverse_transform(encoded_vals)

        # Now predict with the original stacking model
        return self.stacking_model.predict_proba(X_original)

    def explain_instance(self, instance, num_features=10, top_labels=3):
        """
        Explain a single prediction

        Args:
            instance: Single data instance (can be pandas Series or numpy array)
            num_features: Number of top features to show
            top_labels: Number of top predicted classes to explain

        Returns:
            LIME explanation object
        """
        # Encode the instance
        instance_encoded = self.encode_instance(instance)

        # Convert to numpy array
        if isinstance(instance_encoded, pd.Series):
            instance_array = instance_encoded.values
        elif isinstance(instance_encoded, pd.DataFrame):
            instance_array = instance_encoded.values[0]
        else:
            instance_array = instance_encoded

        # Get explanation
        exp = self.explainer.explain_instance(
            instance_array,
            self.predict_proba_wrapper,
            num_features=num_features,
            top_labels=top_labels
        )

        return exp

    def visualize_explanation(self, exp, label=None, figsize=(14, 6)):
        """
        Visualize LIME explanation

        Args:
            exp: LIME explanation object
            label: Class label to explain (None for predicted class)
            figsize: Figure size
        """
        if label is None:
            label = exp.available_labels()[0]

        # Get explanation data
        exp_list = exp.as_list(label=label)
        features = [item[0] for item in exp_list]
        weights = [item[1] for item in exp_list]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Feature importance (manual plotting)
        colors = ['green' if w > 0 else 'red' for w in weights]
        y_pos = np.arange(len(features))

        ax1.barh(y_pos, weights, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize=9)
        ax1.set_xlabel('Weight', fontsize=10)
        ax1.set_title(f'Feature Importance for {self.class_names[label]}', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for i, (feat, w) in enumerate(zip(features, weights)):
            ax1.text(w, i, f' {w:.3f}', va='center', fontsize=8)

        # Plot 2: Prediction probabilities
        probs = exp.predict_proba
        colors_prob = ['green', 'orange', 'red']
        bars = ax2.barh(self.class_names, probs, color=colors_prob, alpha=0.7)
        ax2.set_xlabel('Probability', fontsize=10)
        ax2.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3)

        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob, i, f' {prob:.1%}', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    def explain_multiple_instances(self, X_samples, indices=None, num_features=10):
        """
        Explain multiple instances

        Args:
            X_samples: DataFrame of instances to explain
            indices: List of indices (None for all)
            num_features: Number of features to show
        """
        if indices is None:
            indices = range(min(5, len(X_samples)))

        for idx in indices:
            print(f"\n{'='*80}")
            print(f"EXPLANATION FOR INSTANCE {idx}")
            print(f"{'='*80}\n")

            instance = X_samples.iloc[idx]
            exp = self.explain_instance(instance, num_features=num_features)

            # Get prediction - use original instance
            pred_proba = self.stacking_model.predict_proba(instance.to_frame().T)[0]
            pred_class = np.argmax(pred_proba)

            print(f"Predicted Class: {self.class_names[pred_class]}")
            print(f"Prediction Probabilities:")
            for i, class_name in enumerate(self.class_names):
                print(f"  {class_name}: {pred_proba[i]:.4f}")
            print("\n")

            # Show explanation as list
            print(f"Top {num_features} influential features:")
            print("-" * 60)
            exp_list = exp.as_list(label=pred_class)
            for feature, weight in exp_list:
                print(f"{feature:45s} | Weight: {weight:+.4f}")

            # Visualize
            self.visualize_explanation(exp, label=pred_class)

    def get_explanation_dataframe(self, instance, label=None):
        """
        Get explanation as a pandas DataFrame

        Args:
            instance: Single data instance
            label: Class label (None for predicted class)

        Returns:
            DataFrame with features and their weights
        """
        exp = self.explain_instance(instance)

        if label is None:
            label = exp.available_labels()[0]

        exp_list = exp.as_list(label=label)

        df = pd.DataFrame(exp_list, columns=['Feature', 'Weight'])
        df = df.sort_values('Weight', ascending=False, key=abs)

        return df

    def save_explanation_html(self, instance, filename='lime_explanation.html'):
        """
        Save explanation as interactive HTML

        Args:
            instance: Single data instance
            filename: Output HTML filename
        """
        exp = self.explain_instance(instance)
        exp.save_to_file(filename)
        print(f"Explanation saved to {filename}")

class LightweightSHAPExplainer:
    """Lightweight SHAP explainer for stacking ensemble - RAM friendly"""

    def __init__(self, stacking_model, X_train, X_test, class_names):
        """
        Initialize lightweight SHAP explainer

        Args:
            stacking_model: Trained StackingEnsembleOptuna model
            X_train: Training data (pandas DataFrame)
            X_test: Test data (pandas DataFrame)
            class_names: List of class names
        """
        self.stacking_model = stacking_model
        self.class_names = class_names
        self.X_train_original = X_train.copy()
        self.X_test_original = X_test.copy()

        # Encode categorical features
        self.categorical_cols = ['sex', 'menopausal_status', 'physical_activity']
        self.encoders = {}

        self.X_train_encoded = X_train.copy()
        self.X_test_encoded = X_test.copy()

        for col in self.categorical_cols:
            if col in self.X_train_encoded.columns:
                le = LabelEncoder()
                self.X_train_encoded[col] = le.fit_transform(self.X_train_encoded[col].astype(str))
                self.X_test_encoded[col] = le.transform(self.X_test_encoded[col].astype(str))
                self.encoders[col] = le

        self.feature_names = list(self.X_train_encoded.columns)

        # Model wrapper
        def model_predict(X_encoded):
            X_df = pd.DataFrame(X_encoded, columns=self.feature_names)
            X_original = X_df.copy()
            for col in self.categorical_cols:
                if col in X_original.columns:
                    encoded_vals = X_original[col].astype(int)
                    X_original[col] = self.encoders[col].inverse_transform(encoded_vals)
            return self.stacking_model.predict_proba(X_original)

        self.model_predict = model_predict

        # Use very small background sample - key for RAM efficiency
        background_size = min(20, len(X_train))  # Reduced from 100 to 20
        self.background_data = shap.sample(self.X_train_encoded, background_size, random_state=42)

        print(f"Initializing lightweight SHAP explainer...")
        print(f"Background samples: {background_size} (RAM friendly)")

        # Don't initialize explainer yet - do it on demand
        self.explainer = None
        self.shap_cache = {}  # Cache computed SHAP values

    def _get_explainer(self):
        """Lazy initialization of explainer"""
        if self.explainer is None:
            print("Creating SHAP explainer (first use only)...")
            self.explainer = shap.KernelExplainer(
                self.model_predict,
                self.background_data,
                link="identity"
            )
        return self.explainer

    def explain_instance(self, instance_idx=0, class_idx=None, use_cache=True):
        """
        Explain single instance - most RAM efficient

        Args:
            instance_idx: Index of instance
            class_idx: Class to explain (None = predicted)
            use_cache: Use cached values if available
        """
        # Check cache first
        cache_key = f"instance_{instance_idx}"
        if use_cache and cache_key in self.shap_cache:
            print("Using cached SHAP values...")
            shap_values = self.shap_cache[cache_key]
        else:
            # Encode single instance
            instance_original = self.X_test_original.iloc[instance_idx:instance_idx+1]
            instance_encoded = self.X_test_encoded.iloc[instance_idx:instance_idx+1]

            print(f"Computing SHAP values for instance {instance_idx}...")
            print("This may take 30-60 seconds...")

            explainer = self._get_explainer()
            shap_values = explainer.shap_values(instance_encoded.values)

            # Cache the result
            self.shap_cache[cache_key] = shap_values

        # Get prediction
        instance = self.X_test_original.iloc[instance_idx]
        pred_proba = self.stacking_model.predict_proba(instance.to_frame().T)[0]
        pred_class = np.argmax(pred_proba)

        if class_idx is None:
            class_idx = pred_class

        print(f"\n{'='*80}")
        print(f"SHAP EXPLANATION FOR INSTANCE {instance_idx}")
        print(f"{'='*80}\n")
        print(f"Predicted Class: {self.class_names[pred_class]}")
        print(f"Prediction Probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {pred_proba[i]:.4f}")
        print(f"\nExplaining: {self.class_names[class_idx]}")

        # Get top features
        # Handle SHAP values structure - can vary based on model output
        print(f"\nProcessing SHAP values...")

        if isinstance(shap_values, list):
            # List of arrays, one per class
            print(f"SHAP values structure: List with {len(shap_values)} classes")
            shap_array = shap_values[class_idx]
            shap_values_for_class = shap_array[0] if len(shap_array.shape) > 1 else shap_array
        else:
            # Single array - check dimensions
            print(f"SHAP values structure: Array with shape {shap_values.shape}")

            if len(shap_values.shape) == 3:
                # Shape: (samples, features, classes)
                print(f"Format: (samples={shap_values.shape[0]}, features={shap_values.shape[1]}, classes={shap_values.shape[2]})")
                shap_values_for_class = shap_values[0, :, class_idx]  # Get first sample, all features, specific class
            elif len(shap_values.shape) == 2:
                # Shape: (samples, features) - single class or need to index
                print(f"Format: (samples={shap_values.shape[0]}, features={shap_values.shape[1]})")
                shap_values_for_class = shap_values[0, :]  # Get first sample, all features
            else:
                # Shape: (features,) - already single sample
                shap_values_for_class = shap_values

        print(f"Extracted SHAP values shape: {shap_values_for_class.shape if hasattr(shap_values_for_class, 'shape') else len(shap_values_for_class)}")

        feature_impacts = list(zip(self.feature_names, shap_values_for_class))
        feature_impacts.sort(key=lambda x: abs(float(x[1])), reverse=True)

        print("\nTop 10 Feature Impacts:")
        print("-" * 60)
        for feat, impact in feature_impacts[:10]:
            print(f"{feat:30s} | SHAP value: {impact:+.4f}")

        # Simple bar plot
        top_n = 15
        top_features = feature_impacts[:top_n]
        features = [f[0] for f in top_features]
        values = [f[1] for f in top_features]

        plt.figure(figsize=(10, 6))
        colors = ['green' if v > 0 else 'red' for v in values]
        y_pos = np.arange(len(features))

        plt.barh(y_pos, values, color=colors, alpha=0.7)
        plt.yticks(y_pos, features, fontsize=9)
        plt.xlabel('SHAP Value (impact on prediction)', fontsize=10)
        plt.title(f'Feature Impact - {self.class_names[class_idx]} (Instance {instance_idx})',
                  fontsize=12, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)

        for i, v in enumerate(values):
            plt.text(v, i, f' {v:.3f}', va='center', fontsize=8)

        plt.tight_layout()
        plt.show()

        return shap_values

    def batch_explain(self, n_samples=5, random_samples=False):
        """
        Explain multiple instances efficiently

        Args:
            n_samples: Number of samples to explain
            random_samples: Random selection vs first n
        """
        if random_samples:
            indices = np.random.choice(len(self.X_test_encoded), n_samples, replace=False)
        else:
            indices = range(min(n_samples, len(self.X_test_encoded)))

        print(f"Computing SHAP values for {n_samples} instances...")
        print("Processing one at a time to conserve RAM...")

        for i, idx in enumerate(indices):
            print(f"\n[{i+1}/{n_samples}] Processing instance {idx}...")
            self.explain_instance(idx, use_cache=True)

    def approximate_feature_importance(self, n_samples=30):
        """
        Compute approximate global feature importance using sampling
        Much faster and RAM-friendly than full SHAP computation

        Args:
            n_samples: Number of samples to use
        """
        print(f"Computing approximate feature importance using {n_samples} samples...")

        # Sample instances
        sample_indices = np.random.choice(
            len(self.X_test_encoded),
            min(n_samples, len(self.X_test_encoded)),
            replace=False
        )

        # Compute SHAP values for sampled instances
        all_shap_values = {i: [] for i in range(len(self.class_names))}

        explainer = self._get_explainer()

        for idx in sample_indices:
            instance_encoded = self.X_test_encoded.iloc[idx:idx+1]
            shap_vals = explainer.shap_values(instance_encoded.values)

            # Handle shape (samples, features, classes)
            for class_idx in range(len(self.class_names)):
                if len(shap_vals.shape) == 3:
                    # Shape: (samples, features, classes) -> extract (features,) for this class
                    all_shap_values[class_idx].append(shap_vals[0, :, class_idx])
                elif isinstance(shap_vals, list):
                    # List format: [class][sample][features]
                    all_shap_values[class_idx].append(shap_vals[class_idx][0])
                else:
                    # Fallback for 2D: (samples, features)
                    all_shap_values[class_idx].append(shap_vals[0])

        # Plot for each class
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for class_idx, class_name in enumerate(self.class_names):
            # Calculate mean absolute SHAP
            shap_matrix = np.array(all_shap_values[class_idx])
            mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

            # Get top features
            feature_importance = list(zip(self.feature_names, mean_abs_shap))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            top_n = 12
            top_features = feature_importance[:top_n]
            features = [f[0] for f in top_features]
            importances = [f[1] for f in top_features]

            # Plot
            ax = axes[class_idx]
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, color='steelblue', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel('Mean |SHAP value|', fontsize=9)
            ax.set_title(class_name, fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            for i, v in enumerate(importances):
                ax.text(v, i, f' {v:.3f}', va='center', fontsize=7)

        plt.tight_layout()
        plt.show()

        # Return as DataFrames
        importance_dfs = {}
        for class_idx, class_name in enumerate(self.class_names):
            shap_matrix = np.array(all_shap_values[class_idx])
            mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

            df = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean_Abs_SHAP': mean_abs_shap
            }).sort_values('Mean_Abs_SHAP', ascending=False)
            df['Rank'] = range(1, len(df) + 1)
            importance_dfs[class_name] = df[['Rank', 'Feature', 'Mean_Abs_SHAP']]

        return importance_dfs

    def compare_two_instances(self, idx1=0, idx2=1, class_idx=None):
        """
        Compare SHAP explanations for two instances side-by-side

        Args:
            idx1, idx2: Instance indices
            class_idx: Class to explain (None = predicted for each)
        """
        print(f"Comparing instances {idx1} and {idx2}...")

        # Get SHAP values for both
        shap_vals_1 = self.explain_instance(idx1, class_idx, use_cache=True)
        print("\n")
        shap_vals_2 = self.explain_instance(idx2, class_idx, use_cache=True)

    def clear_cache(self):
        """Clear cached SHAP values to free RAM"""
        self.shap_cache = {}
        print("SHAP cache cleared!")

