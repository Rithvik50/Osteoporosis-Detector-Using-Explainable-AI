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

import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(Path(__file__).resolve().parent.parent/"Ensemble_Stacking"/"patient_info_dataset.csv")

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
