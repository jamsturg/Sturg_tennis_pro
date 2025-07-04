"""
Advanced model training for tennis prediction.
Implements multiple model architectures and ensemble methods.
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TennisModelTrainer:
    """Handles training and optimization of tennis prediction models."""
    
    def __init__(self, models_dir: str = 'models', n_trials: int = 50):
        """Initialize the model trainer.
        
        Args:
            models_dir: Directory to save trained models
            n_trials: Number of optimization trials for hyperparameter tuning
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.n_trials = n_trials
        self.best_params = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the training data."""
        if X.empty:
            return np.array([]), np.array([])
            
        X = X.copy()
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        if fit:
            X_processed = preprocessor.fit_transform(X)
            # Save the preprocessor
            joblib.dump(preprocessor, self.models_dir / 'preprocessor.joblib')
        else:
            preprocessor = joblib.load(self.models_dir / 'preprocessor.joblib')
            X_processed = preprocessor.transform(X)
        
        # Convert y to numpy array if provided
        y_processed = y.values if y is not None else np.array([])
        
        return X_processed, y_processed
    
    def create_base_models(self) -> Dict[str, BaseEstimator]:
        """Create base models for the ensemble."""
        return {
            'xgb': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective='binary:logistic',
                eval_metric='logloss'
            ),
            'lgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective='binary',
                metric='binary_logloss'
            ),
            'catboost': CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=0,
                loss_function='Logloss',
                thread_count=-1
            )
        }
    
    def create_meta_model(self) -> BaseEstimator:
        """Create meta-model for stacking."""
        return LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    
    def create_ensemble(self) -> VotingClassifier:
        """Create an ensemble of models."""
        base_models = self.create_base_models()
        
        # Create voting classifier
        voting = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft',
            n_jobs=-1
        )
        
        return voting
    
    def create_stacked_model(self) -> StackingClassifier:
        """Create a stacked ensemble model."""
        base_models = list(self.create_base_models().items())
        meta_model = self.create_meta_model()
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            stack_method='predict_proba',
            n_jobs=-1
        )
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'xgb'
    ) -> Dict:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            if model_type == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                }
                model = XGBClassifier(**params, random_state=42, n_jobs=-1)
            
            elif model_type == 'lgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = LGBMClassifier(**params, random_state=42, n_jobs=-1)
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(
                model, X, y, cv=tscv,
                scoring='neg_log_loss',
                n_jobs=-1
            )
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'ensemble',
        optimize: bool = False
    ) -> BaseEstimator:
        """Train a prediction model.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train ('xgb', 'lgbm', 'catboost', 'ensemble', 'stacked')
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        
        if X_processed.size == 0:
            raise ValueError("No valid training data provided")
        
        # Train model
        if model_type in ['xgb', 'lgbm', 'catboost']:
            if optimize:
                best_params = self.optimize_hyperparameters(X_processed, y_processed, model_type)
                logger.info(f"Best parameters for {model_type}: {best_params}")
                
                if model_type == 'xgb':
                    model = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
                elif model_type == 'lgbm':
                    model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
                else:  # catboost
                    model = CatBoostClassifier(
                        **best_params,
                        random_seed=42,
                        verbose=0,
                        thread_count=-1
                    )
            else:
                model = self.create_base_models()[model_type]
                
        elif model_type == 'ensemble':
            model = self.create_ensemble()
        elif model_type == 'stacked':
            model = self.create_stacked_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        model.fit(X_processed, y_processed)
        
        # Save the model
        model_path = self.models_dir / f'{model_type}_model.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model
    
    def evaluate_model(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        X_processed, _ = self.preprocess_data(X_test, fit=False)
        
        if X_processed.size == 0:
            return {}
        
        # Make predictions
        y_pred = model.predict(X_processed)
        y_pred_proba = model.predict_proba(X_processed)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info("\nModel Evaluation:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics

# Example usage
if __name__ == "__main__":
    from data_pipeline import TennisDataPipeline
    
    # Initialize data pipeline
    pipeline = TennisDataPipeline()
    
    # Get training data
    X, y = pipeline.get_training_data(days=90)
    
    if not X.empty and not y.empty:
        # Split data into train and test sets
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Initialize model trainer
        trainer = TennisModelTrainer(models_dir='models', n_trials=20)
        
        # Train and evaluate different models
        model_types = ['xgb', 'lgbm', 'ensemble', 'stacked']
        
        for model_type in model_types:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training {model_type.upper()} model")
                logger.info(f"{'='*50}")
                
                # Train model
                model = trainer.train_model(
                    X_train, y_train,
                    model_type=model_type,
                    optimize=True
                )
                
                # Evaluate model
                metrics = trainer.evaluate_model(model, X_test, y_test)
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
