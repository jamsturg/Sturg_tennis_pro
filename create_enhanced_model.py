#!/usr/bin/env python3
"""
Enhanced model creation script with advanced machine learning techniques for tennis prediction.
This creates multiple models with different algorithms and advanced feature engineering.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import json
import math
import pickle
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

class EnhancedTennisModel:
    """
    Advanced tennis prediction model with enhanced features and ensemble methods
    """
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_names = [
            'serve_strength_diff', 'ranking_diff', 'serve_percentage_diff',
            'recent_form_diff', 'rally_performance_diff', 'h2h_advantage',
            'surface_advantage', 'fatigue_index', 'pressure_handling',
            'injury_status', 'weather_impact', 'motivation_level'
        ]
        self.model = None
        self.feature_importance = None
        self.classes_ = [0, 1]
        self.n_features_in_ = len(self.feature_names)
        
    def create_base_models(self):
        """Create individual base models for ensemble"""
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        svm_model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            max_iter=500,
            alpha=0.01,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        return rf_model, gb_model, svm_model, nn_model
    
    def generate_enhanced_training_data(self, n_samples=5000):
        """Generate synthetic training data with advanced tennis features"""
        print(f"Generating {n_samples} samples with {len(self.feature_names)} enhanced features...")
        
        # Basic features (original 6)
        serve_strength_diff = np.random.normal(0, 0.3, n_samples)
        ranking_diff = np.random.normal(0, 2, n_samples)
        serve_percentage_diff = np.random.normal(0, 0.1, n_samples)
        recent_form_diff = np.random.normal(0, 0.4, n_samples)
        rally_performance_diff = np.random.normal(0, 0.2, n_samples)
        h2h_advantage = np.random.normal(0, 0.3, n_samples)
        
        # Enhanced features (new 6)
        surface_advantage = np.random.normal(0, 0.25, n_samples)  # Clay/Grass/Hard court advantage
        fatigue_index = np.random.normal(0, 0.2, n_samples)       # Player fatigue level
        pressure_handling = np.random.normal(0, 0.15, n_samples)  # Performance under pressure
        injury_status = np.random.normal(0, 0.1, n_samples)       # Injury impact
        weather_impact = np.random.normal(0, 0.12, n_samples)     # Weather conditions impact
        motivation_level = np.random.normal(0, 0.18, n_samples)   # Tournament importance
        
        # Combine all features
        X = np.column_stack([
            serve_strength_diff, ranking_diff, serve_percentage_diff,
            recent_form_diff, rally_performance_diff, h2h_advantage,
            surface_advantage, fatigue_index, pressure_handling,
            injury_status, weather_impact, motivation_level
        ])
        
        # Create realistic target variable with enhanced logic
        advantage_score = (
            serve_strength_diff * 0.25 +
            ranking_diff * 0.15 +
            recent_form_diff * 0.30 +
            h2h_advantage * 0.35 +
            surface_advantage * 0.20 +
            fatigue_index * -0.25 +  # Negative because fatigue hurts performance
            pressure_handling * 0.15 +
            injury_status * -0.30 +  # Negative impact
            weather_impact * 0.10 +
            motivation_level * 0.20 +
            np.random.normal(0, 0.08, n_samples)  # Reduced noise for better patterns
        )
        
        # Apply sigmoid-like transformation for more realistic probabilities
        probabilities = 1 / (1 + np.exp(-advantage_score * 2))
        y = (probabilities > 0.5).astype(int)
        
        return X, y, probabilities
    
    def train_model(self, X, y):
        """Train the enhanced model with cross-validation"""
        print(f"Training {self.model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.model_type == 'ensemble':
            # Create ensemble model
            rf_model, gb_model, svm_model, nn_model = self.create_base_models()
            
            self.model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('gb', gb_model),
                    ('svm', svm_model),
                    ('nn', nn_model)
                ],
                voting='soft'  # Use probability voting
            )
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            )
            
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=10,
                random_state=42
            )
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # For ensemble, use RandomForest importance
            rf_estimator = None
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    rf_estimator = estimator
                    break
            if rf_estimator:
                self.feature_importance = rf_estimator.feature_importances_
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X):
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Ensure we have all features
        if X.shape[1] < self.n_features_in_:
            # Pad with zeros for missing features
            missing_features = self.n_features_in_ - X.shape[1]
            X = np.pad(X, ((0, 0), (0, missing_features)), mode='constant')
        elif X.shape[1] > self.n_features_in_:
            # Trim extra features
            X = X[:, :self.n_features_in_]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Ensure we have all features
        if X.shape[1] < self.n_features_in_:
            # Pad with zeros for missing features
            missing_features = self.n_features_in_ - X.shape[1]
            X = np.pad(X, ((0, 0), (0, missing_features)), mode='constant')
        elif X.shape[1] > self.n_features_in_:
            # Trim extra features
            X = X[:, :self.n_features_in_]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


def create_enhanced_models():
    """Create multiple enhanced models and save them"""
    model_types = ['ensemble', 'random_forest', 'gradient_boost']
    model_performances = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Creating {model_type.upper()} model...")
        print(f"{'='*50}")
        
        # Create model instance
        model = EnhancedTennisModel(model_type=model_type)
        
        # Generate training data
        X, y, probabilities = model.generate_enhanced_training_data(n_samples=5000)
        
        # Train model
        performance = model.train_model(X, y)
        model_performances[model_type] = performance
        
        # Save model
        model_filename = f"enhanced_tennis_model_{model_type}.joblib"
        joblib.dump(model, model_filename)
        print(f"✓ Model saved as '{model_filename}'")
        
        # Test the model
        test_features = np.array([[0.1, -0.5, 0.05, 0.2, 0.1, 0.3, 0.15, -0.1, 0.08, 0.02, 0.05, 0.12]])
        prediction = model.predict(test_features)
        probabilities = model.predict_proba(test_features)
        
        print(f"✓ Model test successful:")
        print(f"  - Prediction: {prediction[0]}")
        print(f"  - Probabilities: [{probabilities[0][0]:.3f}, {probabilities[0][1]:.3f}]")
        
        if model.feature_importance is not None:
            print(f"  - Top 3 features: {', '.join([model.feature_names[i] for i in np.argsort(model.feature_importance)[-3:]])}")
    
    # Save performance comparison
    with open('model_performance_comparison.json', 'w') as f:
        json.dump(model_performances, f, indent=2)
    
    # Create backward-compatible model (for existing app)
    print(f"\n{'='*50}")
    print("Creating backward-compatible model...")
    print(f"{'='*50}")
    
    # Load the best performing model
    best_model_type = max(model_performances.keys(), 
                         key=lambda k: model_performances[k]['test_accuracy'])
    best_model = joblib.load(f"enhanced_tennis_model_{best_model_type}.joblib")
    
    # Save as the default model name
    joblib.dump(best_model, 'trained_model.joblib')
    print(f"✓ Best model ({best_model_type}) saved as 'trained_model.joblib'")
    
    return model_performances

if __name__ == "__main__":
    try:
        print("🎾 Enhanced Tennis Prediction Model Creator")
        print("=" * 50)
        
        performances = create_enhanced_models()
        
        print(f"\n🎉 Model creation completed successfully!")
        print(f"\nPerformance Summary:")
        for model_type, perf in performances.items():
            print(f"  {model_type.upper()}:")
            print(f"    - Test Accuracy: {perf['test_accuracy']:.4f}")
            print(f"    - CV Score: {perf['cv_mean']:.4f} ± {perf['cv_std']:.4f}")
        
        best_model = max(performances.keys(), 
                        key=lambda k: performances[k]['test_accuracy'])
        print(f"\n🏆 Best performing model: {best_model.upper()}")
        print("You can now run your enhanced Streamlit app!")
        
    except Exception as e:
        print(f"✗ Error creating enhanced models: {e}")
        exit(1)
