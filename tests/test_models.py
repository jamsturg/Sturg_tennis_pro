"""
Tests for model classes in the application.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Import the models we want to test
from app import BasicModel, EnhancedTennisModel

def test_basic_model_initialization():
    """Test BasicModel initialization and properties."""
    model = BasicModel()
    
    # Test class attributes
    assert hasattr(model, 'classes_')
    assert hasattr(model, 'n_features_in_')
    assert hasattr(model, 'feature_weights')
    
    # Test default values
    assert model.classes_ == [0, 1]
    assert model.n_features_in_ == 6
    assert len(model.feature_weights) == 6

def test_basic_model_predict():
    """Test BasicModel predict method."""
    model = BasicModel()
    
    # Test with single sample
    X = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    predictions = model.predict(X)
    
    # Check output format
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]
    
    # Test with multiple samples
    X_multi = [[0.5] * 6, [0.1] * 6]
    predictions_multi = model.predict(X_multi)
    assert len(predictions_multi) == 2

def test_basic_model_predict_proba():
    """Test BasicModel predict_proba method."""
    model = BasicModel()
    
    # Test with single sample
    X = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    proba = model.predict_proba(X)
    
    # Check output format
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (1, 2)  # One sample, two classes
    assert np.all(proba >= 0) and np.all(proba <= 1)  # Probabilities in [0,1]
    assert np.allclose(proba.sum(axis=1), 1.0)  # Sum to 1

def test_enhanced_model_initialization():
    """Test EnhancedTennisModel initialization and properties."""
    # Test default initialization
    model = EnhancedTennisModel()
    
    # Test class attributes
    assert hasattr(model, 'model_type')
    assert hasattr(model, 'scaler')
    assert hasattr(model, 'feature_names')
    assert hasattr(model, 'model')
    assert hasattr(model, 'classes_')
    assert hasattr(model, 'n_features_in_')
    
    # Test default values
    assert model.model_type == 'ensemble'
    assert len(model.feature_names) == 12  # Should have 12 features
    assert model.classes_ == [0, 1]
    assert model.n_features_in_ == len(model.feature_names)
    assert model.scaler is not None
    assert model.model is None  # Model should be None until trained
    
    # Test with custom model type
    custom_model = EnhancedTennisModel(model_type='logistic')
    assert custom_model.model_type == 'logistic'
    
    # Test that scaler is properly initialized
    assert hasattr(model.scaler, 'transform')

def test_enhanced_model_predict():
    """Test EnhancedTennisModel predict method."""
    # Create model with mock predict
    model = EnhancedTennisModel()
    
    # Mock the model's predict method
    model.model = MagicMock()
    model.model.predict.return_value = np.array([0, 1])
    
    # Mock the scaler's transform method
    model.scaler = MagicMock()
    model.scaler.transform.return_value = np.array([[0.1] * 12, [0.2] * 12])
    
    # Test prediction with numpy array
    X_np = np.random.rand(2, 12)  # 2 samples, 12 features
    predictions_np = model.predict(X_np)
    
    # Test prediction with list
    X_list = X_np.tolist()
    predictions_list = model.predict(X_list)
    
    # Verify calls and output
    assert model.scaler.transform.call_count == 2  # Called for both numpy and list inputs
    assert model.model.predict.call_count == 2  # Called for both numpy and list inputs
    assert isinstance(predictions_np, np.ndarray)
    assert isinstance(predictions_list, np.ndarray)
    assert len(predictions_np) == 2
    assert len(predictions_list) == 2
    assert all(p in [0, 1] for p in predictions_np)
    assert all(p in [0, 1] for p in predictions_list)

def test_enhanced_model_predict_proba():
    """Test EnhancedTennisModel predict_proba method."""
    # Create model with mock predict_proba
    model = EnhancedTennisModel()
    model.model = MagicMock()
    model.model.predict_proba.return_value = np.array([[0.4, 0.6], [0.7, 0.3]])
    
    # Mock the scaler's transform method
    model.scaler = MagicMock()
    model.scaler.transform.return_value = np.array([[0.1] * 12, [0.2] * 12])
    
    # Test prediction probabilities with numpy array
    X_np = np.random.rand(2, 12)  # 2 samples, 12 features
    proba_np = model.predict_proba(X_np)
    
    # Test prediction probabilities with list
    X_list = X_np.tolist()
    proba_list = model.predict_proba(X_list)
    
    # Verify calls and output
    assert model.scaler.transform.call_count == 2  # Called for both numpy and list inputs
    assert model.model.predict_proba.call_count == 2  # Called for both numpy and list inputs
    
    # Check numpy array output
    assert isinstance(proba_np, np.ndarray)
    assert proba_np.shape == (2, 2)  # Two samples, two classes
    assert np.all(proba_np >= 0) and np.all(proba_np <= 1)  # Probabilities in [0,1]
    assert np.allclose(proba_np.sum(axis=1), 1.0)  # Each row sums to 1
    
    # Check list output
    assert isinstance(proba_list, np.ndarray)
    assert proba_list.shape == (2, 2)  # Two samples, two classes
    assert np.all(proba_list >= 0) and np.all(proba_list <= 1)  # Probabilities in [0,1]
    assert np.allclose(proba_list.sum(axis=1), 1.0)  # Each row sums to 1

def test_model_integration(basic_model, enhanced_model):
    """Integration test for model interoperability."""
    # Generate some test data
    X = np.random.rand(5, 6)  # 5 samples, 6 features for basic model
    
    # Test basic model
    basic_predictions = basic_model.predict(X)
    basic_proba = basic_model.predict_proba(X)
    
    assert len(basic_predictions) == 5
    assert basic_proba.shape == (5, 2)
    
    # Test enhanced model with mocked methods
    enhanced_model.predict = MagicMock(return_value=np.array([0, 1, 0, 1, 1]))
    enhanced_model.predict_proba = MagicMock(return_value=np.ones((5, 2)) * 0.5)
    
    X_enhanced = np.random.rand(5, 12)  # 5 samples, 12 features for enhanced model
    enhanced_predictions = enhanced_model.predict(X_enhanced)
    enhanced_proba = enhanced_model.predict_proba(X_enhanced)
    
    enhanced_model.predict.assert_called_once()
    enhanced_model.predict_proba.assert_called_once()
    assert len(enhanced_predictions) == 5
    assert enhanced_proba.shape == (5, 2)