#!/usr/bin/env python3
"""
Create a simple mock model that can be loaded by the Streamlit app.
This creates a basic model structure that works with joblib.
"""

import pickle
import numpy as np

class MockModel:
    """Simple mock model that mimics scikit-learn interface"""
    
    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 6
        # Simple weights for each feature
        self.feature_weights = np.array([0.3, 0.1, 0.2, 0.4, 0.15, 0.5])
        
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Simple linear combination with sigmoid
        scores = np.dot(X, self.feature_weights)
        # Apply sigmoid function
        prob_positive = 1 / (1 + np.exp(-scores))
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])

def create_mock_model():
    """Create and save a mock model"""
    print("Creating mock model...")
    
    model = MockModel()
    
    # Test the model
    test_features = np.array([[0.1, -0.5, 0.05, 0.2, 0.1, 0.3]])
    prediction = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    
    print(f"✓ Model test successful:")
    print(f"  - Prediction: {prediction[0]}")
    print(f"  - Probabilities: [{probabilities[0][0]:.3f}, {probabilities[0][1]:.3f}]")
    
    # Save using pickle (joblib uses pickle under the hood)
    with open('trained_model.joblib', 'wb') as f:
        pickle.dump(model, f)
    
    print("✓ Mock model saved as 'trained_model.joblib'")
    return True

if __name__ == "__main__":
    try:
        create_mock_model()
        print("\n🎉 Mock model creation completed successfully!")
        print("You can now run your Streamlit app.")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        exit(1)
