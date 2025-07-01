#!/usr/bin/env python3
"""
Create a very basic model file that can be loaded by the Streamlit app.
Uses only built-in Python libraries.
"""

import pickle
import math

class BasicModel:
    """Very simple model that mimics scikit-learn interface"""
    
    def __init__(self):
        self.classes_ = [0, 1]
        self.n_features_in_ = 6
        # Simple weights for each feature
        self.feature_weights = [0.3, 0.1, 0.2, 0.4, 0.15, 0.5]
        
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return [1 if prob[1] > 0.5 else 0 for prob in probabilities]
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        # Handle single sample
        if len(X) > 0 and not isinstance(X[0], list):
            X = [X]
            
        results = []
        for sample in X:
            # Simple linear combination
            score = sum(feature * weight for feature, weight in zip(sample, self.feature_weights))
            # Apply sigmoid function
            prob_positive = 1 / (1 + math.exp(-score))
            prob_negative = 1 - prob_positive
            results.append([prob_negative, prob_positive])
        
        return results

def create_basic_model():
    """Create and save a basic model"""
    print("Creating basic model...")
    
    model = BasicModel()
    
    # Test the model
    test_features = [0.1, -0.5, 0.05, 0.2, 0.1, 0.3]
    prediction = model.predict([test_features])
    probabilities = model.predict_proba([test_features])
    
    print(f"✓ Model test successful:")
    print(f"  - Prediction: {prediction[0]}")
    print(f"  - Probabilities: [{probabilities[0][0]:.3f}, {probabilities[0][1]:.3f}]")
    
    # Save using pickle
    with open('trained_model.joblib', 'wb') as f:
        pickle.dump(model, f)
    
    print("✓ Basic model saved as 'trained_model.joblib'")
    return True

if __name__ == "__main__":
    try:
        create_basic_model()
        print("\n🎉 Basic model creation completed successfully!")
        print("Your Streamlit app should now be able to load the model.")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        exit(1)
