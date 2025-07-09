#!/usr/bin/env python3
"""
Simple script to create a trained model for the table tennis prediction app.
"""

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    print("✓ All required packages imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install required packages: pip install scikit-learn joblib numpy pandas")
    exit(1)

def create_model():
    print("Creating trained model...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic training data that matches the app's expected features
    serve_strength_diff = np.random.normal(0, 0.3, n_samples)
    ranking_diff = np.random.normal(0, 2, n_samples)  
    serve_percentage_diff = np.random.normal(0, 0.1, n_samples)
    recent_form_diff = np.random.normal(0, 0.4, n_samples)
    rally_performance_diff = np.random.normal(0, 0.2, n_samples)
    h2h_advantage = np.random.normal(0, 0.3, n_samples)
    
    # Combine features into matrix
    X = np.column_stack([
        serve_strength_diff, 
        ranking_diff, 
        serve_percentage_diff, 
        recent_form_diff, 
        rally_performance_diff, 
        h2h_advantage
    ])
    
    # Generate realistic target variable
    # Players with positive advantages are more likely to win
    advantage_score = (
        serve_strength_diff * 0.3 + 
        ranking_diff * 0.1 + 
        recent_form_diff * 0.4 + 
        h2h_advantage * 0.5 +
        np.random.normal(0, 0.1, n_samples)  # Add some noise
    )
    
    y = (advantage_score > 0).astype(int)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    print(f"Training model on {len(X)} samples with {X.shape[1]} features...")
    model.fit(X, y)
    
    # Save the trained model
    joblib.dump(model, 'trained_model.joblib')
    print("✓ Model saved as 'trained_model.joblib'")
    
    # Test the model to ensure it works
    test_features = np.array([[0.1, -0.5, 0.05, 0.2, 0.1, 0.3]])
    prediction = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    
    print(f"✓ Model test successful:")
    print(f"  - Prediction: {prediction[0]}")
    print(f"  - Probabilities: [{probabilities[0][0]:.3f}, {probabilities[0][1]:.3f}]")
    print(f"  - Feature importance: {model.feature_importances_}")
    
    return True

if __name__ == "__main__":
    try:
        success = create_model()
        if success:
            print("\n🎉 Model creation completed successfully!")
            print("You can now run your Streamlit app.")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        exit(1)
