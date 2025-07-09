"""
Configuration and fixtures for pytest.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the Streamlit functions
sys.modules['streamlit'] = MagicMock()

@pytest.fixture
def mock_requests():
    """Fixture to mock requests.get and requests.post"""
    with patch('requests.get') as mock_get, patch('requests.post') as mock_post:
        yield {'get': mock_get, 'post': mock_post}

@pytest.fixture
def sample_odds_data():
    """Sample odds data for testing"""
    return [
        {
            'id': 'match1',
            'sport_key': 'tennis_atp_wimbledon',
            'commence_time': (datetime.utcnow() + timedelta(days=1)).isoformat(),
            'home_team': 'Player A',
            'away_team': 'Player B',
            'bookmakers': [
                {
                    'key': 'pinnacle',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Player A', 'price': 1.8},
                                {'name': 'Player B', 'price': 2.0}
                            ]
                        }
                    ]
                }
            ]
        }
    ]

@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing"""
    return pd.DataFrame({
        'match_date': pd.date_range(end=datetime.today(), periods=100).date,
        'player1': ['Player A'] * 50 + ['Player B'] * 50,
        'player2': ['Player C'] * 50 + ['Player D'] * 50,
        'player1_odds': np.random.uniform(1.5, 3.0, 100),
        'player2_odds': np.random.uniform(1.5, 3.0, 100),
        'winner': np.random.choice([0, 1], 100),
        'surface': ['Grass'] * 50 + ['Hard'] * 50,
        'tournament': ['Wimbledon'] * 50 + ['US Open'] * 50
    })

@pytest.fixture
def basic_model():
    """Fixture for BasicModel instance"""
    from app import BasicModel
    return BasicModel()

@pytest.fixture
def enhanced_model():
    """Fixture for EnhancedTennisModel instance"""
    from app import EnhancedTennisModel
    model = EnhancedTennisModel()
    # Mock the model's predict method
    model.predict = MagicMock(return_value=np.array([0, 1]))
    model.predict_proba = MagicMock(return_value=np.array([[0.4, 0.6], [0.6, 0.4]]))
    return model

@pytest.fixture
def mock_streamlit():
    """Fixture to mock Streamlit functions"""
    with patch('streamlit.button'), \
         patch('streamlit.selectbox'), \
         patch('streamlit.slider'), \
         patch('streamlit.write'), \
         patch('streamlit.success'), \
         patch('streamlit.warning'), \
         patch('streamlit.error'), \
         patch('streamlit.spinner'):
        yield

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv('ODDS_API_KEY', 'test_api_key_123')
    monkeypatch.setenv('ENVIRONMENT', 'test')