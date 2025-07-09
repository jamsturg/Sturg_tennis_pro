"""
Tests for UI components and interactions in the application.
"""
import pytest
from unittest.mock import MagicMock, patch
import streamlit as st
import pandas as pd
import numpy as np

# Import the UI functions we want to test
from app import (
    prepare_features,
    run_enhanced_backtest,
    create_backtest_visualizations
)

# Mock Streamlit functions
st.session_state = {}

def test_prepare_features():
    """Test feature preparation for model input."""
    # Test with valid inputs
    player1_name = "Player A"
    player2_name = "Player B"
    player1_odds = 1.8
    player2_odds = 2.0
    
    features = prepare_features(player1_name, player2_name, player1_odds, player2_odds)
    
    # Check output format
    assert isinstance(features, list)
    assert len(features) == 6  # Should return 6 features
    assert all(isinstance(x, (int, float)) for x in features)
    
    # Test with None values
    with pytest.raises(ValueError):
        prepare_features(None, "Player B", 1.8, 2.0)
    
    with pytest.raises(ValueError):
        prepare_features("Player A", "Player B", 0, 2.0)  # Invalid odds

def test_run_enhanced_backtest():
    """Test the backtesting functionality."""
    # Create sample data
    data = pd.DataFrame({
        'player1': ['A', 'B', 'C', 'D'],
        'player2': ['B', 'C', 'D', 'A'],
        'player1_odds': [1.8, 2.0, 1.9, 2.1],
        'player2_odds': [2.0, 1.8, 2.0, 1.8],
        'winner': [0, 1, 0, 1],
        'surface': ['Grass'] * 4,
        'tournament': ['Wimbledon'] * 4,
        'match_date': pd.date_range(end='2023-01-04', periods=4)
    })
    
    # Create a mock model
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.6, 0.4]] * len(X))
    
    model = MockModel()
    
    # Test with different strategies
    for strategy in ["flat", "kelly", "proportional"]:
        backtest_df, summary_stats = run_enhanced_backtest(
            data, model, initial_bankroll=1000, strategy=strategy
        )
        
        # Check outputs
        assert isinstance(backtest_df, pd.DataFrame)
        assert not backtest_df.empty
        assert 'bankroll' in backtest_df.columns
        assert 'profit' in backtest_df.columns
        
        assert isinstance(summary_stats, dict)
        assert 'final_bankroll' in summary_stats
        assert 'total_profit' in summary_stats
        assert 'win_rate' in summary_stats

def test_create_backtest_visualizations():
    """Test creation of backtest visualizations."""
    # Create sample backtest data
    backtest_df = pd.DataFrame({
        'match_date': pd.date_range(end='2023-01-10', periods=10),
        'bankroll': [1000 + i*100 for i in range(10)],
        'profit': [i*100 for i in range(10)],
        'bet_size': [100] * 10,
        'odds': [2.0] * 10,
        'outcome': [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        'strategy': ['kelly'] * 10
    })
    
    # Create summary stats
    summary_stats = {
        'final_bankroll': 1900,
        'total_profit': 900,
        'win_rate': 0.7,
        'total_bets': 10,
        'total_wins': 7,
        'total_losses': 3,
        'profit_factor': 2.33,
        'max_drawdown': -200,
        'roi': 0.9
    }
    
    # Test visualization creation
    fig = create_backtest_visualizations(backtest_df, summary_stats)
    
    # Check that we got a Plotly figure back
    assert fig is not None
    assert hasattr(fig, 'to_dict')  # Plotly figure check

@patch('streamlit.button')
@patch('streamlit.selectbox')
@patch('streamlit.slider')
def test_ui_interactions(mock_slider, mock_selectbox, mock_button, sample_odds_data):
    """Test UI component interactions."""
    # Mock UI components
    mock_button.return_value = True
    mock_selectbox.return_value = "tennis_atp_wimbledon"
    mock_slider.return_value = 0.5
    
    # Mock session state
    st.session_state = {
        'last_refresh': None,
        'cached_odds': sample_odds_data,
        'sure_things': [],
        'value_bets': []
    }
    
    # This would test the main app flow, but we're just checking UI interactions
    assert mock_button.called
    assert mock_selectbox.called
    assert mock_slider.called

@patch('streamlit.session_state')
def test_session_state_management(mock_session_state):
    """Test session state management."""
    # Setup mock session state
    mock_session_state.__getitem__.side_effect = lambda x: {
        'last_refresh': None,
        'cached_odds': None,
        'sure_things': [],
        'value_bets': []
    }.get(x, {})
    
    # Test that session state is accessed
    assert 'last_refresh' in mock_session_state
    assert 'cached_odds' in mock_session_state
    assert 'sure_things' in mock_session_state
    assert 'value_bets' in mock_session_state