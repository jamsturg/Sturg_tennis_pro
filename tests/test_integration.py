"""
Integration tests for the complete application workflow.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import streamlit as st

# Import the main app to test its components
from app import (
    fetch_odds_data,
    process_odds_data,
    prepare_features,
    run_enhanced_backtest,
    BasicModel
)

# Mock the API responses
MOCK_ODDS_RESPONSE = {
    "data": [
        {
            "id": "match1",
            "sport_key": "tennis_atp_wimbledon",
            "commence_time": "2025-07-05T14:00:00Z",
            "home_team": "Player A",
            "away_team": "Player B",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Player A", "price": 1.8},
                                {"name": "Player B", "price": 2.0}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}

# Sample historical data for backtesting
SAMPLE_HISTORICAL_DATA = pd.DataFrame({
    'match_date': pd.date_range(end='2023-06-30', periods=100),
    'player1': ['Player ' + str(i) for i in range(50)] * 2,
    'player2': ['Player ' + str(i+50) for i in range(50)] * 2,
    'player1_odds': np.random.uniform(1.5, 3.0, 100),
    'player2_odds': np.random.uniform(1.5, 3.0, 100),
    'winner': np.random.choice([0, 1], 100),
    'surface': ['Grass'] * 50 + ['Hard'] * 50,
    'tournament': ['Wimbledon'] * 50 + ['US Open'] * 50
})

@patch('requests.get')
def test_complete_workflow(mock_get, mock_requests):
    """Test the complete workflow from API call to prediction."""
    # Setup mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_ODDS_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    # 1. Test API call
    odds_data = fetch_odds_data("tennis_atp_wimbledon")
    assert odds_data == MOCK_ODDS_RESPONSE['data']
    
    # 2. Test data processing
    processed_data = process_odds_data(odds_data)
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert 'player1' in processed_data.columns
    assert 'player2' in processed_data.columns
    assert 'player1_odds' in processed_data.columns
    assert 'player2_odds' in processed_data.columns
    
    # 3. Test feature preparation
    features = prepare_features(
        processed_data.iloc[0]['player1'],
        processed_data.iloc[0]['player2'],
        processed_data.iloc[0]['player1_odds'],
        processed_data.iloc[0]['player2_odds']
    )
    assert isinstance(features, list)
    assert len(features) == 6  # 6 features expected
    
    # 4. Test model prediction
    model = BasicModel()
    prediction = model.predict([features])
    assert isinstance(prediction, list)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]  # Should be 0 or 1
    
    # 5. Test backtesting
    backtest_df, summary_stats = run_enhanced_backtest(
        SAMPLE_HISTORICAL_DATA,
        model,
        initial_bankroll=1000,
        strategy="kelly"
    )
    
    assert isinstance(backtest_df, pd.DataFrame)
    assert not backtest_df.empty
    assert 'bankroll' in backtest_df.columns
    assert 'profit' in backtest_df.columns
    
    assert isinstance(summary_stats, dict)
    assert 'final_bankroll' in summary_stats
    assert 'total_profit' in summary_stats
    assert 'win_rate' in summary_stats

@patch('streamlit.session_state')
@patch('streamlit.button')
@patch('streamlit.selectbox')
@patch('streamlit.slider')
def test_ui_workflow(mock_slider, mock_selectbox, mock_button, mock_session_state):
    """Test the UI workflow with mocked components."""
    # Setup mocks
    mock_button.return_value = True
    mock_selectbox.return_value = "tennis_atp_wimbledon"
    mock_slider.return_value = 0.5
    
    # Setup session state
    mock_session_state.__getitem__.side_effect = lambda x: {
        'last_refresh': None,
        'cached_odds': MOCK_ODDS_RESPONSE['data'],
        'sure_things': [],
        'value_bets': []
    }.get(x, {})
    
    # Import main app to trigger the UI code
    import app
    
    # Verify UI components were called
    assert mock_button.called
    assert mock_selectbox.called
    assert mock_slider.called

def test_error_handling():
    """Test error handling in the application."""
    # Test with invalid odds
    with pytest.raises(ValueError):
        prepare_features("Player A", "Player B", 0, 2.0)
    
    # Test with None values
    with pytest.raises(ValueError):
        prepare_features(None, "Player B", 1.8, 2.0)
    
    # Test with empty data
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        run_enhanced_backtest(empty_df, BasicModel())

@patch('requests.get')
def test_api_error_handling(mock_get):
    """Test API error handling."""
    # Test with API error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API Error")
    mock_get.return_value = mock_response
    
    with pytest.raises(Exception):
        fetch_odds_data("tennis_atp_wimbledon")
