"""
Tests for utility functions in the application.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the functions we want to test
from app import (
    convert_to_aest,
    get_matches_with_autocomplete,
    create_odds_visualization,
    convert_odds_to_probability,
    kelly_criterion
)

def test_convert_to_aest():
    """Test UTC to AEST timezone conversion."""
    # Test with a known UTC time with 'Z' timezone
    utc_time_z = "2025-07-04T12:00:00Z"
    aest_time = convert_to_aest(utc_time_z)
    assert aest_time is not None
    assert "AEST" in aest_time
    assert isinstance(aest_time, str)
    
    # Test with a known UTC time with +00:00 timezone
    utc_time_offset = "2025-07-04T12:00:00+00:00"
    aest_time_offset = convert_to_aest(utc_time_offset)
    assert aest_time_offset is not None
    assert "AEST" in aest_time_offset
    
    # Test with None input
    assert convert_to_aest(None) is None
    
    # Test with empty string
    assert convert_to_aest("") is None
    
    # Test with invalid date string (should return None, not raise)
    assert convert_to_aest("invalid-date") is None

def test_convert_odds_to_probability():
    """Test odds to probability conversion."""
    # Test with valid odds
    assert abs(convert_odds_to_probability(2.0) - 0.5) < 1e-9  # 1/2 = 0.5
    assert abs(convert_odds_to_probability(4.0) - 0.25) < 1e-9  # 1/4 = 0.25
    assert abs(convert_odds_to_probability(1.5) - 0.666666666) < 1e-9  # 1/1.5 ≈ 0.666...
    
    # Test with infinity
    assert convert_odds_to_probability(float('inf')) == 0.0
    
    # Test with exact 1.0
    assert convert_odds_to_probability(1.0) == 1.0
    
    # Test with invalid inputs
    with pytest.raises(ValueError, match="Odds must be >= 1.0"):
        convert_odds_to_probability(0.5)  # Odds < 1.0
    
    with pytest.raises(ValueError, match="Odds must be >= 1.0"):
        convert_odds_to_probability(0.0)  # Odds = 0
    
    with pytest.raises(ValueError, match="Odds must be >= 1.0"):
        convert_odds_to_probability(-1.0)  # Negative odds
    
    # Test with non-numeric input
    with pytest.raises(ValueError, match="Odds must be a number"):
        convert_odds_to_probability("2.0")  # String input
    
    with pytest.raises(ValueError, match="Odds must be a number"):
        convert_odds_to_probability(None)  # None input

def test_kelly_criterion():
    """Test Kelly Criterion calculation with various scenarios."""
    # Test with valid inputs (profitable bet)
    bankroll = 1000
    win_prob = 0.6
    odds = 2.0
    
    # Expected: (0.6 * 2.0 - 1) / (2.0 - 1) = 0.2 or 20% (full Kelly)
    # But we use half-Kelly, so 10%
    fraction, amount, message = kelly_criterion(bankroll, win_prob, odds)
    assert abs(fraction - 0.1) < 0.001  # Half of 20%
    assert abs(amount - 100) < 0.01  # 10% of 1000
    assert "Recommended bet:" in message
    
    # Test with break-even bet (should not recommend betting)
    fraction, amount, message = kelly_criterion(bankroll, 0.5, 2.0)
    assert fraction == 0.0
    assert amount == 0.0
    assert "Negative expected value" in message
    
    # Test with zero bankroll
    fraction, amount, message = kelly_criterion(0, win_prob, odds)
    assert amount == 0
    assert "Bankroll must be positive" in message
    
    # Test with negative bankroll
    fraction, amount, message = kelly_criterion(-100, win_prob, odds)
    assert amount == 0
    assert "Bankroll must be positive" in message
    
    # Test with invalid win probability
    fraction, amount, message = kelly_criterion(bankroll, -0.1, odds)
    assert amount == 0
    assert "Win probability must be between 0 and 1" in message
    
    fraction, amount, message = kelly_criterion(bankroll, 1.1, odds)
    assert amount == 0
    assert "Win probability must be between 0 and 1" in message
    
    # Test with invalid odds
    fraction, amount, message = kelly_criterion(bankroll, win_prob, 1.0)
    assert amount == 0
    assert "Decimal odds must be greater than 1.0" in message
    
    fraction, amount, message = kelly_criterion(bankroll, win_prob, 0.5)
    assert amount == 0
    assert "Decimal odds must be greater than 1.0" in message
    
    # Test with very small bankroll (bet amount < 1 unit)
    # For bankroll <= 10, we allow betting the minimum 1 unit if there's a positive edge
    fraction, amount, message = kelly_criterion(5.0, 0.9, 10.0)
    assert amount > 0  # Should bet something (minimum 1 unit)
    assert amount <= 5.0  # But not more than the bankroll
    assert "Recommended bet" in message
    
    # Test with very small edge where we expect no bet
    fraction, amount, message = kelly_criterion(1000, 0.51, 1.9)  # Very small edge
    assert amount == 0  # Should not bet with such a small edge
    assert "Negative expected value" in message  # This is the actual message from the function
    
    # Test with non-numeric inputs
    with pytest.raises(ValueError, match="All inputs must be numbers"):
        kelly_criterion("1000", 0.6, 2.0)
    
    with pytest.raises(ValueError, match="All inputs must be numbers"):
        kelly_criterion(1000, "0.6", 2.0)
    
    with pytest.raises(ValueError, match="All inputs must be numbers"):
        kelly_criterion(1000, 0.6, "2.0")

def test_get_matches_with_autocomplete(sample_odds_data):
    """Test match data processing for autocomplete."""
    matches = get_matches_with_autocomplete(sample_odds_data)
    assert isinstance(matches, list)
    
    if matches:  # Only proceed if we have matches
        match = matches[0]
        # Check for expected keys in the match dictionary
        assert 'match_string' in match
        assert 'home_team' in match
        assert 'away_team' in match
        assert 'event_data' in match
        
        # Check types
        assert isinstance(match['match_string'], str)
        assert isinstance(match['home_team'], str)
        assert isinstance(match['away_team'], str)
        assert isinstance(match['event_data'], dict)
        
        # Check that the match string contains both team names
        assert match['home_team'] in match['match_string']
        assert match['away_team'] in match['match_string']
        
        # Check that the event data contains the original data
        assert 'home_team' in match['event_data']
        assert 'away_team' in match['event_data']
        assert match['event_data']['home_team'] == match['home_team']
        assert match['event_data']['away_team'] == match['away_team']

def test_create_odds_visualization():
    """Test odds visualization creation."""
    # Create sample data
    odds_df = pd.DataFrame({
        'player1': ['Player A', 'Player B'],
        'player2': ['Player C', 'Player D'],
        'player1_odds': [1.8, 2.1],
        'player2_odds': [2.0, 1.8],
        'commence_time': [datetime.utcnow().isoformat()] * 2,
        'sport_key': ['tennis_atp_wimbledon'] * 2
    })
    
    # Test with valid data
    fig = create_odds_visualization(odds_df)
    assert fig is not None
    
    # Test with empty DataFrame
    empty_fig = create_odds_visualization(pd.DataFrame())
    assert empty_fig is not None

def test_visualization_integration(sample_odds_data):
    """Integration test for visualization functions."""
    # Process the data
    matches = get_matches_with_autocomplete(sample_odds_data)
    assert len(matches) > 0
    
    # Create a properly formatted DataFrame for the visualization
    odds_data = []
    for match in sample_odds_data:
        for bookmaker in match['bookmakers']:
            for market in bookmaker['markets']:
                if market['key'] == 'h2h':
                    home_odds = next((o['price'] for o in market['outcomes'] if o['name'] == match['home_team']), None)
                    away_odds = next((o['price'] for o in market['outcomes'] if o['name'] == match['away_team']), None)
                    
                    if home_odds is not None and away_odds is not None:
                        odds_data.append({
                            'Home Team': match['home_team'],
                            'Away Team': match['away_team'],
                            'Home Odds': home_odds,
                            'Away Odds': away_odds,
                            'commence_time': match['commence_time'],
                            'Bookmaker': bookmaker['key']
                        })
    
    odds_df = pd.DataFrame(odds_data)
    
    # Test visualization with properly formatted data
    fig = create_odds_visualization(odds_df)
    
    # Check that we got a valid figure object
    assert fig is not None
    assert hasattr(fig, 'to_dict')  # Basic check that it's a plotly figure
    
    # Check that the figure has the expected structure
    fig_dict = fig.to_dict()
    assert 'data' in fig_dict
    assert 'layout' in fig_dict
    
    # The title might be different based on the data, so let's just check it exists
    assert 'title' in fig_dict['layout']
    assert 'text' in fig_dict['layout']['title']