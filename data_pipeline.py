"""
Enhanced data pipeline for tennis prediction system.
Handles data collection, cleaning, feature engineering, and preparation.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import requests
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TennisDataPipeline:
    """Pipeline for handling tennis match data collection and processing."""
    
    def __init__(self, api_key: Optional[str] = None, data_dir: str = 'data'):
        """Initialize the data pipeline.
        
        Args:
            api_key: API key for data providers
            data_dir: Directory to store cached data
        """
        self.api_key = api_key or os.getenv('TENNIS_API_KEY')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # API endpoints (example - configure based on your data provider)
        self.api_endpoints = {
            'upcoming_matches': 'https://api.tennis.com/v1/matches/upcoming',
            'historical_odds': 'https://api.odds-api.com/v4/sports/tennis/odds',
            'player_stats': 'https://api.tennis.com/v1/players/stats'
        }
        
    def fetch_live_matches(self) -> List[Dict]:
        """Fetch live and upcoming tennis matches."""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
            response = requests.get(
                self.api_endpoints['upcoming_matches'],
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json().get('matches', [])
        except Exception as e:
            logger.error(f"Error fetching live matches: {e}")
            return []
    
    def fetch_historical_odds(self, days: int = 30) -> pd.DataFrame:
        """Fetch historical odds data.
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame containing historical odds data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_odds = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                date_str = current_date.strftime('%Y-%m-%d')
                cache_file = self.data_dir / f'odds_{date_str}.json'
                
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        day_odds = json.load(f)
                else:
                    params = {
                        'date': date_str,
                        'apiKey': self.api_key
                    }
                    response = requests.get(
                        self.api_endpoints['historical_odds'],
                        params=params,
                        timeout=15
                    )
                    response.raise_for_status()
                    day_odds = response.json()
                    
                    # Cache the response
                    with open(cache_file, 'w') as f:
                        json.dump(day_odds, f)
                
                all_odds.extend(day_odds)
                
            except Exception as e:
                logger.error(f"Error fetching odds for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(all_odds)
    
    def process_match_data(self, raw_matches: List[Dict]) -> pd.DataFrame:
        """Process raw match data into a clean DataFrame."""
        if not raw_matches:
            return pd.DataFrame()
            
        processed = []
        for match in raw_matches:
            try:
                # Extract basic match info
                match_data = {
                    'match_id': match.get('id'),
                    'tournament': match.get('tournament', {}).get('name'),
                    'surface': match.get('tournament', {}).get('surface'),
                    'round': match.get('round'),
                    'match_time': match.get('scheduled'),
                    'player1_id': match.get('home_team', {}).get('id'),
                    'player1_name': match.get('home_team', {}).get('name'),
                    'player2_id': match.get('away_team', {}).get('id'),
                    'player2_name': match.get('away_team', {}).get('name'),
                }
                
                # Add odds if available
                if 'odds' in match:
                    for bookmaker in match['odds']:
                        bookie_name = bookmaker.get('bookmaker_name')
                        for market in bookmaker.get('markets', []):
                            if market.get('key') == 'h2h':
                                for outcome in market.get('outcomes', []):
                                    if outcome.get('name') == match_data['player1_name']:
                                        match_data[f'player1_odds_{bookie_name}'] = outcome.get('price')
                                    elif outcome.get('name') == match_data['player2_name']:
                                        match_data[f'player2_odds_{bookie_name}'] = outcome.get('price')
                
                processed.append(match_data)
                
            except Exception as e:
                logger.error(f"Error processing match {match.get('id')}: {e}")
        
        return pd.DataFrame(processed)
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the match data."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Convert match time to datetime
        if 'match_time' in df.columns:
            df['match_time'] = pd.to_datetime(df['match_time'])
            df['hour'] = df['match_time'].dt.hour
            df['day_of_week'] = df['match_time'].dt.dayofweek
        
        # Add surface features
        if 'surface' in df.columns:
            df = pd.get_dummies(df, columns=['surface'], prefix='surface')
        
        # Add round features
        if 'round' in df.columns:
            round_importance = {
                'F': 7, 'SF': 6, 'QF': 5, 'R16': 4,
                'R32': 3, 'R64': 2, 'R128': 1, 'Q': 0.5
            }
            df['round_importance'] = df['round'].map(round_importance).fillna(0)
        
        return df
    
    def get_training_data(self, days: int = 90) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data with features and target."""
        # Fetch and process historical data
        historical_data = self.fetch_historical_odds(days)
        processed_data = self.process_match_data(historical_data)
        
        if processed_data.empty:
            logger.warning("No data available for training")
            return pd.DataFrame(), pd.Series()
        
        # Add features
        feature_df = self.add_features(processed_data)
        
        # For demonstration - in a real system, you would have the actual match outcomes
        # Here we'll create a synthetic target based on odds (lower odds = more likely to win)
        if 'player1_odds' in feature_df.columns and 'player2_odds' in feature_df.columns:
            feature_df['target'] = (feature_df['player1_odds'] < feature_df['player2_odds']).astype(int)
        
        # Separate features and target
        if 'target' in feature_df.columns:
            X = feature_df.drop(columns=['target', 'match_id', 'player1_id', 'player2_id'], errors='ignore')
            y = feature_df['target']
            return X, y
        
        return feature_df, pd.Series()

    def get_live_data(self) -> pd.DataFrame:
        """Get processed data for live predictions."""
        live_matches = self.fetch_live_matches()
        processed_data = self.process_match_data(live_matches)
        
        if not processed_data.empty:
            return self.add_features(processed_data)
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    pipeline = TennisDataPipeline()
    
    # Get training data
    X, y = pipeline.get_training_data(days=30)
    print(f"Training data shape: {X.shape}, Target shape: {y.shape}")
    
    # Get live data for prediction
    live_data = pipeline.get_live_data()
    print(f"Live matches available: {len(live_data)}")
