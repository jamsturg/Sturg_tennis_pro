import pandas as pd
from flask import Flask, request, jsonify
import os
import numpy as np

app = Flask(__name__)

# Global variable to store historical data
historical_matches_df = pd.DataFrame()

def load_historical_data(data_dir="TML-Database"):
    """Loads all historical match data from CSV files in the specified directory."""
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

@app.before_request
def before_first_request():
    """Load data once when the first request comes in."""
    global historical_matches_df
    if historical_matches_df.empty:
        historical_matches_df = load_historical_data()
        if historical_matches_df.empty:
            print("Warning: No historical data loaded. Feature calculation will return zeros.")

def get_player_rank(player_name, df):
    """Fetches the most recent rank for a given player."""
    if df.empty:
        return np.nan
        
    player_matches_winner = df[df.get('winner_name', pd.Series(dtype='object')) == player_name]
    player_matches_loser = df[df.get('loser_name', pd.Series(dtype='object')) == player_name]

    latest_rank_winner = np.nan
    latest_rank_loser = np.nan
    
    if not player_matches_winner.empty and 'winner_rank' in player_matches_winner.columns:
        try:
            latest_rank_winner = player_matches_winner.sort_values(by='tourney_date', ascending=False)['winner_rank'].iloc[0]
        except (KeyError, IndexError):
            latest_rank_winner = np.nan
    
    if not player_matches_loser.empty and 'loser_rank' in player_matches_loser.columns:
        try:
            latest_rank_loser = player_matches_loser.sort_values(by='tourney_date', ascending=False)['loser_rank'].iloc[0]
        except (KeyError, IndexError):
            latest_rank_loser = np.nan

    if pd.isna(latest_rank_winner) and pd.isna(latest_rank_loser):
        return np.nan
    elif pd.isna(latest_rank_winner):
        return latest_rank_loser
    elif pd.isna(latest_rank_loser):
        return latest_rank_winner
    else:
        # Return the better (lower) rank if both are available
        return min(latest_rank_winner, latest_rank_loser)

def get_player_performance_metrics(player_name, df, num_recent_matches=10):
    """
    Calculates various performance metrics for a given player from historical data.
    Returns a dictionary of average metrics.
    """
    if df.empty:
        return {
            'avg_serve_points_won_pct': np.nan,
            'avg_first_serve_pct': np.nan,
            'avg_return_points_won_pct': np.nan,
            'recent_win_percentage': np.nan
        }
    
    # Check if required columns exist
    if 'winner_name' not in df.columns or 'loser_name' not in df.columns:
        return {
            'avg_serve_points_won_pct': np.nan,
            'avg_first_serve_pct': np.nan,
            'avg_return_points_won_pct': np.nan,
            'recent_win_percentage': np.nan
        }
    
    player_matches_winner = df[df['winner_name'] == player_name]
    player_matches_loser = df[df['loser_name'] == player_name]

    player_matches = pd.concat([player_matches_winner, player_matches_loser], ignore_index=True)
    
    if player_matches.empty:
        return {
            'avg_serve_points_won_pct': np.nan,
            'avg_first_serve_pct': np.nan,
            'avg_return_points_won_pct': np.nan,
            'recent_win_percentage': np.nan
        }

    # Sort by date if column exists
    if 'tourney_date' in player_matches.columns:
        player_matches = player_matches.sort_values(by='tourney_date', ascending=False)
    
    recent_matches = player_matches.head(num_recent_matches)

    serve_points_won_pcts = []
    first_serve_pcts = []
    return_points_won_pcts = []

    for _, match in recent_matches.iterrows():
        if match['winner_name'] == player_name:
            # Player is the winner
            total_serve_points = match.get('w_svpt', 0)
            serve_points_won = match.get('w_1stWon', 0) + match.get('w_2ndWon', 0)
            first_serves_in = match.get('w_1stIn', 0)
            
            total_opponent_serve_points = match.get('l_svpt', 0)
            return_points_won = match.get('w_retPtsWon', 0)
            
        else:
            # Player is the loser
            total_serve_points = match.get('l_svpt', 0)
            serve_points_won = match.get('l_1stWon', 0) + match.get('l_2ndWon', 0)
            first_serves_in = match.get('l_1stIn', 0)
            
            total_opponent_serve_points = match.get('w_svpt', 0)
            return_points_won = match.get('l_retPtsWon', 0)

        # Calculate serve points won percentage
        if total_serve_points > 0:
            serve_points_won_pcts.append(serve_points_won / total_serve_points)
        
        # Calculate first serve percentage
        if total_serve_points > 0:
            first_serve_pcts.append(first_serves_in / total_serve_points)
            
        # Calculate return points won percentage
        if total_opponent_serve_points > 0:
            return_points_won_pcts.append(return_points_won / total_opponent_serve_points)

    metrics = {}
    if serve_points_won_pcts:
        metrics['avg_serve_points_won_pct'] = np.mean(serve_points_won_pcts)
    else:
        metrics['avg_serve_points_won_pct'] = np.nan

    if first_serve_pcts:
        metrics['avg_first_serve_pct'] = np.mean(first_serve_pcts)
    else:
        metrics['avg_first_serve_pct'] = np.nan

    if return_points_won_pcts:
        metrics['avg_return_points_won_pct'] = np.mean(return_points_won_pcts)
    else:
        metrics['avg_return_points_won_pct'] = np.nan

    recent_wins = recent_matches[recent_matches['winner_name'] == player_name].shape[0]
    if not recent_matches.empty:
        metrics['recent_win_percentage'] = recent_wins / recent_matches.shape[0]
    else:
        metrics['recent_win_percentage'] = np.nan

    return metrics

def calculate_h2h(player1_name, player2_name, df):
    """Calculates head-to-head advantage."""
    h2h_matches = df[
        ((df['winner_name'] == player1_name) & (df['loser_name'] == player2_name)) |
        ((df['winner_name'] == player2_name) & (df['loser_name'] == player1_name))
    ]

    if h2h_matches.empty:
        return 0.0 # No head-to-head matches

    player1_wins = h2h_matches[h2h_matches['winner_name'] == player1_name].shape[0]
    player2_wins = h2h_matches[h2h_matches['winner_name'] == player2_name].shape[0]

    total_h2h = player1_wins + player2_wins
    if total_h2h == 0:
        return 0.0
    
    return (player1_wins - player2_wins) / total_h2h

def calculate_features(player1_name, player2_name):
    """Calculates feature differences for two players based on historical data."""
    
    # Initialize features with default/placeholder values
    serve_strength_diff = 0.0
    ranking_diff = 0.0
    serve_percentage_diff = 0.0
    recent_form_diff = 0.0
    rally_performance_diff = 0.0
    h2h_advantage = 0.0

    if historical_matches_df.empty:
        return {
            "serve_strength_diff": serve_strength_diff,
            "ranking_diff": ranking_diff,
            "serve_percentage_diff": serve_percentage_diff,
            "recent_form_diff": recent_form_diff,
            "rally_performance_diff": rally_performance_diff,
            "h2h_advantage": h2h_advantage
        }

    # --- Calculate Ranking Difference ---
    rank1 = get_player_rank(player1_name, historical_matches_df)
    rank2 = get_player_rank(player2_name, historical_matches_df)

    if not pd.isna(rank1) and not pd.isna(rank2):
        ranking_diff = rank1 - rank2
    elif pd.isna(rank1) and not pd.isna(rank2):
        # If player1 has no rank, assume they are much lower ranked (higher number)
        ranking_diff = 10000 - rank2 # Large arbitrary difference
    elif not pd.isna(rank1) and pd.isna(rank2):
        # If player2 has no rank, assume they are much lower ranked (higher number)
        ranking_diff = rank1 - 10000 # Large arbitrary difference
    else:
        ranking_diff = 0.0 # Both unknown, assume no difference

    # --- Calculate other features ---
    player1_metrics = get_player_performance_metrics(player1_name, historical_matches_df)
    player2_metrics = get_player_performance_metrics(player2_name, historical_matches_df)

    # Serve Strength Difference (based on serve points won percentage)
    if not pd.isna(player1_metrics['avg_serve_points_won_pct']) and not pd.isna(player2_metrics['avg_serve_points_won_pct']):
        serve_strength_diff = player1_metrics['avg_serve_points_won_pct'] - player2_metrics['avg_serve_points_won_pct']
    
    # Serve Percentage Difference (based on first serve percentage)
    if not pd.isna(player1_metrics['avg_first_serve_pct']) and not pd.isna(player2_metrics['avg_first_serve_pct']):
        serve_percentage_diff = player1_metrics['avg_first_serve_pct'] - player2_metrics['avg_first_serve_pct']

    # Recent Form Difference (based on recent win percentage)
    if not pd.isna(player1_metrics['recent_win_percentage']) and not pd.isna(player2_metrics['recent_win_percentage']):
        recent_form_diff = player1_metrics['recent_win_percentage'] - player2_metrics['recent_win_percentage']

    # Rally Performance Difference (based on return points won percentage)
    if not pd.isna(player1_metrics['avg_return_points_won_pct']) and not pd.isna(player2_metrics['avg_return_points_won_pct']):
        rally_performance_diff = player1_metrics['avg_return_points_won_pct'] - player2_metrics['avg_return_points_won_pct']

    # Head-to-Head Advantage
    h2h_advantage = calculate_h2h(player1_name, player2_name, historical_matches_df)

    return {
        "serve_strength_diff": serve_strength_diff,
        "ranking_diff": ranking_diff,
        "serve_percentage_diff": serve_percentage_diff,
        "recent_form_diff": recent_form_diff,
        "rally_performance_diff": rally_performance_diff,
        "h2h_advantage": h2h_advantage
    }

@app.route('/api/player_features', methods=['POST'])
def player_features_endpoint():
    """API endpoint to get player features for a match."""
    data = request.get_json()
    player1_name = data.get('player1_name')
    player2_name = data.get('player2_name')

    if not player1_name or not player2_name:
        return jsonify({"error": "Missing player1_name or player2_name"}), 400

    features = calculate_features(player1_name, player2_name)
    return jsonify(features)

@app.route('/')
def home():
    return "Player Data API is running. Use /api/player_features to get data."

if __name__ == '__main__':
    # It's generally better to run Flask apps with a production-ready WSGI server like Gunicorn.
    # For development, app.run() is fine.
    # When running with `python player_data_api.py`, it will use Flask's development server.
    # When run by Gunicorn (e.g., `gunicorn -w 4 player_data_api:app`), it will use Gunicorn.
    app.run(host='0.0.0.0', port=5000)