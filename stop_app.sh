#!/bin/bash

# Tennis Predictor Pro - Stop Script
# This script stops both the Streamlit app and the Player Data API

echo "🛑 Stopping Tennis Predictor Pro..."

# Kill Streamlit processes
echo "🔄 Stopping Streamlit app..."
pkill -f "streamlit run app.py"

# Kill Player Data API processes
echo "🔄 Stopping Player Data API..."
pkill -f "python player_data_api.py"

# Wait a moment for processes to stop
sleep 2

# Check if processes are still running
if pgrep -f "streamlit run app.py" > /dev/null; then
    echo "⚠️  Streamlit app may still be running. Force killing..."
    pkill -9 -f "streamlit run app.py"
fi

if pgrep -f "python player_data_api.py" > /dev/null; then
    echo "⚠️  Player Data API may still be running. Force killing..."
    pkill -9 -f "python player_data_api.py"
fi

echo "✅ All services stopped."
