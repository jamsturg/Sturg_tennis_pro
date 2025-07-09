#!/bin/bash

# Tennis Predictor Pro - Startup Script
# This script starts both the Streamlit app and the Player Data API

echo "🎾 Starting Tennis Predictor Pro..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if Streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed. Please install it with: pip install streamlit"
    exit 1
fi

# Kill any existing processes
echo "🔄 Stopping any existing processes..."
pkill -f "streamlit run app.py"
pkill -f "python player_data_api.py"

# Wait a moment for processes to stop
sleep 2

# Start Player Data API in background
echo "🚀 Starting Player Data API on port 5000..."
python player_data_api.py &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start Streamlit app in background
echo "🚀 Starting Streamlit app on port 3000..."
streamlit run app.py --server.port=3000 --server.address=0.0.0.0 &
APP_PID=$!

# Wait a moment for app to start
sleep 3

# Check if both services are running
if ps -p $API_PID > /dev/null; then
    echo "✅ Player Data API is running (PID: $API_PID)"
else
    echo "❌ Failed to start Player Data API"
fi

if ps -p $APP_PID > /dev/null; then
    echo "✅ Streamlit app is running (PID: $APP_PID)"
else
    echo "❌ Failed to start Streamlit app"
fi

echo ""
echo "🎾 Tennis Predictor Pro is now running!"
echo "📊 Main App: http://localhost:3000"
echo "🔌 Player Data API: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running and handle Ctrl+C
trap 'echo "🛑 Stopping services..."; kill $API_PID $APP_PID; exit 0' INT
wait
