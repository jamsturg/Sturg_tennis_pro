#!/bin/bash

# Replit-specific setup script for Tennis Predictor Pro
echo "🎾 Setting up Tennis Predictor Pro for Replit..."

# Install minimal requirements
echo "📦 Installing minimal requirements for Replit..."
pip install -r requirements-replit.txt

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create config.toml with basic configuration
cat > .streamlit/config.toml <<EOL
[server]
port = 3000
enableCORS = false
headless = true

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#31333F"
font = "sans serif"
EOL

# Create secrets.toml template if it doesn't exist
if [ ! -f .streamlit/secrets.toml ]; then
    cat > .streamlit/secrets.toml <<EOL
[api]
odds_api_key = "your_api_key_here"
EOL
    echo "⚠️  Please update .streamlit/secrets.toml with your actual API key"
fi

echo "✅ Setup complete! Run 'streamlit run app.py --server.port=3000 --server.address=0.0.0.0' to start"
