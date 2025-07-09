#!/bin/bash

# Install dependencies using uv if available, otherwise fall back to pip
if command -v uv &> /dev/null
then
    echo "📦 Installing dependencies with uv..."
    uv pip install -r requirements.txt
else
    echo "📦 Installing dependencies with pip..."
    pip install -r requirements.txt
fi

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create config.toml with basic configuration
cat > .streamlit/config.toml <<EOL
[server]
port = 8501
enableCORS = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#31333F"
font = "sans serif"
EOL

echo "✅ Streamlit configuration created successfully!"
