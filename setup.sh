#!/bin/bash

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
