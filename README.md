# 🎾 Tennis Predictor Pro

**Professional AI-Powered Tennis Match Analysis & Prediction System**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yourusername/tennis-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/tennis-predictor/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/yourusername/tennis-predictor/graph/badge.svg?token=YOUR-TOKEN)](https://codecov.io/gh/yourusername/tennis-predictor)

## 🚀 Features

### 🏠 **Dashboard**
- Real-time model and API status monitoring
- Live tennis market overview with AEST timezone
- Quick match previews with odds
- Performance metrics dashboard

### 📊 **Live Odds & Analysis**
- Integration with The Odds API for real-time data
- Interactive Plotly visualizations
- Comprehensive market analysis

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tennis-predictor.git
   cd tennis-predictor
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory and add your API key:
   ```
   ODDS_API_KEY=your_api_key_here
   ENVIRONMENT=development
   ```

## 🚀 Running the Application

### Development Mode
```bash
streamlit run app.py
```

### Production Mode
```bash
python -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

## 🧪 Running Tests

### Install test dependencies
```bash
pip install -r requirements-test.txt
```

### Run all tests
```bash
pytest -v --cov=app tests/
```

### Run specific test files
```bash
pytest tests/test_utils.py -v
pytest tests/test_models.py -v
pytest tests/test_ui.py -v
pytest tests/test_integration.py -v
```

### Generate coverage report
```bash
pytest --cov=app --cov-report=html tests/
open htmlcov/index.html  # View the coverage report
```

## 🏗️ Project Structure

```
tennis-predictor/
├── app.py                 # Main application file
├── requirements.txt       # Production dependencies
├── requirements-test.txt  # Test dependencies
├── setup.py              # Package configuration
├── tests/                # Test files
│   ├── conftest.py       # Test fixtures
│   ├── test_utils.py     # Utility function tests
│   ├── test_models.py    # Model tests
│   ├── test_ui.py        # UI component tests
│   └── test_integration.py # Integration tests
└── .github/workflows/    # GitHub Actions workflows
    └── tests.yml         # CI/CD pipeline
```

## 🤝 Contributing

1. **Fork** the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a **Pull Request**

### Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for better code clarity
- Write docstrings for all public functions and classes
- Keep functions small and focused on a single responsibility

### Testing Guidelines
- Write tests for all new features and bug fixes
- Aim for at least 80% test coverage
- Use descriptive test function names
- Mock external API calls in tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Notes

- The application uses The Odds API for live odds data (API key required)
- All odds and predictions are for informational purposes only
- Betting laws vary by jurisdiction - please ensure compliance with local regulations
- AEST (Brisbane) timezone support
- Auto-refresh functionality

### 🔮 **AI-Powered Match Predictions**
- Advanced machine learning models
- Autocomplete match selection from live odds
- Confidence-based predictions with visual meters
- Value betting detection and analysis
- Risk assessment with color-coded warnings

### 🎯 **Multi-Bet & Parlay Optimizer**
- AI-powered analysis of all available matches
- Sure Thing Parlays (>75% confidence)
- Value Bet Parlays (>10% edge)
- Smart parlay builder with odds calculation
- Kelly Criterion-optimized stake recommendations

### 💰 **Advanced Bankroll Management**
- Multiple betting strategies (Kelly, Fixed, Conservative)
- Comprehensive backtesting with 100+ match simulations
- Interactive performance visualizations
- Real-time Kelly Criterion calculator
- Professional risk assessment tools

### 🧠 **Model Management**
- Multiple AI model support (BasicModel, XGBoost, Neural Networks)
- Feature engineering and management
- Model training and retraining capabilities
- Hyperparameter tuning (Grid Search, Bayesian Optimization)
- Model performance comparison and deployment

### 🤖 **AI Automation Center**
- Automated data refresh and monitoring
- Smart auto-betting with safety controls ⚠️
- Email and webhook notifications
- Live AI activity feed
- Experimental AI trading system

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- Git
- The Odds API key (get from [the-odds-api.com](https://the-odds-api.com/))

### Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/jamsturg/Sturg_tennis_pro.git
cd Sturg_tennis_pro
```

2. **Install with UV (Recommended):**
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

3. **Or install with pip:**
```bash
pip install -r requirements.txt
```

4. **Configure API Key:**
Create `.streamlit/secrets.toml`:
```toml
[api]
odds_api_key = "your_api_key_here"
```

5. **Run the application:**
```bash
streamlit run app.py
```

## 📋 Project Structure

```
Sturg_tennis_pro/
├── app.py                      # Main Streamlit application
├── trained_model.joblib        # Pre-trained AI model
├── processed_ligapro_data.csv  # Historical training data
├── requirements.txt            # Python dependencies
├── pyproject.toml             # UV project configuration
├── .streamlit/
│   └── secrets.toml           # API keys and secrets
├── create_basic_model.py      # Model creation utilities
├── create_model.py           # Advanced model training
└── README.md                 # This file
```

## 🎯 Usage

### 1. Dashboard
- Monitor system status and live markets
- View upcoming matches in AEST timezone
- Quick access to all features

### 2. Live Odds Analysis
- Select tournaments (ATP/WTA Wimbledon, etc.)
- Analyze market trends with interactive charts
- Monitor odds movements and opportunities

### 3. Match Predictions
- Choose between live match selection or manual entry
- Get AI-powered predictions with confidence levels
- Identify value betting opportunities
- Assess risk levels for each prediction

### 4. Multi-Bet & Parlays
- Run AI analysis on all available matches
- Build optimized parlays for different strategies
- Calculate optimal stake sizes using Kelly Criterion

### 5. Bankroll Management
- Backtest strategies on 100+ simulated matches
- Use advanced risk management tools
- Monitor performance with detailed analytics

### 6. Model Management
- Train and retrain AI models
- Add custom features and optimize parameters
- Compare model performance metrics

### 7. AI Automation
- Set up automated data refresh
- Configure alerts and notifications
- **⚠️ CAUTION:** Automated betting features available

## ⚠️ Important Disclaimers

### Risk Warning
- **This software is for educational and research purposes**
- **Sports betting involves significant financial risk**
- **Never bet more than you can afford to lose**
- **Automated betting features should be used with extreme caution**

### Legal Notice
- Ensure sports betting is legal in your jurisdiction
- Users are responsible for compliance with local laws
- This software does not guarantee profits
- Past performance does not indicate future results

## 🔧 Configuration

### API Configuration
The app requires The Odds API for live data:
1. Sign up at [the-odds-api.com](https://the-odds-api.com/)
2. Get your free API key
3. Add it to `.streamlit/secrets.toml`

### Model Configuration
- Default model supports 6 tennis-specific features
- Models can be retrained with new data
- Custom features can be added through the UI

### Timezone Configuration
- Default timezone: Australia/Brisbane (AEST)
- All match times displayed in AEST
- Configurable in the code if needed

## 🧪 Development

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

### Model Development
- Use the Model Management interface
- Add features through the Feature Engineering section
- Train models with the built-in tools

### Testing
```bash
# Run backtests
python -c "from app import run_enhanced_backtest; print('Backtest functionality ready')"

# Test model loading
python -c "from app import load_model; model = load_model('trained_model.joblib'); print('Model loaded successfully' if model else 'Model failed to load')"
```

## 📊 Performance Metrics

The app tracks various performance metrics:
- **Prediction Accuracy**: Model success rate
- **ROI**: Return on investment percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of successful bets
- **Max Drawdown**: Largest loss from peak

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [The Odds API](https://the-odds-api.com/) for live sports data
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [scikit-learn](https://scikit-learn.org/) for machine learning tools

## 📞 Support

For support, please:
1. Check the documentation above
2. Open an issue on GitHub
3. Contact the development team

---

**⚠️ Remember: Bet responsibly and never risk more than you can afford to lose!**

🎾 **Built with ❤️ for tennis enthusiasts and data scientists**
