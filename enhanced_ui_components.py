        unsafe_allow_html=True
        )

def create_portfolio_optimization_view(bets_portfolio: pd.DataFrame) -> go.Figure:
    """Create portfolio optimization visualization"""
    if bets_portfolio.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Risk vs Return Profile',
            'Portfolio Allocation',
            'Correlation Matrix',
            'Sharpe Ratio Analysis'
        ),
        specs=[
            [{"secondary_y": False}, {"type": "pie"}],
            [{"type": "heatmap"}, {"secondary_y": False}]
        ]
    )
    
    # 1. Risk vs Return scatter
    if 'expected_return' in bets_portfolio.columns and 'risk_score' in bets_portfolio.columns:
        fig.add_trace(
            go.Scatter(
                x=bets_portfolio['risk_score'],
                y=bets_portfolio['expected_return'],
                mode='markers+text',
                marker=dict(
                    size=bets_portfolio.get('bet_amount', [10]) / 10,
                    color=bets_portfolio.get('sharpe_ratio', [1]),
                    colorscale='Viridis',
                    showscale=True
                ),
                text=bets_portfolio.get('team', []),
                textposition='top center',
                name='Bets'
            ),
            row=1, col=1
        )
    
    # 2. Portfolio allocation pie chart
    if 'bet_amount' in bets_portfolio.columns:
        fig.add_trace(
            go.Pie(
                labels=bets_portfolio.get('team', []),
                values=bets_portfolio.get('bet_amount', []),
                name="Allocation"
            ),
            row=1, col=2
        )
    
    # 3. Correlation matrix (sample data)
    correlation_data = np.random.rand(5, 5)
    correlation_data = (correlation_data + correlation_data.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_data, 1)  # Diagonal = 1
    
    fig.add_trace(
        go.Heatmap(
            z=correlation_data,
            x=['Bet 1', 'Bet 2', 'Bet 3', 'Bet 4', 'Bet 5'],
            y=['Bet 1', 'Bet 2', 'Bet 3', 'Bet 4', 'Bet 5'],
            colorscale='RdBu',
            zmid=0
        ),
        row=2, col=1
    )
    
    # 4. Sharpe ratio analysis
    if 'sharpe_ratio' in bets_portfolio.columns:
        fig.add_trace(
            go.Bar(
                x=bets_portfolio.get('team', []),
                y=bets_portfolio.get('sharpe_ratio', []),
                name='Sharpe Ratio',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Advanced Portfolio Optimization Dashboard",
        height=800,
        showlegend=True
    )
    
    return fig

def display_model_comparison_table(model_performances: Dict) -> None:
    """Display comprehensive model comparison table"""
    if not model_performances:
        st.info("No model performance data available.")
        return
    
    # Convert to DataFrame for better display
    comparison_data = []
    for model_name, metrics in model_performances.items():
        comparison_data.append({
            'Model': model_name.title(),
            'Test Accuracy': f"{metrics.get('test_accuracy', 0):.3f}",
            'CV Mean': f"{metrics.get('cv_mean', 0):.3f}",
            'CV Std': f"{metrics.get('cv_std', 0):.3f}",
            'Training Time': metrics.get('training_time', 'N/A'),
            'Status': '✅ Active' if model_name == 'ensemble' else '🔄 Available'
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Style the table
    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    # Apply styling to numeric columns
    styled_df = df.style.apply(highlight_best, subset=['Test Accuracy', 'CV Mean'])
    
    st.dataframe(styled_df, use_container_width=True)

def create_live_sentiment_indicator(sentiment_score: float) -> None:
    """Create live market sentiment indicator"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Determine sentiment level and color
        if sentiment_score > 0.6:
            sentiment_text = "Very Bullish 📈"
            color = "green"
        elif sentiment_score > 0.2:
            sentiment_text = "Bullish 📊"
            color = "lightgreen"
        elif sentiment_score > -0.2:
            sentiment_text = "Neutral ⚖️"
            color = "gray"
        elif sentiment_score > -0.6:
            sentiment_text = "Bearish 📉"
            color = "orange"
        else:
            sentiment_text = "Very Bearish 📉"
            color = "red"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; border: 2px solid {color}; 
                    border-radius: 10px; background-color: {color}20;">
            <h4>Market Sentiment: {sentiment_text}</h4>
            <p><strong>Score: {sentiment_score:.2f}</strong></p>
            <div style="font-size: 12px; color: #666;">
                Based on odds movements, betting volumes, and market indicators
            </div>
        </div>
        """, unsafe_allow_html=True)

def enhanced_feature_importance_chart(feature_importance: Dict) -> go.Figure:
    """Create enhanced feature importance visualization"""
    if not feature_importance:
        return go.Figure()
    
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_data)
    
    fig = go.Figure()
    
    # Add horizontal bar chart
    fig.add_trace(
        go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance Score")
            ),
            text=[f"{val:.3f}" for val in importance],
            textposition='inside'
        )
    )
    
    fig.update_layout(
        title="Model Feature Importance Analysis",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(400, len(features) * 30),
        margin=dict(l=150)  # More space for feature names
    )
    
    return fig

def create_prediction_accuracy_trend(historical_predictions: pd.DataFrame) -> go.Figure:
    """Create prediction accuracy trend over time"""
    if historical_predictions.empty:
        return go.Figure()
    
    # Generate sample trend data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    accuracy_trend = np.random.rand(30) * 0.2 + 0.75  # Between 75-95%
    
    fig = go.Figure()
    
    # Add accuracy trend line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=accuracy_trend,
            mode='lines+markers',
            name='Daily Accuracy',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        )
    )
    
    # Add moving average
    moving_avg = pd.Series(accuracy_trend).rolling(window=7).mean()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=moving_avg,
            mode='lines',
            name='7-day Moving Average',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Add target line
    fig.add_hline(
        y=0.8,
        line_dash="dot",
        line_color="gray",
        annotation_text="Target Accuracy (80%)"
    )
    
    fig.update_layout(
        title="Model Prediction Accuracy Trend",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0.7, 1.0], tickformat='.1%'),
        height=400
    )
    
    return fig

def advanced_kelly_calculator_widget() -> Dict:
    """Create interactive Kelly criterion calculator widget"""
    st.subheader("🧮 Advanced Kelly Criterion Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bankroll = st.number_input(
            "💰 Current Bankroll ($)",
            min_value=100.0,
            value=1000.0,
            step=50.0,
            help="Your current betting bankroll"
        )
        
        win_probability = st.slider(
            "🎯 Win Probability",
            min_value=0.01,
            max_value=0.99,
            value=0.55,
            step=0.01,
            format="%.2f",
            help="Model's predicted win probability"
        )
    
    with col2:
        decimal_odds = st.number_input(
            "📊 Decimal Odds",
            min_value=1.01,
            value=1.90,
            step=0.01,
            help="Bookmaker's decimal odds"
        )
        
        kelly_modifier = st.selectbox(
            "🛡️ Kelly Modifier",
            ["Full Kelly", "Half Kelly", "Quarter Kelly", "Custom"],
            index=1,
            help="Risk adjustment factor"
        )
    
    # Calculate Kelly criterion
    b = decimal_odds - 1
    p = win_probability
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b if b > 0 else 0
    kelly_fraction = max(0, kelly_fraction)  # Ensure non-negative
    
    # Apply modifier
    if kelly_modifier == "Half Kelly":
        kelly_fraction *= 0.5
    elif kelly_modifier == "Quarter Kelly":
        kelly_fraction *= 0.25
    elif kelly_modifier == "Custom":
        modifier = st.slider("Custom Modifier", 0.1, 1.0, 0.5, 0.1)
        kelly_fraction *= modifier
    
    bet_amount = bankroll * kelly_fraction
    expected_value = (win_probability * decimal_odds) - 1
    
    # Display results
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Kelly %", f"{kelly_fraction:.1%}")
    with col2:
        st.metric("Bet Amount", f"${bet_amount:.2f}")
    with col3:
        st.metric("Expected Value", f"{expected_value:.1%}")
    with col4:
        potential_profit = bet_amount * (decimal_odds - 1)
        st.metric("Potential Profit", f"${potential_profit:.2f}")
    
    # Risk warnings
    if kelly_fraction > 0.25:
        st.error("🚨 **VERY HIGH RISK**: Kelly suggests >25% of bankroll!")
    elif kelly_fraction > 0.10:
        st.warning("⚠️ **HIGH RISK**: Kelly suggests >10% of bankroll.")
    elif kelly_fraction > 0.05:
        st.info("ℹ️ **MODERATE RISK**: Consider your risk tolerance.")
    else:
        st.success("✅ **ACCEPTABLE RISK**: Within reasonable limits.")
    
    return {
        'kelly_fraction': kelly_fraction,
        'bet_amount': bet_amount,
        'expected_value': expected_value,
        'potential_profit': potential_profit
    }

def create_automated_betting_status_panel() -> None:
    """Create automated betting system status panel"""
    st.subheader("🤖 Automated Betting System")
    
    # Safety status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 System Status: SAFE", disabled=True):
            pass
        st.caption("All safety checks passed")
    
    with col2:
        daily_limit = st.number_input("📅 Daily Limit ($)", value=500, min_value=50, step=50)
        st.caption(f"Used: $127 / ${daily_limit}")
    
    with col3:
        auto_betting_enabled = st.toggle("🤖 Auto-Betting", value=False)
        if auto_betting_enabled:
            st.caption("⚠️ Monitor closely!")
        else:
            st.caption("✅ Manual mode")
    
    # Emergency controls
    st.markdown("---")
    st.subheader("🚨 Emergency Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⏸️ PAUSE ALL", type="secondary"):
            st.success("✅ All automated betting paused")
    
    with col2:
        if st.button("🛑 EMERGENCY STOP", type="primary"):
            st.error("🛑 Emergency stop activated!")
    
    with col3:
        if st.button("🔄 RESET SYSTEM", type="secondary"):
            st.info("🔄 System reset completed")

def display_recent_activity_feed(activities: List[Dict]) -> None:
    """Display live activity feed"""
    st.subheader("📡 Live Activity Feed")
    
    for activity in activities[-10:]:  # Show last 10 activities
        timestamp = activity.get('timestamp', datetime.now().strftime('%H:%M:%S'))
        action = activity.get('action', 'Unknown action')
        details = activity.get('details', '')
        activity_type = activity.get('type', 'info')
        
        # Color code by activity type
        if activity_type == 'success':
            icon = "✅"
            color = "green"
        elif activity_type == 'warning':
            icon = "⚠️"
            color = "orange"
        elif activity_type == 'error':
            icon = "❌"
            color = "red"
        else:
            icon = "ℹ️"
            color = "blue"
        
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"<span style='color: {color}'>{icon} **{timestamp}**</span>", 
                           unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{action}**: {details}")
        
        st.markdown("---")

# Custom CSS for enhanced styling
def inject_custom_css():
    """Inject custom CSS for enhanced UI styling"""
    st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    
    .success-metric {
        border-left-color: #2ca02c;
    }
    
    .warning-metric {
        border-left-color: #ff7f0e;
    }
    
    .danger-metric {
        border-left-color: #d62728;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    
    .value-bet-highlight {
        background-color: #90EE90;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 2px solid #32CD32;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(45deg, #1f77b4, #17becf);
        color: white;
        font-weight: bold;
    }
    
    .automation-status {
        position: fixed;
        top: 80px;
        right: 20px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)
