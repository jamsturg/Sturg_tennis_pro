                predictions.append({
                    'match_id': match.get('match_id', ''),
                    'predicted_winner': 'unknown',
                    'home_win_prob': 0.5,
                    'away_win_prob': 0.5,
                    'confidence': 0.5,
                    'prediction_time': datetime.now().isoformat(),
                    'error': str(e)
                })
        
        predictions_df = pd.DataFrame(predictions)
        
        # Merge predictions with match data
        enhanced_matches = matches_df.merge(
            predictions_df, 
            on='match_id', 
            how='left'
        )
        
        return enhanced_matches
    
    def value_bet_detector(self, matches_df: pd.DataFrame, min_edge: float = 0.05) -> pd.DataFrame:
        """Detect value betting opportunities"""
        value_bets = []
        
        for _, match in matches_df.iterrows():
            try:
                home_model_prob = match.get('home_win_prob', 0.5)
                away_model_prob = match.get('away_win_prob', 0.5)
                home_implied_prob = match.get('home_implied_prob', 0.5)
                away_implied_prob = match.get('away_implied_prob', 0.5)
                
                # Calculate edges
                home_edge = home_model_prob - home_implied_prob
                away_edge = away_model_prob - away_implied_prob
                
                # Check for value bets
                if home_edge > min_edge:
                    value_bets.append({
                        'match_id': match.get('match_id', ''),
                        'selection': 'home',
                        'team': match.get('home_team', ''),
                        'edge': home_edge,
                        'model_prob': home_model_prob,
                        'market_prob': home_implied_prob,
                        'odds': match.get('home_odds', 2.0),
                        'confidence': match.get('confidence', 0.5),
                        'expected_value': (home_model_prob * match.get('home_odds', 2.0)) - 1
                    })
                
                if away_edge > min_edge:
                    value_bets.append({
                        'match_id': match.get('match_id', ''),
                        'selection': 'away',
                        'team': match.get('away_team', ''),
                        'edge': away_edge,
                        'model_prob': away_model_prob,
                        'market_prob': away_implied_prob,
                        'odds': match.get('away_odds', 2.0),
                        'confidence': match.get('confidence', 0.5),
                        'expected_value': (away_model_prob * match.get('away_odds', 2.0)) - 1
                    })
                    
            except Exception as e:
                logger.error(f"Error detecting value for match {match.get('match_id', 'unknown')}: {e}")
        
        return pd.DataFrame(value_bets)
    
    def automated_kelly_calculator(self, value_bets_df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
        """Calculate Kelly criterion bet sizes for value bets"""
        kelly_bets = []
        
        for _, bet in value_bets_df.iterrows():
            try:
                odds = bet.get('odds', 2.0)
                win_prob = bet.get('model_prob', 0.5)
                
                # Kelly formula: f = (bp - q) / b
                # where b = odds - 1, p = win probability, q = lose probability
                b = odds - 1
                p = win_prob
                q = 1 - p
                
                kelly_fraction = (b * p - q) / b if b > 0 else 0
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                
                bet_amount = bankroll * kelly_fraction
                
                kelly_bets.append({
                    'match_id': bet.get('match_id', ''),
                    'selection': bet.get('selection', ''),
                    'team': bet.get('team', ''),
                    'kelly_fraction': kelly_fraction,
                    'bet_amount': bet_amount,
                    'potential_profit': bet_amount * (odds - 1),
                    'edge': bet.get('edge', 0),
                    'expected_value': bet.get('expected_value', 0),
                    'risk_level': 'Low' if kelly_fraction < 0.05 else 'Medium' if kelly_fraction < 0.15 else 'High'
                })
                
            except Exception as e:
                logger.error(f"Error calculating Kelly for bet: {e}")
        
        return pd.DataFrame(kelly_bets)
    
    def risk_management_system(self, kelly_bets_df: pd.DataFrame, 
                             max_daily_risk: float = 0.1, 
                             max_single_bet: float = 0.05) -> pd.DataFrame:
        """Apply risk management rules to betting recommendations"""
        if kelly_bets_df.empty:
            return kelly_bets_df
        
        # Sort by expected value (descending)
        sorted_bets = kelly_bets_df.sort_values('expected_value', ascending=False).copy()
        
        # Apply risk limits
        total_risk = 0
        approved_bets = []
        
        for _, bet in sorted_bets.iterrows():
            kelly_fraction = bet.get('kelly_fraction', 0)
            
            # Check single bet limit
            if kelly_fraction > max_single_bet:
                kelly_fraction = max_single_bet
                bet['kelly_fraction'] = kelly_fraction
                bet['bet_amount'] = bet['bet_amount'] * (max_single_bet / bet.get('kelly_fraction', 1))
                bet['risk_adjusted'] = True
            
            # Check daily risk limit
            if total_risk + kelly_fraction <= max_daily_risk:
                total_risk += kelly_fraction
                bet['approved'] = True
                bet['cumulative_risk'] = total_risk
                approved_bets.append(bet)
            else:
                bet['approved'] = False
                bet['rejection_reason'] = 'Exceeds daily risk limit'
                approved_bets.append(bet)
        
        return pd.DataFrame(approved_bets)
    
    async def run_automation_cycle(self, sport_keys: List[str], model_path: str, 
                                 bankroll: float = 1000) -> Dict:
        """Run complete automation cycle"""
        cycle_start = datetime.now()
        logger.info(f"Starting automation cycle at {cycle_start}")
        
        try:
            # 1. Fetch live odds
            logger.info("Fetching live odds...")
            odds_data = await self.fetch_live_odds_async(sport_keys)
            
            # 2. Process data
            logger.info("Processing live data...")
            matches_df = self.process_live_data(odds_data)
            
            if matches_df.empty:
                logger.warning("No matches found")
                return {'status': 'no_matches', 'timestamp': cycle_start.isoformat()}
            
            # 3. Generate predictions
            logger.info(f"Generating predictions for {len(matches_df)} matches...")
            enhanced_matches = self.automated_prediction_pipeline(matches_df, model_path)
            
            # 4. Detect value bets
            logger.info("Detecting value betting opportunities...")
            value_bets = self.value_bet_detector(enhanced_matches, min_edge=0.03)
            
            # 5. Calculate Kelly criterion
            logger.info("Calculating optimal bet sizes...")
            kelly_bets = self.automated_kelly_calculator(value_bets, bankroll)
            
            # 6. Apply risk management
            logger.info("Applying risk management...")
            final_bets = self.risk_management_system(kelly_bets)
            
            # 7. Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            enhanced_matches.to_csv(f'automation_data/matches_{timestamp}.csv', index=False)
            value_bets.to_csv(f'automation_data/value_bets_{timestamp}.csv', index=False)
            final_bets.to_csv(f'automation_data/recommended_bets_{timestamp}.csv', index=False)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            
            results = {
                'status': 'success',
                'timestamp': cycle_start.isoformat(),
                'duration_seconds': cycle_duration,
                'matches_processed': len(enhanced_matches),
                'value_bets_found': len(value_bets),
                'recommended_bets': len(final_bets[final_bets.get('approved', False)]),
                'total_recommended_amount': final_bets[final_bets.get('approved', False)]['bet_amount'].sum(),
                'max_expected_value': final_bets['expected_value'].max() if not final_bets.empty else 0
            }
            
            logger.info(f"Automation cycle completed successfully in {cycle_duration:.2f}s")
            logger.info(f"Found {results['value_bets_found']} value bets, recommending {results['recommended_bets']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in automation cycle: {e}")
            return {
                'status': 'error',
                'timestamp': cycle_start.isoformat(),
                'error': str(e)
            }
    
    def start_continuous_automation(self, sport_keys: List[str], model_path: str, 
                                  bankroll: float = 1000, interval_minutes: int = 5):
        """Start continuous automation with specified interval"""
        logger.info(f"Starting continuous automation (every {interval_minutes} minutes)")
        
        while True:
            try:
                # Run automation cycle
                results = asyncio.run(self.run_automation_cycle(sport_keys, model_path, bankroll))
                
                # Log results
                logger.info(f"Cycle results: {results}")
                
                # Wait for next cycle
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Automation stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous automation: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


class AdvancedNotificationSystem:
    """Advanced notification system for automation alerts"""
    
    def __init__(self):
        self.notification_channels = []
    
    def add_email_notification(self, email: str, smtp_config: Dict):
        """Add email notification channel"""
        # Implementation for email notifications
        pass
    
    def add_webhook_notification(self, webhook_url: str):
        """Add webhook notification channel"""
        # Implementation for webhook notifications
        pass
    
    def send_value_bet_alert(self, bet_info: Dict):
        """Send alert for high-value betting opportunity"""
        message = f"""
        🎾 HIGH VALUE BET DETECTED! 🎾
        
        Match: {bet_info.get('team', 'Unknown')}
        Edge: {bet_info.get('edge', 0):.1%}
        Expected Value: {bet_info.get('expected_value', 0):.1%}
        Recommended Bet: ${bet_info.get('bet_amount', 0):.2f}
        Odds: {bet_info.get('odds', 'N/A')}
        
        Confidence: {bet_info.get('confidence', 0):.1%}
        Risk Level: {bet_info.get('risk_level', 'Unknown')}
        """
        
        logger.info(f"VALUE BET ALERT: {message}")
    
    def send_system_alert(self, alert_type: str, message: str):
        """Send system status alert"""
        logger.info(f"SYSTEM ALERT [{alert_type}]: {message}")


def main():
    """Main function for testing automation system"""
    # Example usage
    automation = TennisDataAutomation(api_key="your_api_key_here")
    
    sport_keys = ['tennis_atp_wimbledon', 'tennis_wta_wimbledon']
    model_path = 'trained_model.joblib'
    
    # Run single cycle
    print("Running single automation cycle...")
    results = asyncio.run(automation.run_automation_cycle(sport_keys, model_path))
    print(f"Results: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main()
