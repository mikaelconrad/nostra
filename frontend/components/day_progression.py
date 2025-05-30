"""
Day progression mechanics for the game
Handles advancing days, loading data, and updating game state
"""

import dash
from dash import callback, Input, Output, State
import pandas as pd
from datetime import datetime, timedelta

from frontend.game_state import game_instance
from backend.simple_data_collector import DataCollector


class DayProgressionManager:
    """Manages day-to-day progression in the game"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self._price_cache = {}
        self._recommendation_cache = {}
    
    def advance_to_next_day(self, current_game_state):
        """
        Advance game to next day and return updated state
        Returns: (updated_game_state, day_summary)
        """
        # Load game state
        game_instance.from_dict(current_game_state)
        
        # Record end-of-day portfolio value
        current_prices = self.get_prices_for_date(game_instance.current_date)
        game_instance.record_daily_value(
            current_prices['BTC'], 
            current_prices['ETH']
        )
        
        # Advance to next day
        success = game_instance.advance_day()
        
        if not success:
            return current_game_state, None
        
        # Load new day's data
        new_prices = self.get_prices_for_date(game_instance.current_date)
        recommendations = self.get_recommendations_for_date(game_instance.current_date)
        
        # Create day summary
        day_summary = self.create_day_summary(
            game_instance.current_date,
            current_prices,
            new_prices,
            recommendations
        )
        
        return game_instance.to_dict(), day_summary
    
    def get_prices_for_date(self, date):
        """Get cryptocurrency prices for a specific date"""
        if date in self._price_cache:
            return self._price_cache[date]
        
        prices = {}
        
        try:
            # Load BTC price
            btc_data = self.data_collector.load_historical_data('BTC-USD')
            btc_row = btc_data[btc_data['Date'] == date]
            if not btc_row.empty:
                prices['BTC'] = float(btc_row['Close'].iloc[0])
            
            # Load ETH price  
            eth_data = self.data_collector.load_historical_data('ETH-USD')
            eth_row = eth_data[eth_data['Date'] == date]
            if not eth_row.empty:
                prices['ETH'] = float(eth_row['Close'].iloc[0])
            
            self._price_cache[date] = prices
            
        except Exception as e:
            print(f"Error loading prices for {date}: {e}")
            prices = {'BTC': 0, 'ETH': 0}
        
        return prices
    
    def get_recommendations_for_date(self, date):
        """Get AI recommendations for a specific date"""
        if date in self._recommendation_cache:
            return self._recommendation_cache[date]
        
        recommendations = {}
        
        try:
            # Load recommendations
            for symbol in ['BTC-USD', 'ETH-USD']:
                recs = self.data_collector.load_recommendations(symbol)
                if recs:
                    # Find recommendation for date or use most recent
                    date_rec = None
                    for rec in recs:
                        if rec.get('date') == date:
                            date_rec = rec
                            break
                    
                    if not date_rec and recs:
                        # Use most recent recommendation before date
                        for rec in reversed(recs):
                            if rec.get('date') <= date:
                                date_rec = rec
                                break
                    
                    if date_rec:
                        crypto = 'BTC' if 'BTC' in symbol else 'ETH'
                        recommendations[crypto] = date_rec
            
            self._recommendation_cache[date] = recommendations
            
        except Exception as e:
            print(f"Error loading recommendations for {date}: {e}")
        
        return recommendations
    
    def create_day_summary(self, date, old_prices, new_prices, recommendations):
        """Create a summary of the day's market movements"""
        summary = {
            'date': date,
            'prices': new_prices,
            'changes': {},
            'recommendations': recommendations,
            'events': []
        }
        
        # Calculate price changes
        for crypto in ['BTC', 'ETH']:
            if crypto in old_prices and crypto in new_prices:
                old_price = old_prices[crypto]
                new_price = new_prices[crypto]
                change = new_price - old_price
                change_pct = (change / old_price * 100) if old_price > 0 else 0
                
                summary['changes'][crypto] = {
                    'amount': change,
                    'percentage': change_pct
                }
                
                # Add significant events
                if abs(change_pct) > 5:
                    summary['events'].append({
                        'type': 'price_movement',
                        'crypto': crypto,
                        'message': f"{crypto} {'surged' if change_pct > 0 else 'dropped'} {abs(change_pct):.1f}%!"
                    })
        
        # Add recommendation events
        for crypto, rec in recommendations.items():
            if rec.get('signal') in ['Buy', 'Sell']:
                summary['events'].append({
                    'type': 'recommendation',
                    'crypto': crypto,
                    'signal': rec['signal'],
                    'message': f"AI recommends: {rec['signal']} {crypto}"
                })
        
        return summary
    
    def preload_date_range(self, start_date, end_date):
        """Preload data for a date range to improve performance"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Load all historical data once
        btc_data = self.data_collector.load_historical_data('BTC-USD')
        eth_data = self.data_collector.load_historical_data('ETH-USD')
        
        # Cache prices for each day
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            prices = {}
            btc_row = btc_data[btc_data['Date'] == date_str]
            if not btc_row.empty:
                prices['BTC'] = float(btc_row['Close'].iloc[0])
            
            eth_row = eth_data[eth_data['Date'] == date_str]
            if not eth_row.empty:
                prices['ETH'] = float(eth_row['Close'].iloc[0])
            
            if prices:
                self._price_cache[date_str] = prices
            
            current_dt += timedelta(days=1)


# Global instance
day_progression = DayProgressionManager()


def register_progression_callbacks(app):
    """Register callbacks for day progression"""
    
    @app.callback(
        [Output('day-summary-modal', 'is_open'),
         Output('day-summary-content', 'children'),
         Output('game-state-store', 'data', allow_duplicate=True)],
        [Input('next-day-btn', 'n_clicks')],
        [State('game-state-store', 'data')],
        prevent_initial_call=True
    )
    def handle_next_day(n_clicks, game_state):
        """Handle next day button click"""
        if not n_clicks or not game_state:
            return False, "", dash.no_update
        
        # Advance to next day
        updated_state, day_summary = day_progression.advance_to_next_day(game_state)
        
        if not day_summary:
            return False, "", updated_state
        
        # Create day summary display
        summary_content = create_day_summary_display(day_summary)
        
        return True, summary_content, updated_state
    
    @app.callback(
        Output('price-data-store', 'data', allow_duplicate=True),
        [Input('game-state-store', 'data')],
        prevent_initial_call=True
    )
    def update_current_prices(game_state):
        """Update price data when game state changes"""
        if not game_state or game_state.get('state') != 'playing':
            return {}
        
        current_date = game_state.get('current_date')
        return day_progression.get_prices_for_date(current_date)


def create_day_summary_display(summary):
    """Create visual display of day summary"""
    import dash_bootstrap_components as dbc
    from dash import html
    
    content = []
    
    # Date header
    content.append(
        html.H4(
            f"Market Close: {summary['date']}", 
            className="text-center mb-3"
        )
    )
    
    # Price changes
    if summary['changes']:
        price_cards = []
        for crypto, change_data in summary['changes'].items():
            change_pct = change_data['percentage']
            color = 'success' if change_pct >= 0 else 'danger'
            icon = '📈' if change_pct >= 0 else '📉'
            
            price_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{crypto} {icon}"),
                            html.H4(
                                f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
                                className=f"text-{color}"
                            ),
                            html.P(
                                f"CHF {summary['prices'][crypto]:,.2f}",
                                className="mb-0"
                            )
                        ])
                    ])
                ], md=6)
            )
        
        content.append(dbc.Row(price_cards, className="mb-3"))
    
    # Events
    if summary['events']:
        content.append(html.H5("Today's Highlights", className="mt-3 mb-2"))
        
        for event in summary['events']:
            if event['type'] == 'price_movement':
                icon = '🚀' if 'surged' in event['message'] else '⚠️'
                color = 'success' if 'surged' in event['message'] else 'warning'
            else:  # recommendation
                icon = '🤖'
                color = 'info'
            
            content.append(
                dbc.Alert(
                    f"{icon} {event['message']}",
                    color=color,
                    className="mb-2"
                )
            )
    
    return html.Div(content)