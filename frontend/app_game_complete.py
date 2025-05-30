"""
Crypto Trading Simulator - Complete Application
Single-page game interface with full functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

import config
from frontend.game_state import game_instance
from frontend.components import (
    create_setup_layout, register_setup_callbacks,
    create_portfolio_details, create_portfolio_chart,
    create_price_display, create_price_chart,
    create_indicators_content, create_history_content,
    register_trading_callbacks, register_market_callbacks,
    create_final_results_display, register_results_callbacks
)
from backend.simple_data_collector import DataCollector

# Initialize data collector
data_collector = DataCollector()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/game_style.css'],
    title="Crypto Trading Simulator",
    suppress_callback_exceptions=True
)

server = app.server  # For deployment

# Define colors
colors = {
    'background': '#F9F9F9',
    'text': '#333333',
    'primary': '#007BFF',
    'success': '#28A745',
    'danger': '#DC3545',
    'warning': '#FFC107',
    'btc': '#F7931A',
    'eth': '#627EEA',
    'cash': '#85bb65'
}


def create_game_layout():
    """Create the main game layout"""
    return html.Div([
        # Header
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("Crypto Trading Simulator", href="/"),
                dbc.Nav([
                    dbc.NavItem(
                        dbc.Button(
                            "New Game",
                            id="new-game-btn",
                            color="primary",
                            size="sm",
                            className="ms-2"
                        )
                    )
                ], className="ms-auto", navbar=True)
            ]),
            color="dark",
            dark=True,
            className="mb-4"
        ),
        
        # Main content area
        html.Div(id="game-content"),
        
        # Hidden stores for game state
        dcc.Store(id='game-state-store', storage_type='session'),
        dcc.Store(id='price-data-store', storage_type='memory'),
        dcc.Store(id='selected-crypto', data='BTC'),
        
        # Interval for auto-save
        dcc.Interval(id='auto-save-interval', interval=30000),  # 30 seconds
        
        # Hidden div for triggering callbacks
        html.Div(id='hidden-div', style={'display': 'none'})
    ])


def create_playing_layout():
    """Create the main playing interface"""
    return dbc.Container([
        # Top status bar
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(id="current-date-display", className="mb-0"),
                    html.Small(id="days-remaining", className="text-muted")
                ], className="text-center")
            ], md=4),
            dbc.Col([
                html.Div([
                    html.H4("Portfolio Value", className="mb-0"),
                    html.H3(id="portfolio-value", className="text-success mb-0"),
                    html.Small(id="portfolio-change", className="text-muted")
                ], className="text-center")
            ], md=4),
            dbc.Col([
                html.Div([
                    dbc.Button("Next Day →", id="next-day-btn", color="primary", size="lg", className="w-100")
                ])
            ], md=4)
        ], className="mb-4 p-3 bg-light rounded"),
        
        # Main content
        dbc.Row([
            # Left panel - Portfolio
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Your Portfolio", className="mb-0")),
                    dbc.CardBody([
                        html.Div(id="portfolio-details"),
                        html.Hr(),
                        html.Div(id="portfolio-chart", style={"height": "300px"})
                    ])
                ])
            ], md=3),
            
            # Center panel - Market & Trading
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Market", className="mb-0 d-inline"),
                        dbc.ButtonGroup([
                            dbc.Button("BTC", id="select-btc", color="warning", size="sm", active=True),
                            dbc.Button("ETH", id="select-eth", color="info", size="sm")
                        ], className="float-end")
                    ]),
                    dbc.CardBody([
                        # Price display
                        html.Div(id="current-price-display", className="mb-3"),
                        
                        # Price chart
                        dcc.Graph(id="price-chart", style={"height": "300px"}),
                        
                        # Trading interface
                        html.Hr(),
                        html.H6("Trade", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Buy", id="buy-btn", color="success", className="w-100")
                            ], width=6),
                            dbc.Col([
                                dbc.Button("Sell", id="sell-btn", color="danger", className="w-100")
                            ], width=6)
                        ], className="mb-3"),
                        
                        # Trade amount input
                        html.Div([
                            dbc.Label("Amount"),
                            dbc.Input(id="trade-amount", type="number", min=0, step=0.001),
                            html.Small(id="trade-preview", className="text-muted"),
                            dbc.Button("Execute Trade", id="execute-trade", color="primary", className="mt-2 w-100")
                        ], id="trade-controls", style={"display": "none"})
                    ])
                ])
            ], md=6),
            
            # Right panel - Analysis & History
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Indicators", tab_id="indicators"),
                    dbc.Tab(label="History", tab_id="history")
                ], id="right-tabs", active_tab="indicators"),
                html.Div(id="right-panel-content", className="mt-3")
            ], md=3)
        ]),
        
        # Trade confirmation modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Trade Executed")),
            dbc.ModalBody(id="trade-result-body"),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-trade-modal", className="ms-auto")
            ])
        ], id="trade-modal", is_open=False)
    ], fluid=True)


def create_completed_layout():
    """Create the game completion screen"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Simulation Complete!", className="text-center mb-4"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Final Results", className="mb-4"),
                        html.Div(id="final-results"),
                        
                        html.Hr(),
                        
                        html.H4("Performance Chart", className="mb-3"),
                        dcc.Graph(id="performance-chart"),
                        
                        html.Hr(),
                        
                        dbc.Button("Start New Game", id="new-game-final", color="primary", size="lg", className="mt-4")
                    ])
                ], className="shadow")
            ], md=8, className="mx-auto")
        ])
    ], className="mt-5")


# Set the app layout
app.layout = create_game_layout()


# Main callback to render content based on game state
@app.callback(
    Output('game-content', 'children'),
    [Input('game-state-store', 'data'),
     Input('new-game-btn', 'n_clicks'),
     Input('new-game-final', 'n_clicks')]
)
def render_game_content(game_data, new_game_clicks, new_game_final_clicks):
    """Render content based on current game state"""
    triggered = ctx.triggered_id
    
    # Check if new game button was clicked
    if triggered in ['new-game-btn', 'new-game-final']:
        game_instance.reset()
        return create_setup_layout()
    
    # Load game state if exists
    if game_data:
        game_instance.from_dict(game_data)
    
    # Render based on state
    if game_instance.state == config.GameState.SETUP:
        return create_setup_layout()
    elif game_instance.state == config.GameState.PLAYING:
        return create_playing_layout()
    elif game_instance.state == config.GameState.COMPLETED:
        return create_completed_layout()
    else:
        return create_setup_layout()


# Game state management callbacks
@app.callback(
    [Output('game-state-store', 'data'),
     Output('hidden-div', 'children')],
    [Input('start-game', 'n_clicks'),
     Input('next-day-btn', 'n_clicks'),
     Input('execute-trade', 'n_clicks')],
    [State('initial-cash', 'value'),
     State('initial-btc', 'value'),
     State('initial-eth', 'value'),
     State('start-date', 'value'),
     State('duration-select', 'value'),
     State('custom-duration', 'value'),
     State('trade-amount', 'value'),
     State('selected-crypto', 'data'),
     State('buy-btn', 'n_clicks'),
     State('sell-btn', 'n_clicks'),
     State('game-state-store', 'data')]
)
def update_game_state(start_clicks, next_day_clicks, execute_clicks,
                     cash, btc, eth, start_date, duration, custom_duration,
                     trade_amount, selected_crypto, buy_clicks, sell_clicks,
                     current_game_state):
    """Update game state based on user actions"""
    triggered = ctx.triggered_id
    
    if not triggered:
        return dash.no_update, dash.no_update
    
    # Load current state if exists
    if current_game_state:
        game_instance.from_dict(current_game_state)
    
    if triggered == 'start-game' and start_clicks:
        # Initialize game
        if duration == 'custom':
            days = custom_duration or 30
        else:
            days = int(duration)
        
        success = game_instance.initialize_game(
            start_date, days, 
            cash or 0, btc or 0, eth or 0
        )
        
        if success:
            return game_instance.to_dict(), ""
    
    elif triggered == 'next-day-btn' and next_day_clicks:
        # Advance to next day
        game_instance.advance_day()
        return game_instance.to_dict(), ""
    
    elif triggered == 'execute-trade' and execute_clicks and trade_amount:
        # Determine trade type
        trade_type = 'buy' if buy_clicks > sell_clicks else 'sell'
        
        # Get current price
        prices = load_prices_for_date(game_instance.current_date)
        price = prices.get(selected_crypto, 0)
        
        # Execute trade
        success, message = game_instance.execute_trade(
            trade_type, selected_crypto, trade_amount, price
        )
        
        return game_instance.to_dict(), message
    
    return dash.no_update, dash.no_update


# Price data loading callback
@app.callback(
    Output('price-data-store', 'data'),
    [Input('game-state-store', 'data')]
)
def load_price_data(game_data):
    """Load price data for current date"""
    if not game_data or game_data.get('state') != config.GameState.PLAYING:
        return {}
    
    current_date = game_data.get('current_date')
    return load_prices_for_date(current_date)


def load_prices_for_date(date):
    """Helper function to load prices for a specific date"""
    prices = {}
    
    try:
        # Load BTC price
        btc_data = data_collector.load_historical_data('BTC-USD')
        btc_price = btc_data[btc_data['Date'] == date]['Close'].iloc[0]
        prices['BTC'] = float(btc_price)
        
        # Load ETH price
        eth_data = data_collector.load_historical_data('ETH-USD')
        eth_price = eth_data[eth_data['Date'] == date]['Close'].iloc[0]
        prices['ETH'] = float(eth_price)
        
    except Exception as e:
        print(f"Error loading prices for {date}: {e}")
        prices = {'BTC': 0, 'ETH': 0}
    
    return prices


# Date and time display callbacks
@app.callback(
    [Output('current-date-display', 'children'),
     Output('days-remaining', 'children')],
    [Input('game-state-store', 'data')]
)
def update_date_display(game_data):
    """Update date and days remaining display"""
    if not game_data:
        return "", ""
    
    current_date = game_data.get('current_date')
    end_date = game_data.get('end_date')
    
    if current_date and end_date:
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_remaining = (end_dt - current_dt).days
        
        return current_dt.strftime('%B %d, %Y'), f"{days_remaining} days remaining"
    
    return "", ""


# Crypto selection callback
@app.callback(
    [Output('select-btc', 'active'),
     Output('select-eth', 'active'),
     Output('selected-crypto', 'data')],
    [Input('select-btc', 'n_clicks'),
     Input('select-eth', 'n_clicks')]
)
def update_crypto_selection(btc_clicks, eth_clicks):
    """Update selected cryptocurrency"""
    triggered = ctx.triggered_id
    
    if triggered == 'select-eth':
        return False, True, 'ETH'
    else:
        return True, False, 'BTC'


# Register component callbacks
register_setup_callbacks(app)
register_trading_callbacks(app, data_collector)
register_market_callbacks(app, data_collector)
register_results_callbacks(app)


if __name__ == '__main__':
    app.run(debug=config.DEBUG, port=config.FRONTEND_PORT)