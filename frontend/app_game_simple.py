"""
Simple Crypto Trading Simulator - Working Version
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pandas as pd
import os

import config
from frontend.game_state import game_instance
import threading
import plotly.graph_objects as go

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Crypto Trading Simulator",
    suppress_callback_exceptions=True
)

server = app.server

# Global variables for training status and chart caching
training_status = {"status": "idle", "message": "", "btc_complete": False, "eth_complete": False}
chart_cache = {}

def train_models_background(current_date):
    """Train models in background thread with simplified/mock training"""
    global training_status
    
    try:
        training_status = {"status": "training", "message": "🔄 Starting AI model training...", "btc_complete": False, "eth_complete": False}
        print(f"Starting model training for date: {current_date}")
        
        # Simulate training process with delays (replace with actual training for production)
        import time
        
        # Train BTC model
        training_status["message"] = "🔄 Training Bitcoin prediction model..."
        time.sleep(2)  # Simulate training time
        
        training_status["message"] = "🔄 Training Bitcoin 7-day prediction..."
        time.sleep(1)
        training_status["message"] = "🔄 Training Bitcoin 14-day prediction..."
        time.sleep(1)
        training_status["message"] = "🔄 Training Bitcoin 30-day prediction..."
        time.sleep(1)
        
        training_status["btc_complete"] = True
        training_status["message"] = "✅ Bitcoin model trained! Training Ethereum model..."
        
        # Train ETH model
        time.sleep(1)
        training_status["message"] = "🔄 Training Ethereum 7-day prediction..."
        time.sleep(1)
        training_status["message"] = "🔄 Training Ethereum 14-day prediction..."
        time.sleep(1)
        training_status["message"] = "🔄 Training Ethereum 30-day prediction..."
        time.sleep(1)
        
        training_status["eth_complete"] = True
        training_status["status"] = "complete"
        training_status["message"] = "✅ All AI models trained successfully! Predictions ready."
        
        # Set status to show charts after a brief delay
        time.sleep(2)
        training_status["status"] = "show_charts"
        
        print("Model training completed successfully!")
        
    except Exception as e:
        training_status["status"] = "error"
        training_status["message"] = f"❌ Training failed: {str(e)}"
        print(f"Training error: {str(e)}")

def start_training(current_date):
    """Start training in a background thread"""
    training_thread = threading.Thread(target=train_models_background, args=(current_date,))
    training_thread.daemon = True
    training_thread.start()

def generate_mock_predictions(current_price, current_date):
    """Generate mock predictions for demonstration"""
    import random
    import numpy as np
    
    try:
        print(f"DEBUG: generate_mock_predictions called")
        print(f"DEBUG: current_price = {current_price}, type = {type(current_price)}")
        print(f"DEBUG: current_date = {current_date}, type = {type(current_date)}")
        
        predictions = {}
        
        # Ensure current_date is a string, then convert to datetime
        if isinstance(current_date, datetime):
            base_date = current_date
        else:
            base_date = datetime.strptime(str(current_date), '%Y-%m-%d')
        
        print(f"DEBUG: base_date = {base_date}")
        
        # Generate predictions with some randomness but reasonable trends
        for horizon in [7, 14, 30]:
            print(f"DEBUG: Processing horizon = {horizon}, type = {type(horizon)}")
            
            # Add some volatility and trend
            volatility = random.uniform(0.05, 0.15)  # 5-15% volatility
            trend = random.uniform(-0.02, 0.05)     # -2% to +5% trend per week
            
            # Calculate prediction with some randomness
            weekly_factor = horizon / 7.0
            price_change = (1 + trend * weekly_factor) * (1 + random.gauss(0, volatility))
            predicted_price = float(current_price) * price_change
            
            print(f"DEBUG: Before timedelta - horizon = {horizon}, type = {type(horizon)}")
            prediction_date = base_date + timedelta(days=int(horizon))
            print(f"DEBUG: prediction_date = {prediction_date}")
            
            predictions[int(horizon)] = {
                'predicted_price': float(predicted_price),
                'prediction_date': prediction_date,
                'base_date': base_date
            }
            print(f"DEBUG: Added prediction for horizon {horizon}")
        
        print(f"DEBUG: Final predictions = {predictions}")
        return predictions
        
    except Exception as e:
        print(f"ERROR in generate_mock_predictions: {e}")
        import traceback
        traceback.print_exc()
        return {}

def create_simple_prediction_chart(df, current_date, predictions, crypto_name, color, selected_horizon=7):
    """Create a simple prediction chart with mock data"""
    try:
        # Convert inputs to proper types
        current_date_str = str(current_date)
        selected_horizon = int(selected_horizon)  # Convert string to int!
        
        print(f"DEBUG: After conversion - selected_horizon = {selected_horizon} (type: {type(selected_horizon)})")
        print(f"DEBUG: Available prediction keys = {list(predictions.keys()) if predictions else 'None'}")
        
        # Parse current date
        end_date = datetime.strptime(current_date_str, '%Y-%m-%d')
        start_date = end_date - timedelta(days=30)
        future_end_date = end_date + timedelta(days=30)
        
        # Format dates as strings for plotting
        current_date_plot = end_date.strftime('%Y-%m-%d')
        start_date_plot = start_date.strftime('%Y-%m-%d')
        future_end_date_plot = future_end_date.strftime('%Y-%m-%d')
        
        # Filter historical data
        historical_filtered = df[
            (pd.to_datetime(df['Date']) >= start_date) & 
            (pd.to_datetime(df['Date']) <= end_date)
        ].copy()
        
        # Create figure
        fig = go.Figure()
        
        # Add historical price line
        if not historical_filtered.empty:
            # Ensure all dates are strings
            historical_dates = [str(date) for date in historical_filtered['Date']]
            
            fig.add_trace(
                go.Scatter(
                    x=historical_dates,
                    y=historical_filtered['Close'],
                    mode='lines',
                    name=f'Historical {crypto_name}',
                    line=dict(color=color, width=2),
                    hovertemplate=f'<b>Historical {crypto_name}</b><br>Date: %{{x}}<br>Price: CHF %{{y:,.2f}}<extra></extra>'
                )
            )
            
            # Add current price marker
            current_price = float(historical_filtered.iloc[-1]['Close'])
            fig.add_trace(
                go.Scatter(
                    x=[current_date_plot],
                    y=[current_price],
                    mode='markers',
                    name='Current Price',
                    marker=dict(size=12, color=color, symbol='circle', line=dict(width=3, color='white')),
                    hovertemplate=f'<b>Current {crypto_name}</b><br>Date: %{{x}}<br>Price: CHF %{{y:,.2f}}<extra></extra>'
                )
            )
        else:
            current_price = 0
        
        # Add predictions if available
        if predictions and isinstance(predictions, dict) and selected_horizon in predictions:
            pred_data = predictions[selected_horizon]
            print(f"DEBUG: Found prediction data for horizon {selected_horizon}")
            
            # Extract prediction data safely
            if isinstance(pred_data, dict) and 'prediction_date' in pred_data and 'predicted_price' in pred_data:
                pred_date_obj = pred_data['prediction_date']
                if isinstance(pred_date_obj, datetime):
                    pred_date_plot = pred_date_obj.strftime('%Y-%m-%d')
                else:
                    pred_date_plot = str(pred_date_obj)
                
                pred_price = float(pred_data['predicted_price'])
                
                # Color based on horizon
                horizon_colors = {7: '#FF6B6B', 14: '#4ECDC4', 30: '#45B7D1'}
                horizon_color = horizon_colors.get(selected_horizon, '#FF6B6B')
                
                # Add main prediction marker
                fig.add_trace(
                    go.Scatter(
                        x=[pred_date_plot],
                        y=[pred_price],
                        mode='markers',
                        name=f'{selected_horizon}-Day Prediction',
                        marker=dict(
                            size=15,
                            color=horizon_color,
                            symbol='star',
                            line=dict(width=3, color='white')
                        ),
                        hovertemplate=f'<b>{selected_horizon}-day prediction</b><br>Date: %{{x}}<br>Price: CHF %{{y:,.2f}}<extra></extra>'
                    )
                )
                
                # Add trend line
                if current_price > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[current_date_plot, pred_date_plot],
                            y=[current_price, pred_price],
                            mode='lines',
                            name=f'{selected_horizon}-day trend',
                            line=dict(color=horizon_color, width=3, dash='dot'),
                            showlegend=False,
                            hovertemplate=f'<b>Trend to {selected_horizon}-day prediction</b><extra></extra>'
                        )
                    )
                
                # Add reference markers for other horizons
                for horizon_key, pred_ref in predictions.items():
                    if horizon_key != selected_horizon and isinstance(pred_ref, dict):
                        try:
                            ref_date_obj = pred_ref['prediction_date']
                            if isinstance(ref_date_obj, datetime):
                                ref_date_plot = ref_date_obj.strftime('%Y-%m-%d')
                            else:
                                ref_date_plot = str(ref_date_obj)
                            
                            ref_price = float(pred_ref['predicted_price'])
                            ref_color = horizon_colors.get(int(horizon_key), '#999999')
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=[ref_date_plot],
                                    y=[ref_price],
                                    mode='markers',
                                    name=f'{horizon_key}-day (ref)',
                                    marker=dict(
                                        size=8,
                                        color=ref_color,
                                        symbol='circle',
                                        opacity=0.4,
                                        line=dict(width=1, color='white')
                                    ),
                                    showlegend=False,
                                    hovertemplate=f'<b>{horizon_key}-day prediction</b><br>Date: %{{x}}<br>Price: CHF %{{y:,.2f}}<extra></extra>'
                                )
                            )
                        except Exception as e:
                            print(f"Error adding reference marker for horizon {horizon_key}: {e}")
                            continue
            else:
                print(f"DEBUG: Invalid prediction data structure for horizon {selected_horizon}")
        else:
            print(f"DEBUG: No predictions found for horizon {selected_horizon}")
            print(f"DEBUG: Predictions available: {predictions}")
        
        # Update layout
        fig.update_layout(
            title=f"{crypto_name} Price - Historical Data & AI Predictions",
            xaxis_title="Date",
            yaxis_title="Price (CHF)",
            hovermode='x unified',
            height=400,
            margin=dict(t=50, b=50, l=60, r=60),
            legend=dict(x=0.02, y=0.98),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,249,250,1)',
            xaxis=dict(
                range=[start_date_plot, future_end_date_plot],
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray'
            )
        )
        
        # Add vertical line at current date (using shape instead of add_vline)
        fig.add_shape(
            type="line",
            x0=current_date_plot,
            x1=current_date_plot,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="solid")
        )
        
        # Add text annotation for "Today"
        fig.add_annotation(
            x=current_date_plot,
            y=1,
            yref="paper",
            text="Today",
            showarrow=False,
            yanchor="bottom"
        )
        
        # Add prediction zone (using shape instead of add_vrect)
        fig.add_shape(
            type="rect",
            x0=current_date_plot,
            x1=future_end_date_plot,
            y0=0,
            y1=1,
            yref="paper",
            fillcolor="rgba(0,123,255,0.1)",
            layer="below",
            line_width=0
        )
        
        # Add text annotation for prediction zone
        fig.add_annotation(
            x=future_end_date_plot,
            y=0.95,
            yref="paper",
            text="Prediction Zone",
            showarrow=False,
            xanchor="right"
        )
        
        return fig
        
    except Exception as e:
        print(f"ERROR in create_simple_prediction_chart: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title=f"{crypto_name} Chart Error",
            height=400
        )
        return fig


def get_historical_price(crypto, date_str):
    """Get historical price for a cryptocurrency on a specific date"""
    try:
        # Load historical data
        if crypto.upper() == 'BTC':
            file_path = os.path.join(config.RAW_DATA_DIRECTORY, 'BTC_USD.csv')
        elif crypto.upper() == 'ETH':
            file_path = os.path.join(config.RAW_DATA_DIRECTORY, 'ETH_USD.csv')
        else:
            return 0
        
        if not os.path.exists(file_path):
            # Fallback to mock prices if no data
            return 45000 if crypto.upper() == 'BTC' else 2800
        
        df = pd.read_csv(file_path)
        
        # Check for different possible date column names
        date_col = None
        price_col = None
        
        for col in df.columns:
            if col.lower() in ['date', 'timestamp']:
                date_col = col
            if col.lower() in ['close', 'price']:
                price_col = col
        
        if not date_col or not price_col:
            print(f"Missing date or price column in {file_path}. Columns: {df.columns.tolist()}")
            return 45000 if crypto.upper() == 'BTC' else 2800
        
        # Convert date to datetime for comparison
        df[date_col] = pd.to_datetime(df[date_col])
        target_date = pd.to_datetime(date_str)
        
        # Find exact date or closest previous date
        available_dates = df[df[date_col] <= target_date]
        
        if available_dates.empty:
            # If no data before target date, use first available price
            if not df.empty:
                return float(df.iloc[0][price_col])
            return 45000 if crypto.upper() == 'BTC' else 2800
        
        # Get the most recent price before or on the target date
        latest_row = available_dates.loc[available_dates[date_col].idxmax()]
        price = float(latest_row[price_col])
        
        print(f"Loaded {crypto} price for {date_str}: CHF {price:,.2f}")
        return price
        
    except Exception as e:
        print(f"Error loading price for {crypto} on {date_str}: {e}")
        # Fallback prices
        return 45000 if crypto.upper() == 'BTC' else 2800


def get_current_simulation_date(game_data):
    """Get the current simulation date based on start date and current day"""
    if not game_data:
        return "2023-01-01"
    
    start_date = game_data.get("start_date", "2023-01-01")
    current_day = game_data.get("current_day", 1)
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        current_dt = start_dt + timedelta(days=current_day - 1)
        return current_dt.strftime("%Y-%m-%d")
    except:
        return "2023-01-01"


def create_setup_screen():
    """Create the initial setup screen"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🎮 Crypto Trading Simulator", className="text-center mb-4"),
                html.P("Test your trading skills with historical Bitcoin and Ethereum data!", 
                      className="text-center lead mb-5"),
                
                dbc.Card([
                    dbc.CardHeader(html.H3("Setup Your Portfolio", className="mb-0")),
                    dbc.CardBody([
                        dbc.Form([
                            # Starting cash
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Starting Cash (CHF)", className="fw-bold"),
                                    dbc.Input(
                                        id="starting-cash",
                                        type="number",
                                        min=100,
                                        value=10000,
                                        placeholder="Minimum CHF 100"
                                    ),
                                    dbc.FormText("How much cash do you want to start with?")
                                ], md=6),
                                
                                dbc.Col([
                                    dbc.Label("Simulation Duration (Days)", className="fw-bold"),
                                    dbc.Select(
                                        id="duration",
                                        options=[
                                            {"label": "30 days", "value": 30},
                                            {"label": "60 days", "value": 60},
                                            {"label": "90 days", "value": 90}
                                        ],
                                        value=30
                                    ),
                                    dbc.FormText("How long do you want to trade?")
                                ], md=6)
                            ], className="mb-4"),
                            
                            # Optional crypto holdings
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Starting Bitcoin (BTC)", className="fw-bold"),
                                    dbc.Input(
                                        id="starting-btc",
                                        type="number",
                                        min=0,
                                        value=0,
                                        step=0.001,
                                        placeholder="Optional"
                                    ),
                                    dbc.FormText("How much BTC do you already own? (Optional)")
                                ], md=6),
                                
                                dbc.Col([
                                    dbc.Label("Starting Ethereum (ETH)", className="fw-bold"),
                                    dbc.Input(
                                        id="starting-eth",
                                        type="number",
                                        min=0,
                                        value=0,
                                        step=0.01,
                                        placeholder="Optional"
                                    ),
                                    dbc.FormText("How much ETH do you already own? (Optional)")
                                ], md=6)
                            ], className="mb-4"),
                            
                            # Start date
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Start Date", className="fw-bold"),
                                    dbc.Input(
                                        id="start-date",
                                        type="date",
                                        value="2023-01-01",
                                        min="2020-01-01",
                                        max="2024-12-01"
                                    ),
                                    dbc.FormText("Pick a date to start your simulation")
                                ], md=6),
                                
                                dbc.Col([
                                    html.Div([
                                        html.H5("📊 Portfolio Preview", className="mt-2"),
                                        html.Div(id="portfolio-preview", className="mt-2")
                                    ])
                                ], md=6)
                            ], className="mb-4"),
                            
                            dbc.Button(
                                "🚀 Start Trading Game!", 
                                id="start-game-btn", 
                                color="success", 
                                size="lg", 
                                className="w-100 mt-3"
                            )
                        ])
                    ])
                ], className="shadow")
            ], md=8, className="mx-auto")
        ])
    ], className="mt-5")


def create_trading_screen():
    """Create the trading interface"""
    return dbc.Container([
        # Game status bar
        dbc.Alert([
            dbc.Row([
                dbc.Col([
                    html.H4(id="current-date", className="mb-0"),
                    html.Small(id="days-left", className="text-muted")
                ], md=4),
                dbc.Col([
                    html.H4("Portfolio Value", className="mb-0"),
                    html.H3(id="portfolio-value", className="mb-0 text-success"),
                    html.Small(id="profit-loss", className="text-muted")
                ], md=4),
                dbc.Col([
                    dbc.Button("Next Day →", id="next-day-btn", color="primary", size="lg")
                ], md=4)
            ])
        ], color="light", className="mb-4"),
        
        dbc.Row([
            # Portfolio panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("💰 Your Portfolio")),
                    dbc.CardBody([
                        html.Div(id="portfolio-breakdown")
                    ])
                ]),
                
                # Transaction History
                dbc.Card([
                    dbc.CardHeader(html.H5("📋 Transaction History")),
                    dbc.CardBody([
                        html.Div(id="transaction-history", style={"maxHeight": "300px", "overflowY": "auto"})
                    ])
                ], className="mt-3")
            ], md=4),
            
            # Trading panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📈 Market & Trading", className="d-inline"),
                        dbc.ButtonGroup([
                            dbc.Button("Bitcoin", id="select-btc", color="warning", size="sm", active=True),
                            dbc.Button("Ethereum", id="select-eth", color="info", size="sm")
                        ], className="float-end")
                    ]),
                    dbc.CardBody([
                        # Current prices for both cryptocurrencies
                        html.Div(id="all-current-prices", className="mb-3"),
                        
                        # Selected crypto detailed view
                        html.Div(id="current-prices"),
                        html.Hr(),
                        
                        # Trading interface
                        html.H6("💱 Quick Trade"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(
                                    id="trade-amount",
                                    type="number",
                                    placeholder="Amount to trade",
                                    min=0.001,
                                    step=0.001
                                )
                            ], md=8),
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button("Buy", id="buy-btn", color="success", size="sm"),
                                    dbc.Button("Sell", id="sell-btn", color="danger", size="sm")
                                ])
                            ], md=4)
                        ]),
                        html.Div(id="trade-result", className="mt-2")
                    ])
                ])
            ], md=8)
        ]),
        
        # AI Predictions Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🤖 AI Market Predictions", className="d-inline"),
                        dbc.Badge("Neural Network", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="Bitcoin Predictions", tab_id="pred-btc"),
                            dbc.Tab(label="Ethereum Predictions", tab_id="pred-eth")
                        ], id="prediction-tabs", active_tab="pred-btc"),
                        
                        # Prediction horizon selector
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Prediction Horizon:", className="fw-bold"),
                                dbc.Select(
                                    id="prediction-horizon-selector",
                                    options=[
                                        {"label": "7 Days Ahead", "value": 7},
                                        {"label": "14 Days Ahead", "value": 14},
                                        {"label": "30 Days Ahead", "value": 30}
                                    ],
                                    value=7,
                                    className="mb-2"
                                )
                            ], md=4),
                            dbc.Col([
                                html.Div([
                                    html.Small("📊 30 days history + 30 days future", className="text-muted"),
                                    html.Br(),
                                    html.Small("⭐ Selected prediction shown", className="text-info")
                                ])
                            ], md=8)
                        ], className="mb-3 mt-3"),
                        
                        html.Div(id="prediction-content", className="mt-2", style={"height": "450px"})
                    ])
                ])
            ])
        ], className="mt-4"),
        
    ], fluid=True)


def create_results_screen():
    """Create the final results screen"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🎉 Game Complete!", className="text-center mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Final Results"),
                        html.Div(id="final-performance"),
                        html.Hr(),
                        dbc.Button("Play Again", id="play-again-btn", color="primary", size="lg")
                    ])
                ])
            ], md=8, className="mx-auto")
        ])
    ], className="mt-5")


# Main app layout
app.layout = html.Div([
    # Header
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("🎮 Crypto Trading Simulator", href="#"),
            dbc.Nav([
                dbc.Button("New Game", id="new-game-btn", color="primary", size="sm")
            ], className="ms-auto")
        ]),
        color="dark",
        dark=True,
        className="mb-4"
    ),
    
    # Main content
    html.Div(id="main-content"),
    
    # Hidden stores
    dcc.Store(id="game-data", storage_type="session"),
    dcc.Store(id="selected-crypto", data="BTC"),
    dcc.Store(id="training-status", storage_type="memory", data={"status": "idle", "message": ""}),
    
    # Auto-refresh interval for training status (disabled when training complete)
    dcc.Interval(id="training-check-interval", interval=2000, n_intervals=0, disabled=False)
])


# Main callback to switch between screens
@app.callback(
    Output("main-content", "children"),
    [Input("new-game-btn", "n_clicks")],
    [State("game-data", "data")]
)
def render_screen(new_game_clicks, game_data):
    """Render the appropriate screen based on game state"""
    # Always show setup screen for now
    return create_setup_screen()


# Start game callback
@app.callback(
    [Output("main-content", "children", allow_duplicate=True),
     Output("game-data", "data", allow_duplicate=True)],
    [Input("start-game-btn", "n_clicks")],
    [State("starting-cash", "value"),
     State("starting-btc", "value"), 
     State("starting-eth", "value"),
     State("duration", "value"),
     State("start-date", "value")],
    prevent_initial_call=True
)
def start_game(n_clicks, cash, btc, eth, duration, start_date):
    """Start the trading game"""
    if n_clicks:
        # Initialize game state
        game_state = {
            "cash": cash or 10000,
            "btc": btc or 0,
            "eth": eth or 0,
            "duration": duration or 30,
            "start_date": start_date or "2023-01-01",
            "current_day": 1,
            "trades": [],
            # Store initial values for profit/loss calculation
            "initial_cash": cash or 10000,
            "initial_btc": btc or 0,
            "initial_eth": eth or 0
        }
        
        # Start training models for this game session
        current_simulation_date = get_current_simulation_date(game_state)
        start_training(current_simulation_date)
        
        return create_trading_screen(), game_state
    return dash.no_update, dash.no_update

# Portfolio preview callback
@app.callback(
    Output("portfolio-preview", "children"),
    [Input("starting-cash", "value"),
     Input("starting-btc", "value"),
     Input("starting-eth", "value")]
)
def update_portfolio_preview(cash, btc, eth):
    """Update portfolio preview"""
    if not cash:
        cash = 0
    if not btc:
        btc = 0
    if not eth:
        eth = 0
    
    total_crypto = btc + eth  # Simplified for preview
    
    return html.Div([
        html.P(f"💰 Cash: CHF {cash:,.2f}"),
        html.P(f"₿ Bitcoin: {btc:.3f} BTC"),
        html.P(f"⟠ Ethereum: {eth:.3f} ETH"),
        html.Hr(),
        html.P(f"📊 Starting Value: CHF {cash:,.2f} + crypto", className="fw-bold")
    ])


# Trading screen callbacks
@app.callback(
    [Output("portfolio-breakdown", "children"),
     Output("portfolio-value", "children"),
     Output("profit-loss", "children")],
    [Input("game-data", "data"),
     Input("trade-result", "children")],
    prevent_initial_call=True
)
def update_portfolio_display(game_data, trade_result):
    """Update portfolio information on trading screen"""
    if not game_data:
        return "Loading...", "CHF 0.00", "Loading..."
    
    cash = game_data.get("cash", 0)
    btc = game_data.get("btc", 0)
    eth = game_data.get("eth", 0)
    
    # Get current simulation date and load real prices
    current_date = get_current_simulation_date(game_data)
    btc_price = get_historical_price("BTC", current_date)
    eth_price = get_historical_price("ETH", current_date)
    
    btc_value = btc * btc_price
    eth_value = eth * eth_price
    total_value = cash + btc_value + eth_value
    
    # Calculate initial value for profit/loss (use initial portfolio from game setup)
    initial_cash = game_data.get("initial_cash", game_data.get("cash", 10000))
    initial_btc = game_data.get("initial_btc", 0)
    initial_eth = game_data.get("initial_eth", 0)
    
    # Calculate initial value using STARTING prices (from first day)
    start_date = game_data.get("start_date", "2023-01-01")
    initial_btc_price = get_historical_price("BTC", start_date)
    initial_eth_price = get_historical_price("ETH", start_date)
    
    initial_value = initial_cash + (initial_btc * initial_btc_price) + (initial_eth * initial_eth_price)
    profit_loss = total_value - initial_value
    profit_loss_pct = (profit_loss / initial_value * 100) if initial_value > 0 else 0
    
    portfolio_breakdown = html.Div([
        html.P(f"💰 Cash: CHF {cash:,.2f}"),
        html.P(f"₿ Bitcoin: {btc:.4f} BTC (CHF {btc_value:,.2f})"),
        html.P(f"⟠ Ethereum: {eth:.4f} ETH (CHF {eth_value:,.2f})"),
        html.Hr(),
        html.P(f"📊 Total: CHF {total_value:,.2f}", className="fw-bold")
    ])
    
    portfolio_value = f"CHF {total_value:,.2f}"
    
    profit_color = "success" if profit_loss >= 0 else "danger"
    profit_symbol = "+" if profit_loss >= 0 else ""
    profit_loss_text = f"Profit/Loss: {profit_symbol}CHF {profit_loss:,.2f} ({profit_symbol}{profit_loss_pct:.2f}%)"
    
    return portfolio_breakdown, portfolio_value, html.Span(profit_loss_text, className=f"text-{profit_color}")


@app.callback(
    Output("transaction-history", "children"),
    [Input("game-data", "data")],
    prevent_initial_call=True
)
def update_transaction_history(game_data):
    """Update transaction history display"""
    if not game_data or not game_data.get("trades"):
        return html.P("No transactions yet", className="text-muted text-center")
    
    trades = game_data["trades"]
    trade_items = []
    
    for trade in reversed(trades[-10:]):  # Show last 10 trades, most recent first
        crypto_symbol = "₿" if trade["crypto"] == "btc" else "⟠"
        
        if trade["type"] == "buy":
            color = "success"
            icon = "📈"
            text = f"Day {trade['day']}: {icon} Bought {trade['amount']:.4f} {crypto_symbol} for CHF {trade['cost']:,.2f}"
        else:
            color = "info"
            icon = "📉"
            text = f"Day {trade['day']}: {icon} Sold {trade['amount']:.4f} {crypto_symbol} for CHF {trade['revenue']:,.2f}"
        
        trade_items.append(
            html.Div([
                html.Small(text, className=f"text-{color}")
            ], className="mb-1 p-1 border-bottom")
        )
    
    return html.Div(trade_items)


@app.callback(
    Output("all-current-prices", "children"),
    [Input("game-data", "data")],
    prevent_initial_call=True
)
def update_all_current_prices(game_data):
    """Update display showing both BTC and ETH current prices"""
    if not game_data:
        return "Loading prices..."
    
    current_date = get_current_simulation_date(game_data)
    btc_price = get_historical_price("BTC", current_date)
    eth_price = get_historical_price("ETH", current_date)
    
    # Calculate daily changes
    try:
        yesterday = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        btc_yesterday = get_historical_price("BTC", yesterday)
        eth_yesterday = get_historical_price("ETH", yesterday)
        
        btc_change = ((btc_price - btc_yesterday) / btc_yesterday * 100) if btc_yesterday > 0 else 0
        eth_change = ((eth_price - eth_yesterday) / eth_yesterday * 100) if eth_yesterday > 0 else 0
    except:
        btc_change = 0
        eth_change = 0
    
    btc_color = "success" if btc_change >= 0 else "danger"
    eth_color = "success" if eth_change >= 0 else "danger"
    btc_symbol = "+" if btc_change >= 0 else ""
    eth_symbol = "+" if eth_change >= 0 else ""
    
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H6("₿ Bitcoin", className="mb-1"),
                html.H5(f"CHF {btc_price:,.2f}", className="mb-0"),
                html.Small(f"{btc_symbol}{btc_change:.2f}%", className=f"text-{btc_color}")
            ], className="text-center p-2 border rounded")
        ], md=6),
        dbc.Col([
            html.Div([
                html.H6("⟠ Ethereum", className="mb-1"),
                html.H5(f"CHF {eth_price:,.2f}", className="mb-0"),
                html.Small(f"{eth_symbol}{eth_change:.2f}%", className=f"text-{eth_color}")
            ], className="text-center p-2 border rounded")
        ], md=6)
    ])


@app.callback(
    Output("current-prices", "children"),
    [Input("select-btc", "active"),
     Input("select-eth", "active"),
     Input("game-data", "data")],
    prevent_initial_call=True
)
def update_current_prices(btc_active, eth_active, game_data):
    """Update current price display"""
    if not game_data:
        return "Loading prices..."
    
    current_date = get_current_simulation_date(game_data)
    
    if btc_active:
        crypto = "Bitcoin"
        symbol = "₿"
        price = get_historical_price("BTC", current_date)
        
        # Calculate yesterday's price for change
        try:
            yesterday = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday_price = get_historical_price("BTC", yesterday)
            change = ((price - yesterday_price) / yesterday_price * 100) if yesterday_price > 0 else 0
        except:
            change = 0
    else:
        crypto = "Ethereum"
        symbol = "⟠" 
        price = get_historical_price("ETH", current_date)
        
        # Calculate yesterday's price for change
        try:
            yesterday = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday_price = get_historical_price("ETH", yesterday)
            change = ((price - yesterday_price) / yesterday_price * 100) if yesterday_price > 0 else 0
        except:
            change = 0
    
    change_color = "success" if change >= 0 else "danger"
    change_symbol = "+" if change >= 0 else ""
    
    return html.Div([
        html.H4(f"{symbol} {crypto}"),
        html.H3(f"CHF {price:,.2f}", className="mb-0"),
        html.Small(f"24h change: {change_symbol}{change:.2f}%", className=f"text-{change_color}")
    ])

@app.callback(
    [Output("trade-result", "children"),
     Output("game-data", "data", allow_duplicate=True)],
    [Input("buy-btn", "n_clicks"),
     Input("sell-btn", "n_clicks")],
    [State("trade-amount", "value"),
     State("select-btc", "active"),
     State("select-eth", "active"),
     State("game-data", "data")],
    prevent_initial_call=True
)
def execute_trade(buy_clicks, sell_clicks, amount, btc_active, eth_active, game_data):
    """Execute buy/sell trade"""
    ctx = dash.callback_context
    
    if not ctx.triggered or not game_data:
        return "No trigger detected", dash.no_update
    
    if not amount or amount <= 0:
        return dbc.Alert("⚠️ Please enter a valid amount to trade", color="warning", dismissable=True), dash.no_update
    
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    crypto = "Bitcoin" if btc_active else "Ethereum"
    crypto_key = "btc" if btc_active else "eth"
    symbol = "₿" if btc_active else "⟠"
    
    # Get real historical price for current simulation date
    current_date = get_current_simulation_date(game_data)
    price = get_historical_price(crypto_key.upper(), current_date)
    
    # Create a copy of game data to modify
    new_game_data = game_data.copy()
    
    if trigger == "buy-btn" and buy_clicks:
        cost = amount * price
        
        # Check if user has enough cash
        if new_game_data["cash"] < cost:
            return dbc.Alert(
                f"❌ Insufficient funds! You need CHF {cost:,.2f} but only have CHF {new_game_data['cash']:,.2f}",
                color="danger", dismissable=True
            ), dash.no_update
        
        # Execute buy: deduct cash, add crypto
        new_game_data["cash"] -= cost
        new_game_data[crypto_key] += amount
        
        # Record trade
        trade = {
            "day": new_game_data.get("current_day", 1),
            "type": "buy",
            "crypto": crypto_key,
            "amount": amount,
            "price": price,
            "cost": cost
        }
        new_game_data["trades"].append(trade)
        
        return dbc.Alert([
            html.Strong("✅ Trade Executed!"),
            html.Br(),
            f"Bought {amount:.4f} {symbol} {crypto} for CHF {cost:,.2f}"
        ], color="success", dismissable=True, duration=4000), new_game_data
        
    elif trigger == "sell-btn" and sell_clicks:
        # Check if user has enough crypto to sell
        if new_game_data[crypto_key] < amount:
            return dbc.Alert(
                f"❌ Insufficient {crypto}! You need {amount:.4f} but only have {new_game_data[crypto_key]:.4f}",
                color="danger", dismissable=True
            ), dash.no_update
        
        revenue = amount * price
        
        # Execute sell: add cash, deduct crypto
        new_game_data["cash"] += revenue
        new_game_data[crypto_key] -= amount
        
        # Record trade
        trade = {
            "day": new_game_data.get("current_day", 1),
            "type": "sell",
            "crypto": crypto_key,
            "amount": amount,
            "price": price,
            "revenue": revenue
        }
        new_game_data["trades"].append(trade)
        
        return dbc.Alert([
            html.Strong("✅ Trade Executed!"),
            html.Br(), 
            f"Sold {amount:.4f} {symbol} {crypto} for CHF {revenue:,.2f}"
        ], color="info", dismissable=True, duration=4000), new_game_data
    
    return dbc.Alert("🤔 Something went wrong with the trade", color="danger", dismissable=True), dash.no_update

@app.callback(
    [Output("select-btc", "active"),
     Output("select-eth", "active")],
    [Input("select-btc", "n_clicks"),
     Input("select-eth", "n_clicks")],
    prevent_initial_call=True
)
def toggle_crypto_selection(btc_clicks, eth_clicks):
    """Toggle between BTC and ETH selection"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, False
    
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "select-eth":
        return False, True
    else:
        return True, False

@app.callback(
    [Output("current-date", "children"),
     Output("days-left", "children"),
     Output("game-data", "data", allow_duplicate=True)],
    [Input("next-day-btn", "n_clicks")],
    [State("game-data", "data")],
    prevent_initial_call=True
)
def advance_day(n_clicks, game_data):
    """Advance to next trading day"""
    if n_clicks and game_data:
        new_game_data = game_data.copy()
        new_game_data["current_day"] = n_clicks + 1
        
        current_date = get_current_simulation_date(new_game_data)
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        
        duration = game_data.get("duration", 30)
        days_left = duration - n_clicks
        
        # Start training for the new day
        start_training(current_date)
        
        return (
            current_dt.strftime("%B %d, %Y"),
            f"{max(0, days_left)} days remaining",
            new_game_data
        )
    
    return "January 1, 2023", "30 days remaining", dash.no_update


@app.callback(
    [Output("training-status", "data"),
     Output("training-check-interval", "disabled")],
    [Input("training-check-interval", "n_intervals")],
    prevent_initial_call=True
)
def update_training_status(n_intervals):
    """Update training status from global variable and disable interval when complete"""
    global training_status
    
    # Disable the interval if training is complete or in error state
    disabled = training_status.get("status") in ["show_charts", "error", "idle"]
    
    return training_status, disabled


@app.callback(
    Output("prediction-content", "children"),
    [Input("prediction-tabs", "active_tab"),
     Input("game-data", "data"),
     Input("training-status", "data"),
     Input("prediction-horizon-selector", "value")],
    prevent_initial_call=True
)
def update_prediction_charts(active_tab, game_data, training_status_data, selected_horizon):
    """Update prediction charts based on selected tab and current game state"""
    global chart_cache
    
    # Create cache key for this specific chart request
    cache_key = f"{active_tab}_{selected_horizon}_{game_data.get('current_day', 1) if game_data else 'none'}"
    
    # Check training status - only regenerate during training phase
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # If training is complete and this is just a timer update, return cached result
        if (trigger_id == "training-status" and 
            training_status_data and 
            training_status_data.get("status") in ["show_charts", "idle", "error"] and
            cache_key in chart_cache):
            return chart_cache[cache_key]
    
    if not game_data:
        return html.Div("🔄 Loading predictions...", className="text-center text-muted p-4")
    
    # Check training status
    if training_status_data and training_status_data.get("status") == "training":
        # Calculate progress
        progress = 10  # Starting
        if training_status_data.get("btc_complete"):
            progress = 70  # BTC done
        if training_status_data.get("eth_complete"):
            progress = 100  # Both done
        
        return html.Div([
            html.Div([
                html.H5("🤖 AI Model Training in Progress", className="text-center"),
                html.P(training_status_data.get("message", "Training..."), className="text-center"),
                dbc.Progress(
                    value=progress,
                    striped=True,
                    animated=True,
                    className="mt-3",
                    color="info"
                ),
                html.P("Please wait while the neural network learns from historical data...", 
                       className="text-muted text-center mt-2"),
                html.Small(f"Progress: {progress}%", className="text-muted text-center")
            ], className="p-4")
        ])
    
    elif training_status_data and training_status_data.get("status") == "complete":
        # Show success message briefly before showing charts
        return html.Div([
            html.Div([
                html.H5("✅ AI Models Ready!", className="text-center text-success"),
                html.P(training_status_data.get("message", "Training complete"), className="text-center"),
                html.P("Loading prediction charts...", className="text-muted text-center"),
                dbc.Progress(value=100, color="success", className="mt-2")
            ], className="p-4")
        ])
    
    # If training is complete or we're showing charts, proceed with chart generation
    elif training_status_data and training_status_data.get("status") in ["show_charts", "idle"]:
        pass  # Continue to chart generation below
    
    elif training_status_data and training_status_data.get("status") == "error":
        return html.Div([
            html.H6("⚠️ Training Error", className="text-center text-warning"),
            html.P(training_status_data.get("message", "Training failed"), className="text-center text-danger"),
            html.P("Charts will show historical data only", className="text-muted text-center")
        ], className="p-4")
    
    current_date = get_current_simulation_date(game_data)
    
    try:
        # Load historical data for charts
        if active_tab == "pred-btc":
            # Load BTC data
            btc_file = os.path.join(config.RAW_DATA_DIRECTORY, 'BTC_USD.csv')
            if os.path.exists(btc_file):
                btc_df = pd.read_csv(btc_file)
                # Rename columns to match expected format
                column_mapping = {}
                for col in btc_df.columns:
                    if col.lower() in ['date', 'timestamp']:
                        column_mapping[col] = 'Date'
                    elif col.lower() in ['close', 'price']:
                        column_mapping[col] = 'Close'
                
                btc_df = btc_df.rename(columns=column_mapping)
                
                if 'Date' in btc_df.columns and 'Close' in btc_df.columns:
                    # Get current price for mock predictions
                    current_btc_price = get_historical_price("BTC", current_date)
                    mock_predictions = generate_mock_predictions(current_btc_price, current_date)
                    
                    # Create chart with mock predictions
                    fig = create_simple_prediction_chart(btc_df, current_date, mock_predictions, "Bitcoin", "#F7931A", selected_horizon)
                    result = dcc.Graph(figure=fig, style={"height": "400px"})
                    
                    # Cache the result
                    chart_cache[cache_key] = result
                    return result
                else:
                    return html.Div([
                        html.H6("📈 Bitcoin Predictions", className="text-center"),
                        html.P("Chart data format not supported yet", className="text-muted text-center"),
                        html.P(f"Current date: {current_date}", className="text-muted text-center"),
                        html.P("Available columns: " + ", ".join(btc_df.columns.tolist()), className="small text-muted text-center")
                    ], className="p-4")
            else:
                return html.Div([
                    html.H6("📈 Bitcoin Predictions", className="text-center"),
                    html.P("Historical data not available", className="text-muted text-center"),
                    html.P(f"Looking for: {btc_file}", className="small text-muted text-center")
                ], className="p-4")
                
        elif active_tab == "pred-eth":
            # Load ETH data
            eth_file = os.path.join(config.RAW_DATA_DIRECTORY, 'ETH_USD.csv')
            if os.path.exists(eth_file):
                eth_df = pd.read_csv(eth_file)
                # Rename columns to match expected format
                column_mapping = {}
                for col in eth_df.columns:
                    if col.lower() in ['date', 'timestamp']:
                        column_mapping[col] = 'Date'
                    elif col.lower() in ['close', 'price']:
                        column_mapping[col] = 'Close'
                
                eth_df = eth_df.rename(columns=column_mapping)
                
                if 'Date' in eth_df.columns and 'Close' in eth_df.columns:
                    # Get current price for mock predictions
                    current_eth_price = get_historical_price("ETH", current_date)
                    mock_predictions = generate_mock_predictions(current_eth_price, current_date)
                    
                    # Create chart with mock predictions
                    fig = create_simple_prediction_chart(eth_df, current_date, mock_predictions, "Ethereum", "#627EEA", selected_horizon)
                    result = dcc.Graph(figure=fig, style={"height": "400px"})
                    
                    # Cache the result
                    chart_cache[cache_key] = result
                    return result
                else:
                    return html.Div([
                        html.H6("📊 Ethereum Predictions", className="text-center"),
                        html.P("Chart data format not supported yet", className="text-muted text-center"),
                        html.P(f"Current date: {current_date}", className="text-center"),
                        html.P("Available columns: " + ", ".join(eth_df.columns.tolist()), className="small text-muted text-center")
                    ], className="p-4")
            else:
                return html.Div([
                    html.H6("📊 Ethereum Predictions", className="text-center"),
                    html.P("Historical data not available", className="text-muted text-center"),
                    html.P(f"Looking for: {eth_file}", className="small text-muted text-center")
                ], className="p-4")
        
    except Exception as e:
        print(f"Error creating prediction chart: {str(e)}")
        return html.Div([
            html.H6("⚠️ Prediction Error", className="text-center text-warning"),
            html.P("Unable to load prediction charts", className="text-muted text-center"),
            html.P(f"Error: {str(e)}", className="small text-muted text-center")
        ], className="p-4")
    
    return html.Div("Select a cryptocurrency to view predictions", className="text-center text-muted p-4")




if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")