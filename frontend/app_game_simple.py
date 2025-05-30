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

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Crypto Trading Simulator",
    suppress_callback_exceptions=True
)

server = app.server


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
        ])
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
    dcc.Store(id="selected-crypto", data="BTC")
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
        
        return (
            current_dt.strftime("%B %d, %Y"),
            f"{max(0, days_left)} days remaining",
            new_game_data
        )
    
    return "January 1, 2023", "30 days remaining", dash.no_update


if __name__ == "__main__":
    app.run(debug=True, port=8052, host="0.0.0.0")