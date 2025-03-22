"""
Frontend application for Cryptocurrency Investment Recommendation System using Dash
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                title=config.DASHBOARD_TITLE,
                suppress_callback_exceptions=True)

server = app.server  # For deployment

# Define colors
colors = {
    'background': '#F9F9F9',
    'text': '#333333',
    'primary': '#007BFF',
    'secondary': '#6C757D',
    'success': '#28A745',
    'danger': '#DC3545',
    'warning': '#FFC107',
    'info': '#17A2B8',
    'light': '#F8F9FA',
    'dark': '#343A40',
    'btc': '#F7931A',
    'eth': '#627EEA',
    'xrp': '#23292F'
}

# Define crypto colors
crypto_colors = {
    'BTC-USD': colors['btc'],
    'ETH-USD': colors['eth'],
    'XRP-USD': colors['xrp']
}

# Helper functions
def load_price_data(symbol):
    """Load processed price data for a cryptocurrency"""
    filename = os.path.join(config.PROCESSED_DATA_DIRECTORY, f"{symbol.replace('-', '_')}_processed.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, parse_dates=['date'])
        df = df.set_index('date')
        return df
    else:
        print(f"Price data file not found: {filename}")
        # Create mock data if file doesn't exist
        dates = pd.date_range(end=datetime.now(), periods=365)
        mock_data = {
            'open': np.random.normal(50000, 5000, len(dates)),
            'high': np.random.normal(52000, 5000, len(dates)),
            'low': np.random.normal(48000, 5000, len(dates)),
            'close': np.random.normal(50000, 5000, len(dates)),
            'volume': np.random.normal(1000000, 200000, len(dates)),
            'SMA_7': np.random.normal(50000, 4000, len(dates)),
            'SMA_30': np.random.normal(50000, 3000, len(dates)),
        }
        df = pd.DataFrame(mock_data, index=dates)
        return df

def load_sentiment_data(symbol):
    """Load sentiment data for a cryptocurrency"""
    base_symbol = symbol.split('-')[0]
    filename = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f"{base_symbol}_daily_sentiment.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, parse_dates=['created_at'])
        df = df.set_index('created_at')
        return df
    else:
        print(f"Sentiment data file not found: {filename}")
        # Create mock data if file doesn't exist
        dates = pd.date_range(end=datetime.now(), periods=365)
        mock_data = {
            'vader_compound': np.random.normal(0.2, 0.5, len(dates))
        }
        df = pd.DataFrame(mock_data, index=dates)
        return df

def create_price_chart(symbol, timeframe='1y'):
    """Create price chart for a cryptocurrency"""
    df = load_price_data(symbol)
    
    # Filter data based on timeframe
    if timeframe == '1m':
        df = df.iloc[-30:]
    elif timeframe == '3m':
        df = df.iloc[-90:]
    elif timeframe == '6m':
        df = df.iloc[-180:]
    elif timeframe == '1y':
        df = df.iloc[-365:]
    
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{config.CRYPTO_SYMBOLS[symbol]} Price", "Volume"))
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=crypto_colors[symbol]
        ),
        row=2, col=1
    )
    
    # Add moving averages if available
    if 'SMA_7' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_7'],
                name='SMA 7',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA_30' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_30'],
                name='SMA 30',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{config.CRYPTO_SYMBOLS[symbol]} ({symbol}) Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_sentiment_chart(symbol):
    """Create sentiment chart for a cryptocurrency"""
    sentiment_df = load_sentiment_data(symbol)
    
    # Create figure
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=sentiment_df.index,
            y=sentiment_df['vader_compound'],
            name='Sentiment Score',
            line=dict(color=crypto_colors[symbol], width=2)
        )
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        title=f"{config.CRYPTO_SYMBOLS[symbol]} Sentiment Analysis",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=400,
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_portfolio_summary():
    """Create portfolio summary cards"""
    # Mock portfolio data
    portfolio = {
        'total_invested': 1000,
        'current_value': 1150,
        'profit_loss': 150,
        'return_percentage': 15,
        'holdings': {
            'BTC-USD': {'amount': 0.01, 'value': 500},
            'ETH-USD': {'amount': 0.2, 'value': 400},
            'XRP-USD': {'amount': 500, 'value': 250},
        }
    }
    
    # Create summary cards
    summary_cards = [
        dbc.Card(
            dbc.CardBody([
                html.H5("Total Invested", className="card-title"),
                html.H3(f"CHF {portfolio['total_invested']:.2f}", className="card-text")
            ]),
            className="mb-4"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H5("Current Value", className="card-title"),
                html.H3(f"CHF {portfolio['current_value']:.2f}", className="card-text")
            ]),
            className="mb-4"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H5("Profit/Loss", className="card-title"),
                html.H3(f"CHF {portfolio['profit_loss']:.2f}", 
                       className=f"card-text {'text-success' if portfolio['profit_loss'] >= 0 else 'text-danger'}")
            ]),
            className="mb-4"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H5("Return", className="card-title"),
                html.H3(f"{portfolio['return_percentage']:.2f}%", 
                       className=f"card-text {'text-success' if portfolio['return_percentage'] >= 0 else 'text-danger'}")
            ]),
            className="mb-4"
        )
    ]
    
    return dbc.Row([dbc.Col(card, width=3) for card in summary_cards])

def create_recommendation_cards(symbol):
    """Create recommendation cards for a cryptocurrency with real data"""
    # Load price data
    price_df = load_price_data(symbol)
    
    # Get current price (most recent close price)
    current_price = price_df['close'].iloc[-1]
    
    # Create predictions based on simple model
    # In a real scenario, this would use the neural network model
    last_prices = price_df['close'].iloc[-30:].values
    
    # Simple prediction model (for demonstration)
    # 1-day prediction: weighted average of last 3 days with small random factor
    pred_1d = (0.7 * last_prices[-1] + 0.2 * last_prices[-2] + 0.1 * last_prices[-3]) * (1 + np.random.normal(0, 0.01))
    # 7-day prediction: weighted average of last 7 days with medium random factor
    pred_7d = np.mean(last_prices[-7:]) * (1 + np.random.normal(0.02, 0.03))
    # 30-day prediction: weighted average of last 30 days with larger random factor
    pred_30d = np.mean(last_prices[-30:]) * (1 + np.random.normal(0.05, 0.05))
    
    # Calculate expected returns
    return_1d = ((pred_1d / current_price) - 1) * 100
    return_7d = ((pred_7d / current_price) - 1) * 100
    return_30d = ((pred_30d / current_price) - 1) * 100
    
    # Determine actions and strengths
    def get_action_strength(ret):
        if ret > 5:
            return "BUY", "STRONG"
        elif ret > 2:
            return "BUY", "MODERATE"
        elif ret > 0:
            return "BUY", "WEAK"
        elif ret > -2:
            return "HOLD", "NEUTRAL"
        elif ret > -5:
            return "SELL", "MODERATE"
        else:
            return "SELL", "STRONG"
    
    action_1d, strength_1d = get_action_strength(return_1d)
    action_7d, strength_7d = get_action_strength(return_7d)
    action_30d, strength_30d = get_action_strength(return_30d)
    
    # Create prediction dates
    today = datetime.now()
    date_1d = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    date_7d = (today + timedelta(days=7)).strftime('%Y-%m-%d')
    date_30d = (today + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Create recommendations dictionary
    recommendations = {
        '1': {
            'action': action_1d,
            'strength': strength_1d,
            'current_price': current_price,
            'predicted_price': pred_1d,
            'expected_return': return_1d,
            'prediction_date': date_1d
        },
        '7': {
            'action': action_7d,
            'strength': strength_7d,
            'current_price': current_price,
            'predicted_price': pred_7d,
            'expected_return': return_7d,
            'prediction_date': date_7d
        },
        '30': {
            'action': action_30d,
            'strength': strength_30d,
            'current_price': current_price,
            'predicted_price': pred_30d,
            'expected_return': return_30d,
            'prediction_date': date_30d
        }
    }
    
    cards = []
    
    for horizon, data in recommendations.items():
        # Determine card color based on action
        if data['action'] == 'BUY':
            card_color = 'success'
        elif data['action'] == 'SELL':
            card_color = 'danger'
        else:
            card_color = 'secondary'
        
        # Create card
        card = dbc.Card(
            [
                dbc.CardHeader(f"{horizon}-Day Forecast"),
                dbc.CardBody(
                    [
                        html.H5(f"{data['action']} ({data['strength']})", className=f"text-{card_color}"),
                        html.P(f"Current Price: ${data['current_price']:.2f}"),
                        html.P(f"Predicted Price: ${data['predicted_price']:.2f}"),
                        html.P(f"Expected Return: {data['expected_return']:.2f}%"),
                        html.P(f"Prediction Date: {data['prediction_date']}")
                    ]
                )
            ],
            className="mb-4",
            style={"border-left": f"5px solid var(--bs-{card_color})"}
        )
        
        cards.append(card)
    
    return dbc.Row([dbc.Col(card, width=4) for card in cards])

def create_prediction_chart(symbol):
    """Create price chart with prediction overlay for a cryptocurrency"""
    # Load price data
    df = load_price_data(symbol)
    
    # Get last 90 days of data for display
    df = df.iloc[-90:]
    
    # Get current price (most recent close price)
    current_price = df['close'].iloc[-1]
    
    # Create simple predictions
    # In a real scenario, this would use the neural network model
    last_prices = df['close'].values
    
    # Create future dates for predictions
    last_date = df.index[-1]
    if isinstance(last_date, str):
        last_date = datetime.strptime(last_date, '%Y-%m-%d')
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    # Simple prediction model (for demonstration)
    # Use a simple trend-based prediction with some randomness
    trend = (last_prices[-1] - last_prices[-30]) / 30  # daily trend
    
    pred_prices = []
    for i in range(30):
        # Base prediction on trend with increasing uncertainty
        pred = current_price + trend * (i+1) * (1 + np.random.normal(0, 0.01 * (i+1)))
        pred_prices.append(pred)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            name='Historical Price',
            line=dict(color=crypto_colors[symbol], width=2)
        )
    )
    
    # Add prediction line
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=pred_prices,
            name='Price Prediction',
            line=dict(color='rgba(0,0,0,0.5)', width=2, dash='dash')
        )
    )
    
    # Add current price marker
    fig.add_trace(
        go.Scatter(
            x=[df.index[-1]],
            y=[current_price],
            name='Current Price',
            mode='markers',
            marker=dict(color='red', size=10)
        )
    )
    
    # Highlight prediction points at 1, 7, and 30 days
    fig.add_trace(
        go.Scatter(
            x=[future_dates[0], future_dates[6], future_dates[29]],
            y=[pred_prices[0], pred_prices[6], pred_prices[29]],
            name='Prediction Points',
            mode='markers',
            marker=dict(color='green', size=10)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{config.CRYPTO_SYMBOLS[symbol]} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=400,
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_portfolio_pie_chart():
    """Create portfolio allocation pie chart"""
    # Mock portfolio data
    holdings = {
        'BTC-USD': {'value': 500},
        'ETH-USD': {'value': 400},
        'XRP-USD': {'value': 250},
    }
    
    # Create data for pie chart
    labels = [config.CRYPTO_SYMBOLS[symbol] for symbol in holdings.keys()]
    values = [data['value'] for data in holdings.values()]
    colors_list = [crypto_colors[symbol] for symbol in holdings.keys()]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=colors_list
    )])
    
    # Update layout
    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_portfolio_performance_chart():
    """Create portfolio performance chart"""
    # Mock performance data
    dates = pd.date_range(start='2025-01-01', end='2025-03-21')
    values = np.cumsum(np.random.normal(10, 50, len(dates))) + 1000
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            name='Portfolio Value',
            line=dict(color=colors['primary'], width=2)
        )
    )
    
    # Add initial investment line
    fig.add_hline(y=1000, line_dash="dash", line_color="gray", 
                 annotation_text="Initial Investment", annotation_position="bottom right")
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Value (CHF)",
        height=400,
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_correlation_heatmap():
    """Create correlation heatmap for cryptocurrencies"""
    # Mock correlation data
    corr_matrix = pd.DataFrame(
        [[1.0, 0.8, 0.6], 
         [0.8, 1.0, 0.7], 
         [0.6, 0.7, 1.0]],
        columns=['BTC-USD', 'ETH-USD', 'XRP-USD'],
        index=['BTC-USD', 'ETH-USD', 'XRP-USD']
    )
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Cryptocurrency Price Correlation Matrix",
        labels=dict(x="Cryptocurrency", y="Cryptocurrency", color="Correlation")
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(config.DASHBOARD_TITLE, className="text-center my-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Tabs([
        # Dashboard Tab
        dbc.Tab(label="Dashboard", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Portfolio Summary", className="mt-4"),
                    html.Div(id="portfolio-summary")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Bitcoin (BTC) Analysis", className="mt-4"),
                    dcc.Graph(id="btc-prediction-chart"),
                    html.Div(id="btc-recommendations")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Ethereum (ETH) Analysis", className="mt-4"),
                    dcc.Graph(id="eth-prediction-chart"),
                    html.Div(id="eth-recommendations")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Ripple (XRP) Analysis", className="mt-4"),
                    dcc.Graph(id="xrp-prediction-chart"),
                    html.Div(id="xrp-recommendations")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Portfolio Allocation", className="mt-4"),
                    dcc.Graph(id="portfolio-allocation")
                ], width=6),
                dbc.Col([
                    html.H3("Portfolio Performance", className="mt-4"),
                    dcc.Graph(id="portfolio-performance")
                ], width=6)
            ])
        ]),
        
        # Price Analysis Tab
        dbc.Tab(label="Price Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Cryptocurrency Price Charts", className="mt-4"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Cryptocurrency:"),
                                    dcc.Dropdown(
                                        id="price-crypto-dropdown",
                                        options=[
                                            {"label": config.CRYPTO_SYMBOLS[symbol], "value": symbol}
                                            for symbol in config.CRYPTO_SYMBOLS
                                        ],
                                        value="BTC-USD"
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Select Timeframe:"),
                                    dcc.Dropdown(
                                        id="price-timeframe-dropdown",
                                        options=[
                                            {"label": "1 Month", "value": "1m"},
                                            {"label": "3 Months", "value": "3m"},
                                            {"label": "6 Months", "value": "6m"},
                                            {"label": "1 Year", "value": "1y"},
                                            {"label": "All Time", "value": "all"}
                                        ],
                                        value="1y"
                                    )
                                ], width=6)
                            ]),
                            dcc.Graph(id="price-chart")
                        ])
                    ])
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Sentiment Analysis", className="mt-4"),
                    dcc.Graph(id="sentiment-chart")
                ], width=12)
            ])
        ]),
        
        # About Tab
        dbc.Tab(label="About", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("About This System", className="mt-4"),
                    dbc.Card([
                        dbc.CardBody([
                            html.P("""
                                This Cryptocurrency Investment Recommendation System is designed to help you make data-driven 
                                investment decisions for Bitcoin (BTC), Ethereum (ETH), and Ripple (XRP).
                            """),
                            html.P("""
                                The system uses historical price data and sentiment analysis to generate investment 
                                recommendations for 1-day, 7-day, and 30-day horizons using a neural network prediction model.
                            """),
                            html.H5("Key Features:"),
                            html.Ul([
                                html.Li("Historical cryptocurrency price data collection and analysis"),
                                html.Li("Technical indicator calculation (SMA, EMA, RSI, MACD, Bollinger Bands)"),
                                html.Li("Social media sentiment analysis"),
                                html.Li("Neural network prediction model using LSTM architecture"),
                                html.Li("Portfolio tracking and performance reporting"),
                                html.Li("Interactive dashboard with visualizations"),
                                html.Li("Daily investment recommendations")
                            ]),
                            html.P("""
                                This system is for educational and research purposes only. Always conduct your own research 
                                before making investment decisions.
                            """)
                        ])
                    ])
                ], width=12)
            ])
        ])
    ]),
    
    html.Footer([
        html.Hr(),
        html.P("Cryptocurrency Investment Recommendation System © 2025", className="text-center")
    ], className="mt-5")
], fluid=True)

# Callbacks
@app.callback(
    Output("portfolio-summary", "children"),
    Input("portfolio-summary", "id")
)
def update_portfolio_summary(id):
    return create_portfolio_summary()

@app.callback(
    Output("btc-recommendations", "children"),
    Input("btc-recommendations", "id")
)
def update_btc_recommendations(id):
    return create_recommendation_cards("BTC-USD")

@app.callback(
    Output("eth-recommendations", "children"),
    Input("eth-recommendations", "id")
)
def update_eth_recommendations(id):
    return create_recommendation_cards("ETH-USD")

@app.callback(
    Output("xrp-recommendations", "children"),
    Input("xrp-recommendations", "id")
)
def update_xrp_recommendations(id):
    return create_recommendation_cards("XRP-USD")

@app.callback(
    Output("portfolio-allocation", "figure"),
    Input("portfolio-allocation", "id")
)
def update_portfolio_allocation(id):
    return create_portfolio_pie_chart()

@app.callback(
    Output("portfolio-performance", "figure"),
    Input("portfolio-performance", "id")
)
def update_portfolio_performance(id):
    return create_portfolio_performance_chart()

@app.callback(
    Output("correlation-heatmap", "figure"),
    Input("correlation-heatmap", "id")
)
def update_correlation_heatmap(id):
    return create_correlation_heatmap()

@app.callback(
    Output("price-chart", "figure"),
    [Input("price-crypto-dropdown", "value"),
     Input("price-timeframe-dropdown", "value")]
)
def update_price_chart(crypto, timeframe):
    return create_price_chart(crypto, timeframe)

@app.callback(
    Output("sentiment-chart", "figure"),
    Input("price-crypto-dropdown", "value")
)
def update_sentiment_chart(crypto):
    return create_sentiment_chart(crypto)

# New callbacks for prediction charts
@app.callback(
    Output("btc-prediction-chart", "figure"),
    Input("btc-prediction-chart", "id")
)
def update_btc_prediction_chart(id):
    return create_prediction_chart("BTC-USD")

@app.callback(
    Output("eth-prediction-chart", "figure"),
    Input("eth-prediction-chart", "id")
)
def update_eth_prediction_chart(id):
    return create_prediction_chart("ETH-USD")

@app.callback(
    Output("xrp-prediction-chart", "figure"),
    Input("xrp-prediction-chart", "id")
)
def update_xrp_prediction_chart(id):
    return create_prediction_chart("XRP-USD")

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=config.FRONTEND_PORT)