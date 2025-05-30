"""
Frontend application for Cryptocurrency Investment Recommendation System using Dash
This version uses the REST API instead of direct file access
"""

import sys
import os
import requests
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

# API base URL
API_BASE_URL = f"http://localhost:{config.API_PORT}/api"

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
}

# Define crypto colors
crypto_colors = {
    'BTC': colors['btc'],
    'ETH': colors['eth'],
}

# API Helper functions
def api_get(endpoint):
    """Make GET request to API"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API GET error: {str(e)}")
        return None

def api_post(endpoint, data=None):
    """Make POST request to API"""
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API POST error: {str(e)}")
        return None

# Data loading functions
def load_price_data(symbol, days=365):
    """Load price data for a cryptocurrency via API"""
    data = api_get(f"crypto/{symbol}/data?days={days}")
    if data and 'data' in data:
        df = pd.DataFrame(data['data'])
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df
    return pd.DataFrame()

def load_portfolio_data():
    """Load portfolio data via API"""
    return api_get("portfolio")

def load_recommendations():
    """Load recommendations via API"""
    return api_get("recommendations")

def load_sentiment_data(symbol, days=30):
    """Load sentiment data via API"""
    data = api_get(f"sentiment/{symbol}?days={days}")
    if data and 'sentiment' in data:
        df = pd.DataFrame(data['sentiment'])
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df
    return pd.DataFrame()

# Chart creation functions
def create_price_chart(symbol):
    """Create price chart for a cryptocurrency"""
    df = load_price_data(symbol)
    
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'] if 'open' in df else None,
        high=df['high'] if 'high' in df else None,
        low=df['low'] if 'low' in df else None,
        close=df['close'] if 'close' in df else None,
        name=symbol,
        showlegend=False
    ))
    
    # Add moving averages if available
    if 'sma_7' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_7'],
            name='SMA 7',
            line=dict(color='orange', width=1)
        ))
    
    if 'sma_30' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_30'],
            name='SMA 30',
            line=dict(color='blue', width=1)
        ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (CHF)",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_portfolio_pie_chart():
    """Create portfolio allocation pie chart"""
    data = load_portfolio_data()
    
    if not data or 'holdings' not in data:
        return go.Figure().add_annotation(
            text="No portfolio data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    holdings = data['holdings']
    
    # Prepare data for pie chart
    labels = []
    values = []
    colors_list = []
    
    for symbol, info in holdings.items():
        labels.append(symbol.upper())
        if symbol == 'cash':
            values.append(info)
            colors_list.append(colors['secondary'])
        else:
            values.append(info.get('value', 0))
            colors_list.append(crypto_colors.get(symbol, colors['primary']))
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=colors_list
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
        showlegend=True
    )
    
    return fig

def create_sentiment_chart(symbol):
    """Create sentiment analysis chart"""
    df = load_sentiment_data(symbol)
    
    if df.empty:
        return go.Figure().add_annotation(
            text="No sentiment data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    if 'sentiment_score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sentiment_score'],
            name='Sentiment Score',
            line=dict(color=crypto_colors.get(symbol, colors['primary']))
        ))
    
    fig.update_layout(
        title=f"{symbol} Sentiment Analysis",
        xaxis_title="Date",
        yaxis_title="Sentiment Score (0-100)",
        template="plotly_white",
        height=400,
        yaxis_range=[0, 100]
    )
    
    return fig

# Layout components
def create_navbar():
    """Create navigation bar"""
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(config.DASHBOARD_TITLE, className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Dashboard", href="/", active=True)),
                dbc.NavItem(dbc.NavLink("Portfolio", href="/portfolio")),
                dbc.NavItem(dbc.NavLink("Analysis", href="/analysis")),
                dbc.NavItem(dbc.NavLink("Settings", href="/settings")),
            ], className="ms-auto", navbar=True),
        ]),
        color="dark",
        dark=True,
        className="mb-4"
    )

def create_metrics_cards():
    """Create metrics cards for portfolio overview"""
    data = load_portfolio_data()
    
    if not data or 'metrics' not in data:
        return html.Div("No portfolio data available")
    
    metrics = data['metrics']
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Invested", className="text-muted"),
                    html.H3(f"CHF {metrics.get('total_invested', 0):,.2f}")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Current Value", className="text-muted"),
                    html.H3(f"CHF {metrics.get('current_value', 0):,.2f}")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Return", className="text-muted"),
                    html.H3(
                        f"CHF {metrics.get('total_return', 0):,.2f}",
                        style={'color': colors['success'] if metrics.get('total_return', 0) >= 0 else colors['danger']}
                    )
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Return %", className="text-muted"),
                    html.H3(
                        f"{metrics.get('return_percentage', 0):.2f}%",
                        style={'color': colors['success'] if metrics.get('return_percentage', 0) >= 0 else colors['danger']}
                    )
                ])
            ])
        ], md=3),
    ], className="mb-4")

def create_recommendations_section():
    """Create recommendations section"""
    data = load_recommendations()
    
    if not data or 'recommendations' not in data:
        return html.Div("No recommendations available")
    
    recommendations = data['recommendations']
    
    cards = []
    for symbol, rec in recommendations.items():
        rec_type = rec.get('recommendation', 'HOLD').upper()
        color = colors['success'] if rec_type == 'BUY' else colors['danger'] if rec_type == 'SELL' else colors['warning']
        
        cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5(symbol, style={'color': crypto_colors.get(symbol, colors['primary'])})),
                    dbc.CardBody([
                        html.H4(rec_type, style={'color': color}),
                        html.P(f"Confidence: {rec.get('confidence', 0)*100:.1f}%"),
                        html.Hr(),
                        html.P("Predicted Returns:", className="mb-2"),
                        html.Ul([
                            html.Li(f"1 Day: {rec.get('predicted_returns', {}).get('1_day', 0):.2f}%"),
                            html.Li(f"7 Days: {rec.get('predicted_returns', {}).get('7_day', 0):.2f}%"),
                            html.Li(f"30 Days: {rec.get('predicted_returns', {}).get('30_day', 0):.2f}%"),
                        ])
                    ])
                ])
            ], md=4)
        )
    
    return dbc.Row(cards)

# Main layout
def create_layout():
    """Create main application layout"""
    return html.Div([
        create_navbar(),
        dbc.Container([
            # Metrics overview
            html.H2("Portfolio Overview", className="mb-4"),
            create_metrics_cards(),
            
            # Portfolio allocation
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='portfolio-pie-chart', figure=create_portfolio_pie_chart())
                ], md=6),
                dbc.Col([
                    html.H4("Investment Recommendations"),
                    create_recommendations_section()
                ], md=6)
            ], className="mb-4"),
            
            # Price charts
            html.H2("Price Analysis", className="mb-4"),
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id=f'{symbol}-price-chart', figure=create_price_chart(symbol)),
                    label=symbol,
                    tab_style={'backgroundColor': colors['light']},
                    active_tab_style={'backgroundColor': crypto_colors[symbol], 'color': 'white'}
                )
                for symbol in ['BTC', 'ETH']
            ], className="mb-4"),
            
            # Sentiment analysis
            html.H2("Sentiment Analysis", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_sentiment_chart(symbol))
                ], md=4)
                for symbol in ['BTC', 'ETH']
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ], fluid=True)
    ])

# Set the layout
app.layout = create_layout

# Callbacks for interactivity
@app.callback(
    [Output('portfolio-pie-chart', 'figure'),
     Output('BTC-price-chart', 'figure'),
     Output('ETH-price-chart', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_charts(n):
    """Update all charts on interval"""
    return (
        create_portfolio_pie_chart(),
        create_price_chart('BTC'),
        create_price_chart('ETH')
    )

# Run the app
if __name__ == '__main__':
    logger.info(f"Starting Dash frontend on port {config.FRONTEND_PORT}")
    app.run(debug=True, port=config.FRONTEND_PORT)