"""
Market view component for displaying cryptocurrency prices and charts
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

from backend.simple_data_collector import DataCollector


def create_market_overview(btc_price, eth_price, btc_change, eth_change):
    """Create market overview cards"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("Bitcoin", className="text-muted mb-1"),
                        html.H4(f"CHF {btc_price:,.2f}", className="mb-0"),
                        html.Small(
                            f"{'+' if btc_change >= 0 else ''}{btc_change:.2f}%",
                            className=f"text-{'success' if btc_change >= 0 else 'danger'}"
                        )
                    ])
                ])
            ], className="text-center")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("Ethereum", className="text-muted mb-1"),
                        html.H4(f"CHF {eth_price:,.2f}", className="mb-0"),
                        html.Small(
                            f"{'+' if eth_change >= 0 else ''}{eth_change:.2f}%",
                            className=f"text-{'success' if eth_change >= 0 else 'danger'}"
                        )
                    ])
                ])
            ], className="text-center")
        ], md=6)
    ], className="mb-3")


def create_market_comparison_chart(btc_prices, eth_prices, current_date):
    """Create normalized comparison chart for BTC and ETH"""
    # Get last 30 days of data
    end_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=30)
    
    # Filter data
    btc_mask = (btc_prices['Date'] >= start_date.strftime('%Y-%m-%d')) & \
               (btc_prices['Date'] <= current_date)
    eth_mask = (eth_prices['Date'] >= start_date.strftime('%Y-%m-%d')) & \
               (eth_prices['Date'] <= current_date)
    
    btc_data = btc_prices[btc_mask].copy()
    eth_data = eth_prices[eth_mask].copy()
    
    # Normalize prices to percentage change from start
    btc_normalized = (btc_data['Close'] / btc_data['Close'].iloc[0] - 1) * 100
    eth_normalized = (eth_data['Close'] / eth_data['Close'].iloc[0] - 1) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=btc_data['Date'],
        y=btc_normalized,
        mode='lines',
        name='Bitcoin',
        line=dict(color='#F7931A', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=eth_data['Date'],
        y=eth_normalized,
        mode='lines',
        name='Ethereum',
        line=dict(color='#627EEA', width=2)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="30-Day Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Change (%)",
        hovermode='x unified',
        height=250,
        margin=dict(t=40, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_volume_chart(crypto_data, crypto_name, current_date):
    """Create volume chart for cryptocurrency"""
    # Get last 7 days of data
    end_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=7)
    
    mask = (crypto_data['Date'] >= start_date.strftime('%Y-%m-%d')) & \
           (crypto_data['Date'] <= current_date)
    
    data = crypto_data[mask].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data['Date'],
        y=data['Volume'],
        name='Volume',
        marker_color='#F7931A' if 'BTC' in crypto_name else '#627EEA'
    ))
    
    fig.update_layout(
        title=f"{crypto_name} Trading Volume (7 days)",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=200,
        margin=dict(t=40, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )
    
    return fig


def create_recommendation_display(recommendations, current_date):
    """Create recommendation display based on AI predictions"""
    if not recommendations:
        return html.Div([
            html.H6("AI Recommendations", className="mb-3"),
            html.P("No recommendations available", className="text-muted text-center")
        ])
    
    # Find recommendation for current date
    current_rec = None
    for rec in recommendations:
        if rec.get('date') == current_date:
            current_rec = rec
            break
    
    if not current_rec:
        # Use most recent recommendation
        current_rec = recommendations[-1]
    
    signal = current_rec.get('signal', 'Hold')
    confidence = current_rec.get('confidence', 0.5)
    
    signal_color = 'success' if signal == 'Buy' else 'danger' if signal == 'Sell' else 'warning'
    
    return html.Div([
        html.H6("AI Recommendation", className="mb-3"),
        dbc.Alert([
            html.H4(signal, className="alert-heading"),
            html.P(f"Confidence: {confidence:.1%}", className="mb-0")
        ], color=signal_color),
        
        html.Div([
            html.Small("Based on:", className="text-muted"),
            html.Ul([
                html.Li("Price trends"),
                html.Li("Technical indicators"),
                html.Li("Market sentiment"),
                html.Li("Neural network predictions")
            ], className="small")
        ])
    ])


def register_market_callbacks(app, data_collector):
    """Register callbacks for market view"""
    
    @app.callback(
        [Output("current-price-display", "children"),
         Output("price-chart", "figure")],
        [Input("select-btc", "active"),
         Input("select-eth", "active"),
         State("game-state-store", "data"),
         State("price-data-store", "data")]
    )
    def update_market_display(btc_active, eth_active, game_data, price_data):
        """Update market display based on selected crypto"""
        if not game_data or not price_data:
            return html.Div("Loading..."), go.Figure()
        
        crypto = "BTC" if btc_active else "ETH"
        symbol = "BTC-USD" if btc_active else "ETH-USD"
        
        # Load historical prices
        try:
            historical_prices = data_collector.load_historical_data(symbol)
            current_date = game_data.get('current_date')
            
            # Get current and yesterday's price
            current_price = price_data.get(crypto, 0)
            yesterday_idx = historical_prices[historical_prices['Date'] < current_date].index[-1]
            yesterday_price = historical_prices.loc[yesterday_idx, 'Close']
            
            # Create displays
            price_display = create_price_display(crypto, current_price, yesterday_price)
            price_chart = create_price_chart(crypto, historical_prices, current_date)
            
            return price_display, price_chart
            
        except Exception as e:
            return html.Div(f"Error loading data: {str(e)}"), go.Figure()
    
    @app.callback(
        Output("right-panel-content", "children"),
        [Input("right-tabs", "active_tab"),
         State("game-state-store", "data"),
         State("price-data-store", "data"),
         State("select-btc", "active"),
         State("select-eth", "active")]
    )
    def update_right_panel(active_tab, game_data, price_data, btc_active, eth_active):
        """Update right panel content based on selected tab"""
        if not game_data:
            return html.Div("Loading...")
        
        game_instance.from_dict(game_data)
        
        if active_tab == "indicators":
            crypto = "BTC" if btc_active else "ETH"
            symbol = "BTC-USD" if btc_active else "ETH-USD"
            
            try:
                historical_prices = data_collector.load_historical_data(symbol)
                recommendations = data_collector.load_recommendations(symbol)
                
                return create_indicators_content(crypto, historical_prices, recommendations)
            except:
                return html.Div("Error loading indicators")
                
        elif active_tab == "history":
            return create_history_content(game_instance.transactions)
        
        return html.Div()