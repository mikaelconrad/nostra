"""
Daily trading interface component for the game
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

from config import MIN_TRADE_AMOUNT, TRANSACTION_FEE_PERCENTAGE
from frontend.game_state import game_instance
from backend.simple_data_collector import DataCollector


def create_portfolio_details(portfolio, btc_price, eth_price):
    """Create portfolio details display"""
    total_value = portfolio.get_total_value(btc_price, eth_price)
    btc_value = portfolio.btc_amount * btc_price
    eth_value = portfolio.eth_amount * eth_price
    
    return html.Div([
        html.Div([
            html.Div([
                html.Span("Cash", className="currency"),
                html.Span(f"CHF {portfolio.cash:,.2f}", className="amount")
            ], className="portfolio-item"),
            
            html.Div([
                html.Span("Bitcoin", className="currency"),
                html.Div([
                    html.Span(f"{portfolio.btc_amount:.4f} BTC", className="amount d-block"),
                    html.Small(f"CHF {btc_value:,.2f}", className="text-muted")
                ])
            ], className="portfolio-item"),
            
            html.Div([
                html.Span("Ethereum", className="currency"),
                html.Div([
                    html.Span(f"{portfolio.eth_amount:.3f} ETH", className="amount d-block"),
                    html.Small(f"CHF {eth_value:,.2f}", className="text-muted")
                ])
            ], className="portfolio-item"),
            
            html.Hr(),
            
            html.Div([
                html.Span("Total Value", className="currency fw-bold"),
                html.Span(f"CHF {total_value:,.2f}", className="amount fw-bold")
            ], className="portfolio-item bg-primary text-white")
        ])
    ])


def create_portfolio_chart(portfolio, btc_price, eth_price):
    """Create portfolio allocation pie chart"""
    btc_value = portfolio.btc_amount * btc_price
    eth_value = portfolio.eth_amount * eth_price
    
    fig = go.Figure(data=[go.Pie(
        labels=['Cash', 'Bitcoin', 'Ethereum'],
        values=[portfolio.cash, btc_value, eth_value],
        hole=.3,
        marker_colors=['#85bb65', '#F7931A', '#627EEA']
    )])
    
    fig.update_layout(
        showlegend=False,
        height=250,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_price_display(crypto, current_price, yesterday_price):
    """Create price display with change indicator"""
    change = current_price - yesterday_price
    change_pct = (change / yesterday_price) * 100 if yesterday_price > 0 else 0
    
    change_class = "positive" if change >= 0 else "negative"
    change_symbol = "+" if change >= 0 else ""
    
    return html.Div([
        html.H4(f"{crypto} Price", className="mb-2"),
        html.Div([
            html.H2(f"CHF {current_price:,.2f}", className="price mb-0"),
            html.Div([
                html.Span(f"{change_symbol}{change:.2f} ", className=f"change {change_class}"),
                html.Span(f"({change_symbol}{change_pct:.2f}%)", className=f"change {change_class}")
            ])
        ], className="price-display")
    ])


def create_price_chart(crypto, historical_prices, current_date):
    """Create price chart for the selected cryptocurrency"""
    # Get last 30 days of prices up to current date
    end_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=30)
    
    # Filter prices for date range
    mask = (historical_prices['Date'] >= start_date.strftime('%Y-%m-%d')) & \
           (historical_prices['Date'] <= current_date)
    price_data = historical_prices[mask].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_data['Date'],
        y=price_data['Close'],
        mode='lines',
        name=crypto,
        line=dict(color='#F7931A' if crypto == 'BTC' else '#627EEA', width=2)
    ))
    
    # Add current price marker
    fig.add_trace(go.Scatter(
        x=[current_date],
        y=[price_data.iloc[-1]['Close']],
        mode='markers',
        name='Current',
        marker=dict(size=10, color='red'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"{crypto} Price History (30 days)",
        xaxis_title="Date",
        yaxis_title="Price (CHF)",
        hovermode='x unified',
        height=300,
        margin=dict(t=40, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )
    
    return fig


def create_indicators_content(crypto, historical_prices, recommendations):
    """Create technical indicators display"""
    # Calculate simple indicators
    prices = historical_prices['Close'].values[-20:]  # Last 20 days
    sma_5 = prices[-5:].mean()
    sma_20 = prices.mean()
    current_price = prices[-1]
    
    # RSI calculation (simplified)
    price_changes = pd.Series(prices).diff()
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)
    avg_gain = gains.rolling(window=14).mean().iloc[-1]
    avg_loss = losses.rolling(window=14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Get recommendation if available
    rec_signal = "Hold"
    if recommendations and len(recommendations) > 0:
        latest_rec = recommendations[-1]
        rec_signal = latest_rec.get('signal', 'Hold')
    
    return html.Div([
        html.H5("Technical Indicators", className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Small("5-Day SMA", className="text-muted"),
                html.H6(f"CHF {sma_5:,.2f}")
            ], width=6),
            dbc.Col([
                html.Small("20-Day SMA", className="text-muted"),
                html.H6(f"CHF {sma_20:,.2f}")
            ], width=6)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Small("RSI (14)", className="text-muted"),
                html.H6(f"{rsi:.1f}")
            ], width=6),
            dbc.Col([
                html.Small("Signal", className="text-muted"),
                html.H6(rec_signal, className=f"text-{'success' if rec_signal == 'Buy' else 'danger' if rec_signal == 'Sell' else 'warning'}")
            ], width=6)
        ], className="mb-3"),
        
        html.Hr(),
        
        html.Div([
            html.Small("Market Conditions", className="text-muted"),
            html.P([
                "RSI: ",
                html.Strong("Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"),
                html.Br(),
                "Trend: ",
                html.Strong("Bullish" if current_price > sma_20 else "Bearish")
            ])
        ])
    ])


def create_history_content(transactions):
    """Create transaction history display"""
    if not transactions:
        return html.Div([
            html.H5("Transaction History", className="mb-3"),
            html.P("No transactions yet.", className="text-muted text-center")
        ])
    
    # Create table of recent transactions
    rows = []
    for t in transactions[-10:]:  # Show last 10 transactions
        rows.append(
            html.Tr([
                html.Td(t.date),
                html.Td(t.type.upper(), className=f"text-{'success' if t.type == 'buy' else 'danger'}"),
                html.Td(f"{t.amount:.4f} {t.crypto}"),
                html.Td(f"CHF {t.price:,.2f}"),
                html.Td(f"CHF {t.total_value:,.2f}")
            ])
        )
    
    return html.Div([
        html.H5("Transaction History", className="mb-3"),
        html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Date"),
                        html.Th("Type"),
                        html.Th("Amount"),
                        html.Th("Price"),
                        html.Th("Total")
                    ])
                ]),
                html.Tbody(rows[::-1])  # Reverse to show most recent first
            ], className="table table-sm")
        ], style={"maxHeight": "400px", "overflowY": "auto"})
    ])


def register_trading_callbacks(app, data_collector):
    """Register callbacks for the trading interface"""
    
    @app.callback(
        [Output("portfolio-details", "children"),
         Output("portfolio-chart", "children"),
         Output("portfolio-value", "children"),
         Output("portfolio-change", "children")],
        [Input("game-state-store", "data"),
         Input("price-data-store", "data")]
    )
    def update_portfolio_display(game_data, price_data):
        """Update portfolio display"""
        if not game_data or not price_data:
            return html.Div("Loading..."), html.Div(), "CHF 0.00", ""
        
        game_instance.from_dict(game_data)
        btc_price = price_data.get('BTC', 0)
        eth_price = price_data.get('ETH', 0)
        
        portfolio = game_instance.current_portfolio
        total_value = portfolio.get_total_value(btc_price, eth_price)
        
        # Calculate change from initial
        initial_value = game_instance.initial_portfolio.get_total_value(btc_price, eth_price)
        change = total_value - initial_value
        change_pct = (change / initial_value * 100) if initial_value > 0 else 0
        
        change_color = "text-success" if change >= 0 else "text-danger"
        change_symbol = "+" if change >= 0 else ""
        
        return (
            create_portfolio_details(portfolio, btc_price, eth_price),
            create_portfolio_chart(portfolio, btc_price, eth_price),
            f"CHF {total_value:,.2f}",
            html.Span(f"{change_symbol}CHF {change:,.2f} ({change_symbol}{change_pct:.1f}%)", className=change_color)
        )
    
    @app.callback(
        Output("trade-controls", "style"),
        [Input("buy-btn", "n_clicks"),
         Input("sell-btn", "n_clicks")]
    )
    def show_trade_controls(buy_clicks, sell_clicks):
        """Show/hide trade controls"""
        ctx = dash.callback_context
        if ctx.triggered:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("trade-preview", "children"),
        [Input("trade-amount", "value"),
         State("select-btc", "active"),
         State("select-eth", "active"),
         State("price-data-store", "data"),
         State("buy-btn", "n_clicks"),
         State("sell-btn", "n_clicks")]
    )
    def update_trade_preview(amount, btc_active, eth_active, price_data, buy_clicks, sell_clicks):
        """Update trade preview with fees"""
        if not amount or not price_data:
            return ""
        
        ctx = dash.callback_context
        is_buy = "buy-btn" in ctx.triggered[0]['prop_id'] if ctx.triggered else True
        
        crypto = "BTC" if btc_active else "ETH"
        price = price_data.get(crypto, 0)
        
        total = amount * price
        fee = total * (TRANSACTION_FEE_PERCENTAGE / 100)
        
        if is_buy:
            return f"Cost: CHF {total:,.2f} + CHF {fee:.2f} fee = CHF {total + fee:,.2f}"
        else:
            return f"Receive: CHF {total:,.2f} - CHF {fee:.2f} fee = CHF {total - fee:,.2f}"