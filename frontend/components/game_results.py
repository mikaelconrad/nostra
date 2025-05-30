"""
Game results component for displaying final simulation outcomes
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from frontend.game_state import game_instance


def create_performance_chart(daily_values):
    """Create performance chart showing portfolio value over time"""
    df = pd.DataFrame(daily_values)
    
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#007bff', width=3)
    ))
    
    # Add initial value reference line
    initial_value = df['total_value'].iloc[0]
    fig.add_hline(
        y=initial_value,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Value",
        annotation_position="right"
    )
    
    # Highlight profit/loss areas
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_value'],
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.1)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (CHF)",
        hovermode='x unified',
        height=400,
        showlegend=False,
        margin=dict(t=40, b=40, l=60, r=40),
        yaxis=dict(tickformat=',.0f')
    )
    
    return fig


def create_metrics_cards(metrics):
    """Create metric cards displaying key performance indicators"""
    
    # Determine card colors based on performance
    return_color = "success" if metrics['total_return'] >= 0 else "danger"
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Return", className="text-muted"),
                    html.H3(
                        f"CHF {metrics['total_return']:,.2f}",
                        className=f"text-{return_color}"
                    ),
                    html.P(
                        f"{'+' if metrics['total_return_pct'] >= 0 else ''}{metrics['total_return_pct']:.2f}%",
                        className="mb-0"
                    )
                ])
            ])
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Final Value", className="text-muted"),
                    html.H3(f"CHF {metrics['final_value']:,.2f}"),
                    html.P(f"From CHF {metrics['initial_value']:,.2f}", className="mb-0 text-muted")
                ])
            ])
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Trades", className="text-muted"),
                    html.H3(str(metrics['total_trades'])),
                    html.P(
                        f"{metrics['buy_trades']} buys, {metrics['sell_trades']} sells",
                        className="mb-0 text-muted"
                    )
                ])
            ])
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Max Drawdown", className="text-muted"),
                    html.H3(f"{metrics['max_drawdown']:.2f}%"),
                    html.P(f"Fees: CHF {metrics['total_fees']:.2f}", className="mb-0 text-muted")
                ])
            ])
        ], md=3)
    ], className="mb-4")


def create_trade_summary(transactions):
    """Create summary of trading activity"""
    if not transactions:
        return html.P("No trades were executed during the simulation.", className="text-center text-muted")
    
    # Group trades by crypto
    btc_trades = [t for t in transactions if t.crypto == 'BTC']
    eth_trades = [t for t in transactions if t.crypto == 'ETH']
    
    # Calculate average prices
    btc_buys = [t for t in btc_trades if t.type == 'buy']
    btc_sells = [t for t in btc_trades if t.type == 'sell']
    eth_buys = [t for t in eth_trades if t.type == 'buy']
    eth_sells = [t for t in eth_trades if t.type == 'sell']
    
    def avg_price(trades):
        if not trades:
            return 0
        return sum(t.price for t in trades) / len(trades)
    
    return html.Div([
        html.H5("Trading Summary", className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.H6("Bitcoin Trading"),
                html.Ul([
                    html.Li(f"Total trades: {len(btc_trades)}"),
                    html.Li(f"Buys: {len(btc_buys)} (avg: CHF {avg_price(btc_buys):,.2f})"),
                    html.Li(f"Sells: {len(btc_sells)} (avg: CHF {avg_price(btc_sells):,.2f})")
                ])
            ], md=6),
            
            dbc.Col([
                html.H6("Ethereum Trading"),
                html.Ul([
                    html.Li(f"Total trades: {len(eth_trades)}"),
                    html.Li(f"Buys: {len(eth_buys)} (avg: CHF {avg_price(eth_buys):,.2f})"),
                    html.Li(f"Sells: {len(eth_sells)} (avg: CHF {avg_price(eth_sells):,.2f})")
                ])
            ], md=6)
        ])
    ])


def create_performance_analysis(metrics, daily_values):
    """Create performance analysis section"""
    # Calculate additional metrics
    df = pd.DataFrame(daily_values)
    
    # Best and worst days
    df['daily_return'] = df['total_value'].pct_change()
    best_day = df.loc[df['daily_return'].idxmax()]
    worst_day = df.loc[df['daily_return'].idxmin()]
    
    # Win rate (days with positive returns)
    positive_days = (df['daily_return'] > 0).sum()
    total_days = len(df) - 1  # Exclude first day
    win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
    
    return html.Div([
        html.H5("Performance Analysis", className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.P([
                    html.Strong("Best Day: "),
                    f"{best_day['date']} (+{best_day['daily_return']*100:.2f}%)"
                ]),
                html.P([
                    html.Strong("Worst Day: "),
                    f"{worst_day['date']} ({worst_day['daily_return']*100:.2f}%)"
                ])
            ], md=6),
            
            dbc.Col([
                html.P([
                    html.Strong("Win Rate: "),
                    f"{win_rate:.1f}% of days"
                ]),
                html.P([
                    html.Strong("Duration: "),
                    f"{metrics['days_played']} days"
                ])
            ], md=6)
        ])
    ])


def create_final_results_display(game_data):
    """Create complete final results display"""
    game_instance.from_dict(game_data)
    metrics = game_instance.get_performance_metrics()
    
    return html.Div([
        create_metrics_cards(metrics),
        
        dbc.Card([
            dbc.CardBody([
                create_performance_analysis(metrics, game_instance.daily_values),
                html.Hr(),
                create_trade_summary(game_instance.transactions)
            ])
        ])
    ])


def register_results_callbacks(app):
    """Register callbacks for results display"""
    
    @app.callback(
        [Output("final-results", "children"),
         Output("performance-chart", "figure")],
        [Input("game-state-store", "data")]
    )
    def update_final_results(game_data):
        """Update final results when game completes"""
        if not game_data or game_data.get('state') != 'completed':
            return html.Div(), go.Figure()
        
        game_instance.from_dict(game_data)
        
        results_display = create_final_results_display(game_data)
        performance_chart = create_performance_chart(game_instance.daily_values)
        
        return results_display, performance_chart