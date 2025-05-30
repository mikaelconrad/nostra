"""
Predictive Charts Component for the Crypto Trading Game
Provides three charts for BTC and ETH with 7, 14, and 30 day predictions
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

from frontend.components.enhanced_charts import (
    create_predictive_chart_btc,
    create_predictive_chart_eth,
    create_combined_prediction_charts
)


def create_predictive_charts_layout():
    """Create the layout for predictive charts section"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("AI Market Predictions", className="mb-3"),
                html.P("Our neural network model predicts future prices based on historical data and market sentiment.", 
                       className="text-muted mb-4")
            ])
        ]),
        
        # Navigation tabs for different prediction horizons
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="7-Day Outlook", tab_id="pred-7", active_tab="pred-7"),
                    dbc.Tab(label="14-Day Outlook", tab_id="pred-14"),
                    dbc.Tab(label="30-Day Outlook", tab_id="pred-30")
                ], id="prediction-tabs", active_tab="pred-7"),
                html.Div(id="prediction-content", className="mt-3")
            ])
        ]),
        
        # Individual crypto prediction charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Bitcoin Predictions", className="mb-0"),
                        dbc.Badge("AI Powered", color="primary", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="btc-prediction-chart", style={"height": "400px"})
                    ])
                ])
            ], md=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Ethereum Predictions", className="mb-0"),
                        dbc.Badge("AI Powered", color="info", className="ms-2")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="eth-prediction-chart", style={"height": "400px"})
                    ])
                ])
            ], md=6)
        ], className="mt-4"),
        
        # Prediction summary cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Prediction Summary", className="card-title"),
                        html.Div(id="prediction-summary", className="mt-2")
                    ])
                ])
            ])
        ], className="mt-4")
    ], fluid=True)


def register_predictive_charts_callbacks(app):
    """Register callbacks for predictive charts functionality"""
    
    @app.callback(
        Output('prediction-content', 'children'),
        [Input('prediction-tabs', 'active_tab'),
         Input('game-state-store', 'data'),
         Input('price-data-store', 'data')]
    )
    def update_prediction_content(active_tab, game_data, price_data):
        """Update prediction content based on selected tab"""
        if not game_data or not price_data:
            return html.Div("Loading predictions...", className="text-center text-muted")
        
        try:
            # Extract horizon from tab id
            horizon = int(active_tab.split('-')[1])
            current_date = game_data.get('current_date', '2024-01-01')
            
            # Load price data
            btc_data = pd.DataFrame(price_data['btc'])
            eth_data = pd.DataFrame(price_data['eth'])
            
            # Create combined charts for the specific horizon
            combined_charts = create_combined_prediction_charts(
                btc_data, eth_data, current_date, [horizon]
            )
            
            if horizon in combined_charts:
                return dcc.Graph(
                    figure=combined_charts[horizon],
                    style={"height": "600px"}
                )
            else:
                return html.Div("Predictions not available for this horizon", 
                              className="text-center text-muted")
                
        except Exception as e:
            print(f"Error updating prediction content: {str(e)}")
            return html.Div("Error loading predictions", 
                          className="text-center text-danger")
    
    @app.callback(
        Output('btc-prediction-chart', 'figure'),
        [Input('game-state-store', 'data'),
         Input('price-data-store', 'data')]
    )
    def update_btc_prediction_chart(game_data, price_data):
        """Update BTC prediction chart"""
        if not game_data or not price_data:
            return {}
        
        try:
            current_date = game_data.get('current_date', '2024-01-01')
            btc_data = pd.DataFrame(price_data['btc'])
            
            return create_predictive_chart_btc(btc_data, current_date)
        except Exception as e:
            print(f"Error updating BTC prediction chart: {str(e)}")
            return {}
    
    @app.callback(
        Output('eth-prediction-chart', 'figure'),
        [Input('game-state-store', 'data'),
         Input('price-data-store', 'data')]
    )
    def update_eth_prediction_chart(game_data, price_data):
        """Update ETH prediction chart"""
        if not game_data or not price_data:
            return {}
        
        try:
            current_date = game_data.get('current_date', '2024-01-01')
            eth_data = pd.DataFrame(price_data['eth'])
            
            return create_predictive_chart_eth(eth_data, current_date)
        except Exception as e:
            print(f"Error updating ETH prediction chart: {str(e)}")
            return {}
    
    @app.callback(
        Output('prediction-summary', 'children'),
        [Input('game-state-store', 'data'),
         Input('price-data-store', 'data')]
    )
    def update_prediction_summary(game_data, price_data):
        """Update prediction summary with key insights"""
        if not game_data or not price_data:
            return "Loading..."
        
        try:
            current_date = game_data.get('current_date', '2024-01-01')
            
            # Get current prices
            btc_data = pd.DataFrame(price_data['btc'])
            eth_data = pd.DataFrame(price_data['eth'])
            
            current_btc = btc_data[btc_data['Date'] <= current_date].iloc[-1]['Close']
            current_eth = eth_data[eth_data['Date'] <= current_date].iloc[-1]['Close']
            
            # Create summary cards
            summary_cards = []
            
            # BTC summary
            summary_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Bitcoin", className="card-title text-warning"),
                            html.P(f"Current: CHF {current_btc:,.2f}", className="mb-1"),
                            html.Small("AI predictions available for 7, 14, and 30 days", 
                                     className="text-muted")
                        ])
                    ], className="border-warning")
                ], md=6)
            )
            
            # ETH summary
            summary_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Ethereum", className="card-title text-info"),
                            html.P(f"Current: CHF {current_eth:,.2f}", className="mb-1"),
                            html.Small("AI predictions available for 7, 14, and 30 days", 
                                     className="text-muted")
                        ])
                    ], className="border-info")
                ], md=6)
            )
            
            return dbc.Row(summary_cards)
            
        except Exception as e:
            print(f"Error updating prediction summary: {str(e)}")
            return "Error loading summary"