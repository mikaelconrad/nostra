"""
Game setup component for portfolio initialization and date selection
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pandas as pd

from config import (
    MIN_CASH_AMOUNT, DEFAULT_SIMULATION_LENGTHS, 
    CUSTOM_SIMULATION_MIN_DAYS, CUSTOM_SIMULATION_MAX_DAYS,
    SIMULATION_EARLIEST_START, SIMULATION_BUFFER_DAYS
)
from backend.simple_data_collector import DataCollector


def create_setup_layout():
    """Create the game setup interface"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Crypto Trading Simulator", className="text-center mb-4"),
                html.P("Test your trading skills with historical cryptocurrency data", 
                      className="text-center lead mb-5")
            ])
        ]),
        
        # Setup wizard
        html.Div([
            # Step indicators
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("1", className="step-number active", id="step-1-indicator"),
                        html.Div("Portfolio", className="step-label")
                    ], className="step-item"),
                    html.Div([
                        html.Div("2", className="step-number", id="step-2-indicator"),
                        html.Div("Date & Duration", className="step-label")
                    ], className="step-item"),
                    html.Div([
                        html.Div("3", className="step-number", id="step-3-indicator"),
                        html.Div("Confirm", className="step-label")
                    ], className="step-item")
                ], className="setup-steps d-flex justify-content-center mb-5")
            ]),
            
            # Step content
            html.Div([
                # Step 1: Portfolio Setup
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Initial Portfolio", className="mb-4"),
                            
                            dbc.Form([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Starting Cash (CHF)"),
                                        dbc.Input(
                                            id="initial-cash",
                                            type="number",
                                            min=MIN_CASH_AMOUNT,
                                            value=10000,
                                            step=100
                                        ),
                                        dbc.FormText(f"Minimum: {MIN_CASH_AMOUNT} CHF")
                                    ], md=12, className="mb-3"),
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Bitcoin (BTC)"),
                                        dbc.Input(
                                            id="initial-btc",
                                            type="number",
                                            min=0,
                                            value=0,
                                            step=0.0001
                                        ),
                                        dbc.FormText("Amount of BTC to start with")
                                    ], md=6),
                                    
                                    dbc.Col([
                                        dbc.Label("Ethereum (ETH)"),
                                        dbc.Input(
                                            id="initial-eth",
                                            type="number",
                                            min=0,
                                            value=0,
                                            step=0.001
                                        ),
                                        dbc.FormText("Amount of ETH to start with")
                                    ], md=6)
                                ], className="mb-4"),
                                
                                # Portfolio value preview
                                html.Div(id="portfolio-preview", className="alert alert-info"),
                                
                                dbc.Button("Next →", id="step-1-next", color="primary", size="lg", className="float-end")
                            ])
                        ])
                    ])
                ], id="step-1-content"),
                
                # Step 2: Date Selection
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Simulation Period", className="mb-4"),
                            
                            dbc.Form([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Start Date"),
                                        dbc.Input(
                                            id="start-date",
                                            type="date",
                                            min=SIMULATION_EARLIEST_START,
                                            max=(datetime.now() - timedelta(days=SIMULATION_BUFFER_DAYS)).strftime('%Y-%m-%d'),
                                            value=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                                        )
                                    ], md=6),
                                    
                                    dbc.Col([
                                        dbc.Label("Duration"),
                                        dbc.Select(
                                            id="duration-select",
                                            options=[
                                                {"label": f"{days} days", "value": days}
                                                for days in DEFAULT_SIMULATION_LENGTHS
                                            ] + [{"label": "Custom", "value": "custom"}],
                                            value=30
                                        )
                                    ], md=6)
                                ], className="mb-3"),
                                
                                # Custom duration input
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Input(
                                            id="custom-duration",
                                            type="number",
                                            min=CUSTOM_SIMULATION_MIN_DAYS,
                                            max=CUSTOM_SIMULATION_MAX_DAYS,
                                            placeholder="Enter custom duration",
                                            style={"display": "none"}
                                        )
                                    ], md=12)
                                ], className="mb-3"),
                                
                                # Date preview
                                html.Div(id="date-preview", className="alert alert-info mb-4"),
                                
                                # Market preview
                                html.Div(id="market-preview", className="mb-4"),
                                
                                dbc.ButtonGroup([
                                    dbc.Button("← Back", id="step-2-back", color="secondary", size="lg"),
                                    dbc.Button("Next →", id="step-2-next", color="primary", size="lg")
                                ], className="float-end")
                            ])
                        ])
                    ])
                ], id="step-2-content", style={"display": "none"}),
                
                # Step 3: Confirmation
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Ready to Start!", className="mb-4"),
                            
                            html.Div(id="setup-summary", className="mb-4"),
                            
                            dbc.ButtonGroup([
                                dbc.Button("← Back", id="step-3-back", color="secondary", size="lg"),
                                dbc.Button("Start Simulation", id="start-game", color="success", size="lg")
                            ], className="float-end")
                        ])
                    ])
                ], id="step-3-content", style={"display": "none"})
            ])
        ])
    ], className="mt-5")


def register_setup_callbacks(app):
    """Register callbacks for the setup component"""
    
    @app.callback(
        Output("portfolio-preview", "children"),
        [Input("initial-cash", "value"),
         Input("initial-btc", "value"),
         Input("initial-eth", "value")]
    )
    def update_portfolio_preview(cash, btc, eth):
        """Update portfolio value preview"""
        if cash is None:
            cash = 0
        if btc is None:
            btc = 0
        if eth is None:
            eth = 0
            
        # For now, just show the amounts (prices will be loaded based on selected date)
        return [
            html.H5("Initial Portfolio Summary"),
            html.P([
                f"Cash: {cash:,.2f} CHF",
                html.Br(),
                f"Bitcoin: {btc:.4f} BTC",
                html.Br(),
                f"Ethereum: {eth:.3f} ETH"
            ])
        ]
    
    @app.callback(
        Output("custom-duration", "style"),
        Input("duration-select", "value")
    )
    def toggle_custom_duration(duration):
        """Show/hide custom duration input"""
        if duration == "custom":
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("date-preview", "children"),
        [Input("start-date", "value"),
         Input("duration-select", "value"),
         Input("custom-duration", "value")]
    )
    def update_date_preview(start_date, duration, custom_duration):
        """Update date range preview"""
        if not start_date:
            return "Please select a start date"
        
        if duration == "custom" and custom_duration:
            days = custom_duration
        elif duration != "custom":
            days = int(duration)
        else:
            return "Please select a duration"
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=days)
        
        return [
            html.H5("Simulation Period"),
            html.P([
                f"Start: {start_dt.strftime('%B %d, %Y')}",
                html.Br(),
                f"End: {end_dt.strftime('%B %d, %Y')}",
                html.Br(),
                f"Duration: {days} days"
            ])
        ]
    
    # Navigation callbacks would go here
    # These would handle showing/hiding steps and validating inputs