"""
Crypto Trading Simulator - Main Application
Single-page game interface with state-based rendering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

import config
from frontend.game_state import game_instance
from frontend.components import create_setup_layout, register_setup_callbacks

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
        
        # Interval for auto-save
        dcc.Interval(id='auto-save-interval', interval=30000)  # 30 seconds
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
                            html.Small(id="trade-preview", className="text-muted")
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
            dbc.ModalHeader(dbc.ModalTitle("Confirm Trade")),
            dbc.ModalBody(id="trade-confirmation-body"),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-trade", className="ms-auto", n_clicks=0),
                dbc.Button("Confirm", id="confirm-trade", color="primary", n_clicks=0)
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
                        
                        html.H4("Trading History", className="mb-3"),
                        html.Div(id="trade-history"),
                        
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
     Input('new-game-btn', 'n_clicks')]
)
def render_game_content(game_data, new_game_clicks):
    """Render content based on current game state"""
    ctx = dash.callback_context
    
    # Check if new game button was clicked
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'new-game-btn.n_clicks':
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


# Register component callbacks
register_setup_callbacks(app)


if __name__ == '__main__':
    app.run_server(debug=config.DEBUG, port=config.FRONTEND_PORT)