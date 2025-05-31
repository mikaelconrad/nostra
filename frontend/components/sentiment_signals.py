"""
Sentiment-based trading signals for the crypto trading game
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

def create_sentiment_trading_signals(sentiment_data: Dict) -> html.Div:
    """
    Create trading signals based on sentiment data
    
    Args:
        sentiment_data: Current sentiment data
        
    Returns:
        HTML div with trading signals
    """
    signals = []
    current_sentiment = sentiment_data.get('current_sentiment', 0)
    confidence = sentiment_data.get('confidence', 0)
    signal_strength = sentiment_data.get('signal_strength', 'weak')
    
    # Generate signals based on sentiment thresholds
    if current_sentiment > 0.5 and confidence > 0.7:
        signals.append({
            'type': 'strong_buy',
            'icon': '🚀',
            'title': 'Strong Buy Signal',
            'description': 'Extremely bullish sentiment with high confidence',
            'color': 'success',
            'recommendation': 'Consider increasing position',
            'risk_level': 'Medium'
        })
    elif current_sentiment > 0.3 and confidence > 0.6:
        signals.append({
            'type': 'buy',
            'icon': '📈',
            'title': 'Buy Signal',
            'description': 'Bullish sentiment detected',
            'color': 'success',
            'recommendation': 'Consider buying',
            'risk_level': 'Medium-High'
        })
    elif current_sentiment > 0.1 and confidence > 0.5:
        signals.append({
            'type': 'weak_buy',
            'icon': '⬆️',
            'title': 'Weak Buy Signal',
            'description': 'Moderately positive sentiment',
            'color': 'info',
            'recommendation': 'Monitor for opportunities',
            'risk_level': 'High'
        })
    elif current_sentiment < -0.5 and confidence > 0.7:
        signals.append({
            'type': 'strong_sell',
            'icon': '💥',
            'title': 'Strong Sell Signal',
            'description': 'Extremely bearish sentiment with high confidence',
            'color': 'danger',
            'recommendation': 'Consider reducing position',
            'risk_level': 'Low'
        })
    elif current_sentiment < -0.3 and confidence > 0.6:
        signals.append({
            'type': 'sell',
            'icon': '📉',
            'title': 'Sell Signal',
            'description': 'Bearish sentiment detected',
            'color': 'danger',
            'recommendation': 'Consider selling',
            'risk_level': 'Low-Medium'
        })
    elif current_sentiment < -0.1 and confidence > 0.5:
        signals.append({
            'type': 'weak_sell',
            'icon': '⬇️',
            'title': 'Weak Sell Signal',
            'description': 'Moderately negative sentiment',
            'color': 'warning',
            'recommendation': 'Exercise caution',
            'risk_level': 'Medium'
        })
    else:
        signals.append({
            'type': 'hold',
            'icon': '➡️',
            'title': 'Hold Signal',
            'description': 'Neutral sentiment or low confidence',
            'color': 'secondary',
            'recommendation': 'No clear direction',
            'risk_level': 'Medium'
        })
    
    # Create signal cards
    signal_cards = []
    for signal in signals:
        card = dbc.Card([
            dbc.CardBody([
                html.H5([
                    signal['icon'], " ", signal['title']
                ], className="card-title"),
                html.P(signal['description'], className="card-text"),
                dbc.Row([
                    dbc.Col([
                        html.Strong("Recommendation:"),
                        html.Br(),
                        signal['recommendation']
                    ], md=8),
                    dbc.Col([
                        html.Strong("Risk:"),
                        html.Br(),
                        signal['risk_level']
                    ], md=4)
                ])
            ])
        ], color=signal['color'], outline=True, className="mb-2")
        signal_cards.append(card)
    
    return html.Div([
        html.H6("🎯 Sentiment Trading Signals", className="mb-3"),
        html.Div(signal_cards),
        dbc.Alert([
            html.Strong("⚠️ Important: "),
            "These signals are based on social sentiment analysis and should not be used as the sole basis for trading decisions. "
            "Always combine with technical analysis and risk management."
        ], color="warning", className="mt-3")
    ])

def create_sentiment_events_timeline(sentiment_data: Dict) -> html.Div:
    """
    Create a timeline of significant sentiment events
    
    Args:
        sentiment_data: Sentiment data with events
        
    Returns:
        HTML div with sentiment events timeline
    """
    events = [
        {
            'time': '2 hours ago',
            'event': 'Reddit Rally Detected',
            'description': 'Spike in bullish posts on r/Bitcoin',
            'impact': 'positive',
            'icon': '🚀'
        },
        {
            'time': '6 hours ago',
            'event': 'Fear & Greed Index Update',
            'description': 'Index moved from 45 to 52 (Neutral to Greed)',
            'impact': 'positive',
            'icon': '📊'
        },
        {
            'time': '1 day ago',
            'event': 'Institutional Sentiment',
            'description': 'Increased mentions of institutional adoption',
            'impact': 'positive',
            'icon': '🏢'
        },
        {
            'time': '2 days ago',
            'event': 'Market Correction Sentiment',
            'description': 'Brief spike in bearish sentiment during price dip',
            'impact': 'negative',
            'icon': '📉'
        }
    ]
    
    timeline_items = []
    for event in events:
        color = 'success' if event['impact'] == 'positive' else 'danger' if event['impact'] == 'negative' else 'secondary'
        
        item = dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span(event['icon'], className="me-2"),
                    html.Strong(event['event']),
                    html.Br(),
                    html.Small(event['description'], className="text-muted"),
                    html.Br(),
                    html.Small(event['time'], className="text-muted")
                ])
            ], md=10),
            dbc.Col([
                dbc.Badge(
                    "+" if event['impact'] == 'positive' else "-" if event['impact'] == 'negative' else "=",
                    color=color,
                    className="rounded-pill"
                )
            ], md=2, className="text-end")
        ], className="mb-2 pb-2 border-bottom")
        
        timeline_items.append(item)
    
    return html.Div([
        html.H6("📅 Recent Sentiment Events", className="mb-3"),
        html.Div(timeline_items, style={"max-height": "300px", "overflow-y": "auto"})
    ])

def create_sentiment_strategy_suggestions(sentiment_data: Dict) -> html.Div:
    """
    Create strategy suggestions based on sentiment analysis
    
    Args:
        sentiment_data: Current sentiment data
        
    Returns:
        HTML div with strategy suggestions
    """
    current_sentiment = sentiment_data.get('current_sentiment', 0)
    confidence = sentiment_data.get('confidence', 0)
    
    strategies = []
    
    if current_sentiment > 0.3 and confidence > 0.7:
        strategies.extend([
            {
                'name': 'Momentum Trading',
                'description': 'Take advantage of positive sentiment momentum',
                'suitability': 'High',
                'timeframe': 'Short-term (1-7 days)',
                'risk': 'Medium-High'
            },
            {
                'name': 'Dollar Cost Averaging',
                'description': 'Gradually increase position during positive sentiment',
                'suitability': 'Medium',
                'timeframe': 'Medium-term (1-4 weeks)',
                'risk': 'Medium'
            }
        ])
    elif current_sentiment < -0.3 and confidence > 0.7:
        strategies.extend([
            {
                'name': 'Contrarian Buying',
                'description': 'Buy during negative sentiment if fundamentals are strong',
                'suitability': 'Medium',
                'timeframe': 'Long-term (1-3 months)',
                'risk': 'High'
            },
            {
                'name': 'Cash Preservation',
                'description': 'Hold cash and wait for sentiment improvement',
                'suitability': 'High',
                'timeframe': 'Flexible',
                'risk': 'Low'
            }
        ])
    else:
        strategies.extend([
            {
                'name': 'Range Trading',
                'description': 'Trade within support and resistance levels',
                'suitability': 'Medium',
                'timeframe': 'Short-term (1-7 days)',
                'risk': 'Medium'
            },
            {
                'name': 'Wait and Watch',
                'description': 'Monitor sentiment trends before making moves',
                'suitability': 'High',
                'timeframe': 'Flexible',
                'risk': 'Low'
            }
        ])
    
    strategy_cards = []
    for strategy in strategies:
        suitability_color = 'success' if strategy['suitability'] == 'High' else 'warning' if strategy['suitability'] == 'Medium' else 'danger'
        
        card = dbc.Card([
            dbc.CardBody([
                html.H6(strategy['name'], className="card-title"),
                html.P(strategy['description'], className="card-text small"),
                dbc.Row([
                    dbc.Col([
                        html.Small([html.Strong("Timeframe: "), strategy['timeframe']])
                    ], md=6),
                    dbc.Col([
                        html.Small([html.Strong("Risk: "), strategy['risk']])
                    ], md=6)
                ]),
                dbc.Badge(
                    f"Suitability: {strategy['suitability']}",
                    color=suitability_color,
                    className="mt-2"
                )
            ])
        ], className="mb-2", style={"height": "140px"})
        
        strategy_cards.append(card)
    
    return html.Div([
        html.H6("💡 Strategy Suggestions", className="mb-3"),
        html.Div(strategy_cards),
        dbc.Alert([
            html.Strong("📚 Educational Purpose: "),
            "These strategies are for educational purposes in this trading simulation. "
            "Real trading involves significant risk and requires thorough research."
        ], color="info", className="mt-3")
    ])

def create_sentiment_game_features() -> html.Div:
    """Create sentiment-based game features section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("🎮 Sentiment-Based Game Features", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                # Trading signals
                dbc.Col([
                    html.Div(id="sentiment-trading-signals")
                ], md=4),
                
                # Events timeline
                dbc.Col([
                    html.Div(id="sentiment-events-timeline")
                ], md=4),
                
                # Strategy suggestions
                dbc.Col([
                    html.Div(id="sentiment-strategy-suggestions")
                ], md=4)
            ]),
            
            # Sentiment challenges section
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H6("🏆 Sentiment Trading Challenges", className="mb-3"),
                    dbc.Alert([
                        html.H6("Challenge: Sentiment Accuracy", className="mb-2"),
                        html.P("Make 5 trades based on sentiment signals and achieve >60% accuracy", className="mb-2"),
                        dbc.Progress(value=60, label="3/5 trades", color="info", className="mb-2"),
                        dbc.Badge("Active", color="success")
                    ], color="light", className="mb-2"),
                    
                    dbc.Alert([
                        html.H6("Challenge: Contrarian Trader", className="mb-2"),
                        html.P("Successfully trade against negative sentiment 3 times", className="mb-2"),
                        dbc.Progress(value=33, label="1/3 trades", color="warning", className="mb-2"),
                        dbc.Badge("Active", color="success")
                    ], color="light")
                ], md=6),
                
                dbc.Col([
                    html.H6("📈 Sentiment Performance Tracking", className="mb-3"),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Strong("85%"),
                                html.Br(),
                                html.Small("Sentiment Signal Accuracy")
                            ], className="text-center"),
                            dbc.Col([
                                html.Strong("12"),
                                html.Br(),
                                html.Small("Sentiment-Based Trades")
                            ], className="text-center"),
                            dbc.Col([
                                html.Strong("+15.3%"),
                                html.Br(),
                                html.Small("Sentiment Strategy Return")
                            ], className="text-center")
                        ], className="mb-3"),
                        
                        html.H6("Recent Sentiment Trades:", className="mb-2"),
                        html.Div([
                            dbc.Badge("BTC Buy (Strong Bullish) +5.2%", color="success", className="me-1 mb-1"),
                            dbc.Badge("ETH Hold (Neutral) +0.8%", color="secondary", className="me-1 mb-1"),
                            dbc.Badge("BTC Sell (Bearish) +3.1%", color="info", className="me-1 mb-1")
                        ])
                    ])
                ], md=6)
            ])
        ])
    ], className="mt-4")

# Export functions for use in main app
__all__ = [
    'create_sentiment_trading_signals',
    'create_sentiment_events_timeline', 
    'create_sentiment_strategy_suggestions',
    'create_sentiment_game_features'
]