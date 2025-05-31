"""
Social Sentiment Analysis Panel Component
Displays Reddit and market sentiment data for cryptocurrencies
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional
import json
import os


def create_sentiment_gauge(sentiment_score: float, confidence: float, signal_strength: str = "weak") -> go.Figure:
    """
    Create a sentiment gauge chart
    
    Args:
        sentiment_score: Sentiment score between -1 and 1
        confidence: Confidence level between 0 and 1
        signal_strength: Signal strength ("weak", "moderate", "strong")
    
    Returns:
        Plotly figure for sentiment gauge
    """
    # Convert sentiment score to 0-100 scale (50 = neutral)
    gauge_value = (sentiment_score + 1) * 50
    
    # Determine color based on sentiment
    if sentiment_score > 0.3:
        color = "green"
        sentiment_text = "Bullish"
    elif sentiment_score > 0.1:
        color = "lightgreen"
        sentiment_text = "Slightly Bullish"
    elif sentiment_score > -0.1:
        color = "yellow"
        sentiment_text = "Neutral"
    elif sentiment_score > -0.3:
        color = "orange"
        sentiment_text = "Slightly Bearish"
    else:
        color = "red"
        sentiment_text = "Bearish"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{sentiment_text}<br><span style='font-size:0.8em'>Confidence: {confidence:.0%}</span>"},
        delta = {'reference': 50, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'red'},
                {'range': [20, 40], 'color': 'orange'},
                {'range': [40, 60], 'color': 'yellow'},
                {'range': [60, 80], 'color': 'lightgreen'},
                {'range': [80, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': gauge_value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_sentiment_trend_chart(sentiment_data: List[Dict]) -> go.Figure:
    """
    Create a 7-day sentiment trend chart
    
    Args:
        sentiment_data: List of daily sentiment data
        
    Returns:
        Plotly figure for sentiment trend
    """
    if not sentiment_data:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(sentiment_data)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sentiment'],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.2f}<extra></extra>'
    ))
    
    # Add neutral line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add bullish/bearish zones
    fig.add_hrect(y0=0.1, y1=1, fillcolor="green", opacity=0.1, layer="below")
    fig.add_hrect(y0=-1, y1=-0.1, fillcolor="red", opacity=0.1, layer="below")
    
    fig.update_layout(
        title="7-Day Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=200,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        yaxis=dict(range=[-1, 1])
    )
    
    return fig


def create_reddit_highlights_carousel(posts: List[Dict]) -> html.Div:
    """
    Create a carousel of top Reddit posts
    
    Args:
        posts: List of top Reddit posts
        
    Returns:
        HTML div with Reddit highlights
    """
    if not posts:
        return html.Div([
            html.P("No Reddit data available", className="text-muted text-center")
        ])
    
    carousel_items = []
    
    for i, post in enumerate(posts[:5]):  # Show top 5 posts
        sentiment_badge_color = "success" if post.get('sentiment', 0) > 0.1 else "danger" if post.get('sentiment', 0) < -0.1 else "secondary"
        sentiment_text = "Bullish" if post.get('sentiment', 0) > 0.1 else "Bearish" if post.get('sentiment', 0) < -0.1 else "Neutral"
        
        item = dbc.Card([
            dbc.CardBody([
                html.H6(post.get('title', 'Reddit Post')[:80] + "...", className="card-title"),
                html.P(f"r/{post.get('subreddit', 'cryptocurrency')}", className="text-muted small"),
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(f"⬆ {post.get('score', 0)}", color="info", className="me-2"),
                        dbc.Badge(f"💬 {post.get('comments', 0)}", color="secondary", className="me-2"),
                        dbc.Badge(sentiment_text, color=sentiment_badge_color)
                    ])
                ])
            ])
        ], className="mb-2", style={"height": "120px"})
        
        carousel_items.append(item)
    
    return html.Div(carousel_items)


def create_sentiment_panel_layout():
    """Create the main sentiment analysis panel layout"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("📱 Social Sentiment Pulse", className="d-inline"),
            dbc.Badge("Live Community Data", color="success", className="ms-2")
        ]),
        dbc.CardBody([
            # Sentiment overview row
            dbc.Row([
                # Sentiment gauge
                dbc.Col([
                    html.H6("Current Sentiment", className="text-center mb-2"),
                    dcc.Graph(
                        id="sentiment-gauge",
                        config={'displayModeBar': False},
                        style={"height": "300px"}
                    )
                ], md=4),
                
                # Sentiment trend
                dbc.Col([
                    html.H6("Sentiment Trend", className="mb-2"),
                    dcc.Graph(
                        id="sentiment-trend",
                        config={'displayModeBar': False},
                        style={"height": "200px"}
                    ),
                    # Quick stats
                    html.Div(id="sentiment-stats", className="mt-2")
                ], md=8)
            ], className="mb-3"),
            
            # Reddit highlights and community metrics
            dbc.Row([
                # Reddit highlights
                dbc.Col([
                    html.H6("📰 Top Reddit Posts", className="mb-2"),
                    html.Div(id="reddit-highlights", style={"max-height": "300px", "overflow-y": "auto"})
                ], md=7),
                
                # Community metrics
                dbc.Col([
                    html.H6("📊 Community Metrics", className="mb-2"),
                    html.Div(id="community-metrics")
                ], md=5)
            ], className="mb-3"),
            
            # Sentiment controls and refresh
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("🔄 Refresh", id="refresh-sentiment-btn", color="outline-primary", size="sm"),
                        dbc.Button("📅 Historical", id="historical-sentiment-btn", color="outline-secondary", size="sm"),
                        dbc.Button("⚙️ Settings", id="sentiment-settings-btn", color="outline-info", size="sm")
                    ])
                ], md=6),
                dbc.Col([
                    html.Small(id="sentiment-last-updated", className="text-muted text-end")
                ], md=6)
            ])
        ])
    ], className="mt-4")


def format_api_sentiment_data(api_data: Dict, coin: str) -> Dict:
    """Format aggregated API sentiment data for UI display"""
    try:
        return {
            'current_sentiment': api_data.get('combined_sentiment', 0),
            'confidence': api_data.get('confidence_level', 0.5),
            'signal_strength': api_data.get('signal_strength', 'weak'),
            'trend_data': [],  # Would need historical API call
            'reddit_posts': [],  # Would need separate API call
            'community_metrics': {
                'total_posts': api_data.get('reddit_post_count', 0),
                'total_comments': api_data.get('reddit_comment_count', 0),
                'bullish_posts': api_data.get('reddit_bullish_posts', 0),
                'bearish_posts': api_data.get('reddit_bearish_posts', 0),
                'neutral_posts': api_data.get('reddit_neutral_posts', 0),
                'top_subreddits': ['Bitcoin', 'CryptoCurrency', 'ethereum'],
                'social_volume': api_data.get('reddit_total_score', 0),
                'fear_greed_index': 50  # Would need market data
            },
            'last_updated': api_data.get('last_updated', datetime.now().strftime('%Y-%m-%d %H:%M UTC'))
        }
    except Exception as e:
        print(f"Error formatting API data: {e}")
        return get_mock_sentiment_data(coin)


def format_reddit_sentiment_data(reddit_data: Dict, coin: str) -> Dict:
    """Format Reddit API data for UI display"""
    try:
        posts = reddit_data.get('posts', [])
        comments = reddit_data.get('comments', [])
        
        # Calculate basic sentiment from posts
        post_sentiments = [p.get('sentiment_score', 0) for p in posts if p.get('sentiment_score') is not None]
        avg_sentiment = sum(post_sentiments) / len(post_sentiments) if post_sentiments else 0
        
        # Count sentiment categories
        bullish = sum(1 for p in posts if p.get('sentiment_label') == 'bullish')
        bearish = sum(1 for p in posts if p.get('sentiment_label') == 'bearish')
        neutral = sum(1 for p in posts if p.get('sentiment_label') == 'neutral')
        
        # Format posts for display
        formatted_posts = []
        for post in posts[:5]:
            formatted_posts.append({
                'title': post.get('title', 'Reddit Post'),
                'subreddit': post.get('subreddit', 'cryptocurrency'),
                'score': post.get('score', 0),
                'comments': post.get('num_comments', 0),
                'sentiment': post.get('sentiment_score', 0)
            })
        
        return {
            'current_sentiment': avg_sentiment,
            'confidence': 0.7,  # Default confidence for Reddit data
            'signal_strength': 'moderate' if abs(avg_sentiment) > 0.2 else 'weak',
            'trend_data': [],  # Would need historical data
            'reddit_posts': formatted_posts,
            'community_metrics': {
                'total_posts': len(posts),
                'total_comments': len(comments),
                'bullish_posts': bullish,
                'bearish_posts': bearish,
                'neutral_posts': neutral,
                'top_subreddits': list(set([p.get('subreddit', '') for p in posts])),
                'social_volume': sum(p.get('score', 0) for p in posts),
                'fear_greed_index': 50
            },
            'last_updated': reddit_data.get('collection_metadata', {}).get('collected_at', 
                                        datetime.now().strftime('%Y-%m-%d %H:%M UTC'))
        }
    except Exception as e:
        print(f"Error formatting Reddit data: {e}")
        return get_mock_sentiment_data(coin)


def get_mock_sentiment_data(coin: str) -> Dict:
    """Generate mock sentiment data as fallback"""
    import random
    from datetime import datetime, timedelta
    
    # Generate mock trend data for last 7 days
    trend_data = []
    for i in range(7, 0, -1):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        sentiment = random.uniform(-0.8, 0.8)
        trend_data.append({
            'date': date,
            'sentiment': sentiment,
            'confidence': random.uniform(0.5, 0.9),
            'posts': random.randint(50, 200),
            'comments': random.randint(300, 1000)
        })
    
    # Current sentiment
    current_sentiment = random.uniform(-0.6, 0.8)  # Slightly bias toward positive
    confidence = random.uniform(0.6, 0.9)
    signal_strength = "strong" if confidence > 0.8 else "moderate" if confidence > 0.6 else "weak"
    
    # Mock Reddit posts
    reddit_posts = [
        {
            'title': f'{coin} breaking through resistance levels!',
            'subreddit': 'Bitcoin' if coin == 'BTC' else 'ethereum',
            'score': random.randint(100, 2000),
            'comments': random.randint(50, 300),
            'sentiment': random.uniform(0.3, 0.8)
        },
        {
            'title': f'Analysis: {coin} technical indicators showing strength',
            'subreddit': 'CryptoCurrency',
            'score': random.randint(200, 1500),
            'comments': random.randint(80, 400),
            'sentiment': random.uniform(0.1, 0.6)
        },
        {
            'title': f'Institutional adoption of {coin} continues to grow',
            'subreddit': 'CryptoMarkets',
            'score': random.randint(300, 1200),
            'comments': random.randint(100, 250),
            'sentiment': random.uniform(0.4, 0.7)
        },
        {
            'title': f'Market correction in {coin} - buying opportunity?',
            'subreddit': 'Bitcoin' if coin == 'BTC' else 'ethfinance',
            'score': random.randint(150, 800),
            'comments': random.randint(70, 350),
            'sentiment': random.uniform(-0.2, 0.3)
        },
        {
            'title': f'Long-term outlook for {coin} remains positive',
            'subreddit': 'CryptoCurrency',
            'score': random.randint(400, 1800),
            'comments': random.randint(120, 500),
            'sentiment': random.uniform(0.2, 0.6)
        }
    ]
    
    # Community metrics
    community_metrics = {
        'total_posts': random.randint(150, 300),
        'total_comments': random.randint(800, 2000),
        'bullish_posts': random.randint(60, 120),
        'bearish_posts': random.randint(20, 60),
        'neutral_posts': random.randint(40, 100),
        'top_subreddits': ['Bitcoin', 'CryptoCurrency', 'ethereum', 'CryptoMarkets'][:3],
        'social_volume': random.randint(5000, 15000),
        'fear_greed_index': random.randint(20, 80)
    }
    
    return {
        'current_sentiment': current_sentiment,
        'confidence': confidence,
        'signal_strength': signal_strength,
        'trend_data': trend_data,
        'reddit_posts': reddit_posts,
        'community_metrics': community_metrics,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    }


def get_sentiment_data(coin: str = "BTC", use_real_data: bool = True) -> Dict:
    """
    Get sentiment data (real or mock) for cryptocurrency
    
    Args:
        coin: Cryptocurrency symbol
        use_real_data: Whether to attempt to fetch real data
        
    Returns:
        Dictionary with sentiment data
    """
    # Try to get real data if enabled
    if use_real_data:
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            if os.getenv('ENABLE_REAL_SENTIMENT', 'false').lower() == 'true':
                import requests
                
                # Try to get aggregated sentiment data
                try:
                    response = requests.get(f'http://localhost:5000/api/sentiment/social/aggregated?coin={coin}', timeout=2)
                    if response.status_code == 200:
                        api_data = response.json()
                        if api_data.get('status') == 'success':
                            return format_api_sentiment_data(api_data['data'], coin)
                except requests.exceptions.RequestException:
                    pass  # Fall back to mock data
                
                # Try to get current Reddit data
                try:
                    response = requests.get(f'http://localhost:5000/api/sentiment/social/reddit/current?coin={coin}', timeout=2)
                    if response.status_code == 200:
                        api_data = response.json()
                        if api_data.get('status') == 'success':
                            return format_reddit_sentiment_data(api_data['data'], coin)
                except requests.exceptions.RequestException:
                    pass  # Fall back to mock data
        except Exception as e:
            print(f"Error fetching real sentiment data: {e}")
    
    # Generate mock data as fallback
    import random
    from datetime import datetime, timedelta
    
    # Generate mock trend data for last 7 days
    trend_data = []
    for i in range(7, 0, -1):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        sentiment = random.uniform(-0.8, 0.8)
        trend_data.append({
            'date': date,
            'sentiment': sentiment,
            'confidence': random.uniform(0.5, 0.9),
            'posts': random.randint(50, 200),
            'comments': random.randint(300, 1000)
        })
    
    # Current sentiment
    current_sentiment = random.uniform(-0.6, 0.8)  # Slightly bias toward positive
    confidence = random.uniform(0.6, 0.9)
    signal_strength = "strong" if confidence > 0.8 else "moderate" if confidence > 0.6 else "weak"
    
    # Mock Reddit posts
    reddit_posts = [
        {
            'title': f'{coin} breaking through resistance levels!',
            'subreddit': 'Bitcoin' if coin == 'BTC' else 'ethereum',
            'score': random.randint(100, 2000),
            'comments': random.randint(50, 300),
            'sentiment': random.uniform(0.3, 0.8)
        },
        {
            'title': f'Analysis: {coin} technical indicators showing strength',
            'subreddit': 'CryptoCurrency',
            'score': random.randint(200, 1500),
            'comments': random.randint(80, 400),
            'sentiment': random.uniform(0.1, 0.6)
        },
        {
            'title': f'Institutional adoption of {coin} continues to grow',
            'subreddit': 'CryptoMarkets',
            'score': random.randint(300, 1200),
            'comments': random.randint(100, 250),
            'sentiment': random.uniform(0.4, 0.7)
        },
        {
            'title': f'Market correction in {coin} - buying opportunity?',
            'subreddit': 'Bitcoin' if coin == 'BTC' else 'ethfinance',
            'score': random.randint(150, 800),
            'comments': random.randint(70, 350),
            'sentiment': random.uniform(-0.2, 0.3)
        },
        {
            'title': f'Long-term outlook for {coin} remains positive',
            'subreddit': 'CryptoCurrency',
            'score': random.randint(400, 1800),
            'comments': random.randint(120, 500),
            'sentiment': random.uniform(0.2, 0.6)
        }
    ]
    
    # Community metrics
    community_metrics = {
        'total_posts': random.randint(150, 300),
        'total_comments': random.randint(800, 2000),
        'bullish_posts': random.randint(60, 120),
        'bearish_posts': random.randint(20, 60),
        'neutral_posts': random.randint(40, 100),
        'top_subreddits': ['Bitcoin', 'CryptoCurrency', 'ethereum', 'CryptoMarkets'][:3],
        'social_volume': random.randint(5000, 15000),
        'fear_greed_index': random.randint(20, 80)
    }
    
    return {
        'current_sentiment': current_sentiment,
        'confidence': confidence,
        'signal_strength': signal_strength,
        'trend_data': trend_data,
        'reddit_posts': reddit_posts,
        'community_metrics': community_metrics,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    }


def create_community_metrics_display(metrics: Dict) -> html.Div:
    """Create community metrics display"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Strong(f"{metrics.get('total_posts', 0)}"),
                html.Br(),
                html.Small("Posts Today", className="text-muted")
            ], className="text-center"),
            dbc.Col([
                html.Strong(f"{metrics.get('total_comments', 0)}"),
                html.Br(),
                html.Small("Comments", className="text-muted")
            ], className="text-center")
        ], className="mb-2"),
        
        dbc.Row([
            dbc.Col([
                html.Strong(f"{metrics.get('social_volume', 0):,}"),
                html.Br(),
                html.Small("Social Volume", className="text-muted")
            ], className="text-center"),
            dbc.Col([
                html.Strong(f"{metrics.get('fear_greed_index', 50)}"),
                html.Br(),
                html.Small("Fear & Greed", className="text-muted")
            ], className="text-center")
        ], className="mb-3"),
        
        # Sentiment breakdown
        html.H6("Sentiment Breakdown", className="mb-2"),
        dbc.Progress([
            dbc.Progress(value=metrics.get('bullish_posts', 0), color="success", 
                        label=f"Bullish ({metrics.get('bullish_posts', 0)})"),
            dbc.Progress(value=metrics.get('neutral_posts', 0), color="warning",
                        label=f"Neutral ({metrics.get('neutral_posts', 0)})"),
            dbc.Progress(value=metrics.get('bearish_posts', 0), color="danger",
                        label=f"Bearish ({metrics.get('bearish_posts', 0)})")
        ], className="mb-2"),
        
        # Top subreddits
        html.H6("Active Communities", className="mb-1"),
        html.Div([
            dbc.Badge(f"r/{sub}", color="outline-primary", className="me-1 mb-1")
            for sub in metrics.get('top_subreddits', [])
        ])
    ])


def create_sentiment_stats_display(trend_data: List[Dict]) -> html.Div:
    """Create sentiment statistics display"""
    if not trend_data:
        return html.Div("No data available", className="text-muted")
    
    # Calculate stats
    sentiments = [d['sentiment'] for d in trend_data]
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    # Determine trend
    if len(sentiments) >= 2:
        recent_trend = sentiments[-1] - sentiments[-2]
        if recent_trend > 0.1:
            trend_icon = "📈"
            trend_text = "Improving"
            trend_color = "success"
        elif recent_trend < -0.1:
            trend_icon = "📉"
            trend_text = "Declining"
            trend_color = "danger"
        else:
            trend_icon = "➡️"
            trend_text = "Stable"
            trend_color = "secondary"
    else:
        trend_icon = "➡️"
        trend_text = "Stable"
        trend_color = "secondary"
    
    return dbc.Row([
        dbc.Col([
            html.Strong(f"{avg_sentiment:+.2f}"),
            html.Br(),
            html.Small("7-Day Avg", className="text-muted")
        ], className="text-center"),
        dbc.Col([
            html.Strong([trend_icon, " ", trend_text]),
            html.Br(),
            html.Small("Trend", className="text-muted")
        ], className="text-center")
    ])


# Export the main layout function for use in the main app
__all__ = ['create_sentiment_panel_layout', 'get_sentiment_data', 'get_mock_sentiment_data',
           'create_sentiment_gauge', 'create_sentiment_trend_chart',
           'create_reddit_highlights_carousel', 'create_community_metrics_display',
           'create_sentiment_stats_display', 'format_api_sentiment_data',
           'format_reddit_sentiment_data']