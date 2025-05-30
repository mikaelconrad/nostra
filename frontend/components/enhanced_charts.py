"""
Enhanced charting components for BTC and ETH visualization
Optimized for the crypto trading simulation game
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import CRYPTO_SYMBOLS


def create_dual_price_chart(btc_data, eth_data, current_date, days_back=30):
    """Create a dual-axis price chart showing both BTC and ETH"""
    # Filter data for the specified period
    end_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=days_back)
    
    # Filter datasets
    btc_filtered = btc_data[
        (btc_data['Date'] >= start_date.strftime('%Y-%m-%d')) & 
        (btc_data['Date'] <= current_date)
    ].copy()
    
    eth_filtered = eth_data[
        (eth_data['Date'] >= start_date.strftime('%Y-%m-%d')) & 
        (eth_data['Date'] <= current_date)
    ].copy()
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add BTC price (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=btc_filtered['Date'],
            y=btc_filtered['Close'],
            mode='lines',
            name='Bitcoin',
            line=dict(color='#F7931A', width=2),
            hovertemplate='<b>Bitcoin</b><br>Date: %{x}<br>Price: CHF %{y:,.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add ETH price (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=eth_filtered['Date'],
            y=eth_filtered['Close'],
            mode='lines',
            name='Ethereum',
            line=dict(color='#627EEA', width=2),
            hovertemplate='<b>Ethereum</b><br>Date: %{x}<br>Price: CHF %{y:,.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add current date marker
    if not btc_filtered.empty and not eth_filtered.empty:
        current_btc = btc_filtered.iloc[-1]['Close']
        current_eth = eth_filtered.iloc[-1]['Close']
        
        fig.add_trace(
            go.Scatter(
                x=[current_date],
                y=[current_btc],
                mode='markers',
                name='Current BTC',
                marker=dict(size=8, color='#F7931A', symbol='circle'),
                showlegend=False
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=[current_date],
                y=[current_eth],
                mode='markers',
                name='Current ETH',
                marker=dict(size=8, color='#627EEA', symbol='circle'),
                showlegend=False
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title="Bitcoin vs Ethereum Price Comparison",
        xaxis_title="Date",
        hovermode='x unified',
        height=400,
        margin=dict(t=50, b=50, l=60, r=60),
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Bitcoin Price (CHF)", secondary_y=False, color='#F7931A')
    fig.update_yaxes(title_text="Ethereum Price (CHF)", secondary_y=True, color='#627EEA')
    
    return fig


def create_correlation_heatmap(btc_data, eth_data, window_days=30):
    """Create correlation heatmap between BTC and ETH returns"""
    # Calculate daily returns
    btc_returns = btc_data['Close'].pct_change().dropna()
    eth_returns = eth_data['Close'].pct_change().dropna()
    
    # Create rolling correlation
    correlation_data = []
    for i in range(window_days, len(btc_returns)):
        btc_window = btc_returns.iloc[i-window_days:i]
        eth_window = eth_returns.iloc[i-window_days:i]
        
        if len(btc_window) == len(eth_window):
            corr = np.corrcoef(btc_window, eth_window)[0, 1]
            correlation_data.append({
                'Date': btc_data.iloc[i]['Date'],
                'Correlation': corr
            })
    
    corr_df = pd.DataFrame(correlation_data)
    
    fig = go.Figure(data=go.Scatter(
        x=corr_df['Date'],
        y=corr_df['Correlation'],
        mode='lines+markers',
        name='BTC-ETH Correlation',
        line=dict(color='purple', width=2),
        fill='tonexty',
        fillcolor='rgba(128,0,128,0.1)'
    ))
    
    # Add horizontal reference lines
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="High Correlation")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Medium Correlation")
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Low Correlation")
    
    fig.update_layout(
        title="BTC-ETH Correlation Over Time",
        xaxis_title="Date",
        yaxis_title="Correlation Coefficient",
        yaxis=dict(range=[-1, 1]),
        height=300,
        margin=dict(t=50, b=50, l=60, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


def create_portfolio_evolution_chart(daily_values, benchmark_data=None):
    """Create portfolio evolution chart with optional benchmark comparison"""
    df = pd.DataFrame(daily_values)
    
    fig = go.Figure()
    
    # Add portfolio value
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#007bff', width=3),
        hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: CHF %{y:,.2f}<extra></extra>'
    ))
    
    # Add individual components
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cash'],
        mode='lines',
        name='Cash',
        line=dict(color='#85bb65', width=1),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['btc_value'],
        mode='lines',
        name='Bitcoin Value',
        line=dict(color='#F7931A', width=1),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['eth_value'],
        mode='lines',
        name='Ethereum Value',
        line=dict(color='#627EEA', width=1),
        stackgroup='one'
    ))
    
    # Add benchmark if provided
    if benchmark_data is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_data['date'],
            y=benchmark_data['value'],
            mode='lines',
            name='Benchmark (Hold)',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Value: CHF %{y:,.2f}<extra></extra>'
        ))
    
    # Add initial value reference
    if not df.empty:
        initial_value = df['total_value'].iloc[0]
        fig.add_hline(
            y=initial_value,
            line_dash="dot",
            line_color="black",
            annotation_text="Starting Value"
        )
    
    fig.update_layout(
        title="Portfolio Evolution",
        xaxis_title="Date",
        yaxis_title="Value (CHF)",
        hovermode='x unified',
        height=400,
        margin=dict(t=50, b=50, l=60, r=60),
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


def create_performance_metrics_chart(daily_values):
    """Create performance metrics visualization"""
    df = pd.DataFrame(daily_values)
    
    # Calculate metrics
    df['daily_return'] = df['total_value'].pct_change()
    df['cumulative_return'] = (df['total_value'] / df['total_value'].iloc[0] - 1) * 100
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Cumulative Return (%)', 'Daily Return (%)'],
        vertical_spacing=0.1
    )
    
    # Cumulative return
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['cumulative_return'],
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#007bff', width=2),
            fill='tonexty',
            fillcolor='rgba(0,123,255,0.1)'
        ),
        row=1, col=1
    )
    
    # Daily returns
    colors = ['green' if x >= 0 else 'red' for x in df['daily_return'].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['daily_return'] * 100,
            name='Daily Return',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add zero line for daily returns
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    
    fig.update_layout(
        height=500,
        margin=dict(t=50, b=50, l=60, r=60),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


def create_risk_return_scatter(daily_values, period_days=30):
    """Create risk-return scatter plot for different periods"""
    df = pd.DataFrame(daily_values)
    df['daily_return'] = df['total_value'].pct_change()
    
    # Calculate rolling metrics
    risk_return_data = []
    
    for i in range(period_days, len(df)):
        window_returns = df['daily_return'].iloc[i-period_days:i]
        
        mean_return = window_returns.mean() * 100
        volatility = window_returns.std() * 100
        
        risk_return_data.append({
            'date': df.iloc[i]['date'],
            'return': mean_return,
            'risk': volatility,
            'period': f"Day {i-period_days+1}-{i}"
        })
    
    rr_df = pd.DataFrame(risk_return_data)
    
    fig = go.Figure(data=go.Scatter(
        x=rr_df['risk'],
        y=rr_df['return'],
        mode='markers+lines',
        marker=dict(
            size=8,
            color=range(len(rr_df)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time Period")
        ),
        line=dict(width=1, color='rgba(0,0,0,0.3)'),
        text=rr_df['period'],
        hovertemplate='<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Break-even")
    fig.add_vline(x=rr_df['risk'].mean(), line_dash="dash", line_color="gray", annotation_text="Avg Risk")
    
    fig.update_layout(
        title="Risk-Return Profile Over Time",
        xaxis_title="Risk (Volatility %)",
        yaxis_title="Average Return (%)",
        height=400,
        margin=dict(t=50, b=50, l=60, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


def create_trading_activity_chart(transactions):
    """Create trading activity visualization"""
    if not transactions:
        return go.Figure().add_annotation(
            text="No trading activity yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Process transactions
    trade_data = []
    for t in transactions:
        trade_data.append({
            'date': t.date,
            'type': t.type,
            'crypto': t.crypto,
            'amount': t.amount,
            'value': t.total_value,
            'fee': t.fee
        })
    
    df = pd.DataFrame(trade_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Trade Volume by Date', 'Trading Fees Over Time'],
        vertical_spacing=0.15
    )
    
    # Trade volume
    for crypto in ['BTC', 'ETH']:
        crypto_trades = df[df['crypto'] == crypto]
        if not crypto_trades.empty:
            color = '#F7931A' if crypto == 'BTC' else '#627EEA'
            
            fig.add_trace(
                go.Bar(
                    x=crypto_trades['date'],
                    y=crypto_trades['value'],
                    name=f'{crypto} Trades',
                    marker_color=color,
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # Cumulative fees
    df_sorted = df.sort_values('date')
    df_sorted['cumulative_fees'] = df_sorted['fee'].cumsum()
    
    fig.add_trace(
        go.Scatter(
            x=df_sorted['date'],
            y=df_sorted['cumulative_fees'],
            mode='lines+markers',
            name='Cumulative Fees',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        margin=dict(t=50, b=50, l=60, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig