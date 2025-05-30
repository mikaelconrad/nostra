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
from backend.neural_network_model import CryptoPredictor


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


def create_predictive_chart_btc(historical_data, current_date, days_back=90, prediction_horizons=[7, 14, 30]):
    """
    Create BTC predictive chart showing historical data plus model predictions
    
    Args:
        historical_data: DataFrame with BTC historical data
        current_date: Current simulation date
        days_back: Number of historical days to show
        prediction_horizons: List of prediction horizons [7, 14, 30]
    
    Returns:
        Plotly figure with historical data and predictions
    """
    # Filter historical data
    end_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=days_back)
    
    historical_filtered = historical_data[
        (historical_data['Date'] >= start_date.strftime('%Y-%m-%d')) & 
        (historical_data['Date'] <= current_date)
    ].copy()
    
    fig = go.Figure()
    
    # Add historical price line
    fig.add_trace(
        go.Scatter(
            x=historical_filtered['Date'],
            y=historical_filtered['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#F7931A', width=2),
            hovertemplate='<b>Historical BTC</b><br>Date: %{x}<br>Price: CHF %{y:,.2f}<extra></extra>'
        )
    )
    
    # Generate predictions using the neural network model
    try:
        predictor = CryptoPredictor('BTC-USD')
        predictions = predictor.predict_multiple_horizons(
            horizons=prediction_horizons,
            current_date=current_date
        )
        
        # Add prediction points
        pred_dates = []
        pred_prices = []
        pred_labels = []
        
        for horizon, pred_data in predictions.items():
            pred_dates.append(pred_data['prediction_date'].strftime('%Y-%m-%d'))
            pred_prices.append(pred_data['predicted_price'])
            pred_labels.append(f"{horizon}d prediction")
        
        if pred_dates:
            # Add prediction points
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        size=10,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    text=pred_labels,
                    hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Price: CHF %{y:,.2f}<extra></extra>'
                )
            )
            
            # Add prediction lines from current price to each prediction
            current_price = historical_filtered.iloc[-1]['Close'] if not historical_filtered.empty else 0
            
            for i, (pred_date, pred_price) in enumerate(zip(pred_dates, pred_prices)):
                fig.add_trace(
                    go.Scatter(
                        x=[current_date, pred_date],
                        y=[current_price, pred_price],
                        mode='lines',
                        name=f'{prediction_horizons[i]}-day trend',
                        line=dict(
                            color=['rgba(255,107,107,0.5)', 'rgba(78,205,196,0.5)', 'rgba(69,183,209,0.5)'][i],
                            width=2,
                            dash='dot'
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
    
    except Exception as e:
        print(f"Error generating BTC predictions: {str(e)}")
        # Add a note that predictions are not available
        fig.add_annotation(
            x=0.99, y=0.99,
            text="Predictions unavailable",
            showarrow=False,
            xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )
    
    # Add current date marker
    if not historical_filtered.empty:
        current_price = historical_filtered.iloc[-1]['Close']
        fig.add_trace(
            go.Scatter(
                x=[current_date],
                y=[current_price],
                mode='markers',
                name='Current Price',
                marker=dict(size=12, color='#F7931A', symbol='circle', line=dict(width=3, color='white')),
                hovertemplate='<b>Current BTC</b><br>Date: %{x}<br>Price: CHF %{y:,.2f}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Bitcoin Price - Historical Data & AI Predictions",
        xaxis_title="Date",
        yaxis_title="Price (CHF)",
        hovermode='x unified',
        height=400,
        margin=dict(t=50, b=50, l=60, r=60),
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


def create_predictive_chart_eth(historical_data, current_date, days_back=90, prediction_horizons=[7, 14, 30]):
    """
    Create ETH predictive chart showing historical data plus model predictions
    
    Args:
        historical_data: DataFrame with ETH historical data
        current_date: Current simulation date
        days_back: Number of historical days to show
        prediction_horizons: List of prediction horizons [7, 14, 30]
    
    Returns:
        Plotly figure with historical data and predictions
    """
    # Filter historical data
    end_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=days_back)
    
    historical_filtered = historical_data[
        (historical_data['Date'] >= start_date.strftime('%Y-%m-%d')) & 
        (historical_data['Date'] <= current_date)
    ].copy()
    
    fig = go.Figure()
    
    # Add historical price line
    fig.add_trace(
        go.Scatter(
            x=historical_filtered['Date'],
            y=historical_filtered['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#627EEA', width=2),
            hovertemplate='<b>Historical ETH</b><br>Date: %{x}<br>Price: CHF %{y:,.2f}<extra></extra>'
        )
    )
    
    # Generate predictions using the neural network model
    try:
        predictor = CryptoPredictor('ETH-USD')
        predictions = predictor.predict_multiple_horizons(
            horizons=prediction_horizons,
            current_date=current_date
        )
        
        # Add prediction points
        pred_dates = []
        pred_prices = []
        pred_labels = []
        
        for horizon, pred_data in predictions.items():
            pred_dates.append(pred_data['prediction_date'].strftime('%Y-%m-%d'))
            pred_prices.append(pred_data['predicted_price'])
            pred_labels.append(f"{horizon}d prediction")
        
        if pred_dates:
            # Add prediction points
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        size=10,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    text=pred_labels,
                    hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Price: CHF %{y:,.2f}<extra></extra>'
                )
            )
            
            # Add prediction lines from current price to each prediction
            current_price = historical_filtered.iloc[-1]['Close'] if not historical_filtered.empty else 0
            
            for i, (pred_date, pred_price) in enumerate(zip(pred_dates, pred_prices)):
                fig.add_trace(
                    go.Scatter(
                        x=[current_date, pred_date],
                        y=[current_price, pred_price],
                        mode='lines',
                        name=f'{prediction_horizons[i]}-day trend',
                        line=dict(
                            color=['rgba(255,107,107,0.5)', 'rgba(78,205,196,0.5)', 'rgba(69,183,209,0.5)'][i],
                            width=2,
                            dash='dot'
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
    
    except Exception as e:
        print(f"Error generating ETH predictions: {str(e)}")
        # Add a note that predictions are not available
        fig.add_annotation(
            x=0.99, y=0.99,
            text="Predictions unavailable",
            showarrow=False,
            xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )
    
    # Add current date marker
    if not historical_filtered.empty:
        current_price = historical_filtered.iloc[-1]['Close']
        fig.add_trace(
            go.Scatter(
                x=[current_date],
                y=[current_price],
                mode='markers',
                name='Current Price',
                marker=dict(size=12, color='#627EEA', symbol='circle', line=dict(width=3, color='white')),
                hovertemplate='<b>Current ETH</b><br>Date: %{x}<br>Price: CHF %{y:,.2f}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Ethereum Price - Historical Data & AI Predictions",
        xaxis_title="Date",
        yaxis_title="Price (CHF)",
        hovermode='x unified',
        height=400,
        margin=dict(t=50, b=50, l=60, r=60),
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    return fig


def create_combined_prediction_charts(btc_data, eth_data, current_date, prediction_horizons=[7, 14, 30]):
    """
    Create a combined view of BTC and ETH predictions
    
    Args:
        btc_data: DataFrame with BTC historical data
        eth_data: DataFrame with ETH historical data
        current_date: Current simulation date
        prediction_horizons: List of prediction horizons [7, 14, 30]
    
    Returns:
        Dictionary with individual charts for each horizon
    """
    charts = {}
    
    for horizon in prediction_horizons:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'Bitcoin {horizon}-Day Prediction', f'Ethereum {horizon}-Day Prediction'],
            vertical_spacing=0.1
        )
        
        # Filter data for the last 30 days
        end_date = datetime.strptime(current_date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=30)
        
        btc_filtered = btc_data[
            (btc_data['Date'] >= start_date.strftime('%Y-%m-%d')) & 
            (btc_data['Date'] <= current_date)
        ].copy()
        
        eth_filtered = eth_data[
            (eth_data['Date'] >= start_date.strftime('%Y-%m-%d')) & 
            (eth_data['Date'] <= current_date)
        ].copy()
        
        # Add BTC historical data
        fig.add_trace(
            go.Scatter(
                x=btc_filtered['Date'],
                y=btc_filtered['Close'],
                mode='lines',
                name='BTC Historical',
                line=dict(color='#F7931A', width=2)
            ),
            row=1, col=1
        )
        
        # Add ETH historical data
        fig.add_trace(
            go.Scatter(
                x=eth_filtered['Date'],
                y=eth_filtered['Close'],
                mode='lines',
                name='ETH Historical',
                line=dict(color='#627EEA', width=2)
            ),
            row=2, col=1
        )
        
        # Add predictions
        try:
            # BTC predictions
            btc_predictor = CryptoPredictor('BTC-USD')
            btc_predictions = btc_predictor.predict_multiple_horizons(
                horizons=[horizon],
                current_date=current_date
            )
            
            if horizon in btc_predictions:
                pred_data = btc_predictions[horizon]
                current_btc_price = btc_filtered.iloc[-1]['Close'] if not btc_filtered.empty else 0
                
                fig.add_trace(
                    go.Scatter(
                        x=[current_date, pred_data['prediction_date'].strftime('%Y-%m-%d')],
                        y=[current_btc_price, pred_data['predicted_price']],
                        mode='lines+markers',
                        name=f'BTC {horizon}d Prediction',
                        line=dict(color='#FF6B6B', width=3, dash='dot'),
                        marker=dict(size=8, color='#FF6B6B', symbol='star')
                    ),
                    row=1, col=1
                )
            
            # ETH predictions
            eth_predictor = CryptoPredictor('ETH-USD')
            eth_predictions = eth_predictor.predict_multiple_horizons(
                horizons=[horizon],
                current_date=current_date
            )
            
            if horizon in eth_predictions:
                pred_data = eth_predictions[horizon]
                current_eth_price = eth_filtered.iloc[-1]['Close'] if not eth_filtered.empty else 0
                
                fig.add_trace(
                    go.Scatter(
                        x=[current_date, pred_data['prediction_date'].strftime('%Y-%m-%d')],
                        y=[current_eth_price, pred_data['predicted_price']],
                        mode='lines+markers',
                        name=f'ETH {horizon}d Prediction',
                        line=dict(color='#FF6B6B', width=3, dash='dot'),
                        marker=dict(size=8, color='#FF6B6B', symbol='star')
                    ),
                    row=2, col=1
                )
        
        except Exception as e:
            print(f"Error generating predictions for {horizon} days: {str(e)}")
        
        # Update layout
        fig.update_layout(
            title=f"Crypto Predictions - {horizon} Days Ahead",
            height=600,
            margin=dict(t=50, b=50, l=60, r=60),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,249,250,1)'
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price (CHF)")
        
        charts[horizon] = fig
    
    return charts