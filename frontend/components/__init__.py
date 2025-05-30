"""
Frontend components for the Crypto Trading Simulator
"""

from .game_setup import create_setup_layout, register_setup_callbacks
from .daily_trading import (
    create_portfolio_details, create_portfolio_chart,
    create_price_display, create_price_chart,
    create_indicators_content, create_history_content,
    register_trading_callbacks
)
from .market_view import (
    create_market_overview, create_market_comparison_chart,
    create_volume_chart, create_recommendation_display,
    register_market_callbacks
)
from .game_results import (
    create_performance_chart, create_metrics_cards,
    create_final_results_display, register_results_callbacks
)

__all__ = [
    'create_setup_layout',
    'register_setup_callbacks',
    'create_portfolio_details',
    'create_portfolio_chart',
    'create_price_display',
    'create_price_chart',
    'create_indicators_content',
    'create_history_content',
    'register_trading_callbacks',
    'create_market_overview',
    'create_market_comparison_chart',
    'create_volume_chart',
    'create_recommendation_display',
    'register_market_callbacks',
    'create_performance_chart',
    'create_metrics_cards',
    'create_final_results_display',
    'register_results_callbacks'
]