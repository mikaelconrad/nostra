#!/bin/bash
# Daily update script for cryptocurrency investment recommendation system

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run data collection and processing
python backend/data_collector.py
python backend/data_processor.py
python backend/mock_sentiment_generator.py

# Train models
python backend/neural_network_model.py

# Generate daily report
python backend/portfolio_tracker.py

echo "Daily update completed at $(date)"
