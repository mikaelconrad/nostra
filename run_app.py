#!/usr/bin/env python3
"""
Main application runner for Cryptocurrency Investment Recommendation System
Starts both the API server and the frontend dashboard
"""

import os
import sys
import subprocess
import time
import signal
from threading import Thread

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Global process handles
api_process = None
frontend_process = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Shutting down...")
    if api_process:
        api_process.terminate()
    if frontend_process:
        frontend_process.terminate()
    sys.exit(0)

def start_api():
    """Start the Flask API server"""
    global api_process
    logger.info(f"Starting API server on port {config.API_PORT}...")
    api_process = subprocess.Popen(
        [sys.executable, "api/app.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

def start_frontend():
    """Start the Dash frontend"""
    global frontend_process
    logger.info(f"Starting frontend on port {config.FRONTEND_PORT}...")
    frontend_process = subprocess.Popen(
        [sys.executable, "frontend/app_game_simple.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import flask
        import dash
        import requests
        import pandas
        import numpy
        logger.info("All required packages are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.error("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main application entry point"""
    logger.info("="*50)
    logger.info("Cryptocurrency Trading Game")
    logger.info("="*50)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Start API server
    start_api()
    
    # Wait a bit for API to start
    time.sleep(2)
    
    # Start frontend
    start_frontend()
    
    logger.info("")
    logger.info("Application started successfully!")
    logger.info(f"API: http://localhost:{config.API_PORT}/api/health")
    logger.info(f"Frontend: http://localhost:{config.FRONTEND_PORT}")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    
    # Keep the main process running
    try:
        api_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()