#!/usr/bin/env python3
"""
Run the Crypto Trading Simulator
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontend.app_game_simple import app
import config

if __name__ == '__main__':
    print("="*50)
    print("Starting Crypto Trading Simulator...")
    print(f"Open your browser to: http://localhost:{config.FRONTEND_PORT}")
    print("Press Ctrl+C to stop the server")
    print("="*50)
    
    app.run(
        debug=config.DEBUG,
        port=config.FRONTEND_PORT,
        host='0.0.0.0'
    )