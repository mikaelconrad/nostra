#!/usr/bin/env python3
"""
Crypto Trading Game - Single Application Launcher
Starts the trading simulator on port 8050
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the Crypto Trading Game"""
    print("=" * 50)
    print("🎮 Crypto Trading Game")
    print("=" * 50)
    print("🚀 Starting trading simulator...")
    print("🌐 Opening on: http://localhost:8050")
    print("💡 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Import and run the game
    try:
        from frontend.app_game_simple import app
        app.run(
            host='0.0.0.0',
            port=8050,
            debug=False
        )
    except ImportError as e:
        print(f"❌ Error importing game: {e}")
        print("💡 Make sure you've installed dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting game: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()