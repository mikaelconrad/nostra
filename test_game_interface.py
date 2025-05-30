#!/usr/bin/env python3
"""
Test script to verify the game interface loads correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing game interface...")
print("=" * 50)

try:
    from frontend.app_game_complete import app
    print("✅ Game app imported successfully")
    
    # Check if the layout contains game elements
    layout = app.layout
    layout_str = str(layout)
    
    if "Crypto Trading Simulator" in layout_str:
        print("✅ Game title found in layout")
    else:
        print("❌ Game title NOT found in layout")
    
    if "New Game" in layout_str:
        print("✅ New Game button found in layout")
    else:
        print("❌ New Game button NOT found in layout")
        
    if "game-content" in layout_str:
        print("✅ Game content div found in layout")
    else:
        print("❌ Game content div NOT found in layout")
    
    print("\n🎮 Ready to start game with:")
    print("python test_game_interface.py --run")
    
except Exception as e:
    print(f"❌ Error importing game app: {e}")
    import traceback
    traceback.print_exc()

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--run':
    print("\n🚀 Starting game interface...")
    print("Open browser to: http://localhost:8051")
    app.run(debug=True, port=8051, host='0.0.0.0')