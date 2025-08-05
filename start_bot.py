#!/usr/bin/env python3
"""
Discord AI Chatbot Startup Script
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'discord.py',
        'flask',
        'pandas',
        'numpy',
        'sentence_transformers',
        'faiss-cpu',  # Use CPU version for better compatibility
        'mlx_lm',
        'mlx',
        'beautifulsoup4',  # Correct package name for bs4
        'emoji',
        'markdown',
        'psutil'  # Add missing psutil dependency
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print("   pip install -r requirements.txt")
            return False
    
    return True

def main():
    print("🚀 Discord AI Chatbot Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if setup is complete
    if not os.path.exists('setup_config.json'):
        print("\n📋 First time setup required!")
        print("Starting web UI for configuration...")
        print("Visit http://localhost:5000 to complete setup")
        
        try:
            import web_ui
            web_ui.app.run(debug=True, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled")
        return
    
    # Check setup status
    try:
        import json
        with open('setup_config.json', 'r') as f:
            setup_config = json.load(f)
        
        if not setup_config.get('setup_complete', False):
            print("\n📋 Setup not complete!")
            print("Starting web UI for configuration...")
            print("Visit http://localhost:5000 to complete setup")
            
            try:
                import web_ui
                web_ui.app.run(debug=True, host='0.0.0.0', port=5000)
            except KeyboardInterrupt:
                print("\n👋 Setup cancelled")
            return
        
        # Start the bot
        print("\n🤖 Starting Discord AI Chatbot...")
        print("📊 Web UI: http://localhost:5000")
        print("💬 Bot will respond when mentioned")
        print("Press Ctrl+C to stop")
        
        import ai_chatbot
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your Discord token and try again")

if __name__ == "__main__":
    main() 