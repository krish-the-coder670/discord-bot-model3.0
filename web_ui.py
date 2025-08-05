from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json
import os
from datetime import datetime
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random string

# Configuration files
CONFIG_FILE = "bot_config.json"
SETUP_FILE = "setup_config.json"

def load_config():
    """Load bot configuration"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "bot_name": "AI Chatbot",
        "bot_bio": "Just a regular person in the group chat",
        "response_style": "casual",
        "emoji_usage": True,
        "slang_usage": True,
        "auto_respond": False,
        "personality_traits": {
            "sarcasm_level": 0.5,
            "humor_level": 0.7,
            "formality_level": 0.2
        }
    }

def load_setup_config():
    """Load setup configuration"""
    if os.path.exists(SETUP_FILE):
        with open(SETUP_FILE, 'r') as f:
            return json.load(f)
    return {
        "discord_token": "",
        "huggingface_token": "",
        "csv_file_path": "Downloads/GC data.csv",
        "setup_complete": False
    }

def save_config(config):
    """Save bot configuration"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def save_setup_config(config):
    """Save setup configuration"""
    with open(SETUP_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_chat_stats():
    """Get statistics from the CSV file"""
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    if not os.path.exists(csv_file):
        return {}
    
    try:
        df = pd.read_csv(csv_file)
        
        stats = {
            "total_messages": len(df),
            "unique_users": df['Username'].nunique(),
            "most_active_user": df['Username'].value_counts().index[0] if len(df) > 0 else "None",
            "messages_today": len(df[df['Date'].str.contains(datetime.now().strftime("%Y-%m-%d"))]),
            "top_users": df['Username'].value_counts().head(5).to_dict()
        }
        
        return stats
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}

@app.route('/')
def index():
    """Main dashboard or setup redirect"""
    setup_config = load_setup_config()
    
    if not setup_config.get('setup_complete', False):
        return redirect(url_for('setup'))
    
    config = load_config()
    stats = get_chat_stats()
    return render_template('dashboard.html', config=config, stats=stats)

@app.route('/setup')
def setup():
    """Setup wizard"""
    setup_config = load_setup_config()
    return render_template('setup.html', config=setup_config)

@app.route('/api/save_setup', methods=['POST'])
def save_setup():
    """Save setup configuration"""
    setup_config = load_setup_config()
    
    setup_config['discord_token'] = request.form.get('discord_token', '')
    setup_config['huggingface_token'] = request.form.get('huggingface_token', '')
    setup_config['csv_file_path'] = request.form.get('csv_file_path', 'Downloads/GC data.csv')
    setup_config['setup_complete'] = True
    
    save_setup_config(setup_config)
    
    # Update the bot configuration file
    bot_config = load_config()
    bot_config['discord_token'] = setup_config['discord_token']
    bot_config['huggingface_token'] = setup_config['huggingface_token']
    save_config(bot_config)
    
    return jsonify({"success": True, "message": "Setup completed successfully!"})

@app.route('/api/test_discord_token', methods=['POST'])
def test_discord_token():
    """Test Discord token validity"""
    token = request.form.get('discord_token', '')
    
    if not token:
        return jsonify({"success": False, "message": "Token is required"})
    
    # Basic token format validation
    if not token.startswith('MTA') and not token.startswith('MTI'):
        return jsonify({"success": False, "message": "Invalid token format. Please check your Discord account token."})
    
    return jsonify({"success": True, "message": "Token format looks valid!"})

@app.route('/api/test_csv_path', methods=['POST'])
def test_csv_path():
    """Test CSV file path"""
    csv_path = request.form.get('csv_file_path', '')
    
    if not csv_path:
        return jsonify({"success": False, "message": "CSV path is required"})
    
    if not os.path.exists(csv_path):
        return jsonify({"success": False, "message": f"File not found: {csv_path}"})
    
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return jsonify({"success": False, "message": "CSV file is empty"})
        
        return jsonify({
            "success": True, 
            "message": f"CSV file loaded successfully! Found {len(df)} messages from {df['Username'].nunique()} users."
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading CSV: {str(e)}"})

@app.route('/profile')
def profile():
    """Profile management page"""
    setup_config = load_setup_config()
    if not setup_config.get('setup_complete', False):
        return redirect(url_for('setup'))
    
    config = load_config()
    return render_template('profile.html', config=config)

@app.route('/settings')
def settings():
    """Bot settings page"""
    setup_config = load_setup_config()
    if not setup_config.get('setup_complete', False):
        return redirect(url_for('setup'))
    
    config = load_config()
    return render_template('settings.html', config=config)

@app.route('/analytics')
def analytics():
    """Chat analytics page"""
    setup_config = load_setup_config()
    if not setup_config.get('setup_complete', False):
        return redirect(url_for('setup'))
    
    stats = get_chat_stats()
    return render_template('analytics.html', stats=stats)

@app.route('/api/update_profile', methods=['POST'])
def update_profile():
    """Update bot profile"""
    config = load_config()
    
    config['bot_name'] = request.form.get('bot_name', config['bot_name'])
    config['bot_bio'] = request.form.get('bot_bio', config['bot_bio'])
    
    save_config(config)
    return jsonify({"success": True, "message": "Profile updated successfully!"})

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    """Update bot settings"""
    config = load_config()
    
    config['response_style'] = request.form.get('response_style', config['response_style'])
    config['emoji_usage'] = request.form.get('emoji_usage') == 'true'
    config['slang_usage'] = request.form.get('slang_usage') == 'true'
    config['auto_respond'] = request.form.get('auto_respond') == 'true'
    
    # Update personality traits
    config['personality_traits']['sarcasm_level'] = float(request.form.get('sarcasm_level', 0.5))
    config['personality_traits']['humor_level'] = float(request.form.get('humor_level', 0.7))
    config['personality_traits']['formality_level'] = float(request.form.get('formality_level', 0.2))
    
    save_config(config)
    return jsonify({"success": True, "message": "Settings updated successfully!"})

@app.route('/api/chat_stats')
def chat_stats():
    """Get real-time chat statistics"""
    stats = get_chat_stats()
    return jsonify(stats)

@app.route('/api/get_setup_status')
def get_setup_status():
    """Get setup status"""
    setup_config = load_setup_config()
    return jsonify(setup_config)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 