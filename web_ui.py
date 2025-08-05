from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json
import os
import secrets
from datetime import datetime
import pandas as pd

app = Flask(__name__)
# Generate a secure random secret key
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))

# Security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

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

def validate_discord_token(token):
    """Validate Discord token format"""
    if not token:
        return False, "Discord token is required"
    
    token = token.strip()
    
    # Basic format validation - Discord tokens have specific patterns
    if len(token) < 50:
        return False, "Token too short"
    
    if len(token) > 100:
        return False, "Token too long"
    
    # Check for bot token format (should have dots and proper structure)
    if '.' not in token:
        return False, "Invalid bot token format - should contain dots"
    
    parts = token.split('.')
    if len(parts) != 3:
        return False, "Invalid bot token format - should have 3 parts separated by dots"
    
    return True, "Valid token format"

def sanitize_input(input_str, max_length=255):
    """Sanitize user input to prevent XSS and other attacks"""
    if not input_str:
        return ""
    
    # Strip whitespace and limit length
    sanitized = str(input_str).strip()[:max_length]
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\r', '\n']
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    return sanitized

@app.route('/api/save_setup', methods=['POST'])
def save_setup():
    """Save setup configuration"""
    setup_config = load_setup_config()
    
    # Get and validate inputs
    discord_token = sanitize_input(request.form.get('discord_token', ''), 100)
    huggingface_token = sanitize_input(request.form.get('huggingface_token', ''), 100)
    csv_file_path = sanitize_input(request.form.get('csv_file_path', 'Downloads/GC data.csv'), 255)
    
    # Validate Discord token
    is_valid_token, token_message = validate_discord_token(discord_token)
    if not is_valid_token:
        return jsonify({"success": False, "message": f"Discord token validation failed: {token_message}"})
    
    # Validate CSV path
    is_valid_path, path_message = validate_csv_path(csv_file_path)
    if not is_valid_path:
        return jsonify({"success": False, "message": f"CSV path validation failed: {path_message}"})
    
    setup_config['discord_token'] = discord_token
    setup_config['huggingface_token'] = huggingface_token
    setup_config['csv_file_path'] = csv_file_path
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
    """Test Discord bot token validity"""
    token = request.form.get('discord_token', '')
    
    if not token:
        return jsonify({"success": False, "message": "Token is required"})
    
    # Basic bot token format validation
    # Bot tokens typically start with specific patterns and have dots
    if '.' not in token or len(token) < 50:
        return jsonify({"success": False, "message": "Invalid bot token format. Please use a Discord bot token, not a user token."})
    
    # Check for bot token patterns (they usually have 3 parts separated by dots)
    parts = token.split('.')
    if len(parts) != 3:
        return jsonify({"success": False, "message": "Invalid bot token format. Bot tokens should have 3 parts separated by dots."})
    
    return jsonify({"success": True, "message": "Bot token format looks valid!"})

def validate_csv_path(csv_path):
    """Validate CSV file path to prevent path traversal attacks"""
    if not csv_path:
        return False, "CSV path is required"
    
    # Normalize the path
    csv_path = os.path.normpath(csv_path)
    
    # Check for path traversal attempts
    if '..' in csv_path:
        return False, "Invalid path: path traversal detected"
    
    # Check for absolute paths and ensure they're in allowed directories
    if os.path.isabs(csv_path):
        allowed_dirs = [os.getcwd(), os.path.expanduser('~/Downloads'), os.path.expanduser('~/Documents')]
        if not any(csv_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            return False, "Invalid path: not in allowed directories"
    
    # Check file extension
    if not csv_path.lower().endswith('.csv'):
        return False, "Invalid file type: only CSV files are allowed"
    
    # Limit path length
    if len(csv_path) > 255:
        return False, "Path too long"
    
    return True, "Valid path"

@app.route('/api/test_csv_path', methods=['POST'])
def test_csv_path():
    """Test CSV file path"""
    csv_path = request.form.get('csv_file_path', '').strip()
    
    # Validate the path
    is_valid, message = validate_csv_path(csv_path)
    if not is_valid:
        return jsonify({"success": False, "message": message})
    
    if not os.path.exists(csv_path):
        return jsonify({"success": False, "message": f"File not found: {os.path.basename(csv_path)}"})
    
    try:
        df = pd.read_csv(csv_path, nrows=1000)  # Limit rows read for validation
        if len(df) == 0:
            return jsonify({"success": False, "message": "CSV file is empty"})
        
        return jsonify({
            "success": True, 
            "message": f"CSV file loaded successfully! Found {len(df)} messages from {df['Username'].nunique()} users."
        })
    except Exception as e:
        return jsonify({"success": False, "message": "Error reading CSV: Invalid file format"})

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