from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import json
import os
from datetime import datetime
import pandas as pd
import csv
from io import StringIO
import zipfile
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random string

# Configuration files
CONFIG_FILE = "bot_config.json"
SETUP_FILE = "setup_config.json"
COMMANDS_FILE = "custom_commands.json"
USER_PROFILES_FILE = "user_profiles.json"
SCHEDULED_MESSAGES_FILE = "scheduled_messages.json"
NOTIFICATIONS_FILE = "notifications.json"
MODERATION_SETTINGS_FILE = "moderation_settings.json"

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

def load_custom_commands():
    """Load custom commands"""
    if os.path.exists(COMMANDS_FILE):
        with open(COMMANDS_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_user_profiles():
    """Load user profiles"""
    if os.path.exists(USER_PROFILES_FILE):
        with open(USER_PROFILES_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_scheduled_messages():
    """Load scheduled messages"""
    if os.path.exists(SCHEDULED_MESSAGES_FILE):
        with open(SCHEDULED_MESSAGES_FILE, 'r') as f:
            return json.load(f)
    return []

def load_notifications():
    """Load notifications"""
    if os.path.exists(NOTIFICATIONS_FILE):
        with open(NOTIFICATIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_notifications(notifications):
    """Save notifications"""
    with open(NOTIFICATIONS_FILE, 'w') as f:
        json.dump(notifications, f, indent=2)

def load_moderation_settings():
    """Load moderation settings"""
    if os.path.exists(MODERATION_SETTINGS_FILE):
        with open(MODERATION_SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {
        "spam_detection": True,
        "word_filter": True,
        "caps_limit": 0.7,
        "spam_threshold": 5,
        "filtered_words": ["spam", "scam", "phishing"]
    }

def save_moderation_settings(settings):
    """Save moderation settings"""
    with open(MODERATION_SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def get_sentiment_stats():
    """Get sentiment analysis statistics"""
    profiles = load_user_profiles()
    sentiment_data = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "total_users": len(profiles)
    }
    
    for user_id, profile in profiles.items():
        avg_sentiment = profile.get('avg_sentiment', 0)
        if avg_sentiment > 0.1:
            sentiment_data["positive"] += 1
        elif avg_sentiment < -0.1:
            sentiment_data["negative"] += 1
        else:
            sentiment_data["neutral"] += 1
    
    return sentiment_data

def get_topic_stats():
    """Get conversation topic statistics from bot data"""
    # In a real implementation, this would read from the bot's topic tracking
    # For now, we'll try to read from a topics file or return empty data
    topics_file = "conversation_topics.json"
    if os.path.exists(topics_file):
        with open(topics_file, 'r') as f:
            topics_data = json.load(f)
            # Count topics across all channels
            topic_counts = defaultdict(int)
            for channel_topics in topics_data.values():
                for topic_entry in channel_topics:
                    topic_counts[topic_entry.get('topic', 'unknown')] += 1
            return dict(topic_counts)
    
    # Return empty data if no topics file exists
    return {}

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

# New enhanced routes
@app.route('/commands')
def commands():
    """Custom commands management page"""
    setup_config = load_setup_config()
    if not setup_config.get('setup_complete', False):
        return redirect(url_for('setup'))
    
    commands = load_custom_commands()
    return render_template('commands.html', commands=commands)

@app.route('/api/commands', methods=['GET', 'POST', 'DELETE'])
def api_commands():
    """API for managing custom commands"""
    commands = load_custom_commands()
    
    if request.method == 'GET':
        return jsonify(commands)
    
    elif request.method == 'POST':
        data = request.get_json()
        command_name = data.get('name', '').lower()
        command_response = data.get('response', '')
        
        if not command_name or not command_response:
            return jsonify({"success": False, "message": "Name and response are required"})
        
        commands[command_name] = {
            "response": command_response,
            "description": data.get('description', f"Custom command"),
            "usage_count": 0,
            "created_by": "Web UI",
            "created_at": datetime.now().isoformat()
        }
        
        with open(COMMANDS_FILE, 'w') as f:
            json.dump(commands, f, indent=2)
        
        return jsonify({"success": True, "message": f"Command '{command_name}' added successfully"})
    
    elif request.method == 'DELETE':
        command_name = request.args.get('name', '').lower()
        if command_name in commands:
            del commands[command_name]
            with open(COMMANDS_FILE, 'w') as f:
                json.dump(commands, f, indent=2)
            return jsonify({"success": True, "message": f"Command '{command_name}' deleted"})
        return jsonify({"success": False, "message": "Command not found"})

@app.route('/users')
def users():
    """User profiles and analytics page"""
    setup_config = load_setup_config()
    if not setup_config.get('setup_complete', False):
        return redirect(url_for('setup'))
    
    profiles = load_user_profiles()
    sentiment_stats = get_sentiment_stats()
    return render_template('users.html', profiles=profiles, sentiment_stats=sentiment_stats)

@app.route('/export')
def export_page():
    """Data export page"""
    setup_config = load_setup_config()
    if not setup_config.get('setup_complete', False):
        return redirect(url_for('setup'))
    
    return render_template('export.html')

@app.route('/api/export/<export_type>')
def api_export(export_type):
    """API for exporting different types of data"""
    setup_config = load_setup_config()
    csv_path = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    if export_type == 'chat_logs':
        if os.path.exists(csv_path):
            return send_file(csv_path, as_attachment=True, download_name='chat_logs.csv')
        return jsonify({"error": "Chat logs not found"}), 404
    
    elif export_type == 'user_profiles':
        profiles = load_user_profiles()
        output = StringIO()
        if profiles:
            # Convert to CSV format
            fieldnames = ['user_id', 'username', 'total_messages', 'avg_sentiment', 'first_seen', 'last_active']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for user_id, profile in profiles.items():
                row = {
                    'user_id': user_id,
                    'username': profile.get('username', ''),
                    'total_messages': profile.get('total_messages', 0),
                    'avg_sentiment': profile.get('avg_sentiment', 0),
                    'first_seen': profile.get('first_seen', ''),
                    'last_active': profile.get('last_active', '')
                }
                writer.writerow(row)
        
        output.seek(0)
        return send_file(
            StringIO(output.getvalue()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='user_profiles.csv'
        )
    
    elif export_type == 'commands':
        commands = load_custom_commands()
        output = StringIO()
        json.dump(commands, output, indent=2)
        output.seek(0)
        return send_file(
            StringIO(output.getvalue()),
            mimetype='application/json',
            as_attachment=True,
            download_name='custom_commands.json'
        )
    
    elif export_type == 'all':
        # Create a zip file with all data
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                # Add chat logs
                if os.path.exists(csv_path):
                    zf.write(csv_path, 'chat_logs.csv')
                
                # Add user profiles
                if os.path.exists(USER_PROFILES_FILE):
                    zf.write(USER_PROFILES_FILE, 'user_profiles.json')
                
                # Add commands
                if os.path.exists(COMMANDS_FILE):
                    zf.write(COMMANDS_FILE, 'custom_commands.json')
                
                # Add scheduled messages
                if os.path.exists(SCHEDULED_MESSAGES_FILE):
                    zf.write(SCHEDULED_MESSAGES_FILE, 'scheduled_messages.json')
            
            return send_file(tmp.name, as_attachment=True, download_name='bot_data_export.zip')
    
    return jsonify({"error": "Invalid export type"}), 400

@app.route('/api/sentiment_data')
def api_sentiment_data():
    """API for sentiment analysis data"""
    sentiment_stats = get_sentiment_stats()
    topic_stats = get_topic_stats()
    
    return jsonify({
        "sentiment": sentiment_stats,
        "topics": topic_stats
    })

@app.route('/moderation')
def moderation():
    """Auto-moderation settings page"""
    setup_config = load_setup_config()
    if not setup_config.get('setup_complete', False):
        return redirect(url_for('setup'))
    
    mod_settings = load_moderation_settings()
    return render_template('moderation.html', settings=mod_settings)

@app.route('/api/moderation', methods=['GET', 'POST'])
def api_moderation():
    """API for moderation settings"""
    if request.method == 'GET':
        return jsonify(load_moderation_settings())
    
    elif request.method == 'POST':
        data = request.get_json()
        save_moderation_settings(data)
        return jsonify({"success": True, "message": "Moderation settings updated successfully"})

@app.route('/api/notifications')
def api_notifications():
    """API for getting notifications"""
    notifications = load_notifications()
    # Sort by timestamp, newest first
    notifications.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify(notifications)

@app.route('/api/notifications/<notification_id>/read', methods=['POST'])
def mark_notification_read(notification_id):
    """Mark a notification as read"""
    notifications = load_notifications()
    for notification in notifications:
        if notification.get('id') == notification_id:
            notification['read'] = True
            save_notifications(notifications)
            return jsonify({"success": True, "message": "Notification marked as read"})
    
    return jsonify({"success": False, "message": "Notification not found"}), 404

@app.route('/api/notifications/mark_all_read', methods=['POST'])
def mark_all_notifications_read():
    """Mark all notifications as read"""
    notifications = load_notifications()
    for notification in notifications:
        notification['read'] = True
    save_notifications(notifications)
    return jsonify({"success": True, "message": "All notifications marked as read"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 