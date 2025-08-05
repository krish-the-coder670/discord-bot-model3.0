# Discord AI Chatbot with Web UI

An intelligent Discord chatbot that uses a real Discord account (via discord.py-self) with a web interface for management. The bot learns from chat history and responds in a human-like manner.

## Features

### 🤖 Bot Features
- **Real Discord Account**: Uses discord.py-self to run as a real Discord user
- **Intelligent Responses**: Uses MLX-LM for natural language generation
- **User Recognition**: Analyzes user personalities and adapts responses
- **Markdown Handling**: Cleans Discord markdown and converts emojis to text
- **Semantic Search**: Uses FAISS for context-aware responses
- **Human-like Behavior**: Mimics the style of users in the chat

### 🌐 Web UI Features
- **Dashboard**: Real-time statistics and overview
- **Profile Management**: Edit bot name, bio, and response style
- **Settings Control**: Configure personality traits and behavior
- **Analytics**: Chat statistics and user activity charts
- **Real-time Updates**: Live data refresh and monitoring

## Installation

### Prerequisites
- Python 3.8+
- Discord account token (not bot token)
- pip package manager

### Setup

1. **Clone and install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Get your Discord account token:**
   - Open Discord in your browser
   - Press F12 to open Developer Tools
   - Go to Network tab
   - Look for requests to `discord.com/api/v9/users/@me`
   - Copy the `authorization` header value

3. **Configure the bot:**
   - Edit `ai_chatbot.py`
   - Replace `'YOUR_DISCORD_ACCOUNT_TOKEN'` with your actual token

4. **Start the bot:**
```bash
python ai_chatbot.py
```

5. **Start the web UI:**
```bash
python web_ui.py
```

6. **Access the web interface:**
   - Open http://localhost:5000 in your browser

## Usage

### Bot Commands
- **Mention the bot**: `@YourBotName` to get a response
- **Direct messages**: The bot will respond to DMs
- **Group chats**: The bot learns from all messages in the group

### Web UI Navigation
- **Dashboard**: Overview of chat statistics and bot status
- **Profile**: Edit bot name, bio, and response style
- **Settings**: Configure personality traits and behavior
- **Analytics**: View detailed chat statistics and charts

### Configuration

#### Bot Settings
- **Response Style**: Casual, Formal, Sarcastic, Humorous, Friendly
- **Emoji Usage**: Enable/disable emoji in responses
- **Slang Usage**: Enable/disable informal language
- **Auto-respond**: Respond to all messages (use carefully)

#### Personality Traits
- **Sarcasm Level**: 0-1 scale for sarcastic responses
- **Humor Level**: 0-1 scale for funny responses
- **Formality Level**: 0-1 scale for formal language

## File Structure

```
├── ai_chatbot.py          # Main Discord bot
├── web_ui.py              # Flask web application
├── requirements.txt        # Python dependencies
├── bot_config.json        # Bot configuration (auto-generated)
├── Downloads/
│   └── GC data.csv        # Chat history data
├── templates/             # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── profile.html
│   ├── settings.html
│   └── analytics.html
└── static/               # Static assets
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

## CSV Data Format

The bot reads from `Downloads/GC data.csv` with the following columns:
- `Date`: Message timestamp
- `Username`: Sender's username
- `User tag`: Sender's Discord ID
- `Content`: Message content (cleaned)
- `Mentions`: Mentioned users
- `link`: Attachments/links

## Security Notes

⚠️ **Important**: 
- Never share your Discord account token
- The bot runs as your Discord account
- Be careful with auto-respond settings to avoid spam
- Monitor the bot's behavior in group chats

## Troubleshooting

### Common Issues

1. **Token not working:**
   - Make sure you're using an account token, not a bot token
   - Check if your account has 2FA enabled

2. **Bot not responding:**
   - Check if the bot is mentioned correctly
   - Verify the CSV file path is correct
   - Check console for error messages

3. **Web UI not loading:**
   - Ensure Flask is installed: `pip install flask`
   - Check if port 5000 is available
   - Try a different port in `web_ui.py`

4. **MLX-LM errors:**
   - Install MLX for your platform: https://ml-explore.github.io/mlx/
   - Check Python version compatibility

### Performance Tips

- The bot loads the entire CSV into memory
- Large chat histories may slow down response time
- Consider archiving old messages periodically
- Monitor memory usage with large datasets

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes. Use responsibly and in accordance with Discord's Terms of Service. 