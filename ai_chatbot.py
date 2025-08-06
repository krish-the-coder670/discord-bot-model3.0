import discord
import pandas as pd
import numpy as np
import csv
import os
import re
import json
from datetime import datetime, timedelta
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
import faiss
import mlx.core as mx
import emoji
import markdown
from bs4 import BeautifulSoup
import asyncio
import psutil
import threading
import time
import signal
import sys
from collections import deque, defaultdict
import hashlib

# Discord bot setup with proper bot token
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
bot = discord.Client(intents=intents)

# Configuration files
CONFIG_FILE = "bot_config.json"
SETUP_FILE = "setup_config.json"

def load_config():
    """Load bot configuration from web UI"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

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

# Advanced Memory Management
MESSAGE_CACHE = deque(maxlen=2000)  # Increased cache size for better recall
USER_PERSONALITIES = defaultdict(dict)
CONVERSATION_MEMORY = defaultdict(list)  # Comprehensive conversation tracking
USER_INTERACTION_HISTORY = defaultdict(lambda: {'last_seen': None, 'message_count': 0, 'topics': [], 'preferred_name': None, 'interactions': []})
RESPONSE_CACHE = {}
DEBUG_CHANNEL = None  # Assign this to a specific channel ID after initialization
MEMORY_THRESHOLD = 75  # Adjusted threshold for responsiveness
LAST_MEMORY_WARNING = None



def initialize_files():
    """Initialize CSV file if it doesn't exist"""
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    # Validate and sanitize the file path to prevent path traversal
    csv_file = os.path.normpath(csv_file)
    if os.path.isabs(csv_file):
        # If absolute path, ensure it's within allowed directories
        allowed_dirs = [os.getcwd(), os.path.expanduser('~/Downloads'), os.path.expanduser('~/Documents')]
        if not any(csv_file.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            print(f"Warning: CSV path {csv_file} not in allowed directories. Using default.")
            csv_file = 'Downloads/GC data.csv'
    
    # Ensure the path doesn't contain dangerous patterns
    if '..' in csv_file or csv_file.startswith('/'):
        print(f"Warning: Potentially dangerous CSV path {csv_file}. Using default.")
        csv_file = 'Downloads/GC data.csv'
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("Date,Username,User tag,Content,Mentions,link\n")
    except (OSError, IOError) as e:
        print(f"Error initializing CSV file: {e}. Using default location.")
        csv_file = 'Downloads/GC data.csv'
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("Date,Username,User tag,Content,Mentions,link\n")

def clean_markdown(text):
    """Remove Discord markdown formatting"""
    # Remove bold, italic, underline, strikethrough
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'__(.*?)__', r'\1', text)      # Underline
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
    text = re.sub(r'`(.*?)`', r'\1', text)        # Code
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks
    text = re.sub(r'<@!?(\d+)>', '', text)        # User mentions
    text = re.sub(r'<#(\d+)>', '', text)          # Channel mentions
    text = re.sub(r'<@&(\d+)>', '', text)         # Role mentions
    text = re.sub(r'https?://\S+', '', text)      # URLs
    return text.strip()

def extract_emojis(text):
    """Extract emojis and convert to descriptions"""
    return emoji.demojize(text, delimiters=("[", "]"))

def sanitize_csv_field(field):
    """Sanitize field to prevent CSV injection attacks"""
    if field is None:
        return ""
    
    field_str = str(field)
    # Remove or escape dangerous characters that could lead to CSV injection
    dangerous_chars = ['=', '+', '-', '@', '\t', '\r', '\n']
    for char in dangerous_chars:
        if field_str.startswith(char):
            field_str = "'" + field_str  # Escape with single quote
    
    # Limit field length to prevent DoS
    if len(field_str) > 1000:
        field_str = field_str[:997] + "..."
    
    return field_str

def update_csv(message):
    """Update CSV with new message, cleaning markdown and handling emojis"""
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    # Validate and sanitize the file path
    csv_file = os.path.normpath(csv_file)
    if '..' in csv_file:
        print("Warning: Dangerous CSV path detected. Using default.")
        csv_file = 'Downloads/GC data.csv'
    
    try:
        cleaned_content = clean_markdown(message.content)
        emoji_content = extract_emojis(cleaned_content)
        
        # Sanitize all fields to prevent CSV injection
        row_data = [
            sanitize_csv_field(message.created_at.strftime("%Y-%m-%d %H:%M:%S")),
            sanitize_csv_field(message.author.name),
            sanitize_csv_field(str(message.author.id)),
            sanitize_csv_field(emoji_content),
            sanitize_csv_field(",".join([str(m.id) for m in message.mentions])),
            sanitize_csv_field(message.attachments[0].url if message.attachments else "")
        ]
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields for safety
            writer.writerow(row_data)
    except (OSError, IOError) as e:
        print(f"Error writing to CSV: {e}")
    except Exception as e:
        print(f"Unexpected error updating CSV: {e}")

def update_user_preference(message):
    """Detect user's preferred way to be addressed and update interaction history"""
    user_id = message.author.id
    # Example logic to determine preferred name
    if 'call me' in message.content.lower():
        words = message.content.split()
        if 'call' in words:
            index = words.index('call')
            if index + 2 < len(words) and words[index + 1] == 'me':
                preferred_name = words[index + 2].strip().strip('.,!')
                USER_INTERACTION_HISTORY[user_id]['preferred_name'] = preferred_name
                return preferred_name
    return None

def analyze_user_personality(username):
    """Analyze user's communication patterns from CSV data"""
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    if not os.path.exists(csv_file):
        return "neutral"
    
    df = pd.read_csv(csv_file)
    user_messages = df[df['Username'] == username]['Content'].tolist()
    
    if not user_messages:
        return "neutral"
    
    # Analyze common patterns
    personality = {
        'uses_emojis': any('[' in msg and ']' in msg for msg in user_messages),
        'uses_caps': any(msg.isupper() for msg in user_messages),
        'uses_abbreviations': any(len(msg.split()) < 3 for msg in user_messages),
        'uses_slang': any(word in ' '.join(user_messages).lower() for word in ['bruh', 'fr', 'ngl', 'smh', 'jk']),
        'message_length': np.mean([len(msg) for msg in user_messages]),
        'common_words': []
    }
    
    # Get most common words
    all_words = ' '.join(user_messages).lower().split()
    word_counts = {}
    for word in all_words:
        if len(word) > 2:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    personality['common_words'] = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return personality

def generate_user_style_response(username, base_response):
    """Generate response reflecting user's communication style"""
    personality = analyze_user_personality(username)

    # Tailor response based on traits
    if personality['uses_emojis']:
        base_response += " " + np.random.choice(['🙂', '😄', '😂'], p=[0.5, 0.3, 0.2])

    if personality['uses_slang']:
        slang_mappings = {"I think": "ngl", "I believe": "fr", "really": "rly", "you know": "y'know", "ts": "this shit"}
        for key, val in slang_mappings.items():
            base_response = base_response.replace(key, val)

    if personality['uses_caps']:
        base_response = base_response.upper()

    base_response += " " + "!" * np.random.randint(1, 3)  # Add excitement

    return base_response

# Semantic search setup
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatIP(384)  # Match MiniLM dimension

def build_index():
    """Build FAISS index from CSV data"""
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    if len(df) == 0:
        print("CSV file is empty")
        return
    
    embeddings = embedder.encode(df['Content'].tolist())
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save index
    index_file = "embeddings.index"
    faiss.write_index(index, index_file)
    print(f"Built index with {len(df)} messages")

# MLX model setup
model, tokenizer = None, None
MODEL_STATUS = {'loading': False, 'loaded': False, 'error': None}
SYSTEM_MONITOR = {'cpu': 0.0, 'memory': 0.0, 'uptime': 0.0}  # More detailed monitoring

async def send_debug_message(channel, message):
    """Send debug message in code format"""
    if channel:
        try:
            # Sanitize debug message to prevent information leakage
            sanitized_message = str(message)[:500]  # Limit length
            await channel.send(f"`{sanitized_message}`")
        except Exception as e:
            print(f"Failed to send debug message: {e}")

async def monitor_system_resources():
    """Monitor system resources and send warnings"""
    global LAST_MEMORY_WARNING, SYSTEM_MONITOR
    
    while True:
        try:
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            SYSTEM_MONITOR['cpu'] = cpu_percent
            SYSTEM_MONITOR['memory'] = memory_percent
            
            # Check memory threshold
            if memory_percent > MEMORY_THRESHOLD:
                current_time = datetime.now()
                if not LAST_MEMORY_WARNING or (current_time - LAST_MEMORY_WARNING) > timedelta(minutes=5):
                    if DEBUG_CHANNEL:
                        await send_debug_message(DEBUG_CHANNEL, 
                            f"Warning: Low memory, responses will be delayed! (Memory: {memory_percent:.1f}%)")
                    LAST_MEMORY_WARNING = current_time
            
            await asyncio.sleep(30)  # Check every 30 seconds
        except Exception as e:
            print(f"System monitor error: {e}")
            await asyncio.sleep(60)

async def periodic_memory_cleanup():
    """Periodically clean up memory to prevent leaks"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            # Clean up old conversation memory
            current_time = datetime.now()
            for channel_id in list(CONVERSATION_MEMORY.keys()):
                messages = CONVERSATION_MEMORY[channel_id]
                # Keep only messages from last 7 days
                recent_messages = [
                    msg for msg in messages 
                    if hasattr(msg.get('timestamp'), 'date') and 
                    (current_time - msg['timestamp']).days < 7
                ]
                CONVERSATION_MEMORY[channel_id] = recent_messages[-100:]  # Keep max 100 recent messages
            
            # Clean up old user interaction history
            for user_id in list(USER_INTERACTION_HISTORY.keys()):
                history = USER_INTERACTION_HISTORY[user_id]
                if history.get('last_seen'):
                    # Remove users not seen in 30 days
                    if (current_time - history['last_seen']).days > 30:
                        del USER_INTERACTION_HISTORY[user_id]
                    else:
                        # Limit interaction history size
                        if 'interactions' in history:
                            history['interactions'] = history['interactions'][-50:]
                        if 'topics' in history:
                            history['topics'] = history['topics'][-20:]
            
            # Clean up response cache (keep only recent entries)
            if len(RESPONSE_CACHE) > 1000:
                # Keep only the most recent 500 entries
                cache_items = list(RESPONSE_CACHE.items())
                RESPONSE_CACHE.clear()
                RESPONSE_CACHE.update(dict(cache_items[-500:]))
            
            # Save cleaned memory to disk
            save_conversation_memory()
            
            print("Memory cleanup completed")
            
        except Exception as e:
            print(f"Memory cleanup error: {e}")
            await asyncio.sleep(3600)  # Wait an hour before retrying

def save_conversation_memory():
    """Save conversation memory to disk using secure JSON serialization"""
    memory_file = "conversation_memory.json"
    try:
        # Convert datetime objects to strings for JSON serialization
        conversations_serializable = {}
        for channel_id, messages in CONVERSATION_MEMORY.items():
            conversations_serializable[str(channel_id)] = []
            for msg in messages:
                msg_copy = msg.copy()
                if 'timestamp' in msg_copy and hasattr(msg_copy['timestamp'], 'isoformat'):
                    msg_copy['timestamp'] = msg_copy['timestamp'].isoformat()
                conversations_serializable[str(channel_id)].append(msg_copy)
        
        user_history_serializable = {}
        for user_id, history in USER_INTERACTION_HISTORY.items():
            history_copy = history.copy()
            if 'last_seen' in history_copy and history_copy['last_seen']:
                history_copy['last_seen'] = history_copy['last_seen'].isoformat()
            # Limit interactions to prevent excessive memory usage
            if 'interactions' in history_copy:
                history_copy['interactions'] = history_copy['interactions'][-50:]
            user_history_serializable[str(user_id)] = history_copy
        
        data = {
            'conversations': conversations_serializable,
            'user_history': user_history_serializable,
            'personalities': dict(USER_PERSONALITIES),
            'response_cache': dict(RESPONSE_CACHE)
        }
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving memory: {e}")

def load_conversation_memory():
    """Load conversation memory from disk using secure JSON deserialization"""
    memory_file = "conversation_memory.json"
    # Also check for old pickle file and migrate if needed
    old_memory_file = "conversation_memory.pkl"
    
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert back to proper data types
            for channel_id, messages in data.get('conversations', {}).items():
                channel_id = int(channel_id)
                for msg in messages:
                    if 'timestamp' in msg and isinstance(msg['timestamp'], str):
                        try:
                            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
                        except ValueError:
                            msg['timestamp'] = datetime.now()
                CONVERSATION_MEMORY[channel_id] = messages
                
            for user_id, history in data.get('user_history', {}).items():
                user_id = int(user_id)
                if 'last_seen' in history and isinstance(history['last_seen'], str):
                    try:
                        history['last_seen'] = datetime.fromisoformat(history['last_seen'])
                    except ValueError:
                        history['last_seen'] = None
                USER_INTERACTION_HISTORY[user_id] = history
                
            USER_PERSONALITIES.update(data.get('personalities', {}))
            RESPONSE_CACHE.update(data.get('response_cache', {}))
            print("Loaded conversation memory successfully")
        except Exception as e:
            print(f"Error loading memory: {e}")
    elif os.path.exists(old_memory_file):
        print("Found old pickle memory file. Please restart the bot to migrate to secure JSON format.")
        # Don't load the old pickle file for security reasons

def get_response_cache_key(prompt, context):
    """Generate cache key for responses"""
    combined = f"{prompt}:{context}"
    return hashlib.md5(combined.encode()).hexdigest()

def self_awareness_cycle():
    """Virtual self-awareness cycle to analyze environment and users."""
    try:
        for channel_id, messages in CONVERSATION_MEMORY.items():
            if messages:
                recent_activities = messages[-5:]  # Only analyze last 5 messages to reduce load
                print(f"Analyzing recent activities in channel: {channel_id}")
                # Example: update interaction history with inferred sentiments or activities
                for msg in recent_activities:
                    user_id = discord.utils.get(bot.get_all_members(), id=msg['author_id'])
                    action_summary = f"{msg['author']} talked about {', '.join(USER_INTERACTION_HISTORY[user_id]['topics'][:3])}"
                    print(f"User activity summary: {action_summary}")
        
    except Exception as e:
        print(f"Self-awareness cycle error: {e}")


def track_conversation_thread(message):
    """Track conversation threads for better context"""
    channel_id = message.channel.id
    
    # Add to conversation memory
    CONVERSATION_MEMORY[channel_id].append({
        'timestamp': message.created_at,
        'author': message.author.name,
        'content': message.content,
        'message_id': message.id
    })
    
    # Keep only last 100 messages per channel
    if len(CONVERSATION_MEMORY[channel_id]) > 100:
        CONVERSATION_MEMORY[channel_id] = CONVERSATION_MEMORY[channel_id][-100:]
    
    # Update user interaction history
    user_id = message.author.id
    USER_INTERACTION_HISTORY[user_id]['last_seen'] = datetime.now()
    USER_INTERACTION_HISTORY[user_id]['message_count'] += 1
    USER_INTERACTION_HISTORY[user_id]['interactions'].append(message.content)
    
    # Extract topics from message
    words = message.content.lower().split()
    topics = [word for word in words if len(word) > 4 and not word.startswith('http')]
    USER_INTERACTION_HISTORY[user_id]['topics'].extend(topics)
    USER_INTERACTION_HISTORY[user_id]['topics'] = list(set(USER_INTERACTION_HISTORY[user_id]['topics'][-20:]))

def get_advanced_context(message, channel_id):
    """Get advanced context including conversation threads"""
    context_parts = []
    
    # Get recent conversation in this channel
    if channel_id in CONVERSATION_MEMORY:
        recent_convo = CONVERSATION_MEMORY[channel_id][-10:]
        for msg in recent_convo:
            context_parts.append(f"{msg['author']}: {msg['content']}")
    
    # Get user's interaction history
    user_id = message.author.id
    if user_id in USER_INTERACTION_HISTORY:
        user_info = USER_INTERACTION_HISTORY[user_id]
        context_parts.append(f"\n[User {message.author.name} info: {user_info['message_count']} messages, topics: {', '.join(user_info['topics'][:5])}]")
    
    return "\n".join(context_parts)

async def load_mlx_model():
    """Load MLX model with Hugging Face token if available"""
    global model, tokenizer, MODEL_STATUS
    
    MODEL_STATUS['loading'] = True
    
    if DEBUG_CHANNEL:
        await send_debug_message(DEBUG_CHANNEL, "MLX is now turning on.")
    
    setup_config = load_setup_config()
    hf_token = setup_config.get('huggingface_token', '')
    
    try:
        start_time = time.time()
        
        if hf_token:
            # Use token for authenticated access
            model, tokenizer = load("mlx-community/llama-1B-instruct", token=hf_token)
        else:
            # Load without token
            model, tokenizer = load("mlx-community/llama-1B-instruct")
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        load_time = time.time() - start_time
        MODEL_STATUS['loading'] = False
        MODEL_STATUS['loaded'] = True
        
        if DEBUG_CHANNEL:
            await send_debug_message(DEBUG_CHANNEL, 
                f"MLX model loaded successfully in {load_time:.1f} seconds!")
        
        print("MLX model loaded successfully")
    except Exception as e:
        MODEL_STATUS['loading'] = False
        MODEL_STATUS['error'] = str(e)
        
        if DEBUG_CHANNEL:
            await send_debug_message(DEBUG_CHANNEL, 
                f"Error loading MLX model: {str(e)[:100]}...")
        
        print(f"Error loading MLX model: {e}")
        print("Please check your Hugging Face token or internet connection")

def generate_response(prompt, history, user_context=""):
    """Generate human-like response based on chat history and user context"""
    if model is None or tokenizer is None:
        return "Sorry, the AI model isn't loaded properly. Please check the setup."
    
    # Create context-aware prompt
    full_prompt = f"""<s>[INST] <<SYS>>
    You are a member of a Discord group chat. Act like a real person, not an AI assistant.
    Use casual language, slang, and emojis naturally. Respond like the people in the chat history.
    Be conversational, sometimes sarcastic, and match the group's energy.
    <</SYS>>

    Recent Chat History:
    {history}

    User Context: {user_context}
    
    Current Message: {prompt} [/INST]"""
    
    tokens = tokenizer(
        full_prompt,
        return_tensors="np",
        padding=True,
        max_length=1024,
        truncation=True
    )
    tokens = {k: mx.array(v) for k, v in tokens.items()}
    
    output = generate(
        model,
        tokenizer,
        tokens,
        temp=0.8,
        max_tokens=200,
        verbose=False
    )
    
    return output

def get_user_context(message):
    """Get context about the user who sent the message"""
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    if not os.path.exists(csv_file):
        return ""
    
    df = pd.read_csv(csv_file)
    user_messages = df[df['Username'] == message.author.name]['Content'].tolist()
    
    if not user_messages:
        return ""
    
    # Get user's recent messages and style
    recent_messages = user_messages[-5:] if len(user_messages) >= 5 else user_messages
    user_style = analyze_user_personality(message.author.name)
    
    context = f"User {message.author.name} typically: "
    if user_style['uses_emojis']:
        context += "uses emojis, "
    if user_style['uses_slang']:
        context += "uses slang, "
    if user_style['uses_caps']:
        context += "types in caps, "
    
    context += f"average message length: {user_style['message_length']:.1f} characters"
    
    user_id = message.author.id
    preferred_name = USER_INTERACTION_HISTORY[user_id]['preferred_name'] if user_id in USER_INTERACTION_HISTORY else message.author.name
    context += f"\nPreferred name: {preferred_name}"
    
    return context

# Bot events
@bot.event
async def on_ready():
    global DEBUG_CHANNEL
    print(f'Logged in as {bot.user}')
    print(f'Bot ID: {bot.user.id}')

    # Try to find a debug channel by name or use the first available channel
    config = load_config()
    debug_channel_id = config.get('debug_channel_id')
    
    if debug_channel_id:
        DEBUG_CHANNEL = discord.utils.get(bot.get_all_channels(), id=int(debug_channel_id))
    
    if not DEBUG_CHANNEL:
        # Try to find a channel named 'debug' or 'bot-debug'
        for channel in bot.get_all_channels():
            if hasattr(channel, 'name') and channel.name.lower() in ['debug', 'bot-debug', 'general']:
                DEBUG_CHANNEL = channel
                break
    
    if DEBUG_CHANNEL:
        await send_debug_message(DEBUG_CHANNEL, "Bot is now online.")
    else:
        print("No debug channel found. Debug messages will be printed to console only.")

    # Initialize files and build index
    initialize_files()
    build_index()

    # Start system monitor task
    asyncio.create_task(monitor_system_resources())
    
    # Start memory cleanup task
    asyncio.create_task(periodic_memory_cleanup())

    # Load conversation memory
    load_conversation_memory()

    # Load MLX model
    await load_mlx_model()

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Track and store new message
    track_conversation_thread(message)

    # Infrequent self-awareness operation
    if len(MESSAGE_CACHE) % 10 == 0:  # Example of infrequent check
        self_awareness_cycle()

    # Update user preference if mentioned
    preferred_name = update_user_preference(message)

    # Update storage with cleaned content
    update_csv(message)
    
    # Add new embedding to index
    cleaned_content = clean_markdown(message.content)
    if cleaned_content:
        new_embedding = embedder.encode([cleaned_content])
        index.add(new_embedding)
        
        # Save updated index
        index_file = "embeddings.index"
        faiss.write_index(index, index_file)
    
    # Respond to mentions or when directly addressed
    if (bot.user.mentioned_in(message) or 
        message.content.lower().startswith(f"<@{bot.user.id}>") or
        message.content.lower().startswith(f"<@!{bot.user.id}>")):
        
        # Semantic search for context
        query_embed = embedder.encode([clean_markdown(message.content)])
        _, indices = index.search(query_embed, 10)
        
        # Retrieve context from CSV
        setup_config = load_setup_config()
        csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            context_messages = []
            
            for idx in indices[0]:
                if idx < len(df):
                    row = df.iloc[idx]
                    context_messages.append(f"{row['Username']}: {row['Content']}")
            
            context = "\n".join(context_messages[-5:])  # Last 5 relevant messages
        else:
            context = ""
        
        # Get user context
        user_context = get_user_context(message)
        
        # Generate response
        response = generate_response(clean_markdown(message.content), context, user_context)
        
        # Apply user's style to response
        styled_response = generate_user_style_response(message.author.name, response)
        
        # Add some randomness to make it more human
        if np.random.random() < 0.3:
            styled_response += " 💀"
        if np.random.random() < 0.2:
            styled_response = styled_response.lower()
        
        await message.channel.send(styled_response)

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print("\n🔄 Shutting down gracefully...")
    try:
        save_conversation_memory()
        print("💾 Memory saved successfully")
    except Exception as e:
        print(f"⚠️  Error saving memory: {e}")
    
    print("👋 Bot shutdown complete")
    sys.exit(0)

# Main execution block
if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Validate setup completion
    setup_config = load_setup_config()

    if not setup_config.get('setup_complete', False):
        print("❌ Setup not complete! Run the web UI first:")
        print("   python web_ui.py")
        print("   Then visit http://localhost:5000")
        exit(1)

    discord_token = setup_config.get('discord_token', '')

    if not discord_token:
        print("❌ Discord token missing! Complete setup in the web UI")
        exit(1)

    print("🚀 Launching Discord AI Chatbot...")
    print("📊 Manage via Web UI: http://localhost:5000")
    print("🤖 The bot is live and will engage upon mention")
    print("💡 Press Ctrl+C to stop gracefully")

    try:
        # Execute the bot using proper bot token
        bot.run(discord_token)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        print("💡 Ensure Discord bot token is correct and the bot has proper permissions")
        try:
            save_conversation_memory()
        except:
            pass
