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
import gc
import logging
from functools import lru_cache
import weakref
from typing import Dict, List, Optional, Tuple, Any
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from dataclasses import dataclass, asdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging for performance monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Performance Configuration
@dataclass
class PerformanceConfig:
    """Hardware-specific performance configuration"""
    # Memory settings (MB)
    max_memory_usage: int = 8192  # 8GB default
    cache_size: int = 1000
    message_cache_size: int = 2000
    embedding_cache_size: int = 5000
    
    # CPU settings
    max_workers: int = min(4, multiprocessing.cpu_count())
    batch_size: int = 32
    
    # Model settings
    max_tokens: int = 150  # Reduced for faster generation
    temperature: float = 0.7  # Slightly lower for consistency
    
    # Hardware tier: 'low', 'medium', 'high'
    hardware_tier: str = 'medium'
    
    # Optimization flags
    enable_gpu: bool = False
    enable_quantization: bool = True
    enable_memory_mapping: bool = True
    enable_batch_processing: bool = True
    
    @classmethod
    def for_hardware_tier(cls, tier: str) -> 'PerformanceConfig':
        """Create optimized config for specific hardware tier"""
        if tier == 'low':
            return cls(
                max_memory_usage=4096,  # 4GB
                cache_size=500,
                message_cache_size=1000,
                embedding_cache_size=2000,
                max_workers=2,
                batch_size=16,
                max_tokens=100,
                temperature=0.6,
                hardware_tier='low',
                enable_quantization=True,
                enable_memory_mapping=True,
                enable_batch_processing=False
            )
        elif tier == 'high':
            return cls(
                max_memory_usage=16384,  # 16GB
                cache_size=2000,
                message_cache_size=5000,
                embedding_cache_size=10000,
                max_workers=8,
                batch_size=64,
                max_tokens=200,
                temperature=0.8,
                hardware_tier='high',
                enable_gpu=True,
                enable_quantization=False,
                enable_memory_mapping=True,
                enable_batch_processing=True
            )
        else:  # medium
            return cls()

# Global performance configuration
PERF_CONFIG = PerformanceConfig.for_hardware_tier('medium')

# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        
    def end_timer(self, operation: str):
        """End timing and record duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0
        
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation"""
        times = self.metrics.get(operation, [])
        return sum(times) / len(times) if times else 0
        
    def log_memory_usage(self):
        """Log current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage: {memory_mb:.1f} MB")
        return memory_mb

# Global performance monitor
perf_monitor = PerformanceMonitor()

# Optimized Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
bot = discord.Client(intents=intents)

# Configuration files
CONFIG_FILE = "bot_config.json"
SETUP_FILE = "setup_config.json"
CACHE_DB = "bot_cache.db"

# Optimized caching system using SQLite for persistence
class OptimizedCache:
    """High-performance caching system with SQLite backend"""
    
    def __init__(self, db_path: str = CACHE_DB, max_size: int = None):
        self.db_path = db_path
        self.max_size = max_size or PERF_CONFIG.cache_size
        self.memory_cache = {}
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database for persistent caching"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        timestamp REAL,
                        access_count INTEGER DEFAULT 1
                    )
                ''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)')
        except Exception as e:
            logger.error(f"Cache DB initialization error: {e}")
            
    @lru_cache(maxsize=1000)
    def _generate_key(self, key: str) -> str:
        """Generate optimized cache key"""
        return hashlib.md5(key.encode()).hexdigest()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU eviction"""
        cache_key = self._generate_key(key)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
            
        # Check persistent cache
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT value FROM cache WHERE key = ?', 
                    (cache_key,)
                )
                row = cursor.fetchone()
                if row:
                    value = pickle.loads(row[0])
                    # Update memory cache
                    if len(self.memory_cache) < self.max_size:
                        self.memory_cache[cache_key] = value
                    # Update access count
                    conn.execute(
                        'UPDATE cache SET access_count = access_count + 1 WHERE key = ?',
                        (cache_key,)
                    )
                    return value
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            
        return None
        
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        cache_key = self._generate_key(key)
        
        # Update memory cache
        if len(self.memory_cache) < self.max_size:
            self.memory_cache[cache_key] = value
        elif cache_key not in self.memory_cache:
            # Evict least recently used item
            self.memory_cache.pop(next(iter(self.memory_cache)))
            self.memory_cache[cache_key] = value
            
        # Update persistent cache
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache (key, value, timestamp)
                    VALUES (?, ?, ?)
                ''', (cache_key, pickle.dumps(value), time.time()))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            
    def cleanup(self):
        """Clean up expired and least used cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Remove expired entries (older than 24 hours)
                cutoff_time = time.time() - 86400
                conn.execute('DELETE FROM cache WHERE timestamp < ?', (cutoff_time,))
                
                # Keep only top accessed entries if over limit
                conn.execute('''
                    DELETE FROM cache WHERE key NOT IN (
                        SELECT key FROM cache 
                        ORDER BY access_count DESC, timestamp DESC 
                        LIMIT ?
                    )
                ''', (self.max_size * 2,))
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# Global optimized cache
cache = OptimizedCache()

def load_config():
    """Load bot configuration with caching"""
    cached_config = cache.get("bot_config")
    if cached_config:
        return cached_config
        
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                cache.set("bot_config", config, ttl=300)  # 5 minute cache
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    default_config = {
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
    cache.set("bot_config", default_config, ttl=300)
    return default_config

def load_setup_config():
    """Load setup configuration with caching"""
    cached_config = cache.get("setup_config")
    if cached_config:
        return cached_config
        
    if os.path.exists(SETUP_FILE):
        try:
            with open(SETUP_FILE, 'r') as f:
                config = json.load(f)
                cache.set("setup_config", config, ttl=300)
                return config
        except Exception as e:
            logger.error(f"Error loading setup config: {e}")
    
    default_config = {
        "discord_token": "",
        "huggingface_token": "",
        "csv_file_path": "Downloads/GC data.csv",
        "setup_complete": False
    }
    cache.set("setup_config", default_config, ttl=300)
    return default_config

# Optimized memory management
class OptimizedMemoryManager:
    """Advanced memory management for resource-constrained environments"""
    
    def __init__(self):
        self.memory_threshold = PERF_CONFIG.max_memory_usage * 0.8  # 80% threshold
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        current_memory = self.get_memory_usage()
        time_since_cleanup = time.time() - self.last_cleanup
        
        return (current_memory > self.memory_threshold or 
                time_since_cleanup > self.cleanup_interval)
                
    def cleanup_memory(self):
        """Perform aggressive memory cleanup"""
        logger.info("Starting memory cleanup...")
        
        # Clear caches
        global MESSAGE_CACHE, RESPONSE_CACHE, USER_PERSONALITIES
        
        # Keep only recent messages
        if len(MESSAGE_CACHE) > PERF_CONFIG.message_cache_size // 2:
            MESSAGE_CACHE = deque(
                list(MESSAGE_CACHE)[-PERF_CONFIG.message_cache_size // 2:],
                maxlen=PERF_CONFIG.message_cache_size
            )
            
        # Clear old response cache entries
        if len(RESPONSE_CACHE) > PERF_CONFIG.cache_size // 2:
            cache_items = list(RESPONSE_CACHE.items())
            RESPONSE_CACHE.clear()
            RESPONSE_CACHE.update(dict(cache_items[-PERF_CONFIG.cache_size // 2:]))
            
        # Clean up user personalities
        current_time = time.time()
        for user_id in list(USER_PERSONALITIES.keys()):
            if current_time - USER_PERSONALITIES[user_id].get('last_update', 0) > 86400:
                del USER_PERSONALITIES[user_id]
                
        # Force garbage collection
        gc.collect()
        
        self.last_cleanup = time.time()
        logger.info(f"Memory cleanup completed. Current usage: {self.get_memory_usage():.1f} MB")

# Global memory manager
memory_manager = OptimizedMemoryManager()

# Optimized data structures
MESSAGE_CACHE = deque(maxlen=PERF_CONFIG.message_cache_size)
USER_PERSONALITIES = {}
CONVERSATION_MEMORY = defaultdict(lambda: deque(maxlen=100))
USER_INTERACTION_HISTORY = defaultdict(lambda: {
    'last_seen': None, 
    'message_count': 0, 
    'topics': deque(maxlen=20), 
    'preferred_name': None, 
    'interactions': deque(maxlen=50)
})
RESPONSE_CACHE = {}
DEBUG_CHANNEL = None
MEMORY_THRESHOLD = 75
LAST_MEMORY_WARNING = None

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=PERF_CONFIG.max_workers)

def initialize_files():
    """Initialize CSV file with optimized I/O"""
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    # Validate and sanitize path
    csv_file = os.path.normpath(csv_file)
    if os.path.isabs(csv_file):
        allowed_dirs = [os.getcwd(), os.path.expanduser('~/Downloads'), os.path.expanduser('~/Documents')]
        if not any(csv_file.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            logger.warning(f"CSV path {csv_file} not in allowed directories. Using default.")
            csv_file = 'Downloads/GC data.csv'
    
    if '..' in csv_file or csv_file.startswith('/'):
        logger.warning(f"Potentially dangerous CSV path {csv_file}. Using default.")
        csv_file = 'Downloads/GC data.csv'
    
    try:
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Username", "User tag", "Content", "Mentions", "link"])
                
        logger.info(f"CSV file initialized: {csv_file}")
    except (OSError, IOError) as e:
        logger.error(f"Error initializing CSV file: {e}")
        csv_file = 'Downloads/GC data.csv'
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Username", "User tag", "Content", "Mentions", "link"])

@lru_cache(maxsize=1000)
def clean_markdown(text: str) -> str:
    """Optimized markdown cleaning with caching"""
    if not text:
        return ""
        
    # Pre-compiled regex patterns for better performance
    patterns = [
        (re.compile(r'\*\*(.*?)\*\*'), r'\1'),  # Bold
        (re.compile(r'\*(.*?)\*'), r'\1'),      # Italic
        (re.compile(r'__(.*?)__'), r'\1'),      # Underline
        (re.compile(r'~~(.*?)~~'), r'\1'),      # Strikethrough
        (re.compile(r'`(.*?)`'), r'\1'),        # Code
        (re.compile(r'```.*?```', re.DOTALL), ''),  # Code blocks
        (re.compile(r'<@!?(\d+)>'), ''),        # User mentions
        (re.compile(r'<#(\d+)>'), ''),          # Channel mentions
        (re.compile(r'<@&(\d+)>'), ''),         # Role mentions
        (re.compile(r'https?://\S+'), ''),      # URLs
    ]
    
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    
    return text.strip()

@lru_cache(maxsize=500)
def extract_emojis(text: str) -> str:
    """Optimized emoji extraction with caching"""
    if not text:
        return ""
    return emoji.demojize(text, delimiters=("[", "]"))

def sanitize_csv_field(field) -> str:
    """Optimized CSV field sanitization"""
    if field is None:
        return ""
    
    field_str = str(field)
    
    # Quick length check
    if len(field_str) > 1000:
        field_str = field_str[:997] + "..."
    
    # Escape dangerous characters
    if field_str and field_str[0] in ['=', '+', '-', '@', '\t', '\r', '\n']:
        field_str = "'" + field_str
    
    return field_str

def update_csv(message):
    """Optimized CSV update with batch writing"""
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    # Validate path
    csv_file = os.path.normpath(csv_file)
    if '..' in csv_file:
        logger.warning("Dangerous CSV path detected. Using default.")
        csv_file = 'Downloads/GC data.csv'
    
    try:
        # Optimize content processing
        cleaned_content = clean_markdown(message.content)
        emoji_content = extract_emojis(cleaned_content)
        
        # Prepare row data
        row_data = [
            sanitize_csv_field(message.created_at.strftime("%Y-%m-%d %H:%M:%S")),
            sanitize_csv_field(message.author.name),
            sanitize_csv_field(str(message.author.id)),
            sanitize_csv_field(emoji_content),
            sanitize_csv_field(",".join([str(m.id) for m in message.mentions])),
            sanitize_csv_field(message.attachments[0].url if message.attachments else "")
        ]
        
        # Batch write for better performance
        with open(csv_file, 'a', newline='', encoding='utf-8', buffering=8192) as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row_data)
            
    except Exception as e:
        logger.error(f"Error updating CSV: {e}")

def update_user_preference(message):
    """Optimized user preference detection"""
    user_id = message.author.id
    content_lower = message.content.lower()
    
    if 'call me' in content_lower:
        words = message.content.split()
        try:
            call_index = [w.lower() for w in words].index('call')
            if call_index + 2 < len(words) and words[call_index + 1].lower() == 'me':
                preferred_name = words[call_index + 2].strip('.,!')
                USER_INTERACTION_HISTORY[user_id]['preferred_name'] = preferred_name
                return preferred_name
        except (ValueError, IndexError):
            pass
    return None

@lru_cache(maxsize=200)
def analyze_user_personality(username: str) -> dict:
    """Optimized user personality analysis with caching"""
    cache_key = f"personality_{username}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
    
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    if not os.path.exists(csv_file):
        return {"style": "neutral"}
    
    try:
        # Read only user's messages efficiently
        df = pd.read_csv(csv_file, usecols=['Username', 'Content'])
        user_messages = df[df['Username'] == username]['Content'].dropna().tolist()
        
        if not user_messages:
            return {"style": "neutral"}
        
        # Efficient analysis
        total_messages = len(user_messages)
        message_text = ' '.join(user_messages[-50:])  # Analyze last 50 messages
        
        personality = {
            'uses_emojis': any('[' in msg and ']' in msg for msg in user_messages[-20:]),
            'uses_caps': sum(1 for msg in user_messages[-20:] if msg.isupper()) / min(20, total_messages) > 0.1,
            'uses_abbreviations': sum(1 for msg in user_messages[-20:] if len(msg.split()) < 3) / min(20, total_messages) > 0.3,
            'uses_slang': any(word in message_text.lower() for word in ['bruh', 'fr', 'ngl', 'smh', 'jk', 'lol', 'lmao']),
            'message_length': np.mean([len(msg) for msg in user_messages[-20:]]),
            'style': 'casual' if any(word in message_text.lower() for word in ['bruh', 'fr', 'ngl']) else 'neutral'
        }
        
        # Cache result
        cache.set(cache_key, personality, ttl=1800)  # 30 minute cache
        return personality
        
    except Exception as e:
        logger.error(f"Error analyzing personality for {username}: {e}")
        return {"style": "neutral"}

def generate_user_style_response(username: str, base_response: str) -> str:
    """Optimized response styling"""
    personality = analyze_user_personality(username)
    
    if not base_response:
        return base_response
    
    # Apply style modifications efficiently
    if personality.get('uses_emojis', False) and np.random.random() < 0.3:
        emojis = ['🙂', '😄', '😂', '👍', '💯']
        base_response += " " + np.random.choice(emojis)
    
    if personality.get('uses_slang', False) and np.random.random() < 0.4:
        slang_replacements = {
            "I think": "ngl I think",
            "really": "fr",
            "you know": "y'know",
            "that's": "that's",
            "going to": "gonna"
        }
        for original, slang in slang_replacements.items():
            if original in base_response:
                base_response = base_response.replace(original, slang, 1)
                break
    
    return base_response

# Optimized semantic search with batching
class OptimizedEmbedder:
    """Memory-efficient sentence embedder with caching"""
    
    def __init__(self):
        self.model = None
        self.embedding_cache = {}
        self.batch_embeddings = []
        self.batch_texts = []
        
    def _load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            if PERF_CONFIG.hardware_tier == 'low':
                # Use CPU only for low-end hardware
                self.model.device = 'cpu'
            logger.info("Sentence transformer model loaded")
    
    @lru_cache(maxsize=PERF_CONFIG.embedding_cache_size)
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text with caching"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        self._load_model()
        embedding = self.model.encode([text])[0]
        
        # Cache with size limit
        if len(self.embedding_cache) < PERF_CONFIG.embedding_cache_size:
            self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts efficiently"""
        if not texts:
            return np.array([])
        
        self._load_model()
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[text_hash]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts in batch
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, batch_size=PERF_CONFIG.batch_size)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if len(self.embedding_cache) < PERF_CONFIG.embedding_cache_size:
                    self.embedding_cache[text_hash] = embedding
        
        # Combine results
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        
        # Place new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings if uncached_texts else []):
            all_embeddings[i] = embedding
        
        return np.array(all_embeddings)

# Global optimized embedder
embedder = OptimizedEmbedder()

# Optimized FAISS index with memory mapping
class OptimizedFAISSIndex:
    """Memory-efficient FAISS index with persistence"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.index_file = "embeddings_optimized.index"
        self.metadata_file = "embeddings_metadata.json"
        self.texts = []
        self.metadata = {}
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                
                # Load metadata
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, 'r') as f:
                        self.metadata = json.load(f)
                        self.texts = self.metadata.get('texts', [])
            else:
                self._create_new_index()
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new optimized FAISS index"""
        if PERF_CONFIG.hardware_tier == 'low':
            # Use flat index for low-end hardware
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # Use IVF index for better performance on medium/high-end hardware
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, max(10, len(self.texts) // 10)))
        
        logger.info("Created new FAISS index")
    
    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        """Add embeddings to index efficiently"""
        if embeddings.size == 0:
            return
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if embeddings.shape[0] >= self.index.nlist:
                self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.texts.extend(texts)
        
        logger.info(f"Added {len(texts)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings"""
        if self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        return scores[0], indices[0]
    
    def save_index(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            self.metadata = {
                'texts': self.texts,
                'dimension': self.dimension,
                'total_vectors': self.index.ntotal
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
                
            logger.info("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

# Global optimized index
faiss_index = OptimizedFAISSIndex()

def build_index():
    """Build optimized FAISS index from CSV data"""
    perf_monitor.start_timer("build_index")
    
    setup_config = load_setup_config()
    csv_file = setup_config.get('csv_file_path', 'Downloads/GC data.csv')
    
    if not os.path.exists(csv_file):
        logger.warning(f"CSV file not found: {csv_file}")
        return
    
    try:
        # Read CSV in chunks for memory efficiency
        chunk_size = 1000 if PERF_CONFIG.hardware_tier == 'low' else 5000
        texts = []
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_texts = chunk['Content'].dropna().tolist()
            texts.extend(chunk_texts)
            
            # Process in batches to avoid memory issues
            if len(texts) >= chunk_size:
                embeddings = embedder.encode_batch(texts)
                faiss_index.add_embeddings(embeddings, texts)
                texts = []  # Clear processed texts
                
                # Memory cleanup
                if memory_manager.should_cleanup():
                    memory_manager.cleanup_memory()
        
        # Process remaining texts
        if texts:
            embeddings = embedder.encode_batch(texts)
            faiss_index.add_embeddings(embeddings, texts)
        
        # Save index
        faiss_index.save_index()
        
        duration = perf_monitor.end_timer("build_index")
        logger.info(f"Built index in {duration:.2f} seconds with {faiss_index.index.ntotal} messages")
        
    except Exception as e:
        logger.error(f"Error building index: {e}")

# Optimized MLX model management
class OptimizedMLXModel:
    """Memory-efficient MLX model with lazy loading and caching"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.loading = False
        self.generation_cache = {}
        
    async def load_model(self):
        """Load MLX model with optimization"""
        if self.model_loaded or self.loading:
            return
        
        self.loading = True
        logger.info("Loading MLX model...")
        
        try:
            setup_config = load_setup_config()
            hf_token = setup_config.get('huggingface_token', '')
            
            # Use smaller model for low-end hardware
            model_name = "mlx-community/llama-1B-instruct"
            if PERF_CONFIG.hardware_tier == 'low':
                model_name = "mlx-community/llama-1B-instruct"  # Keep same for now
            
            if hf_token:
                self.model, self.tokenizer = load(model_name, token=hf_token)
            else:
                self.model, self.tokenizer = load(model_name)
            
            # Optimize tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            
            self.model_loaded = True
            logger.info("MLX model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading MLX model: {e}")
        finally:
            self.loading = False
    
    def generate_response(self, prompt: str, history: str = "", user_context: str = "") -> str:
        """Generate optimized response with caching"""
        if not self.model_loaded:
            return "Model not loaded. Please wait..."
        
        # Check cache first
        cache_key = hashlib.md5(f"{prompt}:{history[:200]}:{user_context}".encode()).hexdigest()
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]
        
        perf_monitor.start_timer("text_generation")
        
        try:
            # Create optimized prompt
            system_prompt = """You are a member of a Discord group chat. Act like a real person, not an AI assistant.
Use casual language, slang, and emojis naturally. Respond like the people in the chat history.
Be conversational, sometimes sarcastic, and match the group's energy. Keep responses short and natural."""
            
            # Limit context length for performance
            max_context_length = 512 if PERF_CONFIG.hardware_tier == 'low' else 1024
            full_context = f"{history}\n{user_context}\n{prompt}"
            if len(full_context) > max_context_length:
                full_context = full_context[-max_context_length:]
            
            full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{full_context} [/INST]"
            
            # Tokenize with optimization
            tokens = self.tokenizer(
                full_prompt,
                return_tensors="np",
                padding=True,
                max_length=max_context_length,
                truncation=True
            )
            tokens = {k: mx.array(v) for k, v in tokens.items()}
            
            # Generate with optimized parameters
            output = generate(
                self.model,
                self.tokenizer,
                tokens,
                temp=PERF_CONFIG.temperature,
                max_tokens=PERF_CONFIG.max_tokens,
                verbose=False
            )
            
            # Cache result
            if len(self.generation_cache) < PERF_CONFIG.cache_size:
                self.generation_cache[cache_key] = output
            
            duration = perf_monitor.end_timer("text_generation")
            logger.info(f"Generated response in {duration:.2f} seconds")
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I'm having trouble generating a response right now."

# Global optimized model
mlx_model = OptimizedMLXModel()

def get_user_context(message) -> str:
    """Optimized user context retrieval"""
    user_id = message.author.id
    username = message.author.name
    
    # Check cache first
    cache_key = f"user_context_{user_id}"
    cached_context = cache.get(cache_key)
    if cached_context:
        return cached_context
    
    # Get user interaction history
    user_history = USER_INTERACTION_HISTORY.get(user_id, {})
    personality = analyze_user_personality(username)
    
    context_parts = []
    
    # Add personality info
    if personality.get('uses_emojis'):
        context_parts.append("uses emojis")
    if personality.get('uses_slang'):
        context_parts.append("uses slang")
    if personality.get('uses_caps'):
        context_parts.append("types in caps")
    
    # Add interaction info
    preferred_name = user_history.get('preferred_name', username)
    message_count = user_history.get('message_count', 0)
    
    context = f"User {preferred_name}"
    if context_parts:
        context += f" typically: {', '.join(context_parts)}"
    if message_count > 0:
        context += f". Has sent {message_count} messages."
    
    # Cache result
    cache.set(cache_key, context, ttl=600)  # 10 minute cache
    
    return context

def track_conversation_thread(message):
    """Optimized conversation tracking"""
    channel_id = message.channel.id
    user_id = message.author.id
    
    # Add to conversation memory with size limit
    conversation_data = {
        'timestamp': message.created_at,
        'author': message.author.name,
        'content': message.content[:500],  # Limit content length
        'message_id': message.id,
        'author_id': user_id
    }
    
    CONVERSATION_MEMORY[channel_id].append(conversation_data)
    
    # Update user interaction history efficiently
    user_history = USER_INTERACTION_HISTORY[user_id]
    user_history['last_seen'] = datetime.now()
    user_history['message_count'] += 1
    user_history['interactions'].append(message.content[:200])  # Limit length
    
    # Extract and store topics efficiently
    words = message.content.lower().split()
    topics = [word for word in words if len(word) > 4 and not word.startswith('http')][:5]
    user_history['topics'].extend(topics)

def get_advanced_context(message, channel_id: int) -> str:
    """Optimized context retrieval"""
    context_parts = []
    
    # Get recent conversation (limited for performance)
    if channel_id in CONVERSATION_MEMORY:
        recent_convo = list(CONVERSATION_MEMORY[channel_id])[-5:]  # Last 5 messages
        for msg in recent_convo:
            if len(msg['content']) > 100:
                content = msg['content'][:100] + "..."
            else:
                content = msg['content']
            context_parts.append(f"{msg['author']}: {content}")
    
    # Add user info
    user_id = message.author.id
    if user_id in USER_INTERACTION_HISTORY:
        user_info = USER_INTERACTION_HISTORY[user_id]
        topics = list(user_info['topics'])[-3:]  # Last 3 topics
        if topics:
            context_parts.append(f"[{message.author.name} often talks about: {', '.join(topics)}]")
    
    return "\n".join(context_parts)

# Optimized system monitoring
async def monitor_system_resources():
    """Enhanced system monitoring with optimization triggers"""
    global LAST_MEMORY_WARNING
    
    while True:
        try:
            # Get system stats
            memory_usage = perf_monitor.log_memory_usage()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check thresholds and trigger optimizations
            if memory_usage > PERF_CONFIG.max_memory_usage * 0.8:
                current_time = datetime.now()
                if not LAST_MEMORY_WARNING or (current_time - LAST_MEMORY_WARNING) > timedelta(minutes=5):
                    logger.warning(f"High memory usage: {memory_usage:.1f} MB")
                    memory_manager.cleanup_memory()
                    LAST_MEMORY_WARNING = current_time
            
            # Cleanup cache periodically
            if memory_usage > PERF_CONFIG.max_memory_usage * 0.7:
                cache.cleanup()
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"System monitor error: {e}")
            await asyncio.sleep(60)

async def periodic_optimization():
    """Periodic optimization tasks"""
    while True:
        try:
            await asyncio.sleep(1800)  # Run every 30 minutes
            
            logger.info("Running periodic optimization...")
            
            # Memory cleanup
            memory_manager.cleanup_memory()
            
            # Cache cleanup
            cache.cleanup()
            
            # Save FAISS index
            faiss_index.save_index()
            
            # Log performance metrics
            avg_generation_time = perf_monitor.get_average_time("text_generation")
            if avg_generation_time > 0:
                logger.info(f"Average text generation time: {avg_generation_time:.2f}s")
            
            logger.info("Periodic optimization completed")
            
        except Exception as e:
            logger.error(f"Periodic optimization error: {e}")

# Bot events with optimizations
@bot.event
async def on_ready():
    global DEBUG_CHANNEL
    logger.info(f'Bot logged in as {bot.user} (ID: {bot.user.id})')
    
    # Find debug channel
    config = load_config()
    debug_channel_id = config.get('debug_channel_id')
    
    if debug_channel_id:
        DEBUG_CHANNEL = discord.utils.get(bot.get_all_channels(), id=int(debug_channel_id))
    
    if not DEBUG_CHANNEL:
        for channel in bot.get_all_channels():
            if hasattr(channel, 'name') and channel.name.lower() in ['debug', 'bot-debug', 'general']:
                DEBUG_CHANNEL = channel
                break
    
    # Initialize components
    initialize_files()
    
    # Start background tasks
    asyncio.create_task(monitor_system_resources())
    asyncio.create_task(periodic_optimization())
    
    # Load model
    await mlx_model.load_model()
    
    # Build index in background
    asyncio.create_task(asyncio.to_thread(build_index))
    
    logger.info("Bot initialization completed")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    perf_monitor.start_timer("message_processing")
    
    try:
        # Track conversation efficiently
        track_conversation_thread(message)
        
        # Update CSV in background
        if PERF_CONFIG.enable_batch_processing:
            executor.submit(update_csv, message)
        else:
            update_csv(message)
        
        # Check for user preference updates
        update_user_preference(message)
        
        # Add to search index in background
        cleaned_content = clean_markdown(message.content)
        if cleaned_content:
            executor.submit(
                lambda: faiss_index.add_embeddings(
                    embedder.encode_batch([cleaned_content]),
                    [cleaned_content]
                )
            )
        
        # Check if bot should respond
        should_respond = (
            bot.user.mentioned_in(message) or
            message.content.lower().startswith(f"<@{bot.user.id}>") or
            message.content.lower().startswith(f"<@!{bot.user.id}>")
        )
        
        if should_respond:
            # Get context efficiently
            channel_id = message.channel.id
            context = get_advanced_context(message, channel_id)
            user_context = get_user_context(message)
            
            # Generate response
            response = mlx_model.generate_response(
                clean_markdown(message.content),
                context,
                user_context
            )
            
            # Apply user styling
            styled_response = generate_user_style_response(message.author.name, response)
            
            # Add randomness for human-like behavior
            if np.random.random() < 0.2:
                styled_response += " 💀"
            
            await message.channel.send(styled_response)
        
        duration = perf_monitor.end_timer("message_processing")
        if duration > 1.0:  # Log slow processing
            logger.warning(f"Slow message processing: {duration:.2f}s")
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        perf_monitor.end_timer("message_processing")

def signal_handler(sig, frame):
    """Graceful shutdown with cleanup"""
    logger.info("Shutting down gracefully...")
    
    try:
        # Save caches
        faiss_index.save_index()
        cache.cleanup()
        
        # Close thread pool
        executor.shutdown(wait=True)
        
        logger.info("Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    sys.exit(0)

# Performance testing and benchmarking
def run_performance_benchmark():
    """Run performance benchmarks"""
    logger.info("Running performance benchmarks...")
    
    # Test text generation speed
    test_prompts = [
        "Hello, how are you?",
        "What's your favorite color?",
        "Tell me a joke",
        "What do you think about AI?",
        "How's the weather?"
    ]
    
    generation_times = []
    for prompt in test_prompts:
        start_time = time.time()
        response = mlx_model.generate_response(prompt, "", "")
        duration = time.time() - start_time
        generation_times.append(duration)
        logger.info(f"Generated response for '{prompt}' in {duration:.2f}s")
    
    avg_time = sum(generation_times) / len(generation_times)
    logger.info(f"Average generation time: {avg_time:.2f}s")
    
    # Test memory usage
    memory_usage = perf_monitor.log_memory_usage()
    logger.info(f"Current memory usage: {memory_usage:.1f} MB")
    
    # Test embedding speed
    test_texts = ["This is a test message"] * 100
    start_time = time.time()
    embeddings = embedder.encode_batch(test_texts)
    embedding_time = time.time() - start_time
    logger.info(f"Encoded {len(test_texts)} texts in {embedding_time:.2f}s")
    
    return {
        'avg_generation_time': avg_time,
        'memory_usage': memory_usage,
        'embedding_time': embedding_time
    }

# Main execution
if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Validate setup
    setup_config = load_setup_config()
    
    if not setup_config.get('setup_complete', False):
        logger.error("Setup not complete! Run the web UI first:")
        logger.error("   python web_ui.py")
        logger.error("   Then visit http://localhost:5000")
        exit(1)
    
    discord_token = setup_config.get('discord_token', '')
    if not discord_token:
        logger.error("Discord token missing! Complete setup in the web UI")
        exit(1)
    
    logger.info("🚀 Launching Optimized Discord AI Chatbot...")
    logger.info(f"📊 Performance tier: {PERF_CONFIG.hardware_tier}")
    logger.info(f"💾 Max memory usage: {PERF_CONFIG.max_memory_usage} MB")
    logger.info(f"🔧 Max workers: {PERF_CONFIG.max_workers}")
    logger.info("📊 Manage via Web UI: http://localhost:5000")
    logger.info("🤖 The bot is live and will engage upon mention")
    logger.info("💡 Press Ctrl+C to stop gracefully")
    
    try:
        # Run performance benchmark if requested
        if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
            asyncio.run(mlx_model.load_model())
            run_performance_benchmark()
            exit(0)
        
        # Start the bot
        bot.run(discord_token)
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Startup Error: {e}")
        try:
            faiss_index.save_index()
            cache.cleanup()
        except:
            pass