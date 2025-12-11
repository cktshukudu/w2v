import os
import pdfplumber
import pytesseract
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
import pandas as pd
import spacy
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import zipfile
import tempfile
import string
import json
from io import StringIO
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import shutil
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import secrets
import time
import random
from typing import Dict, List, Tuple, Any

# Import our new scraper
from scraper import PaperScraper

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'emb', 'csv'}
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

# Custom stopwords
custom_stopwords = [
    "keywords", "keyword", "abstract", "doi", "authors", "author", "journal", "Abstract", "Authors", "Keywords",
    "http", "elsevier", "api", "sciencedirect", "available", "www", "ieee", "proceeding", "american", "vol",
    "volume issue1", "com procedia", "journalofbusinessresearch68", "proceeding", "european", "nber",
    "journalofclinicalepidemiology152", "socialsciences", "com", "procedia", "copyright", "technologyinsociety48",
    "online", "heliyon9", "energyreports", "procedia computer science", "et al", "cid", "org", "org j", "int j",
    "int", "e e", "j", "e", "al", "et", "springer science business",
]

# Initialize NLP components
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "Word2Vec"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "V@lidation@uj2025"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        print("Error connecting to PostgreSQL database:", e)
        return None

def init_db():
    """Initialize database tables"""
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to database. Skipping initialization.")
        return
        
    cur = conn.cursor()
    
    try:
        # Create users table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(200) NOT NULL,
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                is_admin BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        # Create user_projects table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_projects (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                project_name VARCHAR(100) NOT NULL,
                description TEXT,
                is_public BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, project_name)
            )
        """)
        
        # Create saved_heatmaps table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS saved_heatmaps (
                id SERIAL PRIMARY KEY,
                project_id INTEGER REFERENCES user_projects(id) ON DELETE CASCADE,
                heatmap_name VARCHAR(100) NOT NULL,
                category_x VARCHAR(100),
                category_y VARCHAR(100),
                words_x JSONB,
                words_y JSONB,
                similarity_matrix JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user_sessions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                session_token VARCHAR(100) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Create user_activity table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_activity (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                activity_type VARCHAR(50) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert admin user if not exists
        cur.execute("""
            INSERT INTO users (username, email, password_hash, first_name, last_name, is_admin) 
            VALUES ('admin', 'admin@example.com', %s, 'Admin', 'User', TRUE)
            ON CONFLICT (username) DO NOTHING
        """, (generate_password_hash('admin123'),))
        
        conn.commit()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def login_required(f):
    """Decorator to require login for routes"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_csv_file(file_path):
    """Validate CSV file before processing"""
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_LINES = 500000  # Maximum lines to process
    
    file_size = os.path.getsize(file_path)
    
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large ({file_size/1024/1024:.2f}MB). Maximum allowed: {MAX_FILE_SIZE/1024/1024}MB")
    
    # Count lines to estimate processing time
    line_count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_count += 1
            if line_count > MAX_LINES:
                raise ValueError(f"File has too many lines ({line_count}+). Maximum allowed: {MAX_LINES}")
    
    return True

def preprocess_text(text):
    """Preprocess text by removing URLs, tokenizing, and cleaning"""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Use a regular expression to tokenize based on word boundaries
    words = re.findall(r'\b\w+\b', text)
    # Remove numbers, dates, special characters, and stopwords
    words = [word for word in words if word.isalnum() and not word.isnumeric()]
    words = [word for word in words if word not in custom_stopwords]
    words = [word for word in words if word not in stopwords.words('english')]
    # Apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]  
    return ' '.join(words)

def extract_phrases(text, max_phrase_length=3):
    """Extract phrases from text using spaCy"""
    # Process the text with spaCy
    doc = nlp(text)
    
    # Create a dictionary to store phrase counts
    phrase_count_dict = {}
    
    # Iterate over all tokens in the document
    for i, token in enumerate(doc):
        if token.is_stop:
            continue

        words = [token.lemma_]
        for j in range(i + 1, min(i + max_phrase_length, len(doc))):
            if doc[j].is_stop or doc[j].is_punct:
                break

            words.append(doc[j].lemma_)
            phrase = ' '.join(words)

            if any(char.isdigit() or char in string.punctuation or len(phrase) == 1 or '\n' in phrase for char in phrase):
                continue

            phrase_count_dict[phrase] = phrase_count_dict.get(phrase, 0) + 1
    
    return phrase_count_dict

def parse_emb_file(file_path):
    """Parse .emb file and return word vectors dictionary"""
    word_vectors = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip the first line (header with dimensions)
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                try:
                    vector = [float(x) for x in parts[1:]]
                    word_vectors[word] = vector
                except ValueError:
                    continue  # Skip lines with invalid vector data
    except Exception as e:
        raise ValueError(f"Error parsing embedding file: {str(e)}")
    
    return word_vectors

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def parse_csv_categories(file_path):
    """Parse CSV file and extract categories with their words - with better cleaning"""
    categories = {}
    
    try:
        # First, try to determine the file structure with a small sample
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Count columns in first line to understand structure
        first_line_parts = first_line.split(',')
        num_columns = len(first_line_parts)
        
        # If file is too large, use chunked reading
        file_size = os.path.getsize(file_path)
        use_chunks = file_size > 10 * 1024 * 1024  # 10MB threshold
        
        if use_chunks:
            print(f"Large file detected ({file_size/1024/1024:.2f}MB), using chunked processing...")
            
            # Process in chunks to avoid memory issues
            chunk_size = 10000  # Process 10,000 rows at a time
            for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                print(f"Processing chunk {chunk_idx + 1}...")
                
                if 'Category' in chunk.columns and 'Word' in chunk.columns:
                    # Structure: Category, Word
                    for _, row in chunk.iterrows():
                        category = str(row['Category']).strip()
                        word = str(row['Word']).strip()
                        if category and word and is_valid_word(word):
                            if category not in categories:
                                categories[category] = set()
                            categories[category].add(word.lower())
                
                elif 'Category' in chunk.columns:
                    # Structure with multiple word columns
                    category_col = 'Category'
                    word_cols = [col for col in chunk.columns if col != category_col]
                    
                    for _, row in chunk.iterrows():
                        category = str(row[category_col]).strip()
                        if category:
                            if category not in categories:
                                categories[category] = set()
                            
                            for col in word_cols:
                                word = str(row[col]).strip()
                                if word and is_valid_word(word):
                                    categories[category].add(word.lower())
                
                else:
                    # Assume first column is categories, rest are words
                    for _, row in chunk.iterrows():
                        category = str(row.iloc[0]).strip()
                        if category:
                            if category not in categories:
                                categories[category] = set()
                            
                            for i in range(1, min(len(row), 50)):  # Limit to first 50 columns
                                word = str(row.iloc[i]).strip()
                                if word and is_valid_word(word):
                                    categories[category].add(word.lower())
                
                # Early termination for very large files
                if len(categories) > 1000 and chunk_idx > 10:  # Stop after 1000 categories or 10 chunks
                    print("Large number of categories reached, stopping processing...")
                    break
                    
        else:
            # For smaller files, use regular processing
            df = pd.read_csv(file_path)
            
            if 'Category' in df.columns and 'Word' in df.columns:
                # Structure: Category, Word
                for _, row in df.iterrows():
                    category = str(row['Category']).strip()
                    word = str(row['Word']).strip()
                    if category and word and is_valid_word(word):
                        if category not in categories:
                            categories[category] = set()
                        categories[category].add(word.lower())
            
            elif 'Category' in df.columns:
                # Structure with multiple word columns
                category_col = 'Category'
                word_cols = [col for col in df.columns if col != category_col]
                
                for _, row in df.iterrows():
                    category = str(row[category_col]).strip()
                    if category:
                        if category not in categories:
                            categories[category] = set()
                        
                        for col in word_cols:
                            word = str(row[col]).strip()
                            if word and is_valid_word(word):
                                categories[category].add(word.lower())
            
            else:
                # Assume first column is categories, rest are words
                for _, row in df.iterrows():
                    category = str(row.iloc[0]).strip()
                    if category:
                        if category not in categories:
                            categories[category] = set()
                        
                        for i in range(1, min(len(row), 50)):  # Limit to first 50 columns
                            word = str(row.iloc[i]).strip()
                            if word and is_valid_word(word):
                                categories[category].add(word.lower())
        
        # Convert sets to lists for final output
        for category in categories:
            categories[category] = list(categories[category])
            
        # Remove empty categories
        for category in list(categories.keys()):
            if not categories[category]:
                del categories[category]
        
        # Filter out words that are purely numeric or very short
        for category in categories:
            categories[category] = [word for word in categories[category] 
                                  if is_valid_word(word) and len(word) > 1]
                
        print(f"Successfully processed {len(categories)} categories")
        print(f"Sample categories: {list(categories.keys())[:5]}")
                
    except Exception as e:
        print(f"Error parsing CSV with pandas: {e}")
        # Fallback to line-by-line processing for problematic files
        try:
            categories = parse_csv_line_by_line(file_path)
        except Exception as e2:
            raise ValueError(f"Could not parse CSV file: {str(e2)}")
    
    return categories

def is_valid_word(word):
    """Check if a word is valid for processing"""
    if not word or word.lower() in ['nan', 'null', 'none', '']:
        return False
    
    # Remove quotes and extra spaces
    word = word.strip().strip('"').strip("'")
    
    # Skip purely numeric words
    if word.isdigit():
        return False
    
    # Skip words that are just punctuation or special characters
    if not any(char.isalpha() for char in word):
        return False
    
    # Skip very short words (unless they're acronyms)
    if len(word) < 2 and not word.isupper():
        return False
    
    # Skip common data artifacts
    if word.lower() in ['na', 'n/a', '#n/a', 'undefined']:
        return False
    
    return True

def parse_csv_line_by_line(file_path):
    """Fallback CSV parser that processes line by line with better filtering"""
    categories = {}
    line_count = 0
    max_lines = 100000  # Safety limit
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_count += 1
            if line_count > max_lines:
                print(f"Reached maximum line limit ({max_lines}), stopping processing")
                break
                
            line = line.strip()
            if not line:
                continue
                
            # Clean the line and split
            parts = [part.strip().strip('"').strip("'") for part in line.split(',')]
            parts = [p for p in parts if p and p.lower() != 'nan']
            
            if len(parts) >= 2:
                category = parts[0]
                words = parts[1:]
                
                if category and words:
                    if category not in categories:
                        categories[category] = set()
                    
                    for word in words:
                        if is_valid_word(word) and len(word) < 100:  # Skip very long "words"
                            categories[category].add(word.lower())
    
    # Convert sets to lists and filter
    for category in categories:
        categories[category] = [word for word in categories[category] if is_valid_word(word)]
    
    return categories

def create_heatmap_image(words_x, words_y, matrix, category_x, category_y, stats):
    """Create a high-quality heatmap image"""
    # Dimensions
    cell_size = 60
    header_size = 80
    margin = 20
    
    # Calculate image size
    width = len(words_y) * cell_size + header_size + margin * 2
    height = len(words_x) * cell_size + header_size + margin * 2
    
    # Create image
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a nicer font
        font = ImageFont.truetype("arial.ttf", 12)
        font_bold = ImageFont.truetype("arialbd.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw title
    title = f"Word Similarity Heatmap: {category_x} vs {category_y}"
    try:
        title_bbox = draw.textbbox((0, 0), title, font=font_bold)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((width - title_width) // 2, margin // 2), title, fill='#2e59d9', font=font_bold)
    except:
        draw.text((width // 2 - 100, margin // 2), title, fill='#2e59d9')
    
    # Draw headers
    # Y-axis headers (vertical)
    for i, word in enumerate(words_x):
        x = margin + 10
        y = header_size + margin + (i * cell_size) + (cell_size // 2)
        
        # Truncate long words
        display_word = word[:15] + '...' if len(word) > 15 else word
        
        try:
            # Create a rotated text
            text_bbox = draw.textbbox((0, 0), display_word, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_image = Image.new('RGBA', (text_height, text_width), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_image)
            text_draw.text((0, 0), display_word, fill='#2e59d9', font=font)
            rotated_text = text_image.rotate(90, expand=True)
            image.paste(rotated_text, (x, y - text_width // 2), rotated_text)
        except:
            # Fallback: draw text normally
            draw.text((x, y - 10), display_word, fill='#2e59d9')
    
    # X-axis headers
    for j, word in enumerate(words_y):
        x = header_size + margin + (j * cell_size) + (cell_size // 2)
        y = margin + (header_size // 2)
        
        # Truncate long words
        display_word = word[:15] + '...' if len(word) > 15 else word
        
        try:
            text_bbox = draw.textbbox((0, 0), display_word, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text((x - text_width // 2, y - text_height // 2), display_word, fill='#2e59d9', font=font)
        except:
            draw.text((x - 30, y - 5), display_word, fill='#2e59d9')
    
    # Draw heatmap cells
    for i in range(len(words_x)):
        for j in range(len(words_y)):
            similarity = matrix[i][j]
            normalized_value = (similarity - stats['min']) / (stats['max'] - stats['min']) if stats['max'] > stats['min'] else 0.5
            
            # Calculate color (blue gradient)
            hue = 210  # Blue
            saturation = 70
            lightness = 90 - (normalized_value * 40)  # From light to dark
            
            # Convert HSL to RGB
            h = hue / 360
            s = saturation / 100
            l = lightness / 100
            
            # HSL to RGB conversion
            if s == 0:
                r = g = b = l
            else:
                def hue_to_rgb(p, q, t):
                    if t < 0: t += 1
                    if t > 1: t -= 1
                    if t < 1/6: return p + (q - p) * 6 * t
                    if t < 1/2: return q
                    if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                    return p
                
                q = l * (1 + s) if l < 0.5 else l + s - l * s
                p = 2 * l - q
                r = hue_to_rgb(p, q, h + 1/3)
                g = hue_to_rgb(p, q, h)
                b = hue_to_rgb(p, q, h - 1/3)
            
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            color = (r, g, b)
            
            # Draw cell
            x1 = header_size + margin + (j * cell_size)
            y1 = header_size + margin + (i * cell_size)
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='#e3e6f0')
            
            # Draw similarity value
            text_color = 'white' if normalized_value > 0.5 else 'black'
            value_text = f"{similarity:.2f}"
            try:
                text_bbox = draw.textbbox((0, 0), value_text, font=font_small)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.text(
                    (x1 + (cell_size - text_width) // 2, y1 + (cell_size - text_height) // 2),
                    value_text,
                    fill=text_color,
                    font=font_small
                )
            except:
                draw.text((x1 + 15, y1 + 15), value_text, fill=text_color)
    
    # Draw legend
    legend_x = margin
    legend_y = height - 40
    legend_width = 200
    legend_height = 20
    
    # Draw gradient bar
    for x in range(legend_width):
        normalized_x = x / legend_width
        # Calculate color for this position
        lightness = 90 - (normalized_x * 40)
        
        # Convert HSL to RGB (simplified)
        h = 210 / 360
        s = 70 / 100
        l = lightness / 100
        
        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p
            
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        
        draw.line([legend_x + x, legend_y, legend_x + x, legend_y + legend_height], fill=(r, g, b))
    
    # Draw legend labels
    try:
        draw.text((legend_x, legend_y + legend_height + 5), "Low", fill='black', font=font_small)
        draw.text((legend_x + legend_width - 20, legend_y + legend_height + 5), "High", fill='black', font=font_small)
        draw.text((legend_x + legend_width // 2 - 20, legend_y + legend_height + 5), "Similarity", fill='black', font=font_small)
    except:
        draw.text((legend_x, legend_y + legend_height + 5), "Low", fill='black')
        draw.text((legend_x + legend_width - 20, legend_y + legend_height + 5), "High", fill='black')
        draw.text((legend_x + legend_width // 2 - 20, legend_y + legend_height + 5), "Similarity", fill='black')
    
    # Draw stats
    stats_text = f"Range: {stats['min']:.3f} - {stats['max']:.3f} | Mean: {stats['mean']:.3f}"
    try:
        stats_bbox = draw.textbbox((0, 0), stats_text, font=font_small)
        stats_width = stats_bbox[2] - stats_bbox[0]
        draw.text((width - stats_width - margin, height - 20), stats_text, fill='#6c757d', font=font_small)
    except:
        draw.text((width - 250, height - 20), stats_text, fill='#6c757d')
    
    return image

def create_multi_heatmap_image(words_x, words_y, matrix, 
                               category_labels_x, category_labels_y,
                               categories_x, categories_y, stats):
    """Create a high-quality heatmap image for multiple categories"""
    # Dimensions
    cell_size = 40  # Smaller cells for more data
    header_size = 100
    category_header_size = 30
    margin = 20
    
    # Calculate image size
    width = len(words_y) * cell_size + header_size + margin * 2
    height = len(words_x) * cell_size + header_size + category_header_size + margin * 2
    
    # Create image
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use nicer fonts
        font_small = ImageFont.truetype("arial.ttf", 8)
        font_medium = ImageFont.truetype("arial.ttf", 10)
        font_bold = ImageFont.truetype("arialbd.ttf", 12)
    except:
        # Fallback to default font
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_bold = ImageFont.load_default()
    
    # Draw title
    title = f"Multi-Category Semantic Similarity Analysis"
    try:
        title_bbox = draw.textbbox((0, 0), title, font=font_bold)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((width - title_width) // 2, margin // 2), title, fill='#2e59d9', font=font_bold)
    except:
        draw.text((width // 2 - 100, margin // 2), title, fill='#2e59d9')
    
    # Draw category headers for X-axis
    if categories_y and len(categories_y) > 1:
        current_x = header_size + margin
        for category in categories_y:
            # Count words in this category
            category_word_count = sum(1 for label in category_labels_y if label == category)
            category_width = category_word_count * cell_size
            
            # Draw category background
            draw.rectangle([current_x, margin + header_size // 2 - 10,
                          current_x + category_width, margin + header_size // 2 + 10],
                         fill='#e3e6f0', outline='#d1d5db')
            
            # Draw category label
            try:
                cat_bbox = draw.textbbox((0, 0), category, font=font_medium)
                cat_width = cat_bbox[2] - cat_bbox[0]
                cat_height = cat_bbox[3] - cat_bbox[1]
                draw.text((current_x + (category_width - cat_width) // 2, 
                          margin + (header_size - cat_height) // 2),
                         category, fill='#2e59d9', font=font_medium)
            except:
                draw.text((current_x + 10, margin + header_size // 2 - 5), 
                         category, fill='#2e59d9')
            
            current_x += category_width
    
    # Draw category headers for Y-axis
    if categories_x and len(categories_x) > 1:
        current_y = header_size + margin + category_header_size
        for category in categories_x:
            # Count words in this category
            category_word_count = sum(1 for label in category_labels_x if label == category)
            category_height = category_word_count * cell_size
            
            # Draw category background
            draw.rectangle([margin + 10, current_y,
                          margin + header_size - 10, current_y + category_height],
                         fill='#e3e6f0', outline='#d1d5db')
            
            # Draw category label (rotated)
            try:
                # Create text image
                text_image = Image.new('RGBA', (category_height, 100), (0, 0, 0, 0))
                text_draw = ImageDraw.Draw(text_image)
                cat_bbox = text_draw.textbbox((0, 0), category, font=font_medium)
                text_draw.text((0, 0), category, fill='#2e59d9', font=font_medium)
                
                # Rotate text
                rotated_text = text_image.rotate(90, expand=True)
                image.paste(rotated_text, 
                           (margin + 20, current_y + (category_height - 100) // 2),
                           rotated_text)
            except:
                draw.text((margin + 20, current_y + category_height // 2), 
                         category, fill='#2e59d9')
            
            current_y += category_height
    
    # Draw X-axis headers (individual words)
    current_x = header_size + margin
    for j, word in enumerate(words_y):
        x = current_x + (j * cell_size) + (cell_size // 2)
        y = margin + header_size + category_header_size // 2
        
        # Truncate long words
        display_word = word[:10] + '...' if len(word) > 10 else word
        
        try:
            text_bbox = draw.textbbox((0, 0), display_word, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text((x - text_width // 2, y - text_height // 2), 
                     display_word, fill='#2e59d9', font=font_small)
        except:
            draw.text((x - 20, y - 5), display_word, fill='#2e59d9')
    
    # Draw Y-axis headers (individual words)
    current_y = header_size + margin + category_header_size
    for i, word in enumerate(words_x):
        x = margin + 10
        y = current_y + (i * cell_size) + (cell_size // 2)
        
        # Truncate long words
        display_word = word[:10] + '...' if len(word) > 10 else word
        
        try:
            # Create rotated text
            text_bbox = draw.textbbox((0, 0), display_word, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_image = Image.new('RGBA', (text_height, text_width), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_image)
            text_draw.text((0, 0), display_word, fill='#2e59d9', font=font_small)
            rotated_text = text_image.rotate(90, expand=True)
            image.paste(rotated_text, (x, y - text_width // 2), rotated_text)
        except:
            draw.text((x, y - 10), display_word, fill='#2e59d9')
    
    # Draw heatmap cells
    current_y = header_size + margin + category_header_size
    for i in range(len(words_x)):
        current_x = header_size + margin
        for j in range(len(words_y)):
            similarity = matrix[i][j]
            normalized_value = (similarity - stats['min']) / (stats['max'] - stats['min']) if stats['max'] > stats['min'] else 0.5
            
            # Calculate color
            hue = 210  # Blue
            saturation = 70
            lightness = 90 - (normalized_value * 40)
            
            # Convert HSL to RGB
            h = hue / 360
            s = saturation / 100
            l = lightness / 100
            
            # HSL to RGB conversion
            if s == 0:
                r = g = b = l
            else:
                def hue_to_rgb(p, q, t):
                    if t < 0: t += 1
                    if t > 1: t -= 1
                    if t < 1/6: return p + (q - p) * 6 * t
                    if t < 1/2: return q
                    if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                    return p
                
                q = l * (1 + s) if l < 0.5 else l + s - l * s
                p = 2 * l - q
                r = hue_to_rgb(p, q, h + 1/3)
                g = hue_to_rgb(p, q, h)
                b = hue_to_rgb(p, q, h - 1/3)
            
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            color = (r, g, b)
            
            # Draw cell
            x1 = current_x + (j * cell_size)
            y1 = current_y + (i * cell_size)
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='#e3e6f0')
            
            # Draw similarity value (only if cell is big enough)
            if cell_size >= 30:
                text_color = 'white' if normalized_value > 0.5 else 'black'
                value_text = f"{similarity:.2f}"
                try:
                    text_bbox = draw.textbbox((0, 0), value_text, font=font_small)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    draw.text(
                        (x1 + (cell_size - text_width) // 2, y1 + (cell_size - text_height) // 2),
                        value_text,
                        fill=text_color,
                        font=font_small
                    )
                except:
                    draw.text((x1 + 5, y1 + 5), value_text, fill=text_color)
    
    # Draw legend
    legend_x = margin
    legend_y = height - 40
    legend_width = 200
    legend_height = 20
    
    # Draw gradient bar
    for x in range(legend_width):
        normalized_x = x / legend_width
        lightness = 90 - (normalized_x * 40)
        
        h = 210 / 360
        s = 70 / 100
        l = lightness / 100
        
        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p
            
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        
        draw.line([legend_x + x, legend_y, legend_x + x, legend_y + legend_height], fill=(r, g, b))
    
    # Draw legend labels
    try:
        draw.text((legend_x, legend_y + legend_height + 5), "Low", fill='black', font=font_small)
        draw.text((legend_x + legend_width - 20, legend_y + legend_height + 5), "High", fill='black', font=font_small)
        draw.text((legend_x + legend_width // 2 - 20, legend_y + legend_height + 5), "Similarity", fill='black', font=font_small)
    except:
        draw.text((legend_x, legend_y + legend_height + 5), "Low", fill='black')
        draw.text((legend_x + legend_width - 20, legend_y + legend_height + 5), "High", fill='black')
        draw.text((legend_x + legend_width // 2 - 20, legend_y + legend_height + 5), "Similarity", fill='black')
    
    # Draw stats
    stats_text = f"Matrix: {len(words_x)}×{len(words_y)} | Range: {stats['min']:.3f}-{stats['max']:.3f}"
    try:
        stats_bbox = draw.textbbox((0, 0), stats_text, font=font_small)
        stats_width = stats_bbox[2] - stats_bbox[0]
        draw.text((width - stats_width - margin, height - 20), 
                 stats_text, fill='#6c757d', font=font_small)
    except:
        draw.text((width - 250, height - 20), stats_text, fill='#6c757d')
    
    return image

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum file size is 100MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error occurred.'}), 500

@app.route('/')
def index():
    """Home page - public landing page with login option"""
    # If user is logged in, show the main application dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    # If user is not logged in, show public landing page
    conn = get_db_connection()
    if conn is None:
        return render_template('index.html', public_heatmaps=[], user=None)
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get public heatmaps for preview
        cur.execute("""
            SELECT h.*, u.username, p.project_name 
            FROM saved_heatmaps h
            JOIN user_projects p ON h.project_id = p.id
            JOIN users u ON p.user_id = u.id
            WHERE p.is_public = TRUE
            ORDER BY h.created_at DESC
            LIMIT 6
        """)
        public_heatmaps = cur.fetchall()
    except Exception as e:
        print(f"Error fetching public heatmaps: {e}")
        public_heatmaps = []
    finally:
        cur.close()
        conn.close()
    
    return render_template('index.html', public_heatmaps=public_heatmaps, user=None)

@app.route('/upload_page')
@login_required
def upload_page():
    """File upload page"""
    return render_template('upload.html', user=session.get('user_info'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page - now the landing page"""
    # If user is already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed. Please try again later.', 'danger')
            return render_template('login.html')
        
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get user
            cur.execute("""
                SELECT id, username, email, password_hash, first_name, last_name, is_admin 
                FROM users 
                WHERE username = %s AND is_active = TRUE
            """, (username,))
            
            user = cur.fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                # Create session
                session_token = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(days=7)
                
                # Store session in database
                cur.execute("""
                    INSERT INTO user_sessions (user_id, session_token, expires_at)
                    VALUES (%s, %s, %s)
                """, (user['id'], session_token, expires_at))
                
                # Update last login
                cur.execute("""
                    UPDATE users SET last_login = %s WHERE id = %s
                """, (datetime.now(), user['id']))
                
                # Log activity
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (user['id'], 'login', 'User logged in successfully'))
                
                conn.commit()
                
                # Store user info in session
                session['user_id'] = user['id']
                session['session_token'] = session_token
                session['user_info'] = {
                    'username': user['username'],
                    'email': user['email'],
                    'first_name': user['first_name'],
                    'last_name': user['last_name'],
                    'is_admin': user['is_admin']
                }
                
                flash(f'Welcome back, {user["username"]}!', 'success')
                
                # Redirect to intended page or dashboard
                next_page = request.args.get('next')
                return redirect(next_page or url_for('dashboard'))
            else:
                flash('Invalid username or password.', 'danger')
                
        except Exception as e:
            conn.rollback()
            flash('Login failed. Please try again.', 'danger')
        finally:
            cur.close()
            conn.close()
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    # If user is already logged in, redirect to index
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        first_name = request.form.get('first_name', '')
        last_name = request.form.get('last_name', '')
        
        # Basic validation
        if not username or not email or not password:
            flash('Please fill in all required fields.', 'danger')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')
        
        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed. Please try again later.', 'danger')
            return render_template('register.html')
        
        cur = conn.cursor()
        
        try:
            # Check if user exists
            cur.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
            if cur.fetchone():
                flash('Username or email already exists.', 'danger')
                return render_template('register.html')
            
            # Create user
            password_hash = generate_password_hash(password)
            cur.execute("""
                INSERT INTO users (username, email, password_hash, first_name, last_name)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (username, email, password_hash, first_name, last_name))
            
            user_id = cur.fetchone()[0]
            
            # Log activity
            cur.execute("""
                INSERT INTO user_activity (user_id, activity_type, description)
                VALUES (%s, %s, %s)
            """, (user_id, 'registration', 'User registered successfully'))
            
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            conn.rollback()
            flash('Registration failed. Please try again.', 'danger')
            return render_template('register.html')
        finally:
            cur.close()
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """User logout"""
    if 'user_id' in session:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            
            try:
                # Invalidate session
                cur.execute("""
                    UPDATE user_sessions 
                    SET is_active = FALSE 
                    WHERE user_id = %s AND session_token = %s
                """, (session['user_id'], session.get('session_token')))
                
                # Log activity
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'logout', 'User logged out'))
                
                conn.commit()
            except:
                conn.rollback()
            finally:
                cur.close()
                conn.close()
    
    # Clear session
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    conn = get_db_connection()
    if conn is None:
        flash('Database connection failed. Please try again later.', 'danger')
        return render_template('dashboard.html', projects=[], activities=[], user=session.get('user_info'))
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get user projects
        cur.execute("""
            SELECT p.*, COUNT(h.id) as heatmap_count
            FROM user_projects p
            LEFT JOIN saved_heatmaps h ON p.id = h.project_id
            WHERE p.user_id = %s
            GROUP BY p.id
            ORDER BY p.updated_at DESC
        """, (session['user_id'],))
        
        projects = cur.fetchall()
        
        # Get recent activity
        cur.execute("""
            SELECT activity_type, description, created_at
            FROM user_activity
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 10
        """, (session['user_id'],))
        
        activities = cur.fetchall()
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        projects = []
        activities = []
    finally:
        cur.close()
        conn.close()
    
    return render_template('dashboard.html', 
                         projects=projects, 
                         activities=activities,
                         user=session.get('user_info'))

@app.route('/projects')
@login_required
def projects():
    """User projects page"""
    conn = get_db_connection()
    if conn is None:
        flash('Database connection failed. Please try again later.', 'danger')
        return render_template('projects.html', projects=[], user=session.get('user_info'))
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get user projects with heatmap counts
        cur.execute("""
            SELECT p.*, COUNT(h.id) as heatmap_count,
                   MAX(h.created_at) as last_heatmap_date
            FROM user_projects p
            LEFT JOIN saved_heatmaps h ON p.id = h.project_id
            WHERE p.user_id = %s
            GROUP BY p.id
            ORDER BY p.updated_at DESC
        """, (session['user_id'],))
        
        projects = cur.fetchall()
    except Exception as e:
        print(f"Error fetching projects: {e}")
        projects = []
    finally:
        cur.close()
        conn.close()
    
    return render_template('projects.html', 
                         projects=projects,
                         user=session.get('user_info'))

@app.route('/projects/create', methods=['GET', 'POST'])
@login_required
def create_project():
    """Create a new project"""
    if request.method == 'POST':
        project_name = request.form['project_name']
        description = request.form.get('description', '')
        is_public = 'is_public' in request.form
        
        if not project_name:
            flash('Project name is required.', 'danger')
            return render_template('create_project.html', user=session.get('user_info'))
        
        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'danger')
            return render_template('create_project.html', user=session.get('user_info'))
        
        cur = conn.cursor()
        
        try:
            # Check if project name already exists for this user
            cur.execute("""
                SELECT id FROM user_projects 
                WHERE user_id = %s AND project_name = %s
            """, (session['user_id'], project_name))
            
            if cur.fetchone():
                flash('You already have a project with this name.', 'danger')
                return render_template('create_project.html', user=session.get('user_info'))
            
            # Create project
            cur.execute("""
                INSERT INTO user_projects (user_id, project_name, description, is_public)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (session['user_id'], project_name, description, is_public))
            
            project_id = cur.fetchone()[0]
            
            # Log activity
            cur.execute("""
                INSERT INTO user_activity (user_id, activity_type, description)
                VALUES (%s, %s, %s)
            """, (session['user_id'], 'project_creation', f'Created project: {project_name}'))
            
            conn.commit()
            flash(f'Project "{project_name}" created successfully!', 'success')
            return redirect(url_for('project_detail', project_id=project_id))
            
        except Exception as e:
            conn.rollback()
            flash('Error creating project. Please try again.', 'danger')
            return render_template('create_project.html', user=session.get('user_info'))
        finally:
            cur.close()
            conn.close()
    
    return render_template('create_project.html', user=session.get('user_info'))

@app.route('/project/<int:project_id>')
@login_required
def project_detail(project_id):
    """Project detail page with heatmaps"""
    conn = get_db_connection()
    if conn is None:
        flash('Database connection failed. Please try again later.', 'danger')
        return redirect(url_for('projects'))
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get project details
        cur.execute("""
            SELECT p.* 
            FROM user_projects p
            WHERE p.id = %s AND p.user_id = %s
        """, (project_id, session['user_id']))
        
        project = cur.fetchone()
        
        if not project:
            flash('Project not found.', 'danger')
            return redirect(url_for('projects'))
        
        # Get project heatmaps
        cur.execute("""
            SELECT h.* 
            FROM saved_heatmaps h
            WHERE h.project_id = %s
            ORDER BY h.created_at DESC
        """, (project_id,))
        
        heatmaps = cur.fetchall()
    except Exception as e:
        print(f"Error fetching project details: {e}")
        project = None
        heatmaps = []
    finally:
        cur.close()
        conn.close()
    
    if not project:
        flash('Project not found.', 'danger')
        return redirect(url_for('projects'))
    
    return render_template('project_detail.html', 
                         project=project,
                         heatmaps=heatmaps,
                         user=session.get('user_info'))

@app.route('/project/<int:project_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_project(project_id):
    """Edit an existing project"""
    conn = get_db_connection()
    if conn is None:
        flash('Database connection failed.', 'danger')
        return redirect(url_for('projects'))
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get project details
        cur.execute("""
            SELECT * FROM user_projects 
            WHERE id = %s AND user_id = %s
        """, (project_id, session['user_id']))
        
        project = cur.fetchone()
        
        if not project:
            flash('Project not found.', 'danger')
            return redirect(url_for('projects'))
        
        if request.method == 'POST':
            project_name = request.form['project_name']
            description = request.form.get('description', '')
            is_public = 'is_public' in request.form
            
            if not project_name:
                flash('Project name is required.', 'danger')
                return render_template('edit_project.html', project=project, user=session.get('user_info'))
            
            # Check if project name already exists for this user (excluding current project)
            cur.execute("""
                SELECT id FROM user_projects 
                WHERE user_id = %s AND project_name = %s AND id != %s
            """, (session['user_id'], project_name, project_id))
            
            if cur.fetchone():
                flash('You already have another project with this name.', 'danger')
                return render_template('edit_project.html', project=project, user=session.get('user_info'))
            
            # Update project
            cur.execute("""
                UPDATE user_projects 
                SET project_name = %s, description = %s, is_public = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND user_id = %s
            """, (project_name, description, is_public, project_id, session['user_id']))
            
            # Log activity
            cur.execute("""
                INSERT INTO user_activity (user_id, activity_type, description)
                VALUES (%s, %s, %s)
            """, (session['user_id'], 'project_update', f'Updated project: {project_name}'))
            
            conn.commit()
            flash('Project updated successfully!', 'success')
            return redirect(url_for('project_detail', project_id=project_id))
        
    except Exception as e:
        conn.rollback()
        flash('Error updating project.', 'danger')
    finally:
        cur.close()
        conn.close()
    
    return render_template('edit_project.html', project=project, user=session.get('user_info'))

@app.route('/project/<int:project_id>/delete', methods=['POST'])
@login_required
def delete_project(project_id):
    """Delete a project and all its heatmaps"""
    conn = get_db_connection()
    if conn is None:
        return jsonify({'success': False, 'error': 'Database connection failed'})
    
    cur = conn.cursor()
    
    try:
        # Get project name for logging
        cur.execute("SELECT project_name FROM user_projects WHERE id = %s AND user_id = %s", 
                   (project_id, session['user_id']))
        project = cur.fetchone()
        
        if not project:
            return jsonify({'success': False, 'error': 'Project not found'})
        
        project_name = project[0]
        
        # Delete all heatmaps in this project first
        cur.execute("DELETE FROM saved_heatmaps WHERE project_id = %s", (project_id,))
        
        # Delete the project
        cur.execute("DELETE FROM user_projects WHERE id = %s AND user_id = %s", 
                   (project_id, session['user_id']))
        
        # Log activity
        cur.execute("""
            INSERT INTO user_activity (user_id, activity_type, description)
            VALUES (%s, %s, %s)
        """, (session['user_id'], 'project_deletion', f'Deleted project: {project_name}'))
        
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': f'Project "{project_name}" and all its heatmaps have been deleted.'
        })
        
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cur.close()
        conn.close()

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    conn = get_db_connection()
    if conn is None:
        flash('Database connection failed. Please try again later.', 'danger')
        return render_template('profile.html', user_stats=None, user=session.get('user_info'))
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get user stats
        cur.execute("""
            SELECT 
                COUNT(p.id) as project_count,
                COUNT(h.id) as heatmap_count,
                MAX(a.created_at) as last_activity
            FROM users u
            LEFT JOIN user_projects p ON u.id = p.user_id
            LEFT JOIN saved_heatmaps h ON p.id = h.project_id
            LEFT JOIN user_activity a ON u.id = a.user_id
            WHERE u.id = %s
            GROUP BY u.id
        """, (session['user_id'],))
        
        user_stats = cur.fetchone()
    except Exception as e:
        print(f"Error fetching user stats: {e}")
        user_stats = None
    finally:
        cur.close()
        conn.close()
    
    return render_template('profile.html', 
                         user_stats=user_stats,
                         user=session.get('user_info'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    """Handle file upload and processing - now requires login"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        text_files = []
        
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                
                if not allowed_file(filename):
                    continue
                
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                
                # Process PDF files to extract text
                if filename.lower().endswith('.pdf'):
                    try:
                        pdf = pdfplumber.open(file_path)
                        text = ""
                        
                        # Iterate through each page of the PDF
                        for page in pdf.pages:
                            # Extract text from the page
                            page_text = page.extract_text()
                            
                            # Perform OCR on scanned pages
                            if not page_text or not page_text.strip():
                                try:
                                    page_image = page.to_image()
                                    page_text = pytesseract.image_to_string(page_image.original)
                                except:
                                    page_text = ""
                            
                            text += page_text + "\n"
                        
                        # Close the PDF
                        pdf.close()
                        
                        # Exclude headers and footers
                        text = '\n'.join(line for line in text.splitlines() if not line.startswith(('Page ', 'Chapter ', 'Title ')))
                        
                        # Save as text file
                        txt_filename = os.path.splitext(filename)[0] + '.txt'
                        txt_file_path = os.path.join(temp_dir, txt_filename)
                        with open(txt_file_path, 'w', encoding='utf-8') as fp:
                            fp.write(text)
                        
                        text_files.append(txt_file_path)
                        
                    except Exception as e:
                        print(f"Error processing PDF {filename}: {str(e)}")
                        # Continue with other files
                        continue
                
                # Directly use text files
                elif filename.lower().endswith('.txt'):
                    text_files.append(file_path)
        
        if not text_files:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'error': 'No valid text files found after processing'}), 400
        
        # Merge all text files
        merged_text = ""
        for txt_file in text_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    merged_text += f.read() + "\n"
            except UnicodeDecodeError:
                # Try different encoding
                try:
                    with open(txt_file, 'r', encoding='latin-1') as f:
                        merged_text += f.read() + "\n"
                except:
                    continue
        
        if not merged_text.strip():
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'error': 'No readable text content found in files'}), 400
        
        # Clean the merged text
        documents = merged_text.splitlines()
        cleaned_documents = [preprocess_text(doc) for doc in documents if doc.strip()]
        cleaned_text = '\n'.join(cleaned_documents)
        
        # Tokenize for Word2Vec
        tokenized_documents = []
        for doc in cleaned_documents:
            words = nltk.word_tokenize(doc.lower())
            # Remove numbers, dates, and special characters
            words = [word for word in words if word.isalnum() and not word.isnumeric()]
            words = [word for word in words if word not in stopwords.words('english')]
            words = [lemmatizer.lemmatize(word) for word in words]
            if words:  # Only add non-empty documents
                tokenized_documents.append(words)
        
        if not tokenized_documents:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'error': 'No valid tokens found after preprocessing'}), 400
        
        # Train Word2Vec model
        try:
            model = Word2Vec(sentences=tokenized_documents, vector_size=100, window=5, min_count=1, workers=4, sg=0)
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'error': f'Error training Word2Vec model: {str(e)}'}), 500
        
        # Extract phrases
        all_phrases = {}
        MIN_PHRASE_LENGTH = 2
        MAX_PHRASE_LENGTH = 4
        MIN_PHRASE_COUNT = 2
        
        for doc in tokenized_documents:
            for i in range(len(doc)):
                for j in range(i, min(i + MAX_PHRASE_LENGTH, len(doc))):
                    phrase = ' '.join(doc[i:j+1])
                    
                    if len(phrase) < MIN_PHRASE_LENGTH:
                        continue
                    
                    if all(word in stopwords.words('english') for word in doc[i:j+1]):
                        continue
                    
                    all_phrases[phrase] = all_phrases.get(phrase, 0) + 1
        
        # Filter phrases
        filtered_phrases = {}
        for phrase, count in all_phrases.items():
            if count >= MIN_PHRASE_COUNT and len(phrase.split()) <= MAX_PHRASE_LENGTH:
                filtered_phrases[phrase] = count
        
        # Create phrase DataFrame
        phrase_df = pd.DataFrame.from_dict(filtered_phrases, orient='index', columns=['count'])
        phrase_df.sort_values('count', inplace=True, ascending=False)
        
        # Limit to top 1000 phrases
        MAX_PHRASES = 1000
        if len(phrase_df) > MAX_PHRASES:
            phrase_df = phrase_df.head(MAX_PHRASES)
        
        # Create word vectors
        word_vectors = {}
        for word in model.wv.index_to_key:
            word_vectors[word] = model.wv[word]
        
        # Create output files
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save phrase counts
        phrase_csv_path = os.path.join(output_dir, 'phrase_counts.csv')
        phrase_df.to_csv(phrase_csv_path)
        
        # Save word vectors
        vectors_path = os.path.join(output_dir, 'word_vectors.emb')
        with open(vectors_path, 'w', encoding='utf-8') as emb_file:
            emb_file.write(f"{len(word_vectors)} {len(next(iter(word_vectors.values())))}\n")
            for word, vector in word_vectors.items():
                vector_str = ' '.join(str(value) for value in vector)
                emb_file.write(f"{word} {vector_str}\n")
        
        # Create zip file
        zip_path = os.path.join(temp_dir, 'word2vec_results.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(phrase_csv_path, 'phrase_counts.csv')
            zipf.write(vectors_path, 'word_vectors.emb')
        
        # Log user activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'file_processing', f'Processed {len(files)} files for Word2Vec training'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        # Send file
        response = send_file(zip_path, as_attachment=True, download_name='word2vec_results.zip')
        
        # Clean up
        @response.call_on_close
        def cleanup_temp_dir():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
        
        return response
        
    except Exception as e:
        # Clean up on any unexpected error
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/analyze')
@login_required
def analyze():
    """Analysis tools page"""
    return render_template('analyze.html', user=session.get('user_info'))

@app.route('/analyze/load_embeddings', methods=['POST'])
@login_required
def load_embeddings():
    """Load and parse embedding file"""
    temp_dir = None
    try:
        if 'emb_file' not in request.files:
            return jsonify({'error': 'No embedding file provided'}), 400
        
        emb_file = request.files['emb_file']
        
        if emb_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(emb_file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a .emb file.'}), 400
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        emb_path = os.path.join(temp_dir, 'temp.emb')
        emb_file.save(emb_path)
        
        # Parse embedding file
        word_vectors = parse_emb_file(emb_path)
        
        if not word_vectors:
            return jsonify({'error': 'No valid word vectors found in the file'}), 400
        
        # Log user activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'embedding_load', f'Loaded embedding file with {len(word_vectors)} words'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify({
            'word_vectors': word_vectors,
            'word_count': len(word_vectors)
        })
        
    except Exception as e:
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'Error loading embeddings: {str(e)}'}), 500

@app.route('/analyze/load_categories', methods=['POST'])
@login_required
def load_categories():
    """Load and parse categories CSV file with validation"""
    temp_dir = None
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No CSV file provided'}), 400
        
        csv_file = request.files['csv_file']
        
        if csv_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(csv_file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a .csv file.'}), 400
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, 'temp.csv')
        csv_file.save(csv_path)
        
        # Validate file size and structure
        try:
            validate_csv_file(csv_path)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Parse categories
        categories = parse_csv_categories(csv_path)
        
        if not categories:
            return jsonify({'error': 'No valid categories found in the file'}), 400
        
        # Log user activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'categories_load', f'Loaded categories file with {len(categories)} categories'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify({
            'categories': categories,
            'category_count': len(categories)
        })
        
    except Exception as e:
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'Error loading categories: {str(e)}'}), 500

@app.route('/analyze/similarity', methods=['POST'])
@login_required
def analyze_similarity():
    """Find similar words for a given word"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        word = data.get('word', '').strip().lower()
        word_vectors = data.get('word_vectors', {})
        
        if not word:
            return jsonify({'error': 'No word provided'}), 400
        
        if not word_vectors:
            return jsonify({'error': 'No word vectors provided'}), 400
        
        # Check if word exists
        if word not in word_vectors:
            return jsonify({'error': f'Word "{word}" not found in vector space'}), 404
        
        # Calculate similarities
        similarities = []
        target_vector = word_vectors[word]
        
        for other_word, vector in word_vectors.items():
            if other_word != word:
                try:
                    similarity = calculate_cosine_similarity(target_vector, vector)
                    similarities.append({
                        'word': other_word,
                        'similarity': float(similarity)
                    })
                except:
                    continue  # Skip invalid vectors
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Log user activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'similarity_search', f'Searched similar words for "{word}"'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        return jsonify({
            'word': word,
            'similar_words': similarities[:10]  # Return top 10
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/analyze/visualize', methods=['POST'])
@login_required
def visualize_similarity():
    """Calculate similarity between multiple words"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        words = data.get('words', [])
        word_vectors = data.get('word_vectors', {})
        
        if len(words) < 2:
            return jsonify({'error': 'At least two words required'}), 400
        
        if not word_vectors:
            return jsonify({'error': 'No word vectors provided'}), 400
        
        # Check if all words exist
        missing_words = [word for word in words if word not in word_vectors]
        if missing_words:
            return jsonify({'error': f'Words not found: {", ".join(missing_words)}'}), 404
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                try:
                    similarity = calculate_cosine_similarity(
                        word_vectors[words[i]], 
                        word_vectors[words[j]]
                    )
                    similarities.append({
                        'pair': f'{words[i]} - {words[j]}',
                        'similarity': float(similarity)
                    })
                except:
                    continue  # Skip invalid comparisons
        
        # Log user activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'similarity_visualization', f'Visualized similarities for {len(words)} words'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        return jsonify({
            'words': words,
            'similarities': similarities
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/heatmap/generate', methods=['POST'])
@login_required
def generate_heatmap():
    """Generate heatmap data for two categories"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        category_x = data.get('category_x', '')
        category_y = data.get('category_y', '')
        categories = data.get('categories', {})
        word_vectors = data.get('word_vectors', {})
        normalize = data.get('normalize', True)
        
        if not category_x or not category_y:
            return jsonify({'error': 'Both categories are required'}), 400
        
        if not categories:
            return jsonify({'error': 'No categories provided'}), 400
        
        if not word_vectors:
            return jsonify({'error': 'No word vectors provided'}), 400
        
        # Check if categories exist
        if category_x not in categories:
            return jsonify({'error': f'Category "{category_x}" not found'}), 404
        
        if category_y not in categories:
            return jsonify({'error': f'Category "{category_y}" not found'}), 404
        
        # Get words for categories
        words_x = categories[category_x]
        words_y = categories[category_y]
        
        if not words_x or not words_y:
            return jsonify({'error': 'One or both categories are empty'}), 400
        
        # Check if all words exist in vector space
        all_words = words_x + words_y
        missing_words = [word for word in all_words if word not in word_vectors]
        
        if missing_words:
            # Show first 10 missing words for debugging
            missing_sample = missing_words[:10]
            print(f"Missing words in vector space: {missing_sample}")
            
            # Filter out missing words
            words_x = [word for word in words_x if word in word_vectors]
            words_y = [word for word in words_y if word in word_vectors]
            
            if not words_x or not words_y:
                return jsonify({
                    'error': f'No valid words found after filtering. Missing: {", ".join(missing_sample)}'
                }), 404
            
            print(f"Filtered words_x: {len(words_x)} words, words_y: {len(words_y)} words")
        
        # Calculate similarity matrix
        matrix = []
        all_similarities = []
        
        for word_x in words_x:
            row = []
            for word_y in words_y:
                try:
                    similarity = calculate_cosine_similarity(
                        word_vectors[word_x],
                        word_vectors[word_y]
                    )
                    row.append(float(similarity))
                    all_similarities.append(similarity)
                except Exception as e:
                    print(f"Error calculating similarity between {word_x} and {word_y}: {e}")
                    row.append(0.0)
                    all_similarities.append(0.0)
            matrix.append(row)
        
        if not all_similarities:
            return jsonify({'error': 'Could not calculate any similarities'}), 400
        
        # Calculate statistics
        stats = {
            'min': min(all_similarities),
            'max': max(all_similarities),
            'mean': np.mean(all_similarities),
            'std': np.std(all_similarities)
        }
        
        # Normalize matrix if requested
        if normalize and stats['max'] > stats['min']:
            normalized_matrix = []
            for row in matrix:
                normalized_row = [
                    (val - stats['min']) / (stats['max'] - stats['min'])
                    for val in row
                ]
                normalized_matrix.append(normalized_row)
            matrix = normalized_matrix
            # Update stats for normalized data
            stats = {
                'min': 0.0,
                'max': 1.0,
                'mean': np.mean([item for row in matrix for item in row]),
                'std': np.std([item for row in matrix for item in row])
            }
        
        # Create heatmap image
        heatmap_image = create_heatmap_image(words_x, words_y, matrix, category_x, category_y, stats)
        
        # Convert image to base64 for potential preview
        img_buffer = io.BytesIO()
        heatmap_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Log user activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'heatmap_generation', f'Generated heatmap: {category_x} vs {category_y}'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        # Prepare response data
        response_data = {
            'category_x': category_x,
            'category_y': category_y,
            'words_x': words_x,
            'words_y': words_y,
            'matrix': matrix,
            'stats': stats,
            'image_preview': f"data:image/png;base64,{image_base64}",
            'filtered_missing_words': len(missing_words) if 'missing_words' in locals() else 0
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return jsonify({'error': f'Error generating heatmap: {str(e)}'}), 500

@app.route('/heatmap/generate_multi', methods=['POST'])
@login_required
def generate_multi_heatmap():
    """Generate heatmap data for multiple categories on each axis"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        categories = data.get('categories', {})
        category_selection = data.get('category_selection', {})
        word_vectors = data.get('word_vectors', {})
        normalize = data.get('normalize', True)
        
        x_categories = category_selection.get('x_axis', [])
        y_categories = category_selection.get('y_axis', [])
        
        if not x_categories or not y_categories:
            return jsonify({'error': 'Categories required for both axes'}), 400
        
        if not categories:
            return jsonify({'error': 'No categories provided'}), 400
        
        if not word_vectors:
            return jsonify({'error': 'No word vectors provided'}), 400
        
        # Collect words for each axis with category labels
        words_x = []
        words_y = []
        category_labels_x = []
        category_labels_y = []
        
        # Process X-axis categories
        for category in x_categories:
            if category in categories:
                category_words = categories[category]
                # Filter words that exist in vector space
                valid_words = [word for word in category_words if word in word_vectors]
                words_x.extend(valid_words)
                category_labels_x.extend([category] * len(valid_words))
        
        # Process Y-axis categories
        for category in y_categories:
            if category in categories:
                category_words = categories[category]
                # Filter words that exist in vector space
                valid_words = [word for word in category_words if word in word_vectors]
                words_y.extend(valid_words)
                category_labels_y.extend([category] * len(valid_words))
        
        if not words_x or not words_y:
            return jsonify({'error': 'No valid words found after filtering'}), 404
        
        # Calculate similarity matrix
        matrix = []
        all_similarities = []
        
        for i, word_x in enumerate(words_x):
            row = []
            for j, word_y in enumerate(words_y):
                try:
                    similarity = calculate_cosine_similarity(
                        word_vectors[word_x],
                        word_vectors[word_y]
                    )
                    row.append(float(similarity))
                    all_similarities.append(similarity)
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    row.append(0.0)
                    all_similarities.append(0.0)
            matrix.append(row)
        
        if not all_similarities:
            return jsonify({'error': 'Could not calculate any similarities'}), 400
        
        # Calculate statistics
        stats = {
            'min': min(all_similarities),
            'max': max(all_similarities),
            'mean': np.mean(all_similarities),
            'std': np.std(all_similarities)
        }
        
        # Normalize matrix if requested
        if normalize and stats['max'] > stats['min']:
            normalized_matrix = []
            for row in matrix:
                normalized_row = [
                    (val - stats['min']) / (stats['max'] - stats['min'])
                    for val in row
                ]
                normalized_matrix.append(normalized_row)
            matrix = normalized_matrix
            # Update stats for normalized data
            stats = {
                'min': 0.0,
                'max': 1.0,
                'mean': np.mean([item for row in matrix for item in row]),
                'std': np.std([item for row in matrix for item in row])
            }
        
        # Create heatmap image with category labels
        heatmap_image = create_multi_heatmap_image(
            words_x, words_y, matrix, 
            category_labels_x, category_labels_y,
            x_categories, y_categories, stats
        )
        
        # Convert image to base64
        img_buffer = io.BytesIO()
        heatmap_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Log user activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'multi_heatmap_generation', 
                     f'Generated multi-category heatmap: {len(x_categories)}x{len(y_categories)}'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        # Prepare response data
        response_data = {
            'category_x_info': x_categories,
            'category_y_info': y_categories,
            'category_labels_x': category_labels_x,
            'category_labels_y': category_labels_y,
            'words_x': words_x,
            'words_y': words_y,
            'matrix': matrix,
            'stats': stats,
            'image_preview': f"data:image/png;base64,{image_base64}"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error generating multi-category heatmap: {str(e)}")
        return jsonify({'error': f'Error generating heatmap: {str(e)}'}), 500

@app.route('/heatmap/save', methods=['POST'])
@login_required
def save_heatmap():
    """Save heatmap to user's projects"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        project_name = data.get('project_name', '')
        heatmap_name = data.get('heatmap_name', '')
        category_x = data.get('category_x', '')
        category_y = data.get('category_y', '')
        words_x = data.get('words_x', [])
        words_y = data.get('words_y', [])
        matrix = data.get('matrix', [])
        is_public = data.get('is_public', False)
        
        if not project_name or not heatmap_name:
            return jsonify({'error': 'Project name and heatmap name are required'}), 400
        
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cur = conn.cursor()
        
        try:
            # Create or get project
            cur.execute("""
                INSERT INTO user_projects (user_id, project_name, is_public)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, project_name) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (session['user_id'], project_name, is_public))
            
            project_id = cur.fetchone()[0]
            
            # Save heatmap
            cur.execute("""
                INSERT INTO saved_heatmaps (project_id, heatmap_name, category_x, category_y, words_x, words_y, similarity_matrix)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (project_id, heatmap_name, category_x, category_y, words_x, words_y, json.dumps(matrix)))
            
            heatmap_id = cur.fetchone()[0]
            
            # Log activity
            cur.execute("""
                INSERT INTO user_activity (user_id, activity_type, description)
                VALUES (%s, %s, %s)
            """, (session['user_id'], 'heatmap_save', f'Saved heatmap: {heatmap_name}'))
            
            conn.commit()
            
            return jsonify({
                'success': True,
                'heatmap_id': heatmap_id,
                'message': 'Heatmap saved successfully'
            })
            
        except Exception as e:
            conn.rollback()
            return jsonify({'error': f'Error saving heatmap: {str(e)}'}), 500
        finally:
            cur.close()
            conn.close()
            
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/heatmap/save_to_project', methods=['POST'])
@login_required
def save_heatmap_to_project():
    """Save heatmap to a specific project"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        project_id = data.get('project_id')
        heatmap_name = data.get('heatmap_name', '')
        category_x = data.get('category_x', '')
        category_y = data.get('category_y', '')
        words_x = data.get('words_x', [])
        words_y = data.get('words_y', [])
        matrix = data.get('matrix', [])
        
        if not project_id:
            return jsonify({'error': 'Project ID is required'}), 400
        
        if not heatmap_name:
            return jsonify({'error': 'Heatmap name is required'}), 400
        
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cur = conn.cursor()
        
        try:
            # Verify project belongs to user
            cur.execute("SELECT id FROM user_projects WHERE id = %s AND user_id = %s", 
                       (project_id, session['user_id']))
            
            if not cur.fetchone():
                return jsonify({'error': 'Project not found or access denied'}), 404
            
            # Save heatmap
            cur.execute("""
                INSERT INTO saved_heatmaps (project_id, heatmap_name, category_x, category_y, words_x, words_y, similarity_matrix)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (project_id, heatmap_name, category_x, category_y, words_x, words_y, json.dumps(matrix)))
            
            heatmap_id = cur.fetchone()[0]
            
            # Update project timestamp
            cur.execute("""
                UPDATE user_projects 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE id = %s
            """, (project_id,))
            
            # Log activity
            cur.execute("""
                INSERT INTO user_activity (user_id, activity_type, description)
                VALUES (%s, %s, %s)
            """, (session['user_id'], 'heatmap_save', f'Saved heatmap to project: {heatmap_name}'))
            
            conn.commit()
            
            return jsonify({
                'success': True,
                'heatmap_id': heatmap_id,
                'message': 'Heatmap saved successfully'
            })
            
        except Exception as e:
            conn.rollback()
            return jsonify({'error': f'Error saving heatmap: {str(e)}'}), 500
        finally:
            cur.close()
            conn.close()
            
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/heatmap/<int:heatmap_id>/delete', methods=['POST'])
@login_required
def delete_heatmap(heatmap_id):
    """Delete a specific heatmap"""
    conn = get_db_connection()
    if conn is None:
        return jsonify({'success': False, 'error': 'Database connection failed'})
    
    cur = conn.cursor()
    
    try:
        # Verify heatmap belongs to user
        cur.execute("""
            SELECT h.heatmap_name, p.project_name 
            FROM saved_heatmaps h
            JOIN user_projects p ON h.project_id = p.id
            WHERE h.id = %s AND p.user_id = %s
        """, (heatmap_id, session['user_id']))
        
        heatmap = cur.fetchone()
        
        if not heatmap:
            return jsonify({'success': False, 'error': 'Heatmap not found or access denied'})
        
        heatmap_name, project_name = heatmap
        
        # Delete the heatmap
        cur.execute("DELETE FROM saved_heatmaps WHERE id = %s", (heatmap_id,))
        
        # Log activity
        cur.execute("""
            INSERT INTO user_activity (user_id, activity_type, description)
            VALUES (%s, %s, %s)
        """, (session['user_id'], 'heatmap_deletion', f'Deleted heatmap: {heatmap_name} from {project_name}'))
        
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': f'Heatmap "{heatmap_name}" has been deleted.'
        })
        
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cur.close()
        conn.close()

@app.route('/heatmap/download_all', methods=['POST'])
@login_required
def download_all_heatmap():
    """Download ZIP file containing heatmap image and metrics"""
    temp_dir = None
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        category_x = data.get('category_x', '')
        category_y = data.get('category_y', '')
        words_x = data.get('words_x', [])
        words_y = data.get('words_y', [])
        matrix = data.get('matrix', [])
        stats = data.get('stats', {})
        
        if not category_x or not category_y:
            return jsonify({'error': 'Category information missing'}), 400
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create heatmap image
        heatmap_image = create_heatmap_image(words_x, words_y, matrix, category_x, category_y, stats)
        image_path = os.path.join(temp_dir, f'heatmap_{category_x}_{category_y}.png')
        heatmap_image.save(image_path, 'PNG')
        
        # Create metrics CSV
        csv_path = os.path.join(temp_dir, f'heatmap_metrics_{category_x}_{category_y}.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('Category X,Category Y,Word X,Word Y,Similarity\n')
            for i, word_x in enumerate(words_x):
                for j, word_y in enumerate(words_y):
                    f.write(f'"{category_x}","{category_y}","{word_x}","{word_y}",{matrix[i][j]:.6f}\n')
        
        # Create summary report
        report_path = os.path.join(temp_dir, f'heatmap_report_{category_x}_{category_y}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Word Similarity Heatmap Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Categories: {category_x} vs {category_y}\n")
            f.write(f"X-Axis words: {len(words_x)}\n")
            f.write(f"Y-Axis words: {len(words_y)}\n")
            f.write(f"Total comparisons: {len(words_x) * len(words_y)}\n\n")
            f.write("Similarity Statistics:\n")
            f.write(f"  Minimum: {stats.get('min', 0):.4f}\n")
            f.write(f"  Maximum: {stats.get('max', 0):.4f}\n")
            f.write(f"  Mean: {stats.get('mean', 0):.4f}\n")
            f.write(f"  Standard Deviation: {stats.get('std', 0):.4f}\n\n")
            f.write("Generated by Word2Vec Training & Analysis Tool\n")
        
        # Create ZIP file
        zip_path = os.path.join(temp_dir, f'heatmap_results_{category_x}_{category_y}.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(image_path, f'heatmap_{category_x}_{category_y}.png')
            zipf.write(csv_path, f'heatmap_metrics_{category_x}_{category_y}.csv')
            zipf.write(report_path, f'heatmap_report_{category_x}_{category_y}.txt')
        
        # Log user activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'heatmap_download', f'Downloaded heatmap: {category_x} vs {category_y}'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        # Send file
        response = send_file(zip_path, as_attachment=True, 
                           download_name=f'heatmap_results_{category_x}_{category_y}.zip')
        
        # Clean up
        @response.call_on_close
        def cleanup():
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return response
        
    except Exception as e:
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'Error creating download package: {str(e)}'}), 500

@app.route('/heatmap/preview/<int:heatmap_id>')
def heatmap_preview(heatmap_id):
    """Public heatmap preview page - REMOVED: Now redirects to login"""
    flash('Please log in to view heatmaps.', 'info')
    return redirect(url_for('login', next=url_for('heatmap_preview', heatmap_id=heatmap_id)))

@app.route('/visualize')
@login_required
def visualize():
    """Visualization tools page"""
    return render_template('visualize_flexible.html', user=session.get('user_info'))

@app.route('/literature')
@login_required
def literature_search():
    """Literature search page"""
    return render_template('literature_search.html', user=session.get('user_info'))

@app.route('/api/literature/search', methods=['POST'])
@login_required
def api_literature_search():
    """API endpoint for literature search using free sources"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query = data.get('query', '').strip()
        max_results = data.get('max_results', 50)
        start_year = data.get('start_year')
        end_year = data.get('end_year')
        include_arxiv = data.get('include_arxiv', True)
        include_semantic = data.get('include_semantic', True)
        include_crossref = data.get('include_crossref', True)
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        # Initialize scraper
        scraper = PaperScraper()
        
        all_papers = []
        
        # Search arXiv
        if include_arxiv:
            try:
                arxiv_papers = scraper.search_arxiv(query, max_results)
                all_papers.extend(arxiv_papers)
                print(f"Added {len(arxiv_papers)} arXiv papers")
            except Exception as e:
                print(f"Error with arXiv: {e}")
        
        # Search Semantic Scholar
        if include_semantic and len(all_papers) < max_results:
            try:
                semantic_papers = scraper.search_semantic_scholar(query, max_results - len(all_papers))
                # Avoid duplicates
                existing_dois = {p.get('doi') for p in all_papers if p.get('doi')}
                for paper in semantic_papers:
                    if paper.get('doi') not in existing_dois:
                        all_papers.append(paper)
                print(f"Added {len(semantic_papers)} Semantic Scholar papers")
            except Exception as e:
                print(f"Error with Semantic Scholar: {e}")
        
        # Search CrossRef
        if include_crossref and len(all_papers) < max_results:
            try:
                crossref_papers = scraper.search_crossref(query, max_results - len(all_papers))
                # Avoid duplicates
                existing_dois = {p.get('doi') for p in all_papers if p.get('doi')}
                for paper in crossref_papers:
                    if paper.get('doi') not in existing_dois:
                        all_papers.append(paper)
                print(f"Added {len(crossref_papers)} CrossRef papers")
            except Exception as e:
                print(f"Error with CrossRef: {e}")
        
        # Apply year filter if specified
        if start_year and end_year:
            filtered_papers = []
            for paper in all_papers:
                year_str = paper.get('publication_year', '')
                if year_str and year_str.isdigit():
                    year = int(year_str)
                    if start_year <= year <= end_year:
                        filtered_papers.append(paper)
                else:
                    # Include papers without year information
                    filtered_papers.append(paper)
            all_papers = filtered_papers
        
        # Limit results
        all_papers = all_papers[:max_results]
        
        # Log activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'literature_search', f'Searched for: "{query}" - Found {len(all_papers)} papers'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        return jsonify({
            'success': True,
            'papers': all_papers,
            'total_count': len(all_papers),
            'message': f'Search completed using free APIs. Found {len(all_papers)} papers.'
        })
        
    except Exception as e:
        print(f"Error in literature search: {str(e)}")
        return jsonify({'error': f'Error performing search: {str(e)}'}), 500

@app.route('/api/literature/download', methods=['POST'])
@login_required
def api_literature_download():
    """Download selected papers"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        papers = data.get('papers', [])
        project_id = data.get('project_id')
        
        if not papers:
            return jsonify({'error': 'No papers selected for download'}), 400
        
        # Create project folder if project_id is provided
        download_folder = 'downloads'
        if project_id:
            # Verify project belongs to user
            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                cur.execute("SELECT project_name FROM user_projects WHERE id = %s AND user_id = %s", 
                           (project_id, session['user_id']))
                project = cur.fetchone()
                if project:
                    download_folder = os.path.join('projects', str(project_id), 'papers')
                cur.close()
                conn.close()
        
        os.makedirs(download_folder, exist_ok=True)
        
        scraper = PaperScraper()
        
        download_results = []
        downloaded_count = 0
        
        for i, paper in enumerate(papers):
            try:
                success = scraper.try_download_pdf(paper, download_folder)
                download_results.append({
                    'title': paper.get('title', 'Unknown'),
                    'success': success
                })
                
                if success:
                    downloaded_count += 1
                
                # Small delay to be respectful to servers
                time.sleep(0.5)
                
            except Exception as e:
                download_results.append({
                    'title': paper.get('title', 'Unknown'),
                    'success': False,
                    'error': str(e)
                })
        
        # Save metadata
        metadata_path = os.path.join(download_folder, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        # Log activity
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (%s, %s, %s)
                """, (session['user_id'], 'paper_download', f'Downloaded {downloaded_count} papers to project'))
                conn.commit()
            except Exception as e:
                print(f"Error logging activity: {e}")
                conn.rollback()
            finally:
                cur.close()
                conn.close()
        
        return jsonify({
            'success': True,
            'downloaded': downloaded_count,
            'total': len(papers),
            'download_folder': download_folder,
            'results': download_results
        })
        
    except Exception as e:
        print(f"Error downloading papers: {str(e)}")
        return jsonify({'error': f'Error downloading papers: {str(e)}'}), 500

@app.route('/api/literature/process_project', methods=['POST'])
@login_required
def api_process_literature_project():
    """Process downloaded papers for Word2Vec training"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        project_id = data.get('project_id')
        
        if not project_id:
            return jsonify({'error': 'Project ID is required'}), 400
        
        # Verify project belongs to user
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("SELECT project_name FROM user_projects WHERE id = %s AND user_id = %s", 
                       (project_id, session['user_id']))
            project = cur.fetchone()
            
            if not project:
                return jsonify({'error': 'Project not found or access denied'}), 404
            
            project_name = project['project_name']
            
            # Path to project papers
            papers_folder = os.path.join('projects', str(project_id), 'papers')
            
            if not os.path.exists(papers_folder):
                return jsonify({'error': 'No papers found for this project'}), 404
            
            # Process all PDFs in the folder
            pdf_files = [f for f in os.listdir(papers_folder) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                return jsonify({'error': 'No PDF files found in project folder'}), 404
            
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            text_files = []
            
            # Process each PDF
            for pdf_file in pdf_files:
                try:
                    pdf_path = os.path.join(papers_folder, pdf_file)
                    pdf = pdfplumber.open(pdf_path)
                    text = ""
                    
                    # Extract text from each page
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        
                        # Perform OCR on scanned pages
                        if not page_text or not page_text.strip():
                            try:
                                page_image = page.to_image()
                                page_text = pytesseract.image_to_string(page_image.original)
                            except:
                                page_text = ""
                        
                        text += page_text + "\n"
                    
                    pdf.close()
                    
                    # Exclude headers and footers
                    text = '\n'.join(line for line in text.splitlines() 
                                   if not line.startswith(('Page ', 'Chapter ', 'Title ')))
                    
                    # Save as text file
                    txt_filename = os.path.splitext(pdf_file)[0] + '.txt'
                    txt_file_path = os.path.join(temp_dir, txt_filename)
                    with open(txt_file_path, 'w', encoding='utf-8') as fp:
                        fp.write(text)
                    
                    text_files.append(txt_file_path)
                    
                except Exception as e:
                    print(f"Error processing PDF {pdf_file}: {str(e)}")
                    continue
            
            if not text_files:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return jsonify({'error': 'No valid text extracted from PDFs'}), 400
            
            # Merge all text files
            merged_text = ""
            for txt_file in text_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        merged_text += f.read() + "\n"
                except:
                    continue
            
            if not merged_text.strip():
                shutil.rmtree(temp_dir, ignore_errors=True)
                return jsonify({'error': 'No readable text content found'}), 400
            
            # Clean the text
            documents = merged_text.splitlines()
            cleaned_documents = [preprocess_text(doc) for doc in documents if doc.strip()]
            cleaned_text = '\n'.join(cleaned_documents)
            
            # Tokenize for Word2Vec
            tokenized_documents = []
            for doc in cleaned_documents:
                words = nltk.word_tokenize(doc.lower())
                words = [word for word in words if word.isalnum() and not word.isnumeric()]
                words = [word for word in words if word not in stopwords.words('english')]
                words = [lemmatizer.lemmatize(word) for word in words]
                if words:
                    tokenized_documents.append(words)
            
            if not tokenized_documents:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return jsonify({'error': 'No valid tokens found after preprocessing'}), 400
            
            # Train Word2Vec model
            try:
                model = Word2Vec(sentences=tokenized_documents, vector_size=100, 
                               window=5, min_count=1, workers=4, sg=0)
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return jsonify({'error': f'Error training Word2Vec model: {str(e)}'}), 500
            
            # Save model to project folder
            model_path = os.path.join('projects', str(project_id), 'word2vec_model.model')
            model.save(model_path)
            
            # Save word vectors
            vectors_path = os.path.join('projects', str(project_id), 'word_vectors.emb')
            with open(vectors_path, 'w', encoding='utf-8') as emb_file:
                emb_file.write(f"{len(model.wv.index_to_key)} {len(model.wv[model.wv.index_to_key[0]])}\n")
                for word in model.wv.index_to_key:
                    vector_str = ' '.join(str(value) for value in model.wv[word])
                    emb_file.write(f"{word} {vector_str}\n")
            
            # Log activity
            cur.execute("""
                INSERT INTO user_activity (user_id, activity_type, description)
                VALUES (%s, %s, %s)
            """, (session['user_id'], 'literature_processing', 
                 f'Processed {len(pdf_files)} papers from project: {project_name}'))
            
            conn.commit()
            
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return jsonify({
                'success': True,
                'message': f'Successfully processed {len(pdf_files)} papers',
                'vocabulary_size': len(model.wv.index_to_key),
                'model_path': model_path,
                'vectors_path': vectors_path
            })
            
        except Exception as e:
            conn.rollback()
            return jsonify({'error': f'Error processing project: {str(e)}'}), 500
        finally:
            cur.close()
            conn.close()
            
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500
    
@app.route('/api/projects/list')
@login_required
def api_projects_list():
    """API endpoint to get user projects list"""
    conn = get_db_connection()
    if conn is None:
        return jsonify({'success': False, 'error': 'Database connection failed'})
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute("""
            SELECT id, project_name, description
            FROM user_projects
            WHERE user_id = %s
            ORDER BY project_name
        """, (session['user_id'],))
        
        projects = cur.fetchall()
        
        return jsonify({
            'success': True,
            'projects': projects
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    # Initialize database on first run
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)