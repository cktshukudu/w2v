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
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import zipfile
import tempfile
import string
import json
from io import StringIO
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import shutil

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'emb', 'csv'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
    """Parse CSV file and extract categories with their words"""
    categories = {}
    try:
        df = pd.read_csv(file_path)
        
        # Try different possible column structures
        if 'Category' in df.columns and 'Word' in df.columns:
            # Structure: Category, Word
            for _, row in df.iterrows():
                category = str(row['Category']).strip()
                word = str(row['Word']).strip()
                if category and word and word.lower() != 'nan':
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(word.lower())
        
        elif 'Category' in df.columns:
            # Structure with multiple word columns
            category_col = 'Category'
            word_cols = [col for col in df.columns if col != category_col]
            
            for _, row in df.iterrows():
                category = str(row[category_col]).strip()
                if category:
                    if category not in categories:
                        categories[category] = []
                    
                    for col in word_cols:
                        word = str(row[col]).strip()
                        if word and word.lower() != 'nan':
                            categories[category].append(word.lower())
        
        else:
            # Assume first column is categories, rest are words
            for _, row in df.iterrows():
                category = str(row.iloc[0]).strip()
                if category:
                    if category not in categories:
                        categories[category] = []
                    
                    for i in range(1, len(row)):
                        word = str(row.iloc[i]).strip()
                        if word and word.lower() != 'nan':
                            categories[category].append(word.lower())
        
        # Remove empty categories and duplicates
        for category in list(categories.keys()):
            categories[category] = list(set([w for w in categories[category] if w]))
            if not categories[category]:
                del categories[category]
                
    except Exception as e:
        print(f"Error parsing CSV with pandas: {e}")
        # Try alternative parsing method
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                current_category = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line could be a category header
                    parts = line.split(',')
                    if len(parts) == 1 or (len(parts) > 1 and not parts[1].strip()):
                        current_category = parts[0].strip()
                        if current_category:
                            categories[current_category] = []
                    elif current_category:
                        # Add words to current category
                        words = [w.strip().lower() for w in parts if w.strip() and w.lower() != 'nan']
                        categories[current_category].extend(words)
        except Exception as e2:
            raise ValueError(f"Could not parse CSV file: {str(e2)}")
    
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

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum file size is 100MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error occurred.'}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and processing"""
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
        
        # Extract phrases and their counts
        all_phrases = {}
        for doc in tokenized_documents:
            for i in range(len(doc)):
                for j in range(i, min(i + 3, len(doc))):
                    phrase = ' '.join(doc[i:j+1])
                    all_phrases[phrase] = all_phrases.get(phrase, 0) + 1
        
        # Create phrase DataFrame
        phrase_df = pd.DataFrame.from_dict(all_phrases, orient='index', columns=['count'])
        phrase_df.sort_values('count', inplace=True, ascending=False)
        
        # Create word vectors DataFrame
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
        
        # Create zip file with results
        zip_path = os.path.join(temp_dir, 'word2vec_results.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(phrase_csv_path, 'phrase_counts.csv')
            zipf.write(vectors_path, 'word_vectors.emb')
        
        # Send file and manually clean up after response is sent
        response = send_file(zip_path, as_attachment=True, download_name='word2vec_results.zip')
        
        # Use callback to clean up temporary directory after response
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

@app.route('/analyze/load_embeddings', methods=['POST'])
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

@app.route('/analyze/similarity', methods=['POST'])
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
        
        return jsonify({
            'word': word,
            'similar_words': similarities[:10]  # Return top 10
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/analyze/visualize', methods=['POST'])
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
        
        return jsonify({
            'words': words,
            'similarities': similarities
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/heatmap/categories', methods=['POST'])
def get_categories():
    """Extract categories from CSV file"""
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
        
        # Parse CSV file
        categories = parse_csv_categories(csv_path)
        
        if not categories:
            return jsonify({'error': 'No valid categories found in the CSV file'}), 400
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify({
            'categories': list(categories.keys()),
            'category_details': categories
        })
        
    except Exception as e:
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'Error parsing CSV file: {str(e)}'}), 500

@app.route('/heatmap/generate', methods=['POST'])
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
            return jsonify({'error': f'Words not found in vector space: {", ".join(missing_words[:5])}'}), 404
        
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
                except:
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
        
        # Prepare response data
        response_data = {
            'category_x': category_x,
            'category_y': category_y,
            'words_x': words_x,
            'words_y': words_y,
            'matrix': matrix,
            'stats': stats,
            'image_preview': f"data:image/png;base64,{image_base64}"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Error generating heatmap: {str(e)}'}), 500

@app.route('/heatmap/download_all', methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)