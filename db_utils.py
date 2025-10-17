# # app.py
# import os
# import pdfplumber
# import pytesseract
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from gensim.models import Word2Vec
# import pandas as pd
# import spacy
# from flask import Flask, render_template, request, jsonify, send_file
# from werkzeug.utils import secure_filename
# import zipfile
# import tempfile
# import string

# # Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# # Create upload directory if it doesn't exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Custom stopwords
# custom_stopwords = [
#     "keywords", "keyword", "abstract", "doi", "authors", "author", "journal", "Abstract", "Authors", "Keywords",
#     "http", "elsevier", "api", "sciencedirect", "available", "www", "ieee", "proceeding", "american", "vol",
#     "volume issue1", "com procedia", "journalofbusinessresearch68", "proceeding", "european", "nber",
#     "journalofclinicalepidemiology152", "socialsciences", "com", "procedia", "copyright", "technologyinsociety48",
#     "online", "heliyon9", "energyreports", "procedia computer science", "et al", "cid", "org", "org j", "int j",
#     "int", "e e", "j", "e", "al", "et", "springer science business",
# ]

# # Initialize NLP components
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# # Load spaCy model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("Downloading language model...")
#     from spacy.cli import download
#     download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")

# def preprocess_text(text):
#     """Preprocess text by removing URLs, tokenizing, and cleaning"""
#     # Remove URLs
#     text = re.sub(r'https?://\S+', '', text)
#     # Convert text to lowercase
#     text = text.lower()
#     # Use a regular expression to tokenize based on word boundaries
#     words = re.findall(r'\b\w+\b', text)
#     # Remove numbers, dates, special characters, and stopwords
#     words = [word for word in words if word.isalnum() and not word.isnumeric()]
#     words = [word for word in words if word not in custom_stopwords]
#     words = [word for word in words if word not in stopwords.words('english')]
#     # Apply lemmatization
#     words = [lemmatizer.lemmatize(word) for word in words]  
#     return ' '.join(words)

# def extract_phrases(text, max_phrase_length=3):
#     """Extract phrases from text using spaCy"""
#     # Process the text with spaCy
#     doc = nlp(text)
    
#     # Create a dictionary to store phrase counts
#     phrase_count_dict = {}
    
#     # Iterate over all tokens in the document
#     for i, token in enumerate(doc):
#         if token.is_stop:
#             continue

#         words = [token.lemma_]
#         for j in range(i + 1, min(i + max_phrase_length, len(doc))):
#             if doc[j].is_stop or doc[j].is_punct:
#                 break

#             words.append(doc[j].lemma_)
#             phrase = ' '.join(words)

#             if any(char.isdigit() or char in string.punctuation or len(phrase) == 1 or '\n' in phrase for char in phrase):
#                 continue

#             phrase_count_dict[phrase] = phrase_count_dict.get(phrase, 0) + 1
    
#     return phrase_count_dict

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     """Handle file upload and processing"""
#     if 'files' not in request.files:
#         return jsonify({'error': 'No files selected'}), 400
    
#     files = request.files.getlist('files')
#     if not files or files[0].filename == '':
#         return jsonify({'error': 'No files selected'}), 400
    
#     # Create temporary directory for processing
#     temp_dir = tempfile.mkdtemp()
    
#     try:
#         text_files = []
        
#         for file in files:
#             if file and file.filename:
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(temp_dir, filename)
#                 file.save(file_path)
                
#                 # Process PDF files to extract text
#                 if filename.lower().endswith('.pdf'):
#                     try:
#                         pdf = pdfplumber.open(file_path)
#                         text = ""
                        
#                         # Iterate through each page of the PDF
#                         for page in pdf.pages:
#                             # Extract text from the page
#                             page_text = page.extract_text()
                            
#                             # Perform OCR on scanned pages
#                             if not page_text.strip():
#                                 page_image = page.to_image()
#                                 page_text = pytesseract.image_to_string(page_image)
                            
#                             text += page_text
                        
#                         # Close the PDF
#                         pdf.close()
                        
#                         # Exclude headers and footers
#                         text = '\n'.join(line for line in text.splitlines() if not line.startswith(('Page ', 'Chapter ', 'Title ')))
                        
#                         # Save as text file
#                         txt_filename = os.path.splitext(filename)[0] + '.txt'
#                         txt_file_path = os.path.join(temp_dir, txt_filename)
#                         with open(txt_file_path, 'w', encoding='utf-8') as fp:
#                             fp.write(text)
                        
#                         text_files.append(txt_file_path)
                        
#                     except Exception as e:
#                         # Clean up on error
#                         import shutil
#                         shutil.rmtree(temp_dir, ignore_errors=True)
#                         return jsonify({'error': f'Error processing {filename}: {str(e)}'}), 500
                
#                 # Directly use text files
#                 elif filename.lower().endswith('.txt'):
#                     text_files.append(file_path)
        
#         if not text_files:
#             # Clean up on error
#             import shutil
#             shutil.rmtree(temp_dir, ignore_errors=True)
#             return jsonify({'error': 'No valid text files found'}), 400
        
#         # Merge all text files
#         merged_text = ""
#         for txt_file in text_files:
#             with open(txt_file, 'r', encoding='utf-8') as f:
#                 merged_text += f.read() + "\n"
        
#         # Clean the merged text
#         documents = merged_text.splitlines()
#         cleaned_documents = [preprocess_text(doc) for doc in documents]
#         cleaned_text = '\n'.join(cleaned_documents)
        
#         # Tokenize for Word2Vec
#         tokenized_documents = []
#         for doc in cleaned_documents:
#             words = nltk.word_tokenize(doc.lower())
#             # Remove numbers, dates, and special characters
#             words = [word for word in words if word.isalnum() and not word.isnumeric()]
#             words = [word for word in words if word not in stopwords.words('english')]
#             words = [lemmatizer.lemmatize(word) for word in words]
#             tokenized_documents.append(words)
        
#         # Train Word2Vec model
#         model = Word2Vec(sentences=tokenized_documents, vector_size=300, window=5, min_count=1, sg=0)
        
#         # Extract phrases and their counts
#         all_phrases = {}
#         for doc in tokenized_documents:
#             for i in range(len(doc)):
#                 for j in range(i, min(i + 3, len(doc))):
#                     phrase = ' '.join(doc[i:j+1])
#                     all_phrases[phrase] = all_phrases.get(phrase, 0) + 1
        
#         # Create phrase DataFrame
#         phrase_df = pd.DataFrame.from_dict(all_phrases, orient='index', columns=['count'])
#         phrase_df.sort_values('count', inplace=True, ascending=False)
        
#         # Create word vectors DataFrame
#         word_vectors = {}
#         for word in model.wv.index_to_key:
#             word_vectors[word] = model.wv[word]
        
#         # Create output files
#         output_dir = os.path.join(temp_dir, 'output')
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save phrase counts
#         phrase_csv_path = os.path.join(output_dir, 'phrase_counts.csv')
#         phrase_df.to_csv(phrase_csv_path)
        
#         # Save word vectors
#         vectors_path = os.path.join(output_dir, 'word_vectors.emb')
#         with open(vectors_path, 'w', encoding='utf-8') as emb_file:
#             emb_file.write(f"{len(word_vectors)} {len(next(iter(word_vectors.values())))}\n")
#             for word, vector in word_vectors.items():
#                 vector_str = ' '.join(str(value) for value in vector)
#                 emb_file.write(f"{word} {vector_str}\n")
        
#         # Create zip file with results
#         zip_path = os.path.join(temp_dir, 'word2vec_results.zip')
#         with zipfile.ZipFile(zip_path, 'w') as zipf:
#             zipf.write(phrase_csv_path, 'phrase_counts.csv')
#             zipf.write(vectors_path, 'word_vectors.emb')
        
#         # Send file and manually clean up after response is sent
#         response = send_file(zip_path, as_attachment=True, download_name='word2vec_results.zip')
        
#         # Use callback to clean up temporary directory after response
#         @response.call_on_close
#         def cleanup_temp_dir():
#             import shutil
#             try:
#                 shutil.rmtree(temp_dir, ignore_errors=True)
#             except Exception as e:
#                 print(f"Error cleaning up temporary directory: {e}")
        
#         return response
        
#     except Exception as e:
#         # Clean up on any unexpected error
#         import shutil
#         shutil.rmtree(temp_dir, ignore_errors=True)
#         return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)