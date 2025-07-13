from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from text_processor import process_text, extract_text_from_pdfs
import os
import spacy
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    filenames = []
    
    for file in files:
        if file.filename == '':
            continue
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filenames.append(filename)
    
    return jsonify({
        'success': True,
        'message': f'{len(filenames)} files uploaded',
        'filenames': filenames
    })

@app.route('/extract', methods=['POST'])
def extract_text():
    data = request.json
    filenames = data.get('filenames', [])
    results = []
    
    for filename in filenames:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            text = extract_text_from_pdfs(filepath)
            results.append({
                'filename': filename,
                'text': text,
                'success': True
            })
        except Exception as e:
            results.append({
                'filename': filename,
                'error': str(e),
                'success': False
            })
    
    return jsonify({
        'success': True,
        'results': results
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    texts = data.get('texts', [])
    max_phrase_length = int(data.get('maxPhraseLength', 4))
    min_word_length = int(data.get('minWordLength', 2))
    analysis_type = data.get('analysisType', 'both')
    
    combined_text = ' '.join([t['text'] for t in texts if t.get('success', False)])
    
    if not combined_text:
        return jsonify({'success': False, 'message': 'No valid text to analyze'}), 400
    
    try:
        results = process_text(
            combined_text,
            max_phrase_length=max_phrase_length,
            min_word_length=min_word_length,
            analysis_type=analysis_type
        )
        
        # Save results to CSV
        results_file = os.path.join(app.config['RESULTS_FOLDER'], 'analysis_results.csv')
        results['phrases'].to_csv(results_file, index=False)
        
        return jsonify({
            'success': True,
            'results': {
                'phrases': results['phrases'].to_dict('records'),
                'nouns': results['nouns'].to_dict('records'),
                'adjectives': results['adjectives'].to_dict('records'),
                'verbs': results['verbs'].to_dict('records'),
                'properNouns': results['properNouns'].to_dict('records')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/download')
def download_results():
    results_file = os.path.join(app.config['RESULTS_FOLDER'], 'analysis_results.csv')
    if not os.path.exists(results_file):
        return jsonify({'success': False, 'message': 'No results available'}), 404
    
    return send_file(
        results_file,
        as_attachment=True,
        download_name='text_analysis_results.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)