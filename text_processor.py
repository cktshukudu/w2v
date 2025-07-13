import os
import pdfplumber
import pytesseract
import spacy
import pandas as pd
import string
from collections import defaultdict

def extract_text_from_pdfs(pdf_path):
    """Extract text from PDF using pdfplumber and OCR if needed"""
    text = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                
                # Perform OCR if no text found
                if not page_text or not page_text.strip():
                    page_image = page.to_image(resolution=300)
                    page_text = pytesseract.image_to_string(page_image)
                
                text += page_text + "\n"
    
    except Exception as e:
        raise Exception(f"Error processing {pdf_path}: {str(e)}")
    
    return text

def process_text(text, max_phrase_length=4, min_word_length=2, analysis_type='both'):
    """Process text and extract phrases, nouns, adjectives, verbs, and proper nouns"""
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = len(text) + 1000000  # Increase max length
    
    doc = nlp(text)
    
    results = {
        'phrases': defaultdict(int),
        'nouns': defaultdict(int),
        'adjectives': defaultdict(int),
        'verbs': defaultdict(int),
        'properNouns': defaultdict(int)
    }
    
    # Process phrases
    if analysis_type in ['phrases', 'both']:
        for i, token in enumerate(doc):
            if token.is_stop or token.is_punct:
                continue
                
            words = [token.lemma_.lower()]
            
            for j in range(i + 1, min(i + max_phrase_length, len(doc))):
                if doc[j].is_stop or doc[j].is_punct:
                    break
                words.append(doc[j].lemma_.lower())
                
                phrase = ' '.join(words)
                if (len(phrase) >= min_word_length and 
                    not any(c.isdigit() or c in string.punctuation for c in phrase)):
                    results['phrases'][phrase] += 1
    
    # Process parts of speech
    if analysis_type in ['words', 'both']:
        for token in doc:
            if (token.is_stop or token.is_punct or 
                len(token.lemma_) < min_word_length or
                any(c.isdigit() or c in string.punctuation for c in token.lemma_)):
                continue
                
            lemma = token.lemma_.lower()
            pos = token.pos_
            
            if pos == 'NOUN':
                results['nouns'][lemma] += 1
            elif pos == 'ADJ':
                results['adjectives'][lemma] += 1
            elif pos == 'VERB':
                results['verbs'][lemma] += 1
            elif pos == 'PROPN':
                results['properNouns'][lemma] += 1
    
    # Convert to DataFrames and sort
    return {
        'phrases': pd.DataFrame(
            sorted(results['phrases'].items(), key=lambda x: x[1], reverse=True),
            columns=['Phrase', 'Count']
        ),
        'nouns': pd.DataFrame(
            sorted(results['nouns'].items(), key=lambda x: x[1], reverse=True),
            columns=['Noun', 'Count']
        ),
        'adjectives': pd.DataFrame(
            sorted(results['adjectives'].items(), key=lambda x: x[1], reverse=True),
            columns=['Adjective', 'Count']
        ),
        'verbs': pd.DataFrame(
            sorted(results['verbs'].items(), key=lambda x: x[1], reverse=True),
            columns=['Verb', 'Count']
        ),
        'properNouns': pd.DataFrame(
            sorted(results['properNouns'].items(), key=lambda x: x[1], reverse=True),
            columns=['ProperNoun', 'Count']
        )
    }