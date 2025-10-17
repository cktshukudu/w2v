import requests
import os
import sys
from urllib.parse import quote
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
from werkzeug.utils import secure_filename
from datetime import datetime

def perform_search(query, start_year=None, end_year=None, max_pages=100, sort_by='date'):
    """
    Perform Google Scholar search and return results as DataFrame
    """
    # Encode query
    encoded_query = quote(query)
    
    # Validate years
    current_year = datetime.now().year
    if start_year is None:
        start_year = 1900  # Default to earliest possible
    if end_year is None:
        end_year = current_year  # Default to current year
    
    # Ensure years are integers
    try:
        start_year = int(start_year)
        end_year = int(end_year)
    except (ValueError, TypeError):
        raise ValueError("Years must be integers")
    
    # Validate year range
    if start_year < 1900:
        start_year = 1900
    if end_year > current_year:
        end_year = current_year
    if start_year > end_year:
        start_year, end_year = end_year, start_year  # Swap if reversed
    
    # Create URLs with year parameters
    pages = []
    for page_value in range(0, max_pages, 1):
        url = f'https://scholar.google.com/scholar?start={page_value}0&q={encoded_query}&hl=en&as_sdt=0,5'
        
        # Add year filters only if not searching all years
        if start_year != 1900 or end_year != current_year:
            url += f'&as_ylo={start_year}&as_yhi={end_year}'
        
        pages.append(url)
    
    # Scrape pages
    everything = []
    for url in tqdm(pages, desc="Searching Google Scholar"):
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        proxies = {'http': os.getenv('HTTP_PROXY')}
        
        try:
            response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
            doc = BeautifulSoup(response.text, 'html.parser')
            everything.append(doc)
            time.sleep(2)  # Be polite with requests
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            continue
    
    # Extract data
    titles, links, citations, pub_info = [], [], [], []
    
    for doc in everything:
        paper_tags = doc.select('[data-lid]')
        for tag in paper_tags:
            try:
                title = tag.select('h3')[0].get_text()
                link = tag.select('a')[0]['href']
                pub_info_text = tag.select_one('.gs_a').get_text()
                
                # Get citations if available
                cite_tag = tag.select_one('.gs_fl a[href*="cites"]')
                citation_count = 0
                if cite_tag:
                    citation_text = cite_tag.get_text()
                    if 'Cited by' in citation_text:
                        citation_count = int(''.join(filter(str.isdigit, citation_text)))
                
                titles.append(title)
                links.append(link)
                citations.append(citation_count)
                pub_info.append(pub_info_text)
            except Exception as e:
                print(f"Error parsing paper: {str(e)}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame({
        'title': titles,
        'url': links,
        'citations': citations,
        'pub_info': pub_info
    })
    
    # Extract year from publication info with improved parsing
    def extract_year(info):
        # First try to find 4-digit numbers that look like years
        for word in info.split():
            if len(word) == 4 and word.isdigit():
                year = int(word)
                if 1900 <= year <= datetime.now().year:
                    return year
        return None
    
    df['year'] = df['pub_info'].apply(extract_year)
    
    # Fill missing years with the middle of the search range
    if not df['year'].isnull().all():
        median_year = df['year'].median()
        if pd.isna(median_year):
            median_year = (start_year + end_year) // 2
        df['year'] = df['year'].fillna(median_year).astype(int)
    else:
        df['year'] = (start_year + end_year) // 2
    
    # Ensure years are within bounds
    df['year'] = df['year'].clip(lower=start_year, upper=end_year)
    
    # Sort results
    if sort_by == 'date':
        df = df.sort_values('year', ascending=False)
    elif sort_by == 'citations':
        df = df.sort_values('citations', ascending=False)
    else:  # Default sort by relevance (as returned by Google)
        pass
    
    return df

def download_papers(papers, download_dir, max_downloads=500):
    """
    Download papers to specified directory
    """
    downloaded = []
    count = 0
    
    for paper in tqdm(papers[:max_downloads], desc="Downloading papers"):
        try:
            url = paper['url']
            title = secure_filename(paper['title'][:100])  # Limit filename length
            
            # Skip if not PDF (but try anyway if URL doesn't end with .pdf)
            response = requests.head(url, timeout=5, allow_redirects=True)
            content_type = response.headers.get('content-type', '')
            
            if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                downloaded.append({
                    'title': paper['title'],
                    'url': url,
                    'error': 'Not a PDF',
                    'success': False
                })
                continue
                
            # Download the actual content
            response = requests.get(url, stream=True, timeout=10)
            
            if response.status_code == 200:
                filepath = os.path.join(download_dir, f"{title}.pdf")
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify the downloaded file is actually a PDF
                with open(filepath, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        os.remove(filepath)
                        raise ValueError("Downloaded file is not a valid PDF")
                
                downloaded.append({
                    'title': paper['title'],
                    'path': filepath,
                    'url': url,
                    'year': paper.get('year', 'Unknown'),
                    'success': True
                })
                count += 1
        except Exception as e:
            downloaded.append({
                'title': paper.get('title', 'Unknown'),
                'url': url,
                'error': str(e),
                'success': False
            })
            continue
    
    return downloaded

