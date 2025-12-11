# scraper.py
import requests
import os
import json
import time
import re
from typing import List, Dict, Optional
import urllib.parse
import arxiv
from datetime import datetime

class PaperScraper:
    def __init__(self):
        """Initialize paper scraper"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_arxiv(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of arXiv papers
        """
        try:
            print(f"Searching arXiv for: {query}")
            search = arxiv.Search(
                query=query,
                max_results=min(max_results, 500),
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = {
                    'title': result.title,
                    'authors': [str(author) for author in result.authors],
                    'abstract': result.summary,
                    'pdf_url': result.pdf_url,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'doi': result.doi,
                    'source': 'arxiv',
                    'publication_year': result.published.strftime('%Y') if result.published else '',
                    'journal': 'arXiv',
                    'cited_by': 0
                }
                papers.append(paper)
                
            print(f"Found {len(papers)} arXiv papers")
            return papers
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def search_crossref(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search CrossRef API for papers
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of paper metadata
        """
        print(f"Searching CrossRef for: {query}")
        url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': min(max_results, 100),
            'select': 'DOI,title,author,created,container-title,abstract,URL'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            items = data.get('message', {}).get('items', [])
            
            for item in items:
                # Get authors
                authors = 'Unknown'
                if 'author' in item and item['author']:
                    author_list = []
                    for auth in item['author']:
                        name_parts = []
                        if 'given' in auth:
                            name_parts.append(auth['given'])
                        if 'family' in auth:
                            name_parts.append(auth['family'])
                        if name_parts:
                            author_list.append(' '.join(name_parts))
                    if author_list:
                        authors = ', '.join(author_list)
                
                # Get publication year
                pub_year = ''
                if 'created' in item and 'date-parts' in item['created']:
                    pub_year = str(item['created']['date-parts'][0][0])
                elif 'issued' in item and 'date-parts' in item['issued']:
                    pub_year = str(item['issued']['date-parts'][0][0])
                
                # Get title
                title = 'No title'
                if 'title' in item and item['title']:
                    title = item['title'][0]
                
                paper = {
                    'title': title,
                    'authors': authors,
                    'publication_year': pub_year,
                    'doi': item.get('DOI'),
                    'journal': item.get('container-title', [''])[0] if item.get('container-title') else '',
                    'abstract': item.get('abstract', ''),
                    'link': item.get('URL', ''),
                    'source': 'crossref',
                    'cited_by': 0
                }
                papers.append(paper)
                
            print(f"Found {len(papers)} CrossRef papers")
            return papers
        except Exception as e:
            print(f"Error searching CrossRef: {e}")
            return []
    
    def search_semantic_scholar(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search Semantic Scholar API for papers
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of paper metadata
        """
        print(f"Searching Semantic Scholar for: {query}")
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': min(max_results, 100),
            'fields': 'title,authors,abstract,year,venue,url,externalIds,openAccessPdf,citationCount'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            items = data.get('data', [])
            
            for item in items:
                # Get authors
                authors = 'Unknown'
                if 'authors' in item and item['authors']:
                    author_list = [author.get('name', '') for author in item['authors']]
                    authors = ', '.join([a for a in author_list if a])
                
                # Get PDF URL
                pdf_url = None
                if 'openAccessPdf' in item and item['openAccessPdf']:
                    pdf_url = item['openAccessPdf'].get('url')
                
                paper = {
                    'title': item.get('title', 'No title'),
                    'authors': authors,
                    'publication_year': str(item.get('year', '')),
                    'doi': item.get('externalIds', {}).get('DOI'),
                    'journal': item.get('venue', ''),
                    'abstract': item.get('abstract', ''),
                    'pdf_url': pdf_url,
                    'link': item.get('url', ''),
                    'source': 'semantic_scholar',
                    'cited_by': item.get('citationCount', 0)
                }
                papers.append(paper)
                
            print(f"Found {len(papers)} Semantic Scholar papers")
            return papers
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
            return []
    
    def download_pdf(self, url: str, filename: str, folder: str = "papers") -> bool:
        """
        Download PDF from URL
        
        Args:
            url: PDF URL
            filename: Name to save the file as
            folder: Folder to save PDFs in
            
        Returns:
            True if successful, False otherwise
        """
        os.makedirs(folder, exist_ok=True)
        
        # Clean filename
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        if not filename.endswith('.pdf'):
            filename += '.pdf'
            
        filepath = os.path.join(folder, filename)
        
        try:
            response = requests.get(url, headers=self.headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                print(f"URL doesn't point to PDF: {content_type}")
                return False
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"Error downloading PDF from {url}: {e}")
            return False
    
    def try_download_pdf(self, paper: Dict, folder: str = "papers") -> bool:
        """
        Try multiple methods to download PDF for a paper
        
        Args:
            paper: Paper metadata
            folder: Folder to save PDFs
            
        Returns:
            True if downloaded successfully
        """
        title = paper.get('title', 'unknown').replace(' ', '_')[:50]
        source = paper.get('source', 'unknown')
        
        print(f"Trying to download: {title} ({source})")
        
        # Method 1: Try arXiv if we have arXiv ID
        if source == 'arxiv' and 'arxiv_id' in paper:
            pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
            filename = f"arxiv_{paper['arxiv_id']}_{title}"
            if self.download_pdf(pdf_url, filename, folder):
                return True
        
        # Method 2: Try arXiv PDF URL directly
        if 'pdf_url' in paper and paper['pdf_url']:
            filename = f"{source}_{title}"
            if self.download_pdf(paper['pdf_url'], filename, folder):
                return True
        
        # Method 3: Try Unpaywall for legal open access
        doi = paper.get('doi')
        if doi:
            try:
                print(f"Trying Unpaywall for DOI: {doi}")
                unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=paper_search@example.com"
                response = requests.get(unpaywall_url, timeout=10, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('is_oa') and data.get('best_oa_location'):
                        pdf_url = data['best_oa_location']['url']
                        filename = f"oa_{doi.replace('/', '_')}_{title}"
                        if self.download_pdf(pdf_url, filename, folder):
                            return True
            except Exception as e:
                print(f"Unpaywall error: {e}")
        
        print(f"Could not download PDF for: {paper.get('title', 'Unknown')}")
        return False