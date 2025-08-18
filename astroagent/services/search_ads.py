"""
ADS Literature Search Service for AstroAgent Pipeline.

This service provides access to the ADS (Astrophysics Data System) API
for literature searches. NO MOCK DATA OR FALLBACKS - real API only.
"""

import logging
import os
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential


class ADSSearchService:
    """Service for searching ADS literature databases - REAL API ONLY."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize ADS search service."""
        
        # Get API token
        self.api_token = os.getenv('ADS_API_TOKEN')
        if not self.api_token:
            raise ValueError("ADS API token required. Set ADS_API_TOKEN environment variable.")
        
        self.logger = logger or self._setup_logger()
        self.base_url = "https://api.adsabs.harvard.edu/v1"
        
        # NO MOCK MODE - always use real API or fail
        self.logger.info("ADS service initialized - real API only, no fallbacks")
        
        # Rate limiting
        self.requests_per_day = 5000
        self.requests_made_today = 0
        self.last_request_date = datetime.now().date()
        
        # Request session with headers - always real API
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        })
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the service."""
        logger = logging.getLogger('astroagent.services.ads')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _check_rate_limit(self):
        """Check if rate limit allows another request."""
        today = datetime.now().date()
        
        # Reset counter if new day
        if today != self.last_request_date:
            self.requests_made_today = 0
            self.last_request_date = today
        
        # Check if we've hit the limit
        if self.requests_made_today >= self.requests_per_day:
            raise Exception(f"Daily ADS API rate limit exceeded ({self.requests_per_day} requests)")
        
        self.requests_made_today += 1
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to ADS API with retry logic."""
        
        self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"ADS API request failed: {e}")
            raise
    
    def search(self, query: str, max_results: int = 200, 
               start_year: Optional[int] = None, end_year: Optional[int] = None,
               sort: str = "date desc", fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search ADS for papers matching query."""
        
        if not query or not query.strip():
            self.logger.warning("Empty query provided to ADS search")
            return []
        
        # Default fields
        if fields is None:
            fields = [
                'bibcode', 'title', 'author', 'year', 'pub', 'abstract',
                'keyword', 'doctype', 'citation_count', 'read_count'
            ]
        
        # Build query with year constraints
        full_query = query.strip()
        
        # Add year constraints if provided
        if start_year and end_year:
            full_query += f" year:{start_year}-{end_year}"
        elif start_year:
            full_query += f" year:{start_year}-"
        elif end_year:
            full_query += f" year:-{end_year}"
        
        # NO MOCK DATA EVER - real API only
        
        params = {
            'q': full_query,
            'fl': ','.join(fields),
            'rows': min(max_results, 2000),  # ADS max is 2000
            'sort': sort,
            'start': 0
        }
        
        self.logger.info(f"Searching ADS: '{full_query}' (length: {len(full_query)})")
        
        try:
            response = self._make_request('/search/query', params)
            papers = response.get('response', {}).get('docs', [])
            self.logger.info(f"Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            self.logger.error(f"ADS search failed: {str(e)}")
            # NO FALLBACKS - fail cleanly
            raise Exception(f"ADS search failed and no fallbacks allowed: {str(e)}")
    
    def search_recent(self, query: str, years: int = 3, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for recent papers (last N years)."""
        
        current_year = datetime.now().year
        start_year = current_year - years
        
        return self.search(
            query=query,
            max_results=max_results,
            start_year=start_year,
            end_year=current_year,
            sort="date desc"
        )
    
    def get_paper_details(self, bibcode: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific paper by bibcode."""
        
        if not bibcode or not bibcode.strip():
            self.logger.warning("Empty bibcode provided")
            return None
        
        # NO MOCK DATA EVER
        
        fields = [
            'bibcode', 'title', 'author', 'year', 'pub', 'abstract',
            'keyword', 'doctype', 'citation_count', 'read_count',
            'reference', 'citation', 'doi', 'arxiv_class'
        ]
        
        params = {
            'q': f'bibcode:{bibcode}',
            'fl': ','.join(fields),
            'rows': 1
        }
        
        try:
            response = self._make_request('/search/query', params)
            papers = response.get('response', {}).get('docs', [])
            
            if papers:
                return papers[0]
            else:
                self.logger.warning(f"Paper not found: {bibcode}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get paper details for {bibcode}: {str(e)}")
            # NO FALLBACKS - return None to indicate failure
            return None
    
    def search_by_author(self, author_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search papers by author name."""
        
        if not author_name or not author_name.strip():
            self.logger.warning("Empty author name provided")
            return []
        
        query = f'author:"{author_name}"'
        return self.search(query, max_results=max_results, sort="date desc")
    
    def get_citations(self, bibcode: str) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper."""
        
        if not bibcode or not bibcode.strip():
            self.logger.warning("Empty bibcode provided for citations")
            return []
        
        try:
            # Use ADS citations endpoint
            params = {
                'bibcode': bibcode,
                'fl': 'bibcode,title,author,year,pub,citation_count',
                'rows': 100
            }
            
            response = self._make_request('/search/query', params)
            citations = response.get('response', {}).get('docs', [])
            
            self.logger.info(f"Found {len(citations)} citations for {bibcode}")
            return citations
            
        except Exception as e:
            self.logger.error(f"Failed to get citations for {bibcode}: {str(e)}")
            return []
    
    def search_similar(self, bibcode: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Find papers similar to the given paper."""
        
        if not bibcode or not bibcode.strip():
            self.logger.warning("Empty bibcode provided for similarity search")
            return []
        
        try:
            # Get the original paper details first
            paper = self.get_paper_details(bibcode)
            if not paper:
                return []
            
            # Extract keywords and search for similar papers
            keywords = paper.get('keyword', [])
            if keywords:
                # Use top 3 keywords for similarity search
                keyword_query = ' OR '.join(f'keyword:"{kw}"' for kw in keywords[:3])
                return self.search(keyword_query, max_results=max_results, sort="citation_count desc")
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to find similar papers for {bibcode}: {str(e)}")
            return []
    
    def search_arxiv(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search arXiv for papers (through ADS interface)."""
        
        if not query or not query.strip():
            self.logger.warning("Empty query provided for arXiv search")
            return []
        
        # Search ADS for arXiv papers
        arxiv_query = f"{query} AND doctype:eprint"
        return self.search(arxiv_query, max_results=max_results, sort="date desc")
    
    def get_trending_topics(self, domain: str, days: int = 30) -> List[str]:
        """Get trending topics in a domain over the last N days."""
        
        if not domain or not domain.strip():
            self.logger.warning("Empty domain provided for trending topics")
            return []
        
        try:
            # Search recent papers in domain
            current_year = datetime.now().year
            papers = self.search(
                query=domain,
                max_results=100,
                start_year=current_year,
                sort="read_count desc"  # Most read papers
            )
            
            # Extract keywords from top papers
            all_keywords = []
            for paper in papers[:20]:  # Top 20 most read papers
                keywords = paper.get('keyword', [])
                all_keywords.extend(keywords)
            
            # Count keyword frequency
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            
            # Return top trending keywords
            trending = [kw for kw, count in keyword_counts.most_common(10) if count > 1]
            self.logger.info(f"Found {len(trending)} trending topics for {domain}")
            
            return trending
            
        except Exception as e:
            self.logger.error(f"Failed to get trending topics for {domain}: {str(e)}")
            return []
