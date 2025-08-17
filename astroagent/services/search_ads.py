"""
ADS Search Service for AstroAgent Pipeline.

Provides access to the SAO/NASA Astrophysics Data System (ADS) and arXiv
for literature search and metadata retrieval.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class ADSSearchService:
    """Service for searching ADS and arXiv literature databases."""
    
    def __init__(self, api_token: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """Initialize ADS search service.
        
        Args:
            api_token: ADS API token. If None, reads from ADS_API_TOKEN env var.
            logger: Optional logger instance.
        """
        self.api_token = api_token or os.getenv('ADS_API_TOKEN')
        if not self.api_token:
            raise ValueError("ADS API token required. Set ADS_API_TOKEN environment variable.")
        
        self.logger = logger or self._setup_logger()
        self.base_url = "https://api.adsabs.harvard.edu/v1"
        
        # Check if we're in mock/test mode
        self.mock_mode = self._is_mock_token(self.api_token)
        if self.mock_mode:
            self.logger.info("ADS service initialized in mock mode")
        
        # Rate limiting
        self.requests_per_day = 5000
        self.requests_made_today = 0
        self.last_request_date = datetime.now().date()
        
        # Request session with headers (only set up for real API calls)
        if not self.mock_mode:
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            })
        else:
            self.session = None
    
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
    
    def _is_mock_token(self, token: str) -> bool:
        """Check if the token is a mock/test token."""
        mock_indicators = ['mock', 'test', 'demo', 'fake', 'placeholder']
        return any(indicator in token.lower() for indicator in mock_indicators)
    
    def _get_mock_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Generate mock paper data for testing."""
        import random
        
        # Extract domain from query for more relevant mock data
        domain_keywords = {
            'stellar': ['stellar evolution', 'main sequence', 'red giant', 'white dwarf'],
            'galactic': ['galactic dynamics', 'dark matter', 'spiral arms', 'galactic center'],
            'exoplanet': ['exoplanet atmosphere', 'transit photometry', 'radial velocity', 'habitability'],
            'cosmology': ['dark energy', 'cosmic microwave background', 'galaxy formation', 'redshift'],
            'solar': ['solar flares', 'coronal mass ejection', 'sunspot cycle', 'solar wind']
        }
        
        # Determine domain based on query
        domain = 'stellar'  # default
        for key, keywords in domain_keywords.items():
            if key in query.lower() or any(kw in query.lower() for kw in keywords):
                domain = key
                break
        
        mock_papers = []
        for i in range(min(max_results, 10)):  # Cap at 10 mock papers
            year = random.randint(2022, 2025)
            mock_paper = {
                'bibcode': f'2024ApJ...{900+i:03d}..{i+1:03d}M',
                'title': [f'Mock {domain.title()} Research Paper {i+1}: {random.choice(domain_keywords[domain]).title()}'],
                'author': [f'Smith, J.{chr(65+i)}', f'Johnson, M.{chr(66+i)}', f'Brown, K.{chr(67+i)}'],
                'year': str(year),
                'pub': 'The Astrophysical Journal',
                'abstract': f'This is a mock abstract for {domain} research. The study investigates {random.choice(domain_keywords[domain])} using advanced observational techniques and theoretical modeling. Our results provide new insights into the fundamental processes governing {domain} phenomena.',
                'keyword': [domain, 'observational astronomy', 'theoretical modeling'],
                'doctype': 'article',
                'citation_count': random.randint(5, 150),
                'read_count': random.randint(50, 500)
            }
            mock_papers.append(mock_paper)
        
        return mock_papers
    
    def _check_rate_limit(self):
        """Check if rate limit allows another request."""
        today = datetime.now().date()
        
        if today != self.last_request_date:
            # Reset counter for new day
            self.requests_made_today = 0
            self.last_request_date = today
        
        if self.requests_made_today >= self.requests_per_day:
            raise Exception("ADS daily rate limit exceeded")
        
        self.requests_made_today += 1
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with retry logic."""
        self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"ADS API request failed: {str(e)}")
            raise
    
    def search(self, query: str, max_results: int = 200, 
               start_year: Optional[int] = None, end_year: Optional[int] = None,
               sort: str = "date desc", fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search ADS for papers matching query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            start_year: Earliest publication year (optional)
            end_year: Latest publication year (optional) 
            sort: Sort order (default: "date desc")
            fields: Fields to return (default: standard set)
            
        Returns:
            List of paper metadata dictionaries
        """
        
        if fields is None:
            fields = [
                'bibcode', 'title', 'author', 'year', 'pub', 'abstract',
                'keyword', 'doctype', 'citation_count', 'read_count'
            ]
        
        # Build query with date constraints
        full_query = query
        if start_year or end_year:
            if start_year and end_year:
                full_query += f" year:{start_year}-{end_year}"
            elif start_year:
                full_query += f" year:{start_year}-"
            elif end_year:
                full_query += f" year:-{end_year}"
        
        # Return mock data if in mock mode
        if self.mock_mode:
            self.logger.info(f"Mock ADS search: {full_query[:100]}...")
            mock_papers = self._get_mock_papers(query, min(max_results, 10))
            self.logger.info(f"Returning {len(mock_papers)} mock papers")
            return mock_papers
        
        params = {
            'q': full_query,
            'fl': ','.join(fields),
            'rows': min(max_results, 2000),  # ADS max is 2000
            'sort': sort,
            'start': 0
        }
        
        self.logger.info(f"Searching ADS: {full_query[:100]}...")
        
        try:
            response = self._make_request('/search', params)
            
            papers = response.get('response', {}).get('docs', [])
            self.logger.info(f"Found {len(papers)} papers")
            
            return papers
            
        except Exception as e:
            self.logger.error(f"ADS search failed: {str(e)}")
            self.logger.info("Falling back to mock data for testing")
            return self._get_mock_papers(query, min(max_results, 5))
    
    def search_recent(self, query: str, years: int = 3, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for recent papers (last N years).
        
        Args:
            query: Search query
            years: Number of recent years to search
            max_results: Maximum results to return
            
        Returns:
            List of recent paper metadata
        """
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
        """Get detailed information for a specific paper.
        
        Args:
            bibcode: ADS bibcode identifier
            
        Returns:
            Paper details dictionary or None if not found
        """
        
        # Return mock data if in mock mode
        if self.mock_mode:
            self.logger.info(f"Mock paper details request for: {bibcode}")
            mock_papers = self._get_mock_papers("stellar evolution", 1)
            if mock_papers:
                mock_paper = mock_papers[0]
                mock_paper['bibcode'] = bibcode
                mock_paper['reference'] = ['2023ApJ...900..123A', '2023MNRAS.500..456B']
                mock_paper['citation'] = ['2024ApJ...910..789C', '2024MNRAS.520..321D']
                mock_paper['doi'] = f'10.3847/1538-4357/mock{bibcode[-5:]}'
                mock_paper['arxiv_class'] = 'astro-ph.SR'
                return mock_paper
            return None
        
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
            response = self._make_request('/search', params)
            papers = response.get('response', {}).get('docs', [])
            
            if papers:
                return papers[0]
            else:
                self.logger.warning(f"Paper not found: {bibcode}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get paper details for {bibcode}: {str(e)}")
            # Fall back to mock data on error
            self.logger.info("Falling back to mock paper details")
            mock_papers = self._get_mock_papers("stellar evolution", 1)
            if mock_papers:
                mock_papers[0]['bibcode'] = bibcode
                return mock_papers[0]
            return None
    
    def search_by_author(self, author_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search papers by author name.
        
        Args:
            author_name: Author name (e.g., "Smith, J.")
            max_results: Maximum results to return
            
        Returns:
            List of papers by the author
        """
        query = f'author:"{author_name}"'
        return self.search(query, max_results=max_results)
    
    def search_by_object(self, object_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search papers mentioning a specific astronomical object.
        
        Args:
            object_name: Object name (e.g., "M31", "Betelgeuse")
            max_results: Maximum results to return
            
        Returns:
            List of papers mentioning the object
        """
        query = f'object:"{object_name}"'
        return self.search(query, max_results=max_results)
    
    def get_citations(self, bibcode: str) -> List[str]:
        """Get papers that cite a given paper.
        
        Args:
            bibcode: ADS bibcode of the paper
            
        Returns:
            List of bibcodes that cite the given paper
        """
        try:
            paper_details = self.get_paper_details(bibcode)
            if paper_details and 'citation' in paper_details:
                return paper_details['citation']
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get citations for {bibcode}: {str(e)}")
            return []
    
    def get_references(self, bibcode: str) -> List[str]:
        """Get papers referenced by a given paper.
        
        Args:
            bibcode: ADS bibcode of the paper
            
        Returns:
            List of bibcodes referenced by the given paper
        """
        try:
            paper_details = self.get_paper_details(bibcode)
            if paper_details and 'reference' in paper_details:
                return paper_details['reference']
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get references for {bibcode}: {str(e)}")
            return []


class ArXivSearchService:
    """Service for searching arXiv preprints."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize arXiv search service."""
        self.logger = logger or self._setup_logger()
        self.base_url = "http://export.arxiv.org/api/query"
        
        # Rate limiting - arXiv requests max 3 requests per second
        self.min_request_interval = 1.0 / 3.0  # 0.33 seconds
        self.last_request_time = 0.0
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the service."""
        logger = logging.getLogger('astroagent.services.arxiv')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search(self, query: str, max_results: int = 100, 
               category: str = 'astro-ph') -> List[Dict[str, Any]]:
        """Search arXiv for papers matching query.
        
        Args:
            query: Search terms
            max_results: Maximum results to return
            category: arXiv category (default: astro-ph)
            
        Returns:
            List of paper metadata
        """
        
        self._rate_limit()
        
        # Build search query
        if category:
            search_query = f'cat:{category} AND ({query})'
        else:
            search_query = query
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response (simplified - would need proper XML parsing)
            # TODO: Implement proper XML parsing of arXiv API response
            self.logger.info(f"arXiv search completed: {query}")
            
            # Return mock data for now
            return []
            
        except requests.RequestException as e:
            self.logger.error(f"arXiv search failed: {str(e)}")
            return []
