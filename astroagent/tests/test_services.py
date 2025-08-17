"""
Unit tests for AstroAgent Pipeline services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path
import numpy as np

from ..services.search_ads import ADSSearchService, ArXivSearchService
from ..services.literature import LiteratureService
from . import TestFixtures


class TestADSSearchService:
    """Test the ADS search service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_token = "test_token_123"
    
    def test_ads_service_initialization(self):
        """Test ADS service initializes correctly."""
        with patch.dict('os.environ', {'ADS_API_TOKEN': self.mock_token}):
            service = ADSSearchService()
            assert service.api_token == self.mock_token
            assert service.base_url == "https://api.adsabs.harvard.edu/v1"
    
    def test_ads_service_initialization_no_token(self):
        """Test ADS service fails without token."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="ADS API token required"):
                ADSSearchService()
    
    @patch('astroagent.services.search_ads.requests.Session')
    def test_ads_search_success(self, mock_session):
        """Test successful ADS search."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {
                'docs': [TestFixtures.get_mock_paper()]
            }
        }
        mock_response.raise_for_status.return_value = None
        
        mock_session.return_value.get.return_value = mock_response
        
        with patch.dict('os.environ', {'ADS_API_TOKEN': self.mock_token}):
            service = ADSSearchService()
            
            # Reset request counter for test
            service.requests_made_today = 0
            
            results = service.search("dark matter", max_results=10)
            
            assert len(results) == 1
            assert results[0]['title'] == TestFixtures.get_mock_paper()['title']
    
    @patch('astroagent.services.search_ads.requests.Session')
    def test_ads_search_with_date_constraints(self, mock_session):
        """Test ADS search with date constraints."""
        mock_response = Mock()
        mock_response.json.return_value = {'response': {'docs': []}}
        mock_response.raise_for_status.return_value = None
        
        mock_session.return_value.get.return_value = mock_response
        
        with patch.dict('os.environ', {'ADS_API_TOKEN': self.mock_token}):
            service = ADSSearchService()
            service.requests_made_today = 0
            
            # Test with both start and end year
            results = service.search("test", start_year=2020, end_year=2023)
            
            # Check that the request was called with year constraint
            call_args = mock_session.return_value.get.call_args
            assert 'year:2020-2023' in call_args[1]['params']['q']
    
    @patch('astroagent.services.search_ads.requests.Session')
    def test_ads_get_paper_details(self, mock_session):
        """Test getting specific paper details."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {
                'docs': [TestFixtures.get_mock_paper()]
            }
        }
        mock_response.raise_for_status.return_value = None
        
        mock_session.return_value.get.return_value = mock_response
        
        with patch.dict('os.environ', {'ADS_API_TOKEN': self.mock_token}):
            service = ADSSearchService()
            service.requests_made_today = 0
            
            paper = service.get_paper_details("2023ApJ...900...1A")
            
            assert paper is not None
            assert paper['bibcode'] == "2023ApJ...900...1A"
    
    def test_ads_rate_limit_check(self):
        """Test rate limiting functionality."""
        with patch.dict('os.environ', {'ADS_API_TOKEN': self.mock_token}):
            service = ADSSearchService()
            
            # Set request count to limit
            service.requests_made_today = service.requests_per_day
            
            with pytest.raises(Exception, match="rate limit exceeded"):
                service._check_rate_limit()
    
    @patch('astroagent.services.search_ads.requests.Session')
    def test_ads_search_by_author(self, mock_session):
        """Test searching papers by author."""
        mock_response = Mock()
        mock_response.json.return_value = {'response': {'docs': []}}
        mock_response.raise_for_status.return_value = None
        
        mock_session.return_value.get.return_value = mock_response
        
        with patch.dict('os.environ', {'ADS_API_TOKEN': self.mock_token}):
            service = ADSSearchService()
            service.requests_made_today = 0
            
            results = service.search_by_author("Smith, J.")
            
            # Check that the query includes author search
            call_args = mock_session.return_value.get.call_args
            assert 'author:"Smith, J."' in call_args[1]['params']['q']
    
    @patch('astroagent.services.search_ads.requests.Session')
    def test_ads_search_by_object(self, mock_session):
        """Test searching papers by astronomical object."""
        mock_response = Mock()
        mock_response.json.return_value = {'response': {'docs': []}}
        mock_response.raise_for_status.return_value = None
        
        mock_session.return_value.get.return_value = mock_response
        
        with patch.dict('os.environ', {'ADS_API_TOKEN': self.mock_token}):
            service = ADSSearchService()
            service.requests_made_today = 0
            
            results = service.search_by_object("M31")
            
            # Check that the query includes object search
            call_args = mock_session.return_value.get.call_args
            assert 'object:"M31"' in call_args[1]['params']['q']


class TestArXivSearchService:
    """Test the arXiv search service."""
    
    def test_arxiv_service_initialization(self):
        """Test arXiv service initializes correctly."""
        service = ArXivSearchService()
        assert service.base_url == "http://export.arxiv.org/api/query"
        assert service.min_request_interval > 0
    
    def test_arxiv_rate_limiting(self):
        """Test arXiv rate limiting."""
        service = ArXivSearchService()
        
        # Should not raise exception for rate limiting
        service._rate_limit()
        
        # Should work multiple times
        service._rate_limit()
    
    @patch('astroagent.services.search_ads.requests.get')
    def test_arxiv_search(self, mock_get):
        """Test arXiv search functionality."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<feed></feed>"  # Mock XML response
        
        mock_get.return_value = mock_response
        
        service = ArXivSearchService()
        results = service.search("dark matter", max_results=10)
        
        # Currently returns empty list (XML parsing not implemented)
        # This tests that the method runs without error
        assert isinstance(results, list)


class TestLiteratureService:
    """Test the literature analysis service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="lit_test_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_literature_service_initialization(self):
        """Test literature service initializes correctly."""
        service = LiteratureService(cache_dir=str(self.temp_dir))
        assert service.cache_dir == self.temp_dir
        assert service.cache_dir.exists()
    
    @patch('astroagent.services.literature.SentenceTransformer')
    def test_get_text_embedding(self, mock_transformer):
        """Test text embedding generation."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_transformer.return_value = mock_model
        
        service = LiteratureService(cache_dir=str(self.temp_dir))
        
        # Force the service to use our mock by accessing the property
        _ = service.embedding_model  # This triggers the lazy loading with our mock
        
        embedding = service.get_text_embedding("test text")
        
        assert len(embedding) == 3  # Our mock returns 3 elements 
        assert embedding[0] == 0.1
    
    @patch('astroagent.services.literature.SentenceTransformer')
    def test_compute_similarity(self, mock_transformer):
        """Test similarity computation."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.side_effect = [
            [1.0, 0.0, 0.0],  # First text
            [0.0, 1.0, 0.0]   # Second text  
        ]
        mock_transformer.return_value = mock_model
        
        service = LiteratureService(cache_dir=str(self.temp_dir))
        
        similarity = service.compute_similarity("text 1", "text 2")
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    @patch('astroagent.services.literature.SentenceTransformer')
    def test_assess_novelty(self, mock_transformer):
        """Test novelty assessment."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.side_effect = [
            [1.0, 0.0, 0.0],  # Hypothesis
            [0.5, 0.5, 0.0],  # Paper 1
            [0.0, 0.0, 1.0]   # Paper 2
        ]
        mock_transformer.return_value = mock_model
        
        service = LiteratureService(cache_dir=str(self.temp_dir))
        
        hypothesis = "Novel research hypothesis"
        existing_papers = [
            {
                'title': 'Related Paper 1',
                'abstract': 'Some related work',
                'bibcode': '2023ApJ...900...1A',
                'year': 2023
            },
            {
                'title': 'Different Paper 2', 
                'abstract': 'Unrelated work',
                'bibcode': '2023ApJ...900...2B',
                'year': 2023
            }
        ]
        
        assessment = service.assess_novelty(hypothesis, existing_papers)
        
        assert 'novelty_score' in assessment
        assert 'novelty_rating' in assessment
        assert 'similar_papers' in assessment
        assert 1 <= assessment['novelty_rating'] <= 5
        assert 0.0 <= assessment['novelty_score'] <= 1.0
    
    @patch('astroagent.services.literature.TfidfVectorizer')
    def test_extract_trending_topics(self, mock_vectorizer):
        """Test trending topic extraction."""
        # Mock TF-IDF vectorizer
        mock_tfidf = Mock()
        mock_tfidf.fit_transform.return_value.toarray.return_value = [[0.5, 0.3, 0.8]]
        mock_tfidf.get_feature_names_out.return_value = ['dark', 'matter', 'galaxy']
        mock_vectorizer.return_value = mock_tfidf
        
        service = LiteratureService(cache_dir=str(self.temp_dir))
        
        papers = [
            {
                'title': 'Dark Matter in Galaxies',
                'abstract': 'Study of dark matter distribution in galaxy clusters'
            }
        ]
        
        topics = service.extract_trending_topics(papers, top_n=3)
        
        assert len(topics) <= 3
        for topic in topics:
            assert 'term' in topic
            assert 'score' in topic
            assert 'type' in topic
    
    @patch('astroagent.services.literature.SentenceTransformer')
    def test_deduplicate_papers(self, mock_transformer):
        """Test paper deduplication."""
        # Mock embeddings for duplicate papers
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # Paper 1
            np.array([1.0, 0.0, 0.0]),  # Paper 2 (duplicate)
            np.array([0.0, 1.0, 0.0])   # Paper 3 (different)
        ]
        mock_transformer.return_value = mock_model
        
        service = LiteratureService(cache_dir=str(self.temp_dir))
        
        # Force the service to use our mock
        _ = service.embedding_model
        
        papers = [
            {
                'title': 'Paper 1',
                'abstract': 'Abstract 1',
                'citation_count': 10,
                'year': 2023
            },
            {
                'title': 'Paper 1 Duplicate',
                'abstract': 'Abstract 1',
                'citation_count': 5,
                'year': 2023
            },
            {
                'title': 'Paper 3',
                'abstract': 'Different abstract',
                'citation_count': 8,
                'year': 2023
            }
        ]
        
        deduplicated = service.deduplicate_papers(papers, similarity_threshold=0.9)
        
        # Should remove one duplicate
        assert len(deduplicated) == 2
        
        # Should keep the paper with higher citations
        titles = [p['title'] for p in deduplicated]
        assert 'Paper 1' in titles
        assert 'Paper 3' in titles
    
    def test_analyze_citation_network(self):
        """Test citation network analysis."""
        service = LiteratureService(cache_dir=str(self.temp_dir))
        
        papers = [
            {
                'bibcode': '2023ApJ...900...1A',
                'title': 'High Impact Paper',
                'year': 2023,
                'citation_count': 100,
                'reference': ['2022ApJ...800...1A'],
                'citation': ['2024ApJ...950...1A']
            },
            {
                'bibcode': '2022ApJ...800...1A',
                'title': 'Earlier Paper',
                'year': 2022,
                'citation_count': 50
            }
        ]
        
        analysis = service.analyze_citation_network(papers)
        
        assert 'total_papers' in analysis
        assert 'total_citations' in analysis
        assert 'highly_cited' in analysis
        assert 'influential' in analysis
        assert analysis['total_papers'] == 2
        assert analysis['total_citations'] == 150
    
    def test_embedding_cache(self):
        """Test embedding caching functionality."""
        service = LiteratureService(cache_dir=str(self.temp_dir))
        
        # Test hash generation
        text = "test text"
        hash1 = service._get_text_hash(text)
        hash2 = service._get_text_hash(text)
        
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
        # Test different texts have different hashes
        hash3 = service._get_text_hash("different text")
        assert hash1 != hash3
