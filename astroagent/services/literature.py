"""
Literature Service for AstroAgent Pipeline.

Provides literature analysis, novelty checking, trend detection,
and semantic similarity computations.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class LiteratureService:
    """Service for literature analysis and novelty assessment."""
    
    def __init__(self, cache_dir: str = "data/vectors", 
                 logger: Optional[logging.Logger] = None):
        """Initialize literature service.
        
        Args:
            cache_dir: Directory for caching embeddings and analysis results
            logger: Optional logger instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or self._setup_logger()
        
        # Initialize embedding model (lazy loading)
        self._embedding_model = None
        self._tfidf_vectorizer = None
        
        # Cache for paper embeddings
        self.embedding_cache = {}
        self.cache_file = self.cache_dir / "paper_embeddings.json"
        self._load_cache()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the service."""
        logger = logging.getLogger('astroagent.services.literature')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                self.logger.info("Loading sentence embedding model...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {str(e)}")
                raise
        return self._embedding_model
    
    @property  
    def tfidf_vectorizer(self):
        """Lazy load TF-IDF vectorizer."""
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        return self._tfidf_vectorizer
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding cache: {str(e)}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.embedding_cache, f)
        except Exception as e:
            self.logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def _get_text_hash(self, text: str) -> str:
        """Get hash of text for caching."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        text_hash = self._get_text_hash(text)
        
        # Check cache first
        if text_hash in self.embedding_cache:
            return np.array(self.embedding_cache[text_hash])
        
        # Compute embedding
        try:
            embedding = self.embedding_model.encode(text)
            
            # Cache result
            self.embedding_cache[text_hash] = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to compute embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            emb1 = self.get_text_embedding(text1)
            emb2 = self.get_text_embedding(text2)
            
            # Compute cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0, 0]
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {str(e)}")
            return 0.0
    
    def assess_novelty(self, hypothesis: str, existing_papers: List[Dict[str, Any]], 
                      threshold: float = 0.7) -> Dict[str, Any]:
        """Assess novelty of a hypothesis against existing literature.
        
        Args:
            hypothesis: Hypothesis text to evaluate
            existing_papers: List of papers with 'abstract' and/or 'title' fields
            threshold: Similarity threshold above which to flag overlap
            
        Returns:
            Novelty assessment results
        """
        self.logger.info(f"Assessing novelty for hypothesis against {len(existing_papers)} papers")
        
        novelty_scores = []
        similar_papers = []
        
        for paper in existing_papers:
            # Combine title and abstract for comparison
            paper_text = ""
            if 'title' in paper:
                title = paper['title']
                if isinstance(title, list):
                    title = ' '.join(title) if title else ''
                paper_text += str(title) + " "
            if 'abstract' in paper and paper['abstract']:
                abstract = paper['abstract']
                if isinstance(abstract, list):
                    abstract = ' '.join(abstract) if abstract else ''
                paper_text += str(abstract)
            
            if not paper_text.strip():
                continue
                
            similarity = self.compute_similarity(hypothesis, paper_text)
            novelty_scores.append(similarity)
            
            if similarity >= threshold:
                similar_papers.append({
                    'bibcode': paper.get('bibcode', 'Unknown'),
                    'title': paper.get('title', 'Unknown'),
                    'similarity': similarity,
                    'year': paper.get('year', 'Unknown')
                })
        
        # Calculate novelty metrics
        if novelty_scores:
            max_similarity = max(novelty_scores)
            mean_similarity = sum(novelty_scores) / len(novelty_scores)
            novelty_score = 1.0 - max_similarity  # Higher novelty = lower max similarity
        else:
            max_similarity = 0.0
            mean_similarity = 0.0
            novelty_score = 1.0
        
        # Convert to 1-5 scale
        novelty_rating = int(novelty_score * 4) + 1  # Maps 0-1 to 1-5
        
        result = {
            'novelty_score': novelty_score,
            'novelty_rating': novelty_rating,
            'max_similarity': max_similarity,
            'mean_similarity': mean_similarity,
            'similar_papers_count': len(similar_papers),
            'similar_papers': similar_papers[:5],  # Top 5 most similar
            'assessment': self._get_novelty_assessment(novelty_rating, len(similar_papers))
        }
        
        self.logger.info(f"Novelty assessment: rating={novelty_rating}, similar_papers={len(similar_papers)}")
        
        return result
    
    def _get_novelty_assessment(self, rating: int, similar_count: int) -> str:
        """Generate textual novelty assessment."""
        
        if rating >= 4 and similar_count == 0:
            return "Highly novel - no similar work found in literature"
        elif rating >= 4 and similar_count <= 2:
            return "Highly novel - minimal overlap with existing work"
        elif rating == 3:
            return "Moderately novel - some related work exists"
        elif rating == 2:
            return "Limited novelty - significant overlap with prior work"
        else:
            return "Low novelty - substantial duplication of existing research"
    
    def extract_trending_topics(self, papers: List[Dict[str, Any]], 
                              top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract trending topics from recent literature.
        
        Args:
            papers: List of papers with abstracts/titles
            top_n: Number of top topics to return
            
        Returns:
            List of trending topics with scores
        """
        
        if not papers:
            return []
        
        # Combine all text
        all_text = []
        for paper in papers:
            text = ""
            if 'title' in paper:
                title = paper['title']
                if isinstance(title, list):
                    title = ' '.join(title) if title else ''
                text += str(title) + " "
            if 'abstract' in paper and paper['abstract']:
                abstract = paper['abstract']
                if isinstance(abstract, list):
                    abstract = ' '.join(abstract) if abstract else ''
                text += str(abstract)
            
            if text.strip():
                all_text.append(text)
        
        if not all_text:
            return []
        
        try:
            # Compute TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_text)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores across all documents
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top terms
            top_indices = np.argsort(mean_scores)[-top_n:][::-1]
            
            trending_topics = []
            for idx in top_indices:
                trending_topics.append({
                    'term': feature_names[idx],
                    'score': float(mean_scores[idx]),
                    'type': self._classify_term(feature_names[idx])
                })
            
            return trending_topics
            
        except Exception as e:
            self.logger.error(f"Trending topic extraction failed: {str(e)}")
            return []
    
    def _classify_term(self, term: str) -> str:
        """Classify a trending term by type."""
        
        # Simple classification based on patterns
        term_lower = term.lower()
        
        # Observational terms
        obs_keywords = ['survey', 'telescope', 'observation', 'data', 'catalog', 'imaging']
        if any(kw in term_lower for kw in obs_keywords):
            return 'observational'
        
        # Theoretical/method terms  
        theory_keywords = ['model', 'simulation', 'theory', 'algorithm', 'method']
        if any(kw in term_lower for kw in theory_keywords):
            return 'theoretical'
        
        # Object types
        object_keywords = ['star', 'galaxy', 'planet', 'black hole', 'nebula', 'cluster']
        if any(kw in term_lower for kw in object_keywords):
            return 'astronomical_object'
        
        # Physical processes
        physics_keywords = ['formation', 'evolution', 'emission', 'absorption', 'magnetic']
        if any(kw in term_lower for kw in physics_keywords):
            return 'physical_process'
        
        return 'general'
    
    def find_research_gaps(self, domain_papers: List[Dict[str, Any]], 
                          method_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential research gaps by comparing domains and methods.
        
        Args:
            domain_papers: Papers in target research domain
            method_papers: Papers describing relevant methods
            
        Returns:
            List of potential research gaps
        """
        
        # Extract topics from each set
        domain_topics = self.extract_trending_topics(domain_papers, top_n=20)
        method_topics = self.extract_trending_topics(method_papers, top_n=20)
        
        # Find methods not commonly used in domain
        domain_terms = set(topic['term'] for topic in domain_topics)
        method_terms = set(topic['term'] for topic in method_topics)
        
        # Potential gaps: methods not appearing in domain
        gap_methods = method_terms - domain_terms
        
        gaps = []
        for method in list(gap_methods)[:10]:  # Top 10 gaps
            # Get method score from method papers
            method_score = next((t['score'] for t in method_topics if t['term'] == method), 0.0)
            
            gaps.append({
                'type': 'methodological_gap',
                'method': method,
                'score': method_score,
                'description': f"Method '{method}' underutilized in domain literature"
            })
        
        return gaps
    
    def deduplicate_papers(self, papers: List[Dict[str, Any]], 
                          similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on content similarity.
        
        Args:
            papers: List of papers to deduplicate
            similarity_threshold: Threshold above which papers are considered duplicates
            
        Returns:
            Deduplicated list of papers
        """
        
        if len(papers) <= 1:
            return papers
        
        self.logger.info(f"Deduplicating {len(papers)} papers")
        
        # Compute embeddings for all papers
        embeddings = []
        texts = []
        
        for paper in papers:
            text = ""
            if 'title' in paper:
                text += paper['title'] + " "
            if 'abstract' in paper and paper['abstract']:
                text += paper['abstract']
            
            texts.append(text)
            embeddings.append(self.get_text_embedding(text))
        
        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find duplicates
        to_remove = set()
        for i in range(len(papers)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(papers)):
                if j in to_remove:
                    continue
                    
                if similarity_matrix[i, j] >= similarity_threshold:
                    # Keep the one with more citations or more recent
                    paper_i = papers[i]
                    paper_j = papers[j]
                    
                    # Prefer paper with more citations
                    citations_i = paper_i.get('citation_count', 0) or 0
                    citations_j = paper_j.get('citation_count', 0) or 0
                    
                    if citations_j > citations_i:
                        to_remove.add(i)
                    elif citations_i > citations_j:
                        to_remove.add(j)
                    else:
                        # If equal citations, prefer more recent
                        year_i = paper_i.get('year', 0) or 0
                        year_j = paper_j.get('year', 0) or 0
                        
                        if year_j > year_i:
                            to_remove.add(i)
                        else:
                            to_remove.add(j)
        
        # Remove duplicates
        deduplicated = [papers[i] for i in range(len(papers)) if i not in to_remove]
        
        self.logger.info(f"Removed {len(to_remove)} duplicates, {len(deduplicated)} papers remaining")
        
        return deduplicated
    
    def analyze_citation_network(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze citation patterns in a set of papers.
        
        Args:
            papers: List of papers with citation information
            
        Returns:
            Citation network analysis results
        """
        
        # Extract citation data
        all_bibcodes = set()
        citation_graph = {}
        
        for paper in papers:
            bibcode = paper.get('bibcode')
            if not bibcode:
                continue
                
            all_bibcodes.add(bibcode)
            citation_graph[bibcode] = {
                'title': paper.get('title', ''),
                'year': paper.get('year', 0),
                'citation_count': paper.get('citation_count', 0),
                'references': paper.get('reference', []),
                'citations': paper.get('citation', [])
            }
        
        # Find highly cited papers
        highly_cited = sorted(
            [(bc, data['citation_count']) for bc, data in citation_graph.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        # Find influential papers (high citation rate relative to age)
        current_year = 2024
        influential = []
        
        for bibcode, data in citation_graph.items():
            year = data.get('year', current_year)
            age = max(1, current_year - year)
            citation_rate = data.get('citation_count', 0) / age
            
            influential.append((bibcode, citation_rate, data['title']))
        
        influential.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_papers': len(papers),
            'total_citations': sum(data.get('citation_count', 0) for data in citation_graph.values()),
            'highly_cited': [{'bibcode': bc, 'citations': ct} for bc, ct in highly_cited],
            'influential': [{'bibcode': bc, 'rate': rate, 'title': title} 
                          for bc, rate, title in influential[:10]]
        }
    
    def __del__(self):
        """Cleanup - save cache on destruction."""
        try:
            self._save_cache()
        except:
            pass
