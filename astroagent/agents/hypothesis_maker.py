"""
Hypothesis Maker (HM) Agent for AstroAgent Pipeline.

This agent generates novel, testable astrophysics hypotheses based on current
literature and research domain specifications.
"""

import json
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from .common import BaseAgent, AgentExecutionContext, AgentResult, IdeaSchema, generate_ulid
from ..services.search_ads import ADSSearchService
from ..services.literature import LiteratureService


class HypothesisMaker(BaseAgent):
    """Agent responsible for generating novel research hypotheses."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Initialize services
        self.ads_service = ADSSearchService()
        self.literature_service = LiteratureService()
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        self.guardrails = config.get('guardrails', {})
        
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Generate novel hypotheses based on domain tags and research context."""
        
        try:
            # Extract input parameters
            domain_tags = context.input_data.get('domain_tags', [])
            n_hypotheses = context.input_data.get('n_hypotheses', 5)
            recency_years = context.input_data.get('recency_years', 3)
            
            self.logger.info(f"Generating {n_hypotheses} hypotheses for domains: {domain_tags}")
            
            # Gather recent literature context
            literature_context = self._gather_literature_context(domain_tags, recency_years)
            
            # Generate hypotheses using LLM
            hypotheses = self._generate_hypotheses(
                domain_tags=domain_tags,
                n_hypotheses=n_hypotheses,
                literature_context=literature_context,
                context=context
            )
            
            # Validate and clean hypotheses
            validated_hypotheses = []
            for hypothesis in hypotheses:
                if self._validate_hypothesis(hypothesis):
                    validated_hypotheses.append(hypothesis)
                else:
                    self.logger.warning(f"Hypothesis validation failed: {hypothesis.get('title', 'Unknown')}")
            
            # Prepare registry updates
            registry_updates = []
            for hypothesis in validated_hypotheses:
                registry_updates.append({
                    'registry': 'ideas_register',
                    'action': 'append',
                    'data': hypothesis
                })
            
            return AgentResult(
                success=True,
                agent_name=self.name,
                execution_id=context.execution_id,
                output_data={
                    'hypotheses': validated_hypotheses,
                    'count': len(validated_hypotheses)
                },
                registry_updates=registry_updates,
                execution_time_seconds=0,  # Will be filled by base class
                input_hash="",  # Will be filled by base class
                output_hash=""  # Will be filled by base class
            )
            
        except Exception as e:
            self.logger.error(f"Hypothesis generation failed: {str(e)}")
            return AgentResult(
                success=False,
                agent_name=self.name,
                execution_id=context.execution_id,
                error_message=str(e),
                error_type=type(e).__name__,
                execution_time_seconds=0,
                input_hash="",
                output_hash=""
            )
    
    def _gather_literature_context(self, domain_tags: List[str], recency_years: int) -> Dict[str, Any]:
        """Gather recent literature for context."""
        self.logger.info(f"Gathering literature context for last {recency_years} years")
        
        context = {
            'recent_papers': [],
            'trending_topics': [],
            'available_datasets': [],
            'methodological_advances': []
        }
        
        try:
            # Search for recent papers in domain
            for tag in domain_tags:
                papers = self.ads_service.search_recent(
                    query=tag,
                    years=recency_years,
                    max_results=20
                )
                context['recent_papers'].extend(papers)
            
            # Analyze trending topics
            context['trending_topics'] = self.literature_service.extract_trending_topics(
                context['recent_papers']
            )
            
            # Identify available datasets
            context['available_datasets'] = self._identify_available_datasets(domain_tags)
            
        except Exception as e:
            self.logger.warning(f"Literature context gathering failed: {str(e)}")
        
        return context
    
    def _identify_available_datasets(self, domain_tags: List[str]) -> List[str]:
        """Identify datasets relevant to research domains."""
        # Map domain tags to common datasets
        dataset_mapping = {
            'stellar dynamics': ['Gaia DR3', 'SDSS', 'APOGEE'],
            'galactic structure': ['Gaia DR3', '2MASS', 'WISE'],
            'stellar evolution': ['Kepler', 'TESS', 'MIST models'],
            'exoplanets': ['TESS', 'Kepler', 'K2', 'RV surveys'],
            'cosmology': ['Planck', 'SDSS', 'DES', 'HSC'],
            'high energy': ['Fermi', 'Chandra', 'XMM-Newton'],
            'solar system': ['HORIZONS', 'MPC', 'SPICE kernels'],
            'time domain': ['ZTF', 'ASAS-SN', 'CRTS'],
            'variable stars': ['TESS', 'Kepler', 'OGLE', 'Gaia'],
            'galaxy evolution': ['SDSS', 'CANDELS', 'COSMOS']
        }
        
        datasets = set()
        for tag in domain_tags:
            tag_lower = tag.lower()
            for domain, domain_datasets in dataset_mapping.items():
                if any(word in tag_lower for word in domain.split()):
                    datasets.update(domain_datasets)
        
        return list(datasets)
    
    def _generate_hypotheses(self, domain_tags: List[str], n_hypotheses: int, 
                           literature_context: Dict[str, Any], 
                           context: AgentExecutionContext) -> List[Dict[str, Any]]:
        """Generate hypotheses using LLM."""
        
        # Build prompt from configuration
        system_prompt = self.config.get('system_prompt', '')
        user_prompt_template = self.config.get('user_prompt_template', '')
        
        # Fill template with context
        user_prompt = user_prompt_template.format(
            n_hypotheses=n_hypotheses,
            domain_tags=', '.join(domain_tags),
            recency_years=3,
            available_surveys=', '.join(literature_context.get('available_datasets', [])),
            min_effort_days=1,
            max_effort_days=14
        )
        
        # Add literature context to prompt
        if literature_context['recent_papers']:
            recent_titles = [paper.get('title', '') for paper in literature_context['recent_papers'][:10]]
            user_prompt += f"\n\nRecent relevant papers:\n" + "\n".join(f"- {title}" for title in recent_titles)
        
        # TODO: Integrate with actual LLM service
        # For now, return mock data structure
        self.logger.info("LLM integration not implemented, returning mock hypotheses")
        
        hypotheses = []
        for i in range(min(n_hypotheses, 3)):  # Generate up to 3 mock hypotheses
            hypothesis = {
                'idea_id': generate_ulid(),
                'title': f"Mock Hypothesis {i+1} for {domain_tags[0] if domain_tags else 'General'}",
                'hypothesis': (
                    f"We hypothesize that there exists a previously unrecognized correlation "
                    f"between stellar {domain_tags[0] if domain_tags else 'properties'} and "
                    f"environmental factors that can be detected through systematic analysis "
                    f"of multi-wavelength observations."
                ),
                'rationale': (
                    f"Recent advances in {domain_tags[0] if domain_tags else 'astrophysics'} suggest "
                    f"that traditional models may be incomplete. By leveraging large-scale survey "
                    f"data and modern statistical techniques, we can probe previously inaccessible "
                    f"parameter spaces and potentially discover new physical relationships."
                ),
                'domain_tags': domain_tags,
                'novelty_refs': ['2023ApJ...900...1A', '2023MNRAS.520.1234B', '2022A&A...650.L5C'],
                'required_data': literature_context.get('available_datasets', ['Gaia DR3', 'SDSS']),
                'methods': ['Statistical correlation analysis', 'Machine learning classification', 'Monte Carlo simulation'],
                'est_effort_days': 7 + i * 2
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _validate_hypothesis(self, hypothesis: Dict[str, Any]) -> bool:
        """Validate hypothesis against guardrails."""
        try:
            # Check required fields
            required_fields = self.guardrails.get('required_fields', [])
            for field in required_fields:
                if field not in hypothesis or not hypothesis[field]:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check word counts
            hypothesis_text = hypothesis.get('hypothesis', '')
            rationale_text = hypothesis.get('rationale', '')
            
            hypothesis_words = len(hypothesis_text.split())
            rationale_words = len(rationale_text.split())
            
            min_hyp_words = self.guardrails.get('min_hypothesis_words', 0)
            max_hyp_words = self.guardrails.get('max_hypothesis_words', 1000)
            min_rat_words = self.guardrails.get('min_rationale_words', 0)
            max_rat_words = self.guardrails.get('max_rationale_words', 1000)
            
            if not (min_hyp_words <= hypothesis_words <= max_hyp_words):
                self.logger.warning(f"Hypothesis word count ({hypothesis_words}) outside range [{min_hyp_words}, {max_hyp_words}]")
                return False
                
            if not (min_rat_words <= rationale_words <= max_rat_words):
                self.logger.warning(f"Rationale word count ({rationale_words}) outside range [{min_rat_words}, {max_rat_words}]")
                return False
            
            # Check for forbidden phrases
            forbidden_phrases = self.guardrails.get('forbidden_phrases', [])
            full_text = (hypothesis_text + ' ' + rationale_text).lower()
            
            for phrase in forbidden_phrases:
                if phrase.lower() in full_text:
                    self.logger.warning(f"Forbidden phrase detected: {phrase}")
                    return False
            
            # Validate using Pydantic model
            IdeaSchema(**hypothesis)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Hypothesis validation failed: {str(e)}")
            return False
    
    def validate_input(self, context: AgentExecutionContext) -> bool:
        """Validate input requirements for hypothesis generation."""
        required_fields = ['domain_tags']
        
        for field in required_fields:
            if field not in context.input_data:
                self.logger.error(f"Missing required input field: {field}")
                return False
        
        domain_tags = context.input_data['domain_tags']
        if not isinstance(domain_tags, list) or not domain_tags:
            self.logger.error("domain_tags must be a non-empty list")
            return False
        
        n_hypotheses = context.input_data.get('n_hypotheses', 5)
        if not isinstance(n_hypotheses, int) or n_hypotheses <= 0 or n_hypotheses > 20:
            self.logger.error("n_hypotheses must be an integer between 1 and 20")
            return False
        
        return True
    
    def validate_output(self, result: AgentResult) -> bool:
        """Validate output hypotheses."""
        if not result.success:
            return False
        
        hypotheses = result.output_data.get('hypotheses', [])
        if not isinstance(hypotheses, list):
            self.logger.error("Output hypotheses must be a list")
            return False
        
        if len(hypotheses) == 0:
            self.logger.error("No valid hypotheses generated")
            return False
        
        # Validate each hypothesis
        for hypothesis in hypotheses:
            if not self._validate_hypothesis(hypothesis):
                return False
        
        return True
