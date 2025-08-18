"""
Hypothesis Maker (HM) Agent for AstroAgent Pipeline.

This agent generates novel, testable astrophysics hypotheses based on current
literature and research domain specifications.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timezone

from .common import BaseAgent, AgentExecutionContext, AgentResult, IdeaSchema, generate_ulid
from ..services.search_ads import ADSSearchService
from ..services.literature import LiteratureService


class HypothesisMaker(BaseAgent):
    """Agent responsible for generating novel research hypotheses."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Hypothesis Maker agent."""
        super().__init__(config)
        
        # Initialize services
        self.ads_service = ADSSearchService()
        self.literature_service = LiteratureService()
        
        # Lazy import to avoid circular dependency
        from ..orchestration.tools import RegistryManager
        self.registry_manager = RegistryManager(config.get('data_dir', 'data'), self.logger)
        
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Execute hypothesis generation and revision."""
        self.logger.info(f"Starting execution {context.execution_id}")
        
        try:
            # First, check for ideas that need revision
            revised_hypotheses = self._process_revision_ideas(context)
            
            # Then generate new hypotheses if requested
            new_hypotheses = []
            if context.input_data.get('generate_new', True):
                domain_tags = context.input_data.get('domain_tags', ['stellar evolution'])
                n_hypotheses = context.input_data.get('n_hypotheses', 3)
                
                self.logger.info(f"Generating {n_hypotheses} ambitious new hypotheses for domains: {domain_tags}")
                
                # Generate hypotheses WITHOUT literature constraints - be bold and creative!
                new_hypotheses = self._generate_ambitious_hypotheses(domain_tags, n_hypotheses, context)
            
            all_hypotheses = revised_hypotheses + new_hypotheses
            
            self.logger.info(f"Processed {len(revised_hypotheses)} revised ideas and generated {len(new_hypotheses)} new hypotheses")
            
            return AgentResult(
                success=True,
                agent_name=self.name,
                execution_id=context.execution_id,
                output_data={
                    'hypotheses': all_hypotheses,
                    'revised_count': len(revised_hypotheses),
                    'new_count': len(new_hypotheses)
                },
                execution_time_seconds=0,
                input_hash="",
                output_hash=""
            )
            
        except Exception as e:
            self.logger.error(f"Execution {context.execution_id} failed: {e}")
            return AgentResult(
                success=False,
                agent_name=self.name,
                execution_id=context.execution_id,
                error_message=str(e),
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
    
    def _process_revision_ideas(self, context: AgentExecutionContext) -> List[Dict[str, Any]]:
        """Process ideas that need revision and create improved versions."""
        
        try:
            self.logger.info("Checking for ideas that need revision")
            
            # Load ideas from registry
            ideas_df = self.registry_manager.load_registry('ideas_register')
            
            if ideas_df.empty:
                return []
            
            # Find ideas that need revision
            revision_ideas = ideas_df[ideas_df['status'] == 'Needs Revision']
            
            if revision_ideas.empty:
                self.logger.info("No ideas need revision")
                return []
            
            self.logger.info(f"Found {len(revision_ideas)} ideas needing revision")
            
            revised_hypotheses = []
            
            for _, idea_row in revision_ideas.iterrows():
                try:
                    idea = idea_row.to_dict()
                    
                    # Parse reviewer feedback to understand what needs improvement
                    reviewer_notes = idea.get('reviewer_notes', '')
                    
                    # Create revised version
                    revised_idea = self._revise_idea_based_on_feedback(idea, reviewer_notes)
                    
                    if revised_idea:
                        revised_hypotheses.append(revised_idea)
                        
                        # Update original idea status to indicate it's been revised
                        self.registry_manager.update_registry_row(
                            'ideas_register',
                            {'idea_id': idea['idea_id']},
                            {'status': 'Under Revision', 'updated_at': datetime.now(timezone.utc).isoformat()}
                        )
                        
                except Exception as e:
                    self.logger.error(f"Failed to revise idea {idea_row.get('idea_id', 'unknown')}: {str(e)}")
            
            return revised_hypotheses
            
        except Exception as e:
            self.logger.error(f"Failed to process revision ideas: {str(e)}")
            return []
    
    def _revise_idea_based_on_feedback(self, original_idea: Dict[str, Any], reviewer_notes: str) -> Optional[Dict[str, Any]]:
        """Create a revised version of an idea based on reviewer feedback."""
        
        self.logger.info(f"Revising idea: {original_idea.get('title', 'Unknown')}")
        
        # Parse list fields
        for field in ['domain_tags', 'required_data', 'methods', 'literature_refs']:
            if field in original_idea and isinstance(original_idea[field], str):
                try:
                    if original_idea[field] in ['[]', '']:
                        original_idea[field] = []
                    else:
                        import ast
                        original_idea[field] = ast.literal_eval(original_idea[field])
                except (ValueError, SyntaxError):
                    if original_idea[field].strip():
                        original_idea[field] = [original_idea[field]]
                    else:
                        original_idea[field] = []
        
        # Analyze feedback to determine improvements needed
        improvements_needed = self._analyze_reviewer_feedback(reviewer_notes)
        
        # Create revised hypothesis
        revised_idea = {
            'idea_id': generate_ulid(),  # New ID for revised version
            'parent_idea_id': original_idea['idea_id'],  # Link to original
            'version': 'v2',  # Increment version
            'title': self._improve_title_if_needed(original_idea.get('title', ''), improvements_needed),
            'hypothesis': self._improve_hypothesis_if_needed(original_idea.get('hypothesis', ''), improvements_needed),
            'rationale': self._improve_rationale_if_needed(original_idea.get('rationale', ''), improvements_needed),
            'domain_tags': original_idea.get('domain_tags', []),
            'novelty_refs': [],
            'required_data': self._improve_data_requirements_if_needed(original_idea.get('required_data', []), improvements_needed),
            'methods': self._improve_methods_if_needed(original_idea.get('methods', []), improvements_needed),
            'est_effort_days': original_idea.get('est_effort_days', 7),
            'status': 'Proposed',  # Reset status for re-review
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'confidence_score': 0.8,  # Higher confidence for revised ideas
            'literature_refs': []
        }
        
        return revised_idea
    
    def _analyze_reviewer_feedback(self, reviewer_notes: str) -> Dict[str, bool]:
        """Analyze reviewer feedback to determine what improvements are needed."""
        
        improvements = {
            'testability': False,
            'impact': False,
            'feasibility': False,
            'novelty': False,
            'specificity': False
        }
        
        notes_lower = reviewer_notes.lower()
        
        if 'testability' in notes_lower or 'measurable' in notes_lower or 'predictions' in notes_lower:
            improvements['testability'] = True
        
        if 'impact' in notes_lower or 'broader implications' in notes_lower:
            improvements['impact'] = True
            
        if 'feasibility' in notes_lower or 'data access' in notes_lower or 'methodological' in notes_lower:
            improvements['feasibility'] = True
            
        if 'novelty' in notes_lower or 'overlap' in notes_lower or 'literature' in notes_lower:
            improvements['novelty'] = True
            
        if 'specific' in notes_lower or 'vague' in notes_lower or 'clarify' in notes_lower:
            improvements['specificity'] = True
        
        return improvements
    
    def _improve_title_if_needed(self, title: str, improvements: Dict[str, bool]) -> str:
        """Improve title based on feedback."""
        if improvements.get('specificity', False) and title:
            # Make title more specific
            if 'correlation' in title.lower():
                return title.replace('Correlation', 'Quantitative Analysis of the Correlation')
            elif 'role' in title.lower():
                return title.replace('Role', 'Detailed Investigation of the Role')
            elif 'impact' in title.lower():
                return title.replace('Impact', 'Observational Constraints on the Impact')
        
        return title
    
    def _improve_hypothesis_if_needed(self, hypothesis: str, improvements: Dict[str, bool]) -> str:
        """Improve hypothesis based on feedback."""
        if not hypothesis:
            return hypothesis
        
        improved = hypothesis
        
        if improvements.get('testability', False):
            # Add more specific, measurable predictions
            if 'correlation' in hypothesis.lower() and 'significance' not in hypothesis.lower():
                improved += " We predict this correlation will show statistical significance at p < 0.01 level with effect size > 0.3."
            
            if 'higher' in hypothesis.lower() and 'factor' not in hypothesis.lower():
                improved = improved.replace('higher', 'higher by a factor of 1.5-2.0')
        
        if improvements.get('specificity', False):
            # Add more specific conditions and scope
            if 'galaxies' in hypothesis.lower() and 'stellar mass' not in hypothesis.lower():
                improved += " This effect should be most pronounced for galaxies with stellar masses between 10^9 and 10^11 solar masses."
            
            if 'stellar' in hypothesis.lower() and 'metallicity' not in hypothesis.lower():
                improved += " The effect should vary systematically with stellar metallicity (Z) in the range 0.1-2.0 Z_solar."
        
        if improvements.get('impact', False):
            # Add broader implications
            improved += " This relationship would provide new constraints on current models of stellar evolution and galaxy formation, potentially resolving discrepancies in observed vs. predicted scaling relations."
        
        return improved
    
    def _improve_rationale_if_needed(self, rationale: str, improvements: Dict[str, bool]) -> str:
        """Improve rationale based on feedback."""
        if not rationale:
            return rationale
        
        improved = rationale
        
        if improvements.get('novelty', False):
            # Add more literature context
            improved += " While previous studies have examined individual components of this relationship, no systematic investigation has been conducted across the full parameter space using modern, large-scale datasets."
        
        if improvements.get('feasibility', False):
            # Add more detail about data availability and methods
            improved += " Recent data releases from major surveys (Gaia DR3, SDSS-V, JWST) now provide the unprecedented precision and sample sizes needed to detect these subtle effects."
        
        return improved
    
    def _improve_data_requirements_if_needed(self, data_reqs: List[str], improvements: Dict[str, bool]) -> List[str]:
        """Improve data requirements based on feedback."""
        improved = list(data_reqs)  # Copy original list
        
        if improvements.get('feasibility', False):
            # Add more accessible datasets
            if 'JWST' in improved and 'Gaia DR3' not in improved:
                improved.append('Gaia DR3')
            if 'Hubble' in improved and 'SDSS' not in improved:
                improved.append('SDSS')
        
        return improved
    
    def _improve_methods_if_needed(self, methods: List[str], improvements: Dict[str, bool]) -> List[str]:
        """Improve methods based on feedback."""
        improved = list(methods)  # Copy original list
        
        if improvements.get('testability', False):
            # Add more rigorous statistical methods
            if 'Statistical analysis' in improved:
                improved.remove('Statistical analysis')
                improved.extend(['Bayesian hierarchical modeling', 'Monte Carlo uncertainty propagation', 'Robust regression analysis'])
        
        if improvements.get('feasibility', False):
            # Add more standard, well-established methods
            if 'Machine learning' in improved and 'Cross-validation' not in str(improved):
                improved.append('K-fold cross-validation')
        
        return improved
    
    def _identify_available_datasets(self, domain_tags: List[str]) -> List[str]:
        """Identify relevant astronomical datasets."""
        dataset_mapping = {
            'stellar evolution': ['Gaia DR3', 'APOGEE', 'GALAH', 'Kepler'],
            'galactic dynamics': ['Gaia DR3', 'SDSS', 'Dark Energy Survey'],
            'exoplanet': ['TESS', 'Kepler', 'JWST', 'Hubble'],
            'cosmology': ['Planck', 'Dark Energy Survey', 'SDSS'],
            'galaxy formation': ['HST', 'JWST', 'SDSS', 'CANDELS']
        }
        
        datasets = set()
        for tag in domain_tags:
            tag_lower = tag.lower()
            for domain, domain_datasets in dataset_mapping.items():
                if any(word in tag_lower for word in domain.split()):
                    datasets.update(domain_datasets)
        
        return list(datasets)
    
    def _generate_ambitious_hypotheses(self, domain_tags: List[str], n_hypotheses: int, 
                                     context: AgentExecutionContext) -> List[Dict[str, Any]]:
        """Generate ambitious, creative hypotheses WITHOUT literature constraints."""
        
        # Check if we have real API keys, use LLM if available
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and not openai_key.startswith(('demo', 'mock', 'test')):
            try:
                return self._generate_ambitious_hypotheses_with_llm(domain_tags, n_hypotheses)
            except Exception as e:
                self.logger.error(f"LLM generation failed: {e}")
                raise
        
        # No LLM available
        raise ValueError("OpenAI API key required for hypothesis generation")

    def _generate_ambitious_hypotheses_with_llm(self, domain_tags: List[str], n_hypotheses: int) -> List[Dict[str, Any]]:
        """Generate ambitious, creative hypotheses using OpenAI API - NO literature constraints."""
        import openai
        
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        self.logger.info(f"Generating {n_hypotheses} ambitious hypotheses using OpenAI - unconstrained by recent literature")
        
        # Create ambitious but tractable prompt
        prompt = f"""You are an expert astrophysicist tasked with generating {n_hypotheses} AMBITIOUS but TRACTABLE hypotheses about {', '.join(domain_tags)}.

APPROACH:
- Think BIG but stay REALISTIC - hypotheses should be testable with available data
- Generate novel connections between observable phenomena  
- Focus on unexplored relationships that can be measured with current/near-future instruments
- Consider creative uses of existing datasets (Gaia, JWST, Hubble, TESS, etc.)
- Aim for significant discoveries that are achievable within ~1-2 years
- Avoid purely theoretical concepts without observational signatures

CRITICAL: You MUST follow this EXACT format with EXACT word counts:

===HYPOTHESIS_START===
TITLE: [Bold, ambitious scientific title]
HYPOTHESIS: [Write a falsifiable statement that is EXACTLY 50-150 words. Make a specific, measurable prediction about observable phenomena that would be significant if true. Think beyond current models - what surprising relationships or mechanisms might exist? Write at least 50 words but no more than 150 words.]
RATIONALE: [Write a compelling scientific justification that is EXACTLY 100-300 words. Explain why this hypothesis is worth pursuing, what observational approach could test it, and why it represents an important advance. Focus on the potential impact and novel insights, not just recent literature. Write at least 100 words but no more than 300 words.]
===HYPOTHESIS_END===

EXAMPLE OF AMBITIOUS BUT TRACTABLE THINKING:
===HYPOTHESIS_START===
TITLE: Stellar Metallicity Gradients Reveal Hidden Galactic Assembly History Through Chemical Tagging
HYPOTHESIS: The detailed three-dimensional metallicity distribution of stars measured by spectroscopic surveys contains a previously unrecognized signature of galactic assembly history, where discrete chemical abundance patterns correspond to ancient merger events. Systematic analysis of [α/Fe] ratios and heavy element abundances across different galactic radii and heights will reveal distinct stellar populations that trace the timing and mass ratios of past galactic mergers. This chemical archaeological approach predicts that stellar streams and overdensities identified through proper motions will correlate with specific abundance signatures, providing a new method to reconstruct the hierarchical assembly of the Milky Way.
RATIONALE: Current galactic archaeology relies primarily on stellar kinematics and ages, but the chemical abundance information from massive spectroscopic surveys like APOGEE and Gaia-ESO is underutilized for tracing galactic history. Different stellar formation environments during merger events should leave distinct chemical fingerprints that persist for billions of years. With precise stellar abundances for millions of stars and accurate proper motions from Gaia, we can now perform detailed chemical tagging to identify stellar populations formed during specific merger events. This approach would provide independent validation of cosmological simulations and offer new insights into the formation history of spiral galaxies.
===HYPOTHESIS_END===

NOW generate {n_hypotheses} ambitious but tractable hypothesis/hypotheses following this exact format. Think creatively about observable phenomena and novel analysis approaches!

Available major datasets: Gaia, JWST, ALMA, Hubble, TESS, Euclid, Rubin Observatory, Chandra, VLA, LIGO/Virgo
"""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert astrophysicist generating novel, testable research hypotheses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            self.logger.info(f"LLM response received: {len(response_text)} characters")
            
            # Parse using structured markers
            hypotheses = []
            hypothesis_blocks = re.findall(r'===HYPOTHESIS_START===(.*?)===HYPOTHESIS_END===', response_text, re.DOTALL)
            
            for i, block in enumerate(hypothesis_blocks[:n_hypotheses]):
                # Extract structured components
                title_match = re.search(r'TITLE:\s*([^\n]+)', block)
                hypothesis_match = re.search(r'HYPOTHESIS:\s*(.*?)(?=RATIONALE:)', block, re.DOTALL)
                rationale_match = re.search(r'RATIONALE:\s*(.*?)$', block, re.DOTALL)
                
                if not (title_match and hypothesis_match and rationale_match):
                    self.logger.error(f"Failed to parse hypothesis block {i+1}: missing required sections")
                    continue
                
                title = title_match.group(1).strip()
                hypothesis_text = hypothesis_match.group(1).strip()
                rationale = rationale_match.group(1).strip()
                
                # Validate word counts
                hypothesis_words = len(hypothesis_text.split())
                rationale_words = len(rationale.split())
                
                if hypothesis_words < 50 or hypothesis_words > 150:
                    self.logger.error(f"Hypothesis {i+1} word count ({hypothesis_words}) outside range [50, 150]")
                    continue
                    
                if rationale_words < 80 or rationale_words > 300:
                    self.logger.error(f"Rationale {i+1} word count ({rationale_words}) outside range [80, 300]") 
                    continue
                
                # Enforce character limits
                if len(hypothesis_text) > 350:
                    hypothesis_text = hypothesis_text[:347] + "..."
                if len(rationale) > 500:
                    rationale = rationale[:497] + "..."
                
                # Create properly structured hypothesis
                hypothesis = {
                    'idea_id': generate_ulid(),
                    'title': title[:100],
                    'hypothesis': hypothesis_text,
                    'rationale': rationale,
                    'domain_tags': domain_tags,
                    'status': 'Proposed',
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'effort_estimate_days': 7,
                    'est_effort_days': 7,
                    'confidence_score': 0.75,
                    'novelty_score': 4,
                    'novelty_refs': [],
                    'required_data': ['Multi-wavelength observations', 'Survey data', 'Archival observations'],
                    'methods': ['Statistical analysis', 'Machine learning', 'Observational comparison'],
                    'literature_refs': []
                }
                
                hypotheses.append(hypothesis)
                self.logger.info(f"Successfully parsed hypothesis {i+1}: '{title}' ({hypothesis_words} words, {rationale_words} words)")
            
            if not hypotheses:
                raise ValueError("No valid hypotheses could be parsed from LLM response. Response format was incorrect.")
            
            self.logger.info(f"Successfully generated {len(hypotheses)} valid hypotheses")
            return hypotheses
            
        except Exception as e:
            self.logger.error(f"LLM hypothesis generation failed: {e}")
            raise
    
    def _validate_hypothesis(self, hypothesis: Dict[str, Any]) -> bool:
        """Validate hypothesis against guardrails."""
        try:
            # Check required fields
            required_fields = self.config.get('guardrails', {}).get('required_fields', [])
            for field in required_fields:
                if field not in hypothesis:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check word counts
            hypothesis_text = hypothesis.get('hypothesis', '')
            rationale_text = hypothesis.get('rationale', '')
            
            hypothesis_words = len(hypothesis_text.split())
            rationale_words = len(rationale_text.split())
            
            min_hypothesis_words = self.config.get('guardrails', {}).get('min_hypothesis_words', 50)
            max_hypothesis_words = self.config.get('guardrails', {}).get('max_hypothesis_words', 150)
            min_rationale_words = self.config.get('guardrails', {}).get('min_rationale_words', 100)
            max_rationale_words = self.config.get('guardrails', {}).get('max_rationale_words', 300)
            
            if hypothesis_words < min_hypothesis_words or hypothesis_words > max_hypothesis_words:
                self.logger.warning(f"Hypothesis word count ({hypothesis_words}) outside range [{min_hypothesis_words}, {max_hypothesis_words}]")
                return False
            
            if rationale_words < min_rationale_words or rationale_words > max_rationale_words:
                self.logger.warning(f"Rationale word count ({rationale_words}) outside range [{min_rationale_words}, {max_rationale_words}]")
                return False
            
            # Check forbidden phrases
            forbidden_phrases = self.config.get('guardrails', {}).get('forbidden_phrases', [])
            for phrase in forbidden_phrases:
                if phrase.lower() in hypothesis_text.lower() or phrase.lower() in rationale_text.lower():
                    self.logger.warning(f"Contains forbidden phrase: {phrase}")
                    return False
            
            # Validate against schema if available
            try:
                IdeaSchema(**hypothesis)
            except Exception as e:
                self.logger.warning(f"Hypothesis validation failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False