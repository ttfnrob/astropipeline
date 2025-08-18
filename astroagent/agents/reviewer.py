"""
Reviewer (RV) Agent for AstroAgent Pipeline.

This agent evaluates research hypotheses on multiple dimensions including
impact, feasibility, testability, and novelty.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from .common import BaseAgent, AgentExecutionContext, AgentResult, IdeaSchema
from ..services.literature import LiteratureService
from ..services.search_ads import ADSSearchService


class Reviewer(BaseAgent):
    """Agent responsible for reviewing and scoring research hypotheses."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Initialize services
        self.literature_service = LiteratureService()
        self.ads_service = ADSSearchService()
        
        # Lazy import to avoid circular dependency
        from ..orchestration.tools import RegistryManager
        self.registry_manager = RegistryManager(config.get('data_dir', 'data'), logger)
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.3)
        self.scoring_rubric = config.get('scoring_rubric', {})
        self.approval_thresholds = config.get('approval_thresholds', {})
        
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Review and score hypotheses from the ideas register."""
        
        try:
            # Load ideas to review
            input_filter = context.input_data.get('filter', {'status': ['Proposed', 'Under Review']})
            ideas_to_review = self._load_ideas_for_review(input_filter)
            
            if not ideas_to_review:
                self.logger.info("No ideas found for review")
                return AgentResult(
                    success=True,
                    agent_name=self.name,
                    execution_id=context.execution_id,
                    output_data={'reviewed_count': 0},
                    execution_time_seconds=0,
                    input_hash="",
                    output_hash=""
                )
            
            self.logger.info(f"Reviewing {len(ideas_to_review)} ideas")
            
            # Review each idea
            reviewed_ideas = []
            registry_updates = []
            
            for idea in ideas_to_review:
                try:
                    review_result = self._review_idea(idea)
                    reviewed_ideas.append(review_result)
                    
                    # Prepare registry update
                    registry_updates.append({
                        'registry': 'ideas_register',
                        'action': 'update',
                        'filter': {'idea_id': idea['idea_id']},
                        'data': review_result
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to review idea {idea.get('idea_id', 'unknown')}: {str(e)}")
            
            return AgentResult(
                success=True,
                agent_name=self.name,
                execution_id=context.execution_id,
                output_data={
                    'reviewed_ideas': reviewed_ideas,
                    'reviewed_count': len(reviewed_ideas)
                },
                registry_updates=registry_updates,
                execution_time_seconds=0,
                input_hash="",
                output_hash=""
            )
            
        except Exception as e:
            self.logger.error(f"Review process failed: {str(e)}")
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
    
    def _load_ideas_for_review(self, filter_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load ideas from registry based on filter criteria."""
        try:
            # Load ideas from the registry
            self.logger.info("Loading ideas from registry for review")
            
            ideas_df = self.registry_manager.load_registry('ideas_register')
            
            if ideas_df.empty:
                self.logger.info("No ideas found in registry")
                return []
            
            # Apply filter criteria
            filtered_df = ideas_df.copy()
            
            for key, value in filter_criteria.items():
                if key in filtered_df.columns:
                    if isinstance(value, list):
                        filtered_df = filtered_df[filtered_df[key].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[key] == value]
            
            # Convert to list of dictionaries
            ideas_list = filtered_df.to_dict('records')
            
            # Parse list fields that are stored as strings
            for idea in ideas_list:
                for field in ['domain_tags', 'novelty_refs', 'required_data', 'methods', 'literature_refs']:
                    if field in idea and isinstance(idea[field], str):
                        try:
                            # Handle empty list strings
                            if idea[field] in ['[]', '']:
                                idea[field] = []
                            else:
                                # Parse JSON-like list string
                                import ast
                                idea[field] = ast.literal_eval(idea[field])
                        except (ValueError, SyntaxError):
                            # If parsing fails, treat as single item list or empty
                            if idea[field].strip():
                                idea[field] = [idea[field]]
                            else:
                                idea[field] = []
            
            self.logger.info(f"Found {len(ideas_list)} ideas matching filter criteria")
            return ideas_list
            
        except Exception as e:
            self.logger.error(f"Failed to load ideas for review: {str(e)}")
            return []
    
    def _review_idea(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Review a single idea and assign scores."""
        self.logger.info(f"Reviewing idea: {idea.get('title', 'Unknown')}")
        
        # Perform novelty check
        novelty_score = self._assess_novelty(idea)
        
        # Assess other dimensions
        impact_score = self._assess_impact(idea)
        feasibility_score = self._assess_feasibility(idea)
        testability_score = self._assess_testability(idea)
        
        # Calculate total score
        total_score = impact_score + feasibility_score + testability_score + novelty_score
        
        # Determine status based on thresholds
        status = self._determine_status(total_score, {
            'impact': impact_score,
            'feasibility': feasibility_score,
            'testability': testability_score,
            'novelty': novelty_score
        })
        
        # Log detailed scoring for debugging
        self.logger.info(f"Scoring: Impact={impact_score}, Feasibility={feasibility_score}, "
                        f"Testability={testability_score}, Novelty={novelty_score}, "
                        f"Total={total_score}/20 → Status: {status}")
        
        # Generate reviewer notes
        reviewer_notes = self._generate_reviewer_notes(
            idea, impact_score, feasibility_score, testability_score, novelty_score
        )
        
        # Update idea with review results
        reviewed_idea = idea.copy()
        reviewed_idea.update({
            'impact_score': impact_score,
            'feasibility_score': feasibility_score,
            'testability_score': testability_score,
            'novelty_score': novelty_score,
            'total_score': total_score,
            'status': status,
            'reviewer_notes': reviewer_notes,
            'updated_at': pd.Timestamp.now(tz='UTC').isoformat()
        })
        
        return reviewed_idea
    
    def _assess_novelty(self, idea: Dict[str, Any]) -> int:
        """Assess novelty score (1-5) based on comprehensive literature search."""
        try:
            hypothesis = idea.get('hypothesis', '')
            domain_tags = idea.get('domain_tags', [])
            title = idea.get('title', '')
            
            self.logger.info(f"Conducting comprehensive literature search for: {title}")
            
            # Perform comprehensive literature search across ALL time periods
            literature_context = self._comprehensive_literature_search(hypothesis, domain_tags, title)
            
            # Analyze literature overlap and novelty
            novelty_score = self._analyze_literature_novelty(hypothesis, literature_context)
            
            # Store literature references in idea for future use
            idea['literature_refs'] = literature_context.get('relevant_papers', [])
            idea['novelty_analysis'] = literature_context.get('novelty_analysis', '')
            
            self.logger.info(f"Novelty score: {novelty_score}/5 based on {len(literature_context.get('relevant_papers', []))} relevant papers")
            return novelty_score
                
        except Exception as e:
            self.logger.warning(f"Novelty assessment failed: {str(e)}")
            return 3  # Default to moderate score when search fails
    
    def _comprehensive_literature_search(self, hypothesis: str, domain_tags: List[str], title: str) -> Dict[str, Any]:
        """Perform comprehensive literature search across ALL time periods for any hypothesis."""
        
        literature_context = {
            'relevant_papers': [],
            'search_strategies': [],
            'total_papers_found': 0,
            'novelty_analysis': ''
        }
        
        try:
            # Strategy 1: Search by domain tags (broad scope)
            if isinstance(domain_tags, list):
                domains_to_search = domain_tags
            else:
                # Handle case where domain_tags might be a string or other format
                domains_to_search = [str(domain_tags)]
            
            for domain in domains_to_search:
                if domain and isinstance(domain, str) and len(domain.strip()) > 2:
                    clean_domain = domain.strip().replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                    self.logger.info(f"Searching for domain: '{clean_domain}'")
                    if clean_domain and len(clean_domain) > 2:  # Double-check after cleaning
                        papers = self.ads_service.search(
                            query=clean_domain,
                            max_results=50,
                            # NO time constraints - search all papers ever published
                            sort="citation_count desc"  # Get most cited papers
                        )
                        literature_context['relevant_papers'].extend(papers)
                        literature_context['search_strategies'].append(f"Domain search: {clean_domain}")
                    else:
                        self.logger.warning(f"Domain '{domain}' cleaned to empty string: '{clean_domain}'")
            
            # Strategy 2: Search by key terms extracted from hypothesis
            key_terms = self._extract_key_terms_from_hypothesis(hypothesis)
            for term in key_terms[:3]:  # Top 3 most important terms
                if term and len(term.strip()) > 2:  # Ensure term is valid
                    papers = self.ads_service.search(
                        query=term,  # Use term directly, not quoted (ADS handles this better)
                        max_results=30,
                        sort="citation_count desc"
                    )
                    literature_context['relevant_papers'].extend(papers)
                    literature_context['search_strategies'].append(f"Key term search: {term}")
            
            # Strategy 3: Search by methodological approach
            methods = self._extract_methods_from_hypothesis(hypothesis)
            for method in methods[:2]:  # Top 2 methods
                papers = self.ads_service.search(
                    query=f"{method} AND ({' OR '.join(domain_tags)})",
                    max_results=20,
                    sort="date desc"  # Get recent methodological advances
                )
                literature_context['relevant_papers'].extend(papers)
                literature_context['search_strategies'].append(f"Method search: {method}")
            
            # Remove duplicates based on bibcode
            seen_bibcodes = set()
            unique_papers = []
            for paper in literature_context['relevant_papers']:
                bibcode = paper.get('bibcode', '')
                if bibcode and bibcode not in seen_bibcodes:
                    seen_bibcodes.add(bibcode)
                    unique_papers.append(paper)
            
            literature_context['relevant_papers'] = unique_papers
            literature_context['total_papers_found'] = len(unique_papers)
            
            self.logger.info(f"Found {len(unique_papers)} unique relevant papers using {len(literature_context['search_strategies'])} search strategies")
            
        except Exception as e:
            self.logger.error(f"Literature search failed: {e}")
        
        return literature_context
    
    def _extract_key_terms_from_hypothesis(self, hypothesis: str) -> List[str]:
        """Extract key scientific terms from hypothesis for targeted literature search."""
        import re
        
        # Remove common words and extract potential scientific terms
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'that', 'this', 'these', 'those', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'be', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'done', 'doing'}
        
        # First extract scientific compound terms (like "dark matter", "stellar evolution")
        scientific_phrases = re.findall(r'\b(?:dark matter|stellar evolution|galactic dynamics|exoplanet|black hole|neutron star|supernova|gamma ray|x-ray|magnetic field|gravitational wave|cosmic ray|planetary nebula|white dwarf|red giant|main sequence|brown dwarf|quasar|active galactic nuclei|cosmic microwave background|big bang|inflation|dark energy)\b', hypothesis.lower())
        
        # Extract individual scientific words (3+ characters, excluding common words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', hypothesis.lower())
        scientific_words = [word for word in words if word not in common_words and len(word) > 3]
        
        # Combine and prioritize
        key_terms = scientific_phrases + scientific_words[:10]  # Limit individual words
        
        # Remove duplicates and empty terms
        key_terms = [term.strip() for term in set(key_terms) if term and len(term.strip()) > 2]
        
        # Sort by length (longer scientific phrases first)
        key_terms.sort(key=len, reverse=True)
        
        return key_terms[:5]
    
    def _extract_methods_from_hypothesis(self, hypothesis: str) -> List[str]:
        """Extract methodological approaches from hypothesis."""
        
        # Common astrophysical methods and techniques
        method_keywords = [
            'spectroscopy', 'photometry', 'astrometry', 'interferometry',
            'modeling', 'simulation', 'analysis', 'survey', 'observations',
            'statistical analysis', 'machine learning', 'deep learning',
            'time series', 'variability', 'correlation', 'regression'
        ]
        
        hypothesis_lower = hypothesis.lower()
        found_methods = []
        
        for method in method_keywords:
            if method in hypothesis_lower:
                found_methods.append(method)
        
        return found_methods[:3]  # Return top 3 methods
    
    def _analyze_literature_novelty(self, hypothesis: str, literature_context: Dict[str, Any]) -> int:
        """Analyze literature context to determine novelty score."""
        
        papers = literature_context.get('relevant_papers', [])
        
        if len(papers) == 0:
            # No related literature found - either very novel or poorly formed hypothesis
            return 5  # Assume highly novel if no literature exists
        
        # Score based on literature landscape
        if len(papers) < 10:
            # Very few papers - likely novel area
            return 5
        elif len(papers) < 25:
            # Some papers but not saturated field
            return 4
        elif len(papers) < 50:
            # Moderate literature base
            return 3
        elif len(papers) < 100:
            # Well-studied area
            return 2
        else:
            # Highly studied field - harder to be novel
            return 1
    
    def _assess_impact(self, idea: Dict[str, Any]) -> int:
        """Assess potential impact score (1-5) - be encouraging but realistic."""
        try:
            hypothesis = idea.get('hypothesis', '')
            domain_tags = idea.get('domain_tags', [])
            title = idea.get('title', '')
            
            # Start with good base score - be encouraging!
            score = 3  # Base score (was 2)
            
            # Boost for specific, measurable predictions
            specific_indicators = [
                'correlation', 'relationship', 'ratio', 'rate', 'efficiency',
                'distribution', 'variation', 'signature', 'pattern', 'trend'
            ]
            
            # Boost for high-impact domains
            high_impact_domains = [
                'stellar evolution', 'galactic dynamics', 'exoplanets', 'dark matter',
                'black holes', 'neutron stars', 'supernova', 'galaxy formation'
            ]
            
            # Boost for novel analysis approaches
            novel_approaches = [
                'systematic analysis', 'cross-correlation', 'multi-wavelength',
                'time-domain', 'statistical survey', 'machine learning'
            ]
            
            text = (hypothesis + ' ' + title + ' ' + ' '.join(domain_tags)).lower()
            
            # Boost for specific predictions
            specific_matches = sum(1 for indicator in specific_indicators if indicator in text)
            if specific_matches >= 2:
                score += 1  # Multiple specific indicators
            
            # Boost for high-impact domains
            domain_matches = sum(1 for domain in high_impact_domains if domain in text)
            if domain_matches >= 1:
                score += 1  # Working in high-impact area
            
            # Boost for novel approaches
            approach_matches = sum(1 for approach in novel_approaches if approach in text)
            if approach_matches >= 1:
                score += 1  # Novel methodological approach
            
            # Don't penalize - be encouraging
            return min(score, 5)
            
        except Exception as e:
            self.logger.warning(f"Impact assessment failed: {str(e)}")
            return 3  # Default to neutral score
    
    def _assess_feasibility(self, idea: Dict[str, Any]) -> int:
        """Assess feasibility score (1-5) based on data and methods."""
        try:
            required_data = idea.get('required_data', [])
            methods = idea.get('methods', [])
            est_effort_days = idea.get('est_effort_days', 10)
            
            score = 3  # Base score
            
            # Common, accessible datasets boost feasibility
            accessible_datasets = [
                'Gaia', 'SDSS', 'TESS', 'Kepler', '2MASS', 'WISE', 'Pan-STARRS'
            ]
            
            accessible_count = sum(1 for dataset in required_data 
                                 for accessible in accessible_datasets 
                                 if accessible.lower() in dataset.lower())
            
            if accessible_count >= len(required_data) * 0.8:  # 80% accessible
                score += 1
            elif accessible_count < len(required_data) * 0.5:  # Less than 50% accessible
                score -= 1
            
            # Standard methods boost feasibility
            standard_methods = [
                'statistical', 'correlation', 'regression', 'classification',
                'time series', 'spectral analysis', 'photometry'
            ]
            
            method_text = ' '.join(methods).lower()
            if any(method in method_text for method in standard_methods):
                score += 1
            
            # Adjust for effort estimate
            if est_effort_days <= 7:
                score += 1
            elif est_effort_days > 21:
                score -= 1
            
            return max(1, min(score, 5))
            
        except Exception as e:
            self.logger.warning(f"Feasibility assessment failed: {str(e)}")
            return 3  # Default to neutral score
    
    def _assess_testability(self, idea: Dict[str, Any]) -> int:
        """Assess testability score (1-5) based on hypothesis clarity."""
        try:
            hypothesis = idea.get('hypothesis', '')
            methods = idea.get('methods', [])
            
            score = 3  # Base score
            
            # Look for quantitative language
            quantitative_terms = [
                'measure', 'correlation', 'ratio', 'distribution', 'probability',
                'significance', 'threshold', 'magnitude', 'frequency', 'rate'
            ]
            
            hypothesis_lower = hypothesis.lower()
            quant_matches = sum(1 for term in quantitative_terms if term in hypothesis_lower)
            
            if quant_matches >= 3:
                score += 1
            elif quant_matches == 0:
                score -= 1
            
            # Look for clear predictions
            prediction_terms = [
                'expect', 'predict', 'should', 'will', 'increase', 'decrease',
                'correlate', 'differ', 'greater', 'less', 'higher', 'lower'
            ]
            
            pred_matches = sum(1 for term in prediction_terms if term in hypothesis_lower)
            
            if pred_matches >= 2:
                score += 1
            elif pred_matches == 0:
                score -= 1
            
            # Statistical methods boost testability
            stats_methods = [
                'statistical', 'hypothesis test', 'p-value', 'confidence',
                'bayesian', 'monte carlo', 'bootstrap'
            ]
            
            method_text = ' '.join(methods).lower()
            if any(method in method_text for method in stats_methods):
                score += 1
            
            return max(1, min(score, 5))
            
        except Exception as e:
            self.logger.warning(f"Testability assessment failed: {str(e)}")
            return 3  # Default to neutral score
    
    def _determine_status(self, total_score: int, individual_scores: Dict[str, int]) -> str:
        """Determine approval status based on scores and thresholds - BE MORE LENIENT."""
        thresholds = self.approval_thresholds
        
        # Check total score thresholds FIRST (more important than individual mins)
        approved_min = thresholds.get('approved', {}).get('total_min', 10)  # Updated default
        rejected_max = thresholds.get('rejected', {}).get('total_max', 7)    # Updated default
        
        if total_score >= approved_min:
            # Even if some individual scores are low, approve if total is good
            return 'Approved'
        elif total_score <= rejected_max:
            return 'Rejected'
        else:
            # Middle range gets revision
            return 'Needs Revision'
    
    def _generate_reviewer_notes(self, idea: Dict[str, Any], 
                               impact: int, feasibility: int, 
                               testability: int, novelty: int) -> str:
        """Generate detailed, PhD-level reviewer feedback."""
        
        notes = []
        
        # Detailed impact assessment
        if impact >= 4:
            notes.append(f"**HIGH IMPACT RESEARCH**: This hypothesis addresses fundamental questions in {', '.join(idea.get('domain_tags', ['astrophysics']))} with potential to advance theoretical understanding or observational capabilities. Expected to influence future research directions and potentially resolve current controversies in the field.")
        elif impact == 3:
            notes.append(f"**MODERATE IMPACT**: Solid contribution to {', '.join(idea.get('domain_tags', ['astrophysics']))} literature. Results would be of interest to specialists and provide incremental advances in understanding.")
        elif impact <= 2:
            notes.append(f"**LIMITED IMPACT**: While technically sound, the research question lacks broader significance. Consider reframing to address larger theoretical questions or connecting to current controversies in {', '.join(idea.get('domain_tags', ['astrophysics']))}.")
        
        # Detailed feasibility assessment
        if feasibility <= 2:
            data_reqs = idea.get('required_data', [])
            methods = idea.get('methods', [])
            notes.append(f"**FEASIBILITY CONCERNS**: Significant challenges identified. Data access for {', '.join(data_reqs)} may be limited or require specialized facilities. Proposed methods ({', '.join(methods)}) may require computational resources beyond typical research group capabilities. Consider alternative datasets or simplified methodological approaches.")
        elif feasibility >= 4:
            effort_days = idea.get('est_effort_days', 'Unknown')
            notes.append(f"**EXCELLENT FEASIBILITY**: All required datasets are publicly accessible with well-documented APIs. Proposed analysis methods are standard in the field. Estimated timeline of {effort_days} days is realistic for a postdoc-level researcher. No major technical barriers anticipated.")
        else:
            notes.append(f"**MODERATE FEASIBILITY**: Most requirements appear achievable with standard research infrastructure. Some challenges expected but manageable with appropriate planning.")
        
        # Detailed testability assessment
        if testability <= 2:
            hypothesis = idea.get('hypothesis', '')
            notes.append(f"**TESTABILITY DEFICIENCIES**: The hypothesis lacks quantitative predictions and measurable outcomes. Current formulation ('{hypothesis[:100]}...') is too qualitative for rigorous statistical testing. Recommend specifying: (1) Expected effect sizes with uncertainty ranges, (2) Statistical significance thresholds, (3) Alternative hypotheses for comparison, (4) Falsifiability criteria with specific observational signatures.")
        elif testability >= 4:
            notes.append(f"**HIGHLY TESTABLE**: Hypothesis provides clear, quantifiable predictions with well-defined success criteria. Statistical analysis plan is appropriate for the research question. Expected observational signatures are precisely specified with realistic detection thresholds.")
        else:
            notes.append(f"**ADEQUATE TESTABILITY**: Hypothesis can be tested but would benefit from more specific quantitative predictions and clearer statistical frameworks.")
        
        # Detailed novelty assessment
        if novelty <= 2:
            notes.append(f"**NOVELTY LIMITATIONS**: Significant conceptual overlap with existing literature detected. The proposed approach appears to be an incremental extension of established methods. Recommend conducting more thorough literature review to identify truly unexplored parameter space or novel methodological approaches.")
        elif novelty >= 4:
            domain_tags = idea.get('domain_tags', [])
            notes.append(f"**HIGHLY NOVEL CONTRIBUTION**: This research appears to explore genuinely uncharted territory in {', '.join(domain_tags)}. The proposed approach combines concepts or datasets in ways not previously attempted. High potential for discovery of new phenomena or relationships.")
        else:
            notes.append(f"**MODERATE NOVELTY**: Some novel elements present but building on established foundations. Good potential for advancing current understanding.")
        
        # Astrophysics-specific considerations
        astro_notes = self._generate_astrophysics_specific_notes(idea)
        if astro_notes:
            notes.extend(astro_notes)
        
        # Risk assessment
        risk_assessment = self._assess_astrophysics_risks(idea)
        if risk_assessment:
            notes.append(f"**RISK ASSESSMENT**: {risk_assessment}")
        
        # Overall recommendations with detailed reasoning
        total = impact + feasibility + testability + novelty
        
        if total >= 13:
            notes.append(f"**STRONG RECOMMENDATION FOR APPROVAL** (Score: {total}/20): This research meets high standards for impact, feasibility, and scientific rigor. The hypothesis is well-formulated, the approach is sound, and the expected outcomes would make valuable contributions to the field. Recommend immediate progression to experiment design phase.")
        elif 9 <= total <= 12:
            notes.append(f"**CONDITIONAL APPROVAL - REVISIONS NEEDED** (Score: {total}/20): The core research idea has merit but requires refinement before proceeding. Address the specific concerns noted above, particularly in areas scoring below 3. With revisions, this could become a strong contribution to the field.")
        else:
            notes.append(f"**RECOMMENDATION FOR REJECTION** (Score: {total}/20): Fundamental issues prevent approval in current form. The hypothesis requires major reconceptualization, alternative methodological approaches, or substantial additional development before it can meet publication standards.")
        
        return " | ".join(notes)
    
    def _generate_astrophysics_specific_notes(self, idea: Dict[str, Any]) -> List[str]:
        """Generate astrophysics domain-specific review notes."""
        
        notes = []
        domain_tags = idea.get('domain_tags', [])
        hypothesis = idea.get('hypothesis', '').lower()
        methods = idea.get('methods', [])
        data_reqs = idea.get('required_data', [])
        
        # Domain-specific considerations
        if any('exoplanet' in tag.lower() for tag in domain_tags):
            notes.append("**EXOPLANET SCIENCE**: Ensure consideration of selection effects in transit/RV surveys, stellar activity contamination, and atmospheric modeling uncertainties. Consider validation with multiple detection methods.")
            
        if any('stellar' in tag.lower() for tag in domain_tags):
            if 'gaia' in str(data_reqs).lower():
                notes.append("**STELLAR ASTROPHYSICS**: Excellent choice using Gaia data. Consider parallax uncertainties, proper motion systematic errors, and completeness limits. Account for stellar population gradients in the Galaxy.")
            
        if any('galactic' in tag.lower() for tag in domain_tags):
            notes.append("**GALACTIC ASTRONOMY**: Consider impact of dust extinction, distance uncertainties, and kinematic substructure. Validate results across different Galactic environments (disk, halo, bulge).")
            
        # Statistical methodology considerations
        if 'machine learning' in str(methods).lower():
            notes.append("**ML METHODOLOGY**: Ensure proper cross-validation, feature selection, and interpretability. Beware of overfitting with astronomical datasets. Consider physics-informed constraints.")
            
        if 'bayesian' in str(methods).lower():
            notes.append("**BAYESIAN ANALYSIS**: Excellent choice for handling uncertainties. Ensure proper prior selection and MCMC convergence diagnostics. Consider hierarchical modeling for population studies.")
        
        return notes
    
    def _assess_astrophysics_risks(self, idea: Dict[str, Any]) -> str:
        """Assess astrophysics-specific risks."""
        
        risks = []
        hypothesis = idea.get('hypothesis', '').lower()
        data_reqs = idea.get('required_data', [])
        domain_tags = idea.get('domain_tags', [])
        
        # Data-specific risks
        if 'jwst' in str(data_reqs).lower():
            risks.append("JWST data access may be limited by observing time allocation")
            
        if 'gaia' in str(data_reqs).lower() and 'precision' in hypothesis:
            risks.append("Gaia systematic errors may limit precision for faint sources (G > 19)")
            
        # Analysis risks
        if 'correlation' in hypothesis and len(data_reqs) < 2:
            risks.append("Single dataset correlation studies vulnerable to systematic biases")
            
        if any('supernova' in tag.lower() for tag in domain_tags) and 'host galaxy' in hypothesis:
            risks.append("Supernova-host galaxy associations subject to misidentification in crowded fields")
            
        # Sample size risks
        if 'rare' in hypothesis or 'transient' in hypothesis:
            risks.append("Limited sample sizes may reduce statistical power for rare phenomena")
        
        return '; '.join(risks) if risks else "Standard astrophysical analysis risks apply"
    
    def validate_input(self, context: AgentExecutionContext) -> bool:
        """Validate input for review process."""
        # Review can work with empty input (reviews all pending ideas)
        return True
    
    def validate_output(self, result: AgentResult) -> bool:
        """Validate review output."""
        if not result.success:
            return False
        
        reviewed_ideas = result.output_data.get('reviewed_ideas', [])
        
        # Check that all reviewed ideas have required scoring fields
        required_fields = ['impact_score', 'feasibility_score', 'testability_score', 
                          'novelty_score', 'total_score', 'status', 'reviewer_notes']
        
        for idea in reviewed_ideas:
            for field in required_fields:
                if field not in idea:
                    self.logger.error(f"Missing required review field: {field}")
                    return False
            
            # Validate score ranges
            for score_field in ['impact_score', 'feasibility_score', 'testability_score', 'novelty_score']:
                score = idea.get(score_field, 0)
                if not isinstance(score, int) or not (1 <= score <= 5):
                    self.logger.error(f"Invalid score for {score_field}: {score}")
                    return False
            
            # Validate status
            valid_statuses = ['Approved', 'Needs Revision', 'Rejected']
            if idea.get('status') not in valid_statuses:
                self.logger.error(f"Invalid status: {idea.get('status')}")
                return False
        
        return True
