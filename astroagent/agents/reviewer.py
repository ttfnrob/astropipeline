"""
Reviewer (RV) Agent for AstroAgent Pipeline.

This agent evaluates research hypotheses on multiple dimensions including
impact, feasibility, testability, and novelty.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from .common import BaseAgent, AgentExecutionContext, AgentResult, IdeaSchema
from ..services.literature import LiteratureService


class Reviewer(BaseAgent):
    """Agent responsible for reviewing and scoring research hypotheses."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Initialize services
        self.literature_service = LiteratureService()
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.3)
        self.scoring_rubric = config.get('scoring_rubric', {})
        self.approval_thresholds = config.get('approval_thresholds', {})
        
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Review and score hypotheses from the ideas register."""
        
        try:
            # Load ideas to review
            input_filter = context.input_data.get('filter', {'status': 'Proposed'})
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
            # TODO: Implement actual registry loading
            # For now, return mock data
            self.logger.info("Registry loading not implemented, using mock data")
            
            mock_ideas = [
                {
                    'idea_id': '01HZRXP1K2M3N4P5Q6R7S8T9U0',
                    'title': 'Novel Stellar Dynamics Correlation',
                    'hypothesis': 'Stellar velocity dispersions in galaxy clusters correlate with dark matter halo properties in ways not predicted by current models.',
                    'rationale': 'Recent observations suggest discrepancies between predicted and observed stellar kinematics in cluster environments.',
                    'domain_tags': ['stellar dynamics', 'galaxy clusters'],
                    'novelty_refs': ['2023ApJ...900...1A', '2023MNRAS.520.1234B'],
                    'required_data': ['Gaia DR3', 'SDSS'],
                    'methods': ['Statistical analysis', 'N-body simulations'],
                    'est_effort_days': 10,
                    'status': 'Proposed'
                }
            ]
            
            return mock_ideas
            
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
        """Assess novelty score (1-5) based on literature overlap."""
        try:
            # Check overlap with existing literature
            hypothesis = idea.get('hypothesis', '')
            domain_tags = idea.get('domain_tags', [])
            
            # TODO: Implement actual novelty checking using vector similarity
            # For now, return a mock score
            self.logger.info("Novelty assessment not fully implemented, using mock score")
            
            # Simple heuristic: longer, more specific hypotheses tend to be more novel
            word_count = len(hypothesis.split())
            tag_count = len(domain_tags)
            
            if word_count > 25 and tag_count >= 2:
                return 4  # High novelty
            elif word_count > 15 and tag_count >= 1:
                return 3  # Moderate novelty
            else:
                return 2  # Low novelty
                
        except Exception as e:
            self.logger.warning(f"Novelty assessment failed: {str(e)}")
            return 2  # Default to conservative score
    
    def _assess_impact(self, idea: Dict[str, Any]) -> int:
        """Assess potential impact score (1-5)."""
        try:
            hypothesis = idea.get('hypothesis', '')
            domain_tags = idea.get('domain_tags', [])
            
            # Heuristic scoring based on keywords and domain
            impact_keywords = [
                'paradigm', 'breakthrough', 'fundamental', 'revolutionary',
                'discovery', 'novel', 'unprecedented', 'significant'
            ]
            
            high_impact_domains = [
                'cosmology', 'dark matter', 'dark energy', 'exoplanets',
                'gravitational waves', 'black holes', 'galaxy formation'
            ]
            
            score = 2  # Base score
            
            # Boost for impact keywords
            hypothesis_lower = hypothesis.lower()
            keyword_matches = sum(1 for word in impact_keywords if word in hypothesis_lower)
            score += min(keyword_matches, 2)
            
            # Boost for high-impact domains
            domain_matches = sum(1 for tag in domain_tags 
                               for domain in high_impact_domains 
                               if domain in tag.lower())
            score += min(domain_matches, 1)
            
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
        """Determine approval status based on scores and thresholds."""
        thresholds = self.approval_thresholds
        
        # Check individual minimums
        individual_min = thresholds.get('approved', {}).get('individual_min', 3)
        for score_name, score in individual_scores.items():
            if score < individual_min:
                if total_score <= thresholds.get('rejected', {}).get('total_max', 8):
                    return 'Rejected'
                else:
                    return 'Needs Revision'
        
        # Check total score thresholds
        approved_min = thresholds.get('approved', {}).get('total_min', 13)
        revision_min = thresholds.get('revision', {}).get('total_min', 9)
        revision_max = thresholds.get('revision', {}).get('total_max', 12)
        
        if total_score >= approved_min:
            return 'Approved'
        elif revision_min <= total_score <= revision_max:
            return 'Needs Revision'
        else:
            return 'Rejected'
    
    def _generate_reviewer_notes(self, idea: Dict[str, Any], 
                               impact: int, feasibility: int, 
                               testability: int, novelty: int) -> str:
        """Generate detailed reviewer feedback."""
        
        notes = []
        
        # Impact feedback
        if impact >= 4:
            notes.append("HIGH IMPACT: This research addresses important scientific questions.")
        elif impact <= 2:
            notes.append("LOW IMPACT: Consider framing the research in terms of broader implications.")
        
        # Feasibility feedback
        if feasibility <= 2:
            notes.append("FEASIBILITY CONCERNS: Data access or methodological challenges may impede progress.")
        elif feasibility >= 4:
            notes.append("GOOD FEASIBILITY: Required data and methods appear readily accessible.")
        
        # Testability feedback
        if testability <= 2:
            notes.append("TESTABILITY ISSUES: Hypothesis needs more specific, measurable predictions.")
        elif testability >= 4:
            notes.append("WELL-TESTABLE: Clear predictions with defined success criteria.")
        
        # Novelty feedback
        if novelty <= 2:
            notes.append("NOVELTY CONCERNS: Significant overlap with existing literature detected.")
        elif novelty >= 4:
            notes.append("NOVEL APPROACH: Appears to address unexplored research directions.")
        
        # Overall recommendations
        total = impact + feasibility + testability + novelty
        
        if total >= 13:
            notes.append("RECOMMENDATION: Approve for experiment design phase.")
        elif 9 <= total <= 12:
            notes.append("RECOMMENDATION: Revise hypothesis to address identified concerns.")
        else:
            notes.append("RECOMMENDATION: Reject - fundamental issues require major reconceptualization.")
        
        return " | ".join(notes)
    
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
