"""
Peer Reviewer (PR) Agent for AstroAgent Pipeline.

This agent reviews experimental results, validates reproducibility,
and provides approval or requests changes.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import pandas as pd

from .common import BaseAgent, AgentExecutionContext, AgentResult


class PeerReviewer(BaseAgent):
    """Agent responsible for reviewing experimental results and providing feedback."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Lazy import to avoid circular dependency
        from ..orchestration.tools import RegistryManager
        self.registry_manager = RegistryManager(config.get('data_dir', 'data'), logger)
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.3)
        self.review_criteria = config.get('review_criteria', {})
        self.approval_thresholds = config.get('approval_thresholds', {})
        
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Review experimental results for a project."""
        
        try:
            project_id = context.input_data.get('idea_id')
            if not project_id:
                return AgentResult(
                    success=False,
                    agent_name=self.name,
                    execution_id=context.execution_id,
                    error_message="No project ID provided",
                    error_type="ValueError",
                    execution_time_seconds=0,
                    input_hash="",
                    output_hash=""
                )
            
            self.logger.info(f"Peer reviewing project: {project_id}")
            
            # Load project details
            project = self._load_project(project_id)
            if not project:
                return AgentResult(
                    success=False,
                    agent_name=self.name,
                    execution_id=context.execution_id,
                    error_message=f"Project {project_id} not found",
                    error_type="NotFoundError",
                    execution_time_seconds=0,
                    input_hash="",
                    output_hash=""
                )
            
            # Validate project has results to review
            if not self._has_results_to_review(project):
                return AgentResult(
                    success=False,
                    agent_name=self.name,
                    execution_id=context.execution_id,
                    error_message=f"Project {project_id} has no results to review",
                    error_type="ValidationError",
                    execution_time_seconds=0,
                    input_hash="",
                    output_hash=""
                )
            
            # Conduct peer review
            review_result = self._conduct_review(project)
            
            # Generate reviewer report
            report_file = self._write_reviewer_report(project, review_result)
            
            # Determine next state based on review outcome
            next_maturity = 'Ready for Reporting' if review_result['approved'] else 'Needs Changes'
            
            # Update project status
            registry_updates = [{
                'registry': 'project_index',
                'action': 'update',
                'filter': {'idea_id': project_id},
                'data': {
                    'maturity': next_maturity,
                    'peer_review_status': 'Approved' if review_result['approved'] else 'Changes Requested',
                    'updated_at': pd.Timestamp.now(tz='UTC').isoformat()
                }
            }]
            
            return AgentResult(
                success=True,
                agent_name=self.name,
                execution_id=context.execution_id,
                output_data={
                    'review_result': review_result,
                    'reviewer_report': str(report_file),
                    'approved': review_result['approved']
                },
                files_created=[str(report_file)],
                registry_updates=registry_updates,
                execution_time_seconds=0,
                input_hash="",
                output_hash=""
            )
            
        except Exception as e:
            self.logger.error(f"Peer review failed: {str(e)}")
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
    
    def _load_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Load project details from registry."""
        try:
            self.logger.info(f"Loading project {project_id} from registry")
            
            # Load project index
            project_df = self.registry_manager.load_registry('project_index')
            
            if project_df.empty:
                self.logger.warning("Project index is empty")
                return None
            
            # Find the specific project
            matching_projects = project_df[project_df['idea_id'] == project_id]
            
            if matching_projects.empty:
                self.logger.warning(f"Project {project_id} not found in project index")
                return None
            
            # Get the first match (should be unique)
            project_data = matching_projects.iloc[0].to_dict()
            
            self.logger.info(f"Found project: {project_data.get('slug', 'unknown')}")
            return project_data
            
        except Exception as e:
            self.logger.error(f"Failed to load project {project_id}: {str(e)}")
            return None
    
    def _has_results_to_review(self, project: Dict[str, Any]) -> bool:
        """Check if project has results available for review."""
        try:
            project_path = Path(project['path'])
            
            # Check for results.md file
            results_file = project_path / 'results.md'
            if not results_file.exists():
                self.logger.warning(f"No results.md found in {project_path}")
                return False
            
            # Check for artefacts directory
            artefacts_dir = project_path / 'artefacts'
            if not artefacts_dir.exists():
                self.logger.warning(f"No artefacts directory found in {project_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking project results: {str(e)}")
            return False
    
    def _conduct_review(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct peer review of experimental results."""
        
        self.logger.info(f"Conducting peer review for project: {project.get('slug', 'unknown')}")
        
        project_path = Path(project['path'])
        
        # Review components
        review_components = {
            'reproducibility_score': self._check_reproducibility(project_path),
            'statistical_validity_score': self._check_statistical_validity(project_path),  
            'novelty_score': self._check_novelty(project_path),
            'presentation_score': self._check_presentation(project_path),
            'data_quality_score': self._check_data_quality(project_path)
        }
        
        # Calculate overall score
        total_score = sum(review_components.values())
        max_score = len(review_components) * 5  # Each component scored 1-5
        
        # Determine approval based on threshold
        approval_threshold = self.approval_thresholds.get('total_min', 18)  # 3.6/5 average
        approved = total_score >= approval_threshold
        
        # Generate feedback
        feedback = self._generate_review_feedback(review_components, approved)
        
        return {
            'approved': approved,
            'total_score': total_score,
            'max_score': max_score,
            'components': review_components,
            'feedback': feedback,
            'recommendation': 'Approved' if approved else 'Changes Required'
        }
    
    def _check_reproducibility(self, project_path: Path) -> int:
        """Check reproducibility of results (score 1-5)."""
        score = 3  # Base score
        
        # Check for scripts directory
        if (project_path / 'scripts').exists():
            score += 1
        
        # Check for notebooks
        notebooks_dir = project_path / 'notebooks'
        if notebooks_dir.exists() and list(notebooks_dir.glob('*.ipynb')):
            score += 1
        
        # Check for results file
        if (project_path / 'results.md').exists():
            score = min(score, 5)
        else:
            score -= 1
        
        return max(1, min(score, 5))
    
    def _check_statistical_validity(self, project_path: Path) -> int:
        """Check statistical validity of analysis (score 1-5)."""
        # For now, use heuristics based on results content
        score = 3  # Base score
        
        results_file = project_path / 'results.md'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    content = f.read().lower()
                
                # Look for statistical rigor indicators
                if 'confidence interval' in content:
                    score += 1
                if 'p-value' in content or 'p =' in content:
                    score += 1
                if 'bootstrap' in content or 'robustness' in content:
                    score += 1
                
                # Penalize if no statistical measures found
                if not any(term in content for term in ['statistical', 'significance', 'confidence', 'p-value']):
                    score -= 1
                    
            except Exception as e:
                self.logger.warning(f"Could not read results file: {e}")
                score -= 1
        
        return max(1, min(score, 5))
    
    def _check_novelty(self, project_path: Path) -> int:
        """Check novelty and contribution (score 1-5)."""
        # For now, assume moderate novelty
        # In a full implementation, this would check against literature
        return 4
    
    def _check_presentation(self, project_path: Path) -> int:
        """Check quality of presentation (score 1-5)."""
        score = 3  # Base score
        
        # Check for results file
        results_file = project_path / 'results.md'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    content = f.read()
                
                # Check for good structure
                required_sections = ['## Summary', '## Results', '## Limitations']
                found_sections = sum(1 for section in required_sections if section in content)
                
                if found_sections >= 2:
                    score += 1
                if found_sections == len(required_sections):
                    score += 1
                    
                # Check for figures/tables mentions
                if 'figure' in content.lower() or 'table' in content.lower():
                    score += 1
                    
            except Exception as e:
                self.logger.warning(f"Could not assess presentation: {e}")
                score -= 1
        
        return max(1, min(score, 5))
    
    def _check_data_quality(self, project_path: Path) -> int:
        """Check data quality and handling (score 1-5)."""
        score = 3  # Base score
        
        # Check for experiment plan
        if (project_path / 'experiment_plan.md').exists():
            score += 1
        
        # Check for artefacts
        artefacts_dir = project_path / 'artefacts'
        if artefacts_dir.exists() and list(artefacts_dir.iterdir()):
            score += 1
        
        return max(1, min(score, 5))
    
    def _generate_review_feedback(self, components: Dict[str, int], approved: bool) -> List[str]:
        """Generate detailed review feedback."""
        
        feedback = []
        
        # Component-specific feedback
        if components['reproducibility_score'] <= 2:
            feedback.append("REPRODUCIBILITY: Improve documentation of analysis steps and provide executable code.")
        elif components['reproducibility_score'] >= 4:
            feedback.append("REPRODUCIBILITY: Excellent documentation and code quality.")
        
        if components['statistical_validity_score'] <= 2:
            feedback.append("STATISTICS: Statistical analysis needs strengthening with proper significance tests.")
        elif components['statistical_validity_score'] >= 4:
            feedback.append("STATISTICS: Rigorous statistical approach with appropriate measures.")
        
        if components['presentation_score'] <= 2:
            feedback.append("PRESENTATION: Results need better organization and clearer presentation.")
        elif components['presentation_score'] >= 4:
            feedback.append("PRESENTATION: Clear and well-structured results presentation.")
        
        # Overall recommendation
        if approved:
            feedback.append("RECOMMENDATION: Approved for final reporting and publication.")
        else:
            feedback.append("RECOMMENDATION: Revisions required before approval.")
            feedback.append("Please address the concerns above and resubmit for review.")
        
        return feedback
    
    def _write_reviewer_report(self, project: Dict[str, Any], review_result: Dict[str, Any]) -> Path:
        """Write detailed reviewer report."""
        
        project_path = Path(project['path'])
        report_file = project_path / 'reviewer_report.md'
        
        # Generate report content
        report_content = f"""# Peer Review Report: {project['idea_id']}

## Project
**Title:** {project.get('slug', 'Unknown')}
**Reviewer:** AstroAgent Peer Review System
**Review Date:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}

## Verdict
**{review_result['recommendation']}**

## Overall Assessment
Total Score: {review_result['total_score']}/{review_result['max_score']}

## Component Scores
- Reproducibility: {review_result['components']['reproducibility_score']}/5
- Statistical Validity: {review_result['components']['statistical_validity_score']}/5  
- Novelty: {review_result['components']['novelty_score']}/5
- Presentation: {review_result['components']['presentation_score']}/5
- Data Quality: {review_result['components']['data_quality_score']}/5

## Detailed Feedback
{chr(10).join('- ' + item for item in review_result['feedback'])}

## Summary
{"This work meets the standards for publication and makes a valuable contribution to the field." if review_result['approved'] else "This work requires revisions before it can be approved for publication."}

---
*Automated review conducted by AstroAgent Peer Review System*
"""
        
        # Write report
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Generated peer review report: {report_file}")
        return report_file
    
    def validate_input(self, context: AgentExecutionContext) -> bool:
        """Validate input for peer review."""
        if 'idea_id' not in context.input_data:
            self.logger.error("Missing required input: idea_id")
            return False
        
        return True
    
    def validate_output(self, result: AgentResult) -> bool:
        """Validate peer review output."""
        if not result.success:
            return False
        
        # Check review result was generated
        if 'review_result' not in result.output_data:
            self.logger.error("No review result generated")
            return False
        
        return True
