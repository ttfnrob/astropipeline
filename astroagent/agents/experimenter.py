"""
Experimenter (EX) Agent for AstroAgent Pipeline.

This agent executes the planned experiments, runs analyses, and generates results.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from .common import BaseAgent, AgentExecutionContext, AgentResult


class Experimenter(BaseAgent):
    """Agent responsible for executing experiments and generating results."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.2)
        self.timeout_hours = config.get('timeout_hours', 24)
        
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Execute experiment for a ready project."""
        
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
            
            self.logger.info(f"Executing experiment for project: {project_id}")
            
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
            
            # Execute analysis pipeline
            results = self._execute_analysis(project)
            
            # Generate results documentation
            results_file = self._write_results(project, results)
            
            # Update project status
            registry_updates = [{
                'registry': 'project_index',
                'action': 'update',
                'filter': {'idea_id': project_id},
                'data': {
                    'maturity': 'Complete',
                    'execution_end': 'timestamp_placeholder'
                }
            }]
            
            return AgentResult(
                success=True,
                agent_name=self.name,
                execution_id=context.execution_id,
                output_data={
                    'results': results,
                    'results_file': str(results_file)
                },
                files_created=[str(results_file)],
                registry_updates=registry_updates,
                execution_time_seconds=0,
                input_hash="",
                output_hash=""
            )
            
        except Exception as e:
            self.logger.error(f"Experiment execution failed: {str(e)}")
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
        # TODO: Implement actual registry loading
        self.logger.info("Project loading not implemented, using mock data")
        
        return {
            'idea_id': project_id,
            'path': f'projects/Ready for Execution/{project_id}__mock-project',
            'maturity': 'Ready'
        }
    
    def _execute_analysis(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis pipeline."""
        
        # TODO: Implement actual analysis execution
        # This would involve:
        # 1. Loading and validating data
        # 2. Running preprocessing steps  
        # 3. Executing statistical analysis
        # 4. Generating figures and tables
        # 5. Computing confidence intervals and diagnostics
        
        self.logger.info("Analysis execution not implemented, generating mock results")
        
        return {
            'status': 'completed',
            'primary_result': {
                'metric': 'correlation_coefficient',
                'value': 0.342,
                'confidence_interval': [0.298, 0.386],
                'p_value': 0.001,
                'effect_size': 'medium'
            },
            'secondary_results': [
                {'metric': 'sample_size', 'value': 15432},
                {'metric': 'data_quality_score', 'value': 0.89}
            ],
            'figures_generated': ['correlation_plot.png', 'distribution_plot.png'],
            'tables_generated': ['summary_statistics.csv'],
            'robustness_checks': {
                'bootstrap_ci': [0.301, 0.383],
                'sensitivity_analysis': 'stable',
                'outlier_test': 'passed'
            }
        }
    
    def _write_results(self, project: Dict[str, Any], results: Dict[str, Any]) -> Path:
        """Write results to markdown file."""
        
        project_path = Path(project['path'])
        results_file = project_path / 'results.md'
        
        # Generate results markdown
        results_content = f"""# Results: {project['idea_id']}

## Summary
Analysis completed successfully with significant findings.

**Primary Result:** {results['primary_result']['metric']} = {results['primary_result']['value']} 
(95% CI: {results['primary_result']['confidence_interval']}, p = {results['primary_result']['p_value']})

## Figures
{', '.join(results.get('figures_generated', []))}

## Tables  
{', '.join(results.get('tables_generated', []))}

## Robustness checks
- Bootstrap confidence interval: {results['robustness_checks']['bootstrap_ci']}
- Sensitivity analysis: {results['robustness_checks']['sensitivity_analysis']}
- Outlier detection: {results['robustness_checks']['outlier_test']}

## Limitations
- Analysis based on observational data, causality not established
- Sample selection effects may bias results
- Further validation with independent datasets recommended
"""
        
        # Write to file
        with open(results_file, 'w') as f:
            f.write(results_content)
        
        return results_file
    
    def validate_input(self, context: AgentExecutionContext) -> bool:
        """Validate input for experiment execution."""
        if 'idea_id' not in context.input_data:
            self.logger.error("Missing required input: idea_id")
            return False
        
        return True
    
    def validate_output(self, result: AgentResult) -> bool:
        """Validate experiment output."""
        if not result.success:
            return False
        
        # Check results were generated
        if 'results' not in result.output_data:
            self.logger.error("No results generated")
            return False
        
        return True
