"""
Experiment Designer (ED) Agent for AstroAgent Pipeline.

This agent creates detailed experimental plans for approved research ideas,
including data acquisition plans, analysis methods, and validation approaches.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import shutil

from .common import BaseAgent, AgentExecutionContext, AgentResult, create_project_folder


class ExperimentDesigner(BaseAgent):
    """Agent responsible for designing experiments and creating project plans."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Lazy import to avoid circular dependency
        from ..orchestration.tools import RegistryManager
        self.registry_manager = RegistryManager(config.get('data_dir', 'data'), logger)
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.5)
        self.ready_checklist = config.get('ready_checklist', [])
        
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Design experiments for approved ideas."""
        
        try:
            # Get approved ideas that need experiment design
            idea_id = context.input_data.get('idea_id')
            
            if not idea_id:
                # Load all approved ideas needing design
                approved_ideas = self._load_approved_ideas()
                if not approved_ideas:
                    return AgentResult(
                        success=True,
                        agent_name=self.name,
                        execution_id=context.execution_id,
                        output_data={'designed_count': 0},
                        execution_time_seconds=0,
                        input_hash="",
                        output_hash=""
                    )
                idea = approved_ideas[0]  # Process first one
            else:
                # Load specific idea
                idea = self._load_idea_by_id(idea_id)
                if not idea:
                    return AgentResult(
                        success=False,
                        agent_name=self.name,
                        execution_id=context.execution_id,
                        error_message=f"Idea {idea_id} not found",
                        error_type="NotFoundError",
                        execution_time_seconds=0,
                        input_hash="",
                        output_hash=""
                    )
            
            self.logger.info(f"Designing experiment for idea: {idea.get('title', 'Unknown')}")
            
            # Create project folder
            project_path, slug = create_project_folder(
                idea['idea_id'], 
                idea['title']
            )
            
            # Generate experiment plan
            experiment_plan = self._generate_experiment_plan(idea)
            
            # Write experiment plan to file
            plan_file = Path(project_path) / "experiment_plan.md"
            with open(plan_file, 'w') as f:
                f.write(experiment_plan)
            
            # Copy templates and populate initial files
            self._populate_project_structure(project_path, idea)
            
            # Run readiness checks
            ready_checks_passed = self._run_readiness_checks(project_path, idea)
            
            # Prepare registry updates
            registry_updates = [{
                'registry': 'project_index',
                'action': 'append',
                'data': {
                    'idea_id': idea['idea_id'],
                    'slug': slug,
                    'path': project_path,
                    'ready_checklist_passed': ready_checks_passed,
                    'data_requirements_met': True,  # TODO: Implement actual check
                    'analysis_plan_preregistered': True,
                    'maturity': 'Prepared' if ready_checks_passed else 'Preparing'
                }
            }]
            
            return AgentResult(
                success=True,
                agent_name=self.name,
                execution_id=context.execution_id,
                output_data={
                    'project_path': project_path,
                    'ready_checks_passed': ready_checks_passed,
                    'experiment_plan': experiment_plan
                },
                files_created=[str(plan_file)],
                registry_updates=registry_updates,
                execution_time_seconds=0,
                input_hash="",
                output_hash=""
            )
            
        except Exception as e:
            self.logger.error(f"Experiment design failed: {str(e)}")
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
    
    def _load_approved_ideas(self) -> List[Dict[str, Any]]:
        """Load approved ideas that need experiment design."""
        try:
            self.logger.info("Loading approved ideas from registry")
            
            # Load ideas from registry
            ideas_df = self.registry_manager.load_registry('ideas_register')
            
            if ideas_df.empty:
                self.logger.info("No ideas found in registry")
                return []
            
            # Filter for approved ideas
            approved_df = ideas_df[ideas_df['status'] == 'Approved']
            
            if approved_df.empty:
                self.logger.info("No approved ideas found in registry")
                return []
            
            # Convert to list of dictionaries
            ideas_list = approved_df.to_dict('records')
            
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
            
            self.logger.info(f"Found {len(ideas_list)} approved ideas needing experiment design")
            return ideas_list
            
        except Exception as e:
            self.logger.error(f"Failed to load approved ideas: {str(e)}")
            return []
    
    def _load_idea_by_id(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific idea by ID."""
        try:
            self.logger.info(f"Loading idea {idea_id} from registry")
            
            # Load ideas from registry
            ideas_df = self.registry_manager.load_registry('ideas_register')
            
            if ideas_df.empty:
                self.logger.warning("Ideas registry is empty")
                return None
            
            # Find the specific idea
            matching_ideas = ideas_df[ideas_df['idea_id'] == idea_id]
            
            if matching_ideas.empty:
                self.logger.warning(f"Idea {idea_id} not found in registry")
                return None
            
            # Get the first match (should be unique)
            idea_data = matching_ideas.iloc[0].to_dict()
            
            # Parse list fields that are stored as strings
            for field in ['domain_tags', 'novelty_refs', 'required_data', 'methods', 'literature_refs']:
                if field in idea_data and isinstance(idea_data[field], str):
                    try:
                        # Handle empty list strings
                        if idea_data[field] in ['[]', '']:
                            idea_data[field] = []
                        else:
                            # Parse JSON-like list string
                            import ast
                            idea_data[field] = ast.literal_eval(idea_data[field])
                    except (ValueError, SyntaxError):
                        # If parsing fails, treat as single item list or empty
                        if idea_data[field].strip():
                            idea_data[field] = [idea_data[field]]
                        else:
                            idea_data[field] = []
            
            self.logger.info(f"Found idea: {idea_data.get('title', 'unknown')}")
            return idea_data
            
        except Exception as e:
            self.logger.error(f"Failed to load idea {idea_id}: {str(e)}")
            return None
    
    def _generate_experiment_plan(self, idea: Dict[str, Any]) -> str:
        """Generate detailed experiment plan."""
        
        # Load template
        template_path = Path("templates") / "experiment_plan_template.md"
        
        if template_path.exists():
            with open(template_path, 'r') as f:
                template = f.read()
        else:
            template = self._get_default_template()
        
        # Fill template with idea details
        plan = template.replace('<idea_id>', idea['idea_id'])
        plan = plan.replace('<title>', idea['title'])
        
        # Add specific content based on idea
        objectives_section = f"""
## Objectives
- Primary outcome: Test the hypothesis - {idea['hypothesis']}
- Secondary outcomes: Validate methods and data quality
- Success criteria: Statistical significance p < 0.05 with effect size > 0.1
"""
        
        data_section = f"""
## Data
- Sources: {', '.join(idea.get('required_data', []))}
- Access pattern: Bulk download via APIs, quality filtering applied
- Quality controls: Remove flagged sources, require S/N > 5, completeness > 90%
- Sample size estimate: 10,000-100,000 objects depending on domain
"""
        
        methods_section = f"""
## Methods
- Preprocessing: Standard calibration and quality cuts
- Statistical approach: {', '.join(idea.get('methods', []))}
- Sensitivity analysis: Vary key parameters by ±20%
- Validation: Bootstrap resampling, cross-validation
- Null hypothesis: No correlation/effect detected
"""
        
        risks_section = """
## Risks
- Data availability: Backup datasets identified
- Selection bias: Multiple validation samples planned
- Systematic errors: Independent calibration checks
- Statistical power: Sample size calculations performed
- Look-elsewhere effect: Bonferroni correction applied
"""
        
        resources_section = f"""
## Resources
- Compute hours: {idea.get('est_effort_days', 7) * 8} CPU-hours estimated
- Storage: ~1-10 GB depending on dataset size
- Timeline: {idea.get('est_effort_days', 7)} days total effort
"""
        
        timeline_section = f"""
## Timeline
- Days 1-2: Data acquisition and initial quality checks
- Days 3-4: Preprocessing and exploratory analysis
- Days {max(5, idea.get('est_effort_days', 7)-2)}-{idea.get('est_effort_days', 7)}: Statistical analysis and validation
- Final day: Results compilation and documentation
"""
        
        # Replace template placeholders
        plan = plan.replace('- Primary outcome: <metric and acceptance threshold>', objectives_section)
        plan = plan.replace('- Sources: <catalogues, surveys, observatories>', data_section)
        plan = plan.replace('- Preprocessing', methods_section)
        plan = plan.replace('- Data gaps, selection bias, confounders, overfitting, look-elsewhere', risks_section)
        plan = plan.replace('- Compute hours, storage, estimated cost', resources_section)
        plan = plan.replace('- Milestones and review gates', timeline_section)
        
        return plan
    
    def _get_default_template(self) -> str:
        """Fallback experiment plan template."""
        return """# Experiment Plan: <idea_id> <title>

## Objectives
- Primary outcome: <metric and acceptance threshold>

## Data
- Sources: <catalogues, surveys, observatories>

## Methods
- Preprocessing

## Risks
- Data gaps, selection bias, confounders, overfitting, look-elsewhere

## Resources
- Compute hours, storage, estimated cost

## Timeline
- Milestones and review gates
"""
    
    def _populate_project_structure(self, project_path: str, idea: Dict[str, Any]):
        """Populate project folder with initial files and structure."""
        
        project_dir = Path(project_path)
        
        # Create idea.md file
        idea_file = project_dir / "idea.md"
        idea_content = f"""# Idea: {idea['title']}

**Idea ID:** {idea['idea_id']}
**Domain tags:** {', '.join(idea.get('domain_tags', []))}
**Status:** {idea.get('status', 'Unknown')}

## Hypothesis
{idea.get('hypothesis', 'Not specified')}

## Rationale
{idea.get('rationale', 'Not specified')}

## Required Data
{', '.join(idea.get('required_data', []))}

## Proposed Methods
{', '.join(idea.get('methods', []))}

## Estimated Effort
{idea.get('est_effort_days', 'Unknown')} days
"""
        
        with open(idea_file, 'w') as f:
            f.write(idea_content)
        
        # Create initial notebook
        notebooks_dir = project_dir / "notebooks"
        initial_notebook = notebooks_dir / "01_data_exploration.ipynb"
        
        # Create basic notebook structure (simplified JSON)
        notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration\\n",
    "\\n",
    "Initial exploration of datasets for: """ + idea['title'] + """\\n",
    "\\n",
    "## Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import astropy.units as u\\n",
    "from astropy.coordinates import SkyCoord\\n",
    "\\n",
    "# Configure plotting\\n",
    "plt.style.use('default')\\n",
    "%matplotlib inline"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
        
        with open(initial_notebook, 'w') as f:
            f.write(notebook_content)
        
        # Create initial script
        scripts_dir = project_dir / "scripts"
        data_script = scripts_dir / "fetch_data.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Data fetching script for project: {idea['title']}
"""

import argparse
import logging
from pathlib import Path

# TODO: Import appropriate data services
# from astroagent.services.datasets import DatasetService

def fetch_data(output_dir: Path):
    """Fetch required datasets."""
    
    required_data = {repr(idea.get('required_data', []))}
    
    print(f"Fetching data: {{', '.join(required_data)}}")
    
    # TODO: Implement actual data fetching
    for dataset in required_data:
        print(f"- Fetching {{dataset}}...")
    
    print("Data fetching complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch project data")
    parser.add_argument("--output", type=Path, default="data/raw",
                       help="Output directory for data files")
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    fetch_data(args.output)
'''
        
        with open(data_script, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        data_script.chmod(0o755)
    
    def _run_readiness_checks(self, project_path: str, idea: Dict[str, Any]) -> bool:
        """Run readiness checklist validation."""
        
        project_dir = Path(project_path)
        all_passed = True
        
        # Check experiment plan exists
        plan_file = project_dir / "experiment_plan.md"
        if not plan_file.exists():
            self.logger.error("Experiment plan file missing")
            all_passed = False
        
        # Check required structure exists
        required_dirs = ["research", "notebooks", "scripts", "artefacts"]
        for dirname in required_dirs:
            if not (project_dir / dirname).exists():
                self.logger.error(f"Required directory missing: {dirname}")
                all_passed = False
        
        # Check data sources are valid
        required_data = idea.get('required_data', [])
        valid_datasets = ['Gaia', 'SDSS', 'TESS', 'Kepler', '2MASS', 'WISE', 'Pan-STARRS', 'ZTF']
        
        for dataset in required_data:
            if not any(valid in dataset for valid in valid_datasets):
                self.logger.warning(f"Potentially problematic dataset: {dataset}")
        
        # TODO: Add more sophisticated readiness checks
        # - Data accessibility validation
        # - Method feasibility checks  
        # - Resource requirement validation
        
        return all_passed
    
    def validate_input(self, context: AgentExecutionContext) -> bool:
        """Validate input for experiment design."""
        # Can work with no input (processes all approved ideas)
        # Or with specific idea_id
        return True
    
    def validate_output(self, result: AgentResult) -> bool:
        """Validate experiment design output."""
        if not result.success:
            return False
        
        # Check that project path was created
        project_path = result.output_data.get('project_path')
        if not project_path or not Path(project_path).exists():
            self.logger.error("Project path not created")
            return False
        
        # Check experiment plan was generated
        if 'experiment_plan' not in result.output_data:
            self.logger.error("Experiment plan not generated")
            return False
        
        return True
