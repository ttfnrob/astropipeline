"""
Continuous Pipeline for AstroAgent System

This module implements a continuous multi-agent workflow that processes ideas
through their complete lifecycle until completion criteria are met.
"""

import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from .registry import ProjectRegistry
from .tools import RegistryManager
from ..agents import create_agent


class ContinuousPipeline:
    """Continuous multi-agent pipeline that processes ideas through completion."""
    
    def __init__(self, config_dir: str = "astroagent/config", data_dir: str = "data"):
        """Initialize continuous pipeline.
        
        Args:
            config_dir: Configuration directory path
            data_dir: Data directory path
        """
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        
        self.logger = self._setup_logger()
        
        # Initialize components
        self.registry_manager = RegistryManager(data_dir)
        self.project_registry = ProjectRegistry(data_dir)
        
        # Pipeline state
        self.is_running = False
        self.start_time = None
        self.completed_ideas = 0
        self.total_cycles = 0
        
        # Status callbacks for real-time updates
        self.status_callbacks: List[Callable[[Dict], None]] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup pipeline logger."""
        logger = logging.getLogger('astroagent.continuous_pipeline')
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def add_status_callback(self, callback: Callable[[Dict], None]):
        """Add callback for status updates."""
        self.status_callbacks.append(callback)
    
    def _broadcast_status(self, status: Dict[str, Any]):
        """Broadcast status to all callbacks."""
        for callback in self.status_callbacks:
            try:
                callback(status)
            except Exception as e:
                self.logger.error(f"Status callback error: {e}")
    
    async def run_continuous(self, 
                           initial_inputs: Dict[str, Any],
                           completion_mode: str = "ideas",
                           completion_target: int = 3,
                           max_duration_minutes: Optional[int] = None) -> Dict[str, Any]:
        """Run continuous pipeline until completion criteria are met.
        
        Args:
            initial_inputs: Initial pipeline inputs (domain_tags, etc.)
            completion_mode: "ideas" (complete N ideas) or "time" (run for duration)
            completion_target: Number of ideas to complete (if mode="ideas")
            max_duration_minutes: Maximum runtime in minutes (if mode="time" or as backup)
            
        Returns:
            Pipeline execution summary
        """
        
        self.logger.info(f"Starting continuous pipeline - {completion_mode} mode")
        self.logger.info(f"Target: {completion_target} {'ideas' if completion_mode == 'ideas' else 'minutes'}")
        
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        self.completed_ideas = 0
        self.total_cycles = 0
        
        try:
            while self.is_running:
                cycle_start = time.time()
                self.total_cycles += 1
                
                self.logger.info(f"Pipeline cycle {self.total_cycles}")
                
                # Execute one pipeline cycle
                cycle_results = await self._execute_cycle(initial_inputs)
                
                # Update completion tracking
                if cycle_results.get('ideas_completed', 0) > 0:
                    self.completed_ideas += cycle_results['ideas_completed']
                
                # Broadcast status update
                await self._send_status_update()
                
                # Check completion criteria
                if await self._check_completion_criteria(completion_mode, completion_target, max_duration_minutes):
                    break
                
                # Brief pause between cycles
                cycle_time = time.time() - cycle_start
                if cycle_time < 2.0:  # Ensure minimum 2 second cycle time
                    await asyncio.sleep(2.0 - cycle_time)
                
        except KeyboardInterrupt:
            self.logger.info("Pipeline stopped by user")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
        finally:
            self.is_running = False
            
        # Final summary
        runtime = datetime.now(timezone.utc) - self.start_time
        summary = {
            'success': True,
            'runtime_seconds': runtime.total_seconds(),
            'completed_ideas': self.completed_ideas,
            'total_cycles': self.total_cycles,
            'ideas_per_hour': (self.completed_ideas / runtime.total_seconds() * 3600) if runtime.total_seconds() > 0 else 0
        }
        
        self.logger.info(f"Pipeline completed: {self.completed_ideas} ideas in {runtime}")
        return summary
    
    async def _execute_cycle(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one complete pipeline cycle."""
        
        cycle_results = {
            'ideas_generated': 0,
            'ideas_reviewed': 0,
            'experiments_designed': 0,
            'experiments_executed': 0,
            'ideas_completed': 0,
            'errors': []
        }
        
        try:
            # Stage 1: Generate new ideas (if needed)
            ideas_generated = await self._generate_ideas_stage(inputs)
            cycle_results['ideas_generated'] = ideas_generated
            
            # Stage 2: Review pending ideas
            ideas_reviewed = await self._review_ideas_stage()
            cycle_results['ideas_reviewed'] = ideas_reviewed
            
            # Stage 3: Design experiments for approved ideas
            experiments_designed = await self._design_experiments_stage()
            cycle_results['experiments_designed'] = experiments_designed
            
            # Stage 4: Execute ready experiments
            experiments_executed = await self._execute_experiments_stage()
            cycle_results['experiments_executed'] = experiments_executed
            
            # Stage 5: Archive completed work
            ideas_completed = await self._archive_completed_stage()
            cycle_results['ideas_completed'] = ideas_completed
            
        except Exception as e:
            self.logger.error(f"Cycle execution error: {e}")
            cycle_results['errors'].append(str(e))
        
        return cycle_results
    
    async def _generate_ideas_stage(self, inputs: Dict[str, Any]) -> int:
        """Generate new ideas if needed."""
        
        # Check how many ideas are in the pipeline
        ideas_df = self.registry_manager.load_registry('ideas_register')
        pending_ideas = len(ideas_df[ideas_df['status'].isin(['Proposed', 'Under Review'])])
        
        # Generate new ideas if we have too few in the pipeline
        target_pipeline_size = inputs.get('pipeline_size', 5)
        if pending_ideas < target_pipeline_size:
            needed = target_pipeline_size - pending_ideas
            
            self.logger.info(f"Generating {needed} new ideas")
            
            try:
                # Create hypothesis maker
                hypothesis_maker = create_agent('hypothesis_maker', str(self.config_dir))
                
                # Generate ideas
                from ..agents.common import AgentExecutionContext
                context = AgentExecutionContext(
                    agent_name='hypothesis_maker',
                    state_name='continuous_generation',
                    input_data={
                        'domain_tags': inputs.get('domain_tags', ['stellar evolution']),
                        'n_hypotheses': needed,
                        'recency_years': inputs.get('recency_years', 3)
                    }
                )
                
                result = hypothesis_maker.run(context)
                
                if result.success:
                    hypotheses = result.output_data.get('hypotheses', [])
                    
                    # Add to registry
                    for hypothesis in hypotheses:
                        self.registry_manager.append_to_registry('ideas_register', hypothesis)
                    
                    return len(hypotheses)
                
            except Exception as e:
                self.logger.error(f"Idea generation failed: {e}")
        
        return 0
    
    async def _review_ideas_stage(self) -> int:
        """Review pending ideas."""
        
        # Get ideas needing review
        ideas_df = self.registry_manager.load_registry('ideas_register')
        pending_review = ideas_df[ideas_df['status'] == 'Proposed']
        
        if len(pending_review) == 0:
            return 0
        
        self.logger.info(f"Reviewing {len(pending_review)} ideas")
        
        try:
            # Create reviewer
            reviewer = create_agent('reviewer', str(self.config_dir))
            
            reviewed_count = 0
            for _, idea_row in pending_review.iterrows():
                idea = idea_row.to_dict()
                
                # Review the idea
                reviewed_idea = reviewer._review_idea(idea)
                
                # Update registry
                self.registry_manager.update_registry_row(
                    'ideas_register',
                    {'idea_id': idea['idea_id']},
                    reviewed_idea
                )
                
                reviewed_count += 1
            
            return reviewed_count
            
        except Exception as e:
            self.logger.error(f"Review stage failed: {e}")
            return 0
    
    async def _design_experiments_stage(self) -> int:
        """Design experiments for approved ideas."""
        
        # Get approved ideas without experiments
        ideas_df = self.registry_manager.load_registry('ideas_register')
        approved_ideas = ideas_df[ideas_df['status'] == 'Approved']
        
        projects_df = self.registry_manager.load_registry('project_index')
        existing_projects = set(projects_df['idea_id'].tolist()) if not projects_df.empty else set()
        
        # Find approved ideas without projects
        needs_experiments = approved_ideas[~approved_ideas['idea_id'].isin(existing_projects)]
        
        if len(needs_experiments) == 0:
            return 0
        
        self.logger.info(f"Designing experiments for {len(needs_experiments)} ideas")
        
        try:
            # Create experiment designer
            experiment_designer = create_agent('experiment_designer', str(self.config_dir))
            
            designed_count = 0
            for _, idea_row in needs_experiments.iterrows():
                idea = idea_row.to_dict()
                
                # Design experiment
                from ..agents.common import AgentExecutionContext
                context = AgentExecutionContext(
                    agent_name='experiment_designer',
                    state_name='continuous_design',
                    input_data={'idea_id': idea['idea_id']}
                )
                
                result = experiment_designer.run(context)
                
                if result.success:
                    # Update project registry
                    project_data = result.output_data
                    self.registry_manager.append_to_registry('project_index', project_data)
                    designed_count += 1
            
            return designed_count
            
        except Exception as e:
            self.logger.error(f"Experiment design stage failed: {e}")
            return 0
    
    async def _execute_experiments_stage(self) -> int:
        """Execute ready experiments."""
        
        # Get ready projects
        projects_df = self.registry_manager.load_registry('project_index')
        ready_projects = projects_df[projects_df['maturity'] == 'Ready for Execution']
        
        if len(ready_projects) == 0:
            return 0
        
        self.logger.info(f"Executing {len(ready_projects)} experiments")
        
        # For now, simulate experiment execution
        # In a real system, this would run actual analysis pipelines
        executed_count = 0
        for _, project_row in ready_projects.iterrows():
            project = project_row.to_dict()
            
            # Simulate execution (in real system, would run actual experiments)
            self.registry_manager.update_registry_row(
                'project_index',
                {'idea_id': project['idea_id']},
                {'maturity': 'Complete', 'execution_end': datetime.now(timezone.utc).isoformat()}
            )
            
            executed_count += 1
        
        return executed_count
    
    async def _archive_completed_stage(self) -> int:
        """Archive completed projects."""
        
        # Get completed projects
        projects_df = self.registry_manager.load_registry('project_index')
        completed_projects = projects_df[projects_df['maturity'] == 'Complete']
        
        if len(completed_projects) == 0:
            return 0
        
        # Move to completed registry
        completed_count = 0
        for _, project_row in completed_projects.iterrows():
            project = project_row.to_dict()
            
            # Add to completed registry
            completed_entry = {
                'idea_id': project['idea_id'],
                'title': f"Project {project['idea_id']}",
                'moved_to_library_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.registry_manager.append_to_registry('completed_index', completed_entry)
            completed_count += 1
        
        self.logger.info(f"Archived {completed_count} completed ideas")
        return completed_count
    
    async def _check_completion_criteria(self, 
                                       mode: str, 
                                       target: int, 
                                       max_duration: Optional[int]) -> bool:
        """Check if completion criteria are met."""
        
        # Check time limit (always applies)
        if max_duration:
            runtime_minutes = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
            if runtime_minutes >= max_duration:
                self.logger.info(f"Time limit reached: {runtime_minutes:.1f} minutes")
                return True
        
        # Check mode-specific criteria
        if mode == "ideas":
            if self.completed_ideas >= target:
                self.logger.info(f"Completed {self.completed_ideas} ideas (target: {target})")
                return True
        elif mode == "time":
            runtime_minutes = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
            if runtime_minutes >= target:
                self.logger.info(f"Time target reached: {runtime_minutes:.1f} minutes")
                return True
        
        return False
    
    async def _send_status_update(self):
        """Send status update to callbacks."""
        runtime = datetime.now(timezone.utc) - self.start_time
        
        status = {
            'is_running': self.is_running,
            'runtime_seconds': runtime.total_seconds(),
            'completed_ideas': self.completed_ideas,
            'total_cycles': self.total_cycles,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self._broadcast_status(status)
    
    def stop(self):
        """Stop the continuous pipeline."""
        self.logger.info("Stopping continuous pipeline...")
        self.is_running = False
