"""
Continuous Pipeline for AstroAgent System

This module implements a continuous multi-agent workflow that processes ideas
through their complete lifecycle until completion criteria are met.
"""

import asyncio
import signal
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
        self.is_paused = False
        self.should_stop = False
        self.start_time = None
        self.completed_ideas = 0
        self.total_cycles = 0
        
        # Lab technician state
        self.lab_technician_enabled = True
        self.lab_technician_interval = 3600  # Run every hour
        self.last_lab_technician_run = None
        
        # Status callbacks for real-time updates
        self.status_callbacks: List[Callable[[Dict], None]] = []
        
        # Signal handling for graceful shutdown
        self._setup_signal_handlers()
        
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
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            self.logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            self.should_stop = True
            self.is_running = False
        
        # Register signal handlers (only if not in a thread)
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            # This might happen in threads, ignore
            pass
    
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
    
    def _broadcast_stage_update(self, stage_name: str, stage_info: Dict[str, Any]):
        """Broadcast stage-specific updates for real-time UI feedback."""
        runtime = datetime.now(timezone.utc) - self.start_time if self.start_time else timedelta(0)
        
        stage_status = {
            'stage': stage_name,
            'stage_info': stage_info,
            'is_running': self.is_running,
            'runtime_seconds': runtime.total_seconds(),
            'completed_ideas': self.completed_ideas,
            'total_cycles': self.total_cycles,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self._broadcast_status(stage_status)
    
    async def run_continuous(self, 
                           initial_inputs: Dict[str, Any],
                           completion_mode: str = "ideas",
                           completion_target: int = 1,
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
        
        # Track baseline completion count at pipeline start
        self.baseline_completed_count = self._get_current_completed_count()
        
        try:
            while self.is_running and not self.should_stop:
                # Handle pause state with frequent checks for stop signal
                while self.is_paused and self.is_running and not self.should_stop:
                    self.logger.info("Pipeline paused, waiting for resume...")
                    # Use shorter sleep intervals to respond faster to Ctrl+C
                    await asyncio.sleep(0.5)
                
                if self.should_stop:
                    self.logger.info("Stop signal received, breaking execution loop...")
                    break
                
                cycle_start = time.time()
                self.total_cycles += 1
                
                self.logger.info(f"Pipeline cycle {self.total_cycles}")
                
                # Execute one pipeline cycle
                try:
                    cycle_results = await asyncio.wait_for(
                        self._execute_cycle(initial_inputs), 
                        timeout=30.0  # Timeout each cycle to allow for interruption
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Cycle execution timed out, continuing...")
                    cycle_results = {}
                except asyncio.CancelledError:
                    self.logger.info("Cycle execution cancelled")
                    break
                
                # Update completion tracking
                if cycle_results.get('ideas_completed', 0) > 0:
                    # Only count NEW completions since pipeline started (projects moved to Library)
                    new_completions = self._get_current_completed_count() - self.baseline_completed_count
                    self.completed_ideas = new_completions
                    self.logger.debug(f"Completed projects in Library: {self._get_current_completed_count()}, Baseline: {self.baseline_completed_count}, New: {new_completions}")
                
                # Broadcast status update
                try:
                    await asyncio.wait_for(self._send_status_update(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Status update timed out")
                
                # Check completion criteria
                if await self._check_completion_criteria(completion_mode, completion_target, max_duration_minutes):
                    break
                
                # Brief pause between cycles with frequent stop checks
                cycle_time = time.time() - cycle_start
                sleep_duration = max(0, 2.0 - cycle_time)
                
                # Break sleep into smaller chunks to respond to stop signals faster
                while sleep_duration > 0 and not self.should_stop:
                    sleep_chunk = min(0.5, sleep_duration)
                    await asyncio.sleep(sleep_chunk)
                    sleep_duration -= sleep_chunk
                
                if self.should_stop:
                    self.logger.info("Stop signal received during cycle sleep")
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Pipeline stopped by user (KeyboardInterrupt)")
            self.should_stop = True
        except asyncio.CancelledError:
            self.logger.info("Pipeline cancelled")
            self.should_stop = True
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
        finally:
            self.is_running = False
            self.logger.info("Pipeline execution loop ended")
            
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
        """Execute one complete pipeline cycle with real-time status updates."""
        
        cycle_results = {
            'ideas_generated': 0,
            'ideas_reviewed': 0,
            'experiments_designed': 0,
            'experiments_executed': 0,
            'peer_reviews_completed': 0,
            'reports_generated': 0,
            'ideas_completed': 0,
            'errors': []
        }
        
        try:
            # Stage 1: Generate new ideas (if needed)
            self._broadcast_stage_update('idea_generation', {'status': 'starting'})
            ideas_generated = await self._generate_ideas_stage(inputs)
            cycle_results['ideas_generated'] = ideas_generated
            self._broadcast_stage_update('idea_generation', {'status': 'completed', 'count': ideas_generated})
            
            # Stage 2: Review pending ideas
            self._broadcast_stage_update('idea_review', {'status': 'starting'})
            ideas_reviewed = await self._review_ideas_stage()
            cycle_results['ideas_reviewed'] = ideas_reviewed
            self._broadcast_stage_update('idea_review', {'status': 'completed', 'count': ideas_reviewed})
            
            # Stage 2.5: Process ideas that need revision
            self._broadcast_stage_update('idea_revision', {'status': 'starting'})
            ideas_revised = await self._revision_stage()
            cycle_results['ideas_revised'] = ideas_revised
            self._broadcast_stage_update('idea_revision', {'status': 'completed', 'count': ideas_revised})
            
            # Stage 3: Design experiments for approved ideas
            self._broadcast_stage_update('experiment_design', {'status': 'starting'})
            experiments_designed = await self._design_experiments_stage()
            cycle_results['experiments_designed'] = experiments_designed
            self._broadcast_stage_update('experiment_design', {'status': 'completed', 'count': experiments_designed})
            
            # Stage 4: Ready check for preparing projects
            self._broadcast_stage_update('ready_check', {'status': 'starting'})
            ready_projects = await self._ready_check_stage()
            cycle_results['ready_projects'] = ready_projects
            self._broadcast_stage_update('ready_check', {'status': 'completed', 'count': ready_projects})
            
            # Stage 5: Execute ready experiments
            self._broadcast_stage_update('experiment_execution', {'status': 'starting'})
            experiments_executed = await self._execute_experiments_stage()
            cycle_results['experiments_executed'] = experiments_executed
            self._broadcast_stage_update('experiment_execution', {'status': 'completed', 'count': experiments_executed})
            
            # Stage 6: Peer review executed projects
            self._broadcast_stage_update('peer_review', {'status': 'starting'})
            peer_reviews_completed = await self._peer_review_stage()
            cycle_results['peer_reviews_completed'] = peer_reviews_completed
            self._broadcast_stage_update('peer_review', {'status': 'completed', 'count': peer_reviews_completed})
            
            # Stage 7: Generate reports for reviewed projects
            self._broadcast_stage_update('report_generation', {'status': 'starting'})
            reports_generated = await self._report_generation_stage()
            cycle_results['reports_generated'] = reports_generated
            self._broadcast_stage_update('report_generation', {'status': 'completed', 'count': reports_generated})
            
            # Stage 8: Archive completed work
            self._broadcast_stage_update('archiving', {'status': 'starting'})
            ideas_completed = await self._archive_completed_stage()
            cycle_results['ideas_completed'] = ideas_completed
            self._broadcast_stage_update('archiving', {'status': 'completed', 'count': ideas_completed})
            
            # Stage 9: Lab technician analysis (periodic)
            lab_tech_results = await self._lab_technician_stage()
            cycle_results['lab_technician_analysis'] = lab_tech_results
            
        except Exception as e:
            self.logger.error(f"Cycle execution error: {e}")
            cycle_results['errors'].append(str(e))
            self._broadcast_stage_update('error', {'status': 'error', 'message': str(e)})
        
        return cycle_results
    
    async def _generate_ideas_stage(self, inputs: Dict[str, Any]) -> int:
        """Generate new ideas only when pipeline needs more to stay active."""
        
        # Check current pipeline state
        ideas_df = self.registry_manager.load_registry('ideas_register')
        projects_df = self.registry_manager.load_registry('project_index')
        
        # Count ideas in each stage
        proposed_ideas = len(ideas_df[ideas_df['status'] == 'Proposed'])
        under_review = len(ideas_df[ideas_df['status'] == 'Under Review'])
        needs_revision = len(ideas_df[ideas_df['status'] == 'Needs Revision'])
        approved_ideas = len(ideas_df[ideas_df['status'] == 'Approved'])
        
        # Count projects in progress
        existing_projects = set(projects_df['idea_id'].tolist()) if not projects_df.empty else set()
        approved_without_projects = len(ideas_df[
            (ideas_df['status'] == 'Approved') & 
            (~ideas_df['idea_id'].isin(existing_projects))
        ])
        
        ready_projects = len(projects_df[projects_df['maturity'] == 'Ready for Execution'])
        
        # Smart generation logic: only generate if we need more ideas in the pipeline
        # DON'T count revision ideas as active work - they need to be processed!
        total_active_work = proposed_ideas + under_review + approved_without_projects + ready_projects
        
        # Pipeline capacity with maximum limits to prevent runaway generation
        min_pipeline_size = inputs.get('min_pipeline_size', 3)  
        max_total_ideas = inputs.get('max_total_ideas', 20)  # Hard limit on total ideas
        max_active_projects = inputs.get('max_active_projects', 10)  # Hard limit on projects
        
        total_ideas = len(ideas_df)
        total_projects = len(projects_df) if not projects_df.empty else 0
        
        self.logger.info(f"Pipeline status: {proposed_ideas} proposed, {under_review} reviewing, "
                        f"{needs_revision} needing revision, {approved_without_projects} awaiting experiments, {ready_projects} ready to execute")
        self.logger.info(f"Limits: {total_ideas}/{max_total_ideas} ideas, {total_projects}/{max_active_projects} projects")
        
        # SAFETY CHECKS - prevent runaway generation
        if total_ideas >= max_total_ideas:
            self.logger.warning(f"🛑 SAFETY: At idea limit ({total_ideas}/{max_total_ideas}). Skipping generation.")
            return 0
            
        if total_projects >= max_active_projects:
            self.logger.warning(f"🛑 SAFETY: At project limit ({total_projects}/{max_active_projects}). Skipping generation.")
            return 0
        
        if total_active_work < min_pipeline_size:
            # Only generate what we need, respecting limits
            needed = min(
                min_pipeline_size - total_active_work,
                max_total_ideas - total_ideas,  # Don't exceed idea limit
                3  # Never generate more than 3 at once
            )
            
            if needed <= 0:
                self.logger.info("Pipeline full or at limits. No new ideas needed.")
                return 0
            
            # Check for recent similar titles to prevent duplicates
            if self._has_recent_similar_ideas(ideas_df, inputs.get('domain_tags', [])):
                self.logger.warning("🛑 DUPLICATE PREVENTION: Similar ideas generated recently. Skipping.")
                return 0
            
            self.logger.info(f"Pipeline needs more work. Generating {needed} new ideas to maintain flow")
            
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
                        'generate_ambitious': True  # Flag for ambitious hypothesis generation
                    }
                )
                
                result = hypothesis_maker.run(context)
                
                if result.success:
                    hypotheses = result.output_data.get('hypotheses', [])
                    
                    # Add to registry
                    for hypothesis in hypotheses:
                        self.registry_manager.append_to_registry('ideas_register', hypothesis)
                    
                    self.logger.info(f"Generated {len(hypotheses)} new hypotheses")
                    return len(hypotheses)
                
            except Exception as e:
                self.logger.error(f"Idea generation failed: {e}")
        else:
            self.logger.info(f"Pipeline has sufficient active work ({total_active_work} items). "
                           "Focusing on advancing existing ideas.")
        
        return 0
    
    async def _review_ideas_stage(self) -> int:
        """Review pending ideas."""
        
        # Get ideas needing review
        ideas_df = self.registry_manager.load_registry('ideas_register')
        pending_review = ideas_df[ideas_df['status'].isin(['Proposed', 'Under Review'])]
        
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
    
    async def _revision_stage(self) -> int:
        """Process ideas that need revision by calling hypothesis maker."""
        
        # Get ideas needing revision
        ideas_df = self.registry_manager.load_registry('ideas_register')
        revision_ideas = ideas_df[ideas_df['status'] == 'Needs Revision']
        
        if len(revision_ideas) == 0:
            return 0
        
        self.logger.info(f"🔄 Processing {len(revision_ideas)} ideas that need revision")
        
        try:
            # Create hypothesis maker for revision processing
            hypothesis_maker = create_agent('hypothesis_maker', str(self.config_dir))
            
            # Call hypothesis maker specifically for revision work
            from ..agents.common import AgentExecutionContext
            context = AgentExecutionContext(
                agent_name='hypothesis_maker',
                state_name='revision_processing',
                input_data={
                    'generate_new': False,  # Only process revisions, don't generate new
                    'focus_on_revision': True
                }
            )
            
            result = hypothesis_maker.run(context)
            
            if result.success:
                revised_hypotheses = result.output_data.get('hypotheses', [])
                
                # Add revised hypotheses to registry
                for revised_hypothesis in revised_hypotheses:
                    self.registry_manager.append_to_registry('ideas_register', revised_hypothesis)
                
                self.logger.info(f"✅ Successfully revised {len(revised_hypotheses)} ideas")
                return len(revised_hypotheses)
            else:
                self.logger.error(f"Revision processing failed: {result.error_message}")
                return 0
            
        except Exception as e:
            self.logger.error(f"Revision stage failed: {e}")
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
                    # Apply any registry updates returned by the agent
                    if hasattr(result, 'registry_updates') and result.registry_updates:
                        from .graph import apply_registry_updates
                        apply_registry_updates(result.registry_updates, self.registry_manager)
                    designed_count += 1
            
            return designed_count
            
        except Exception as e:
            self.logger.error(f"Experiment design stage failed: {e}")
            return 0
    
    async def _execute_experiments_stage(self) -> int:
        """Execute ready experiments using the real experimenter agent."""
        
        # Get ready projects
        projects_df = self.registry_manager.load_registry('project_index')
        ready_projects = projects_df[projects_df['maturity'] == 'Ready']
        
        if len(ready_projects) == 0:
            return 0
        
        self.logger.info(f"Executing {len(ready_projects)} experiments")
        
        try:
            # Create experimenter agent
            experimenter = create_agent('experimenter', str(self.config_dir))
            
            executed_count = 0
            for _, project_row in ready_projects.iterrows():
                project = project_row.to_dict()
                
                self.logger.info(f"Running real experiment for project {project['idea_id']}")
                
                # Execute real experiment
                from ..agents.common import AgentExecutionContext
                context = AgentExecutionContext(
                    agent_name='experimenter',
                    state_name='continuous_execution',
                    input_data={'idea_id': project['idea_id']}
                )
                
                result = experimenter.run(context)
                
                if result.success:
                    # Update project status to executed (ready for peer review)
                    # Note: Executed projects stay in "Ready for Execution" folder for peer review
                    self.registry_manager.update_registry_row(
                        'project_index',
                        {'idea_id': project['idea_id']},
                        {
                            'maturity': 'Executed', 
                            'execution_end': datetime.now(timezone.utc).isoformat(),
                            'compute_hours_used': result.output_data.get('compute_hours_used', 0.0),
                            'storage_gb_used': result.output_data.get('storage_gb_used', 0.0)
                        }
                    )
                    executed_count += 1
                    self.logger.info(f"Successfully executed experiment for {project['idea_id']}")
                else:
                    # Move failed projects back to Preparing folder
                    from ..agents.common import move_project_folder
                    old_path = project.get('path', '')
                    new_path = move_project_folder(
                        project['idea_id'], 
                        project.get('slug', ''), 
                        'Ready',    # from maturity
                        'Failed'    # to maturity  
                    )
                    
                    # Mark as failed and update path
                    self.registry_manager.update_registry_row(
                        'project_index',
                        {'idea_id': project['idea_id']},
                        {
                            'maturity': 'Failed', 
                            'path': new_path,
                            'execution_end': datetime.now(timezone.utc).isoformat()
                        }
                    )
                    self.logger.warning(f"Experiment failed for {project['idea_id']}: {result.error_message}")
                    self.logger.info(f"Moved failed project back to Preparing/ for rework")
            
            return executed_count
            
        except Exception as e:
            self.logger.error(f"Experiment execution stage failed: {e}")
            return 0
    
    def _has_recent_similar_ideas(self, ideas_df, domain_tags):
        """Check if similar ideas were generated recently - but be revision-aware."""
        if ideas_df.empty:
            return False
            
        try:
            # Check existing pipeline state first - this is the key insight!
            proposed_count = len(ideas_df[ideas_df['status'] == 'Proposed'])
            under_review_count = len(ideas_df[ideas_df['status'] == 'Under Review'])
            needs_revision_count = len(ideas_df[ideas_df['status'] == 'Needs Revision'])
            approved_count = len(ideas_df[ideas_df['status'] == 'Approved'])
            
            self.logger.info(f"Workflow state: {proposed_count} proposed, {under_review_count} under review, "
                           f"{needs_revision_count} need revision, {approved_count} approved")
            
            # PRIORITY: If we have ideas needing revision, allow new generation since revisions aren't processed automatically
            if needs_revision_count > 0:
                self.logger.info(f"🔄 Found {needs_revision_count} ideas needing revision - will process revisions alongside new generation")
                # DON'T block - let the hypothesis maker handle revisions in its normal workflow
            
            # If we have sufficient work in pipeline, don't generate more
            total_pending_work = proposed_count + under_review_count + approved_count
            if total_pending_work >= 4:  # Enough work in pipeline
                self.logger.info(f"📋 Found {total_pending_work} ideas in workflow - sufficient pipeline work")
                return True  # Block to focus on advancing existing ideas
            
            # Get recent ideas (last 12 hours - shorter window)
            from datetime import datetime, timedelta, timezone
            import pandas as pd
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=12)  # Reduced from 24h
            
            # Filter recent ideas
            ideas_df['created_at_dt'] = pd.to_datetime(ideas_df['created_at'])
            recent_ideas = ideas_df[ideas_df['created_at_dt'] > cutoff_time]
            
            # Only block if excessive recent generation (higher threshold)
            if len(recent_ideas) > 10:  # Increased from 5 to 10
                self.logger.warning(f"Generated {len(recent_ideas)} ideas in last 12h - likely excessive")
                return True
                        
        except Exception as e:
            self.logger.warning(f"Error checking workflow state: {e}")
            return False  # If we can't check, allow generation
            
        return False  # Default to allowing generation
    
    async def _ready_check_stage(self) -> int:
        """Check projects in Preparing status and promote to Ready if requirements met."""
        
        # Get projects needing ready check (both Prepared and Failed projects that are ready for retry)
        projects_df = self.registry_manager.load_registry('project_index')
        preparing_projects = projects_df[
            (projects_df['maturity'] == 'Prepared') | 
            (projects_df['maturity'] == 'Failed')
        ]
        
        if len(preparing_projects) == 0:
            return 0
        
        self.logger.info(f"Checking readiness for {len(preparing_projects)} projects")
        
        ready_count = 0
        for _, project_row in preparing_projects.iterrows():
            project = project_row.to_dict()
            
            # Check if project is ready for execution
            ready_checks_passed = project.get('ready_checklist_passed', False)
            data_requirements_met = project.get('data_requirements_met', False)  
            analysis_plan_ready = project.get('analysis_plan_preregistered', False)
            
            if ready_checks_passed and data_requirements_met and analysis_plan_ready:
                # Move project folder from Preparing to Ready for Execution
                from ..agents.common import move_project_folder
                from datetime import datetime, timezone
                
                old_path = project.get('path', '')
                new_path = move_project_folder(
                    project['idea_id'], 
                    project.get('slug', ''), 
                    'Prepared',  # from maturity
                    'Ready'      # to maturity
                )
                
                # Promote to Ready status and update path
                self.registry_manager.update_registry_row(
                    'project_index',
                    {'idea_id': project['idea_id']},
                    {
                        'maturity': 'Ready', 
                        'path': new_path,
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }
                )
                self.logger.info(f"Promoted project {project['idea_id'][:8]}... to Ready and moved to Ready for Execution/")
                ready_count += 1
            else:
                self.logger.info(f"Project {project['idea_id'][:8]}... not ready - checks: {ready_checks_passed}, data: {data_requirements_met}, plan: {analysis_plan_ready}")
        
        return ready_count
    
    async def _peer_review_stage(self) -> int:
        """Peer review executed projects."""
        
        # Get executed projects needing peer review
        projects_df = self.registry_manager.load_registry('project_index')
        executed_projects = projects_df[projects_df['maturity'] == 'Executed']
        
        if len(executed_projects) == 0:
            return 0
        
        self.logger.info(f"Peer reviewing {len(executed_projects)} executed projects")
        
        try:
            # Create peer reviewer agent
            peer_reviewer = create_agent('peer_reviewer', str(self.config_dir))
            
            reviewed_count = 0
            for _, project_row in executed_projects.iterrows():
                project = project_row.to_dict()
                
                self.logger.info(f"Peer reviewing project {project['idea_id']}")
                
                # Execute peer review
                from ..agents.common import AgentExecutionContext
                context = AgentExecutionContext(
                    agent_name='peer_reviewer',
                    state_name='continuous_peer_review',
                    input_data={'idea_id': project['idea_id']}
                )
                
                result = peer_reviewer.run(context)
                
                if result.success:
                    # Apply registry updates from peer reviewer
                    if hasattr(result, 'registry_updates') and result.registry_updates:
                        from .graph import apply_registry_updates
                        apply_registry_updates(result.registry_updates, self.registry_manager)
                    
                    # Check if approved for next stage
                    approved = result.output_data.get('approved', False)
                    if approved:
                        # Update to Reviewed status for Reporter
                        self.registry_manager.update_registry_row(
                            'project_index',
                            {'idea_id': project['idea_id']},
                            {'maturity': 'Reviewed', 'reviewed_at': datetime.now(timezone.utc).isoformat()}
                        )
                        self.logger.info(f"Project {project['idea_id']} approved by peer review")
                    else:
                        # Send back to experimenter for revisions
                        self.registry_manager.update_registry_row(
                            'project_index',
                            {'idea_id': project['idea_id']},
                            {'maturity': 'Ready', 'review_feedback': result.output_data.get('feedback', '')}
                        )
                        self.logger.info(f"Project {project['idea_id']} needs revisions")
                    
                    reviewed_count += 1
                else:
                    self.logger.warning(f"Peer review failed for {project['idea_id']}: {result.error_message}")
            
            return reviewed_count
            
        except Exception as e:
            self.logger.error(f"Peer review stage failed: {e}")
            return 0
    
    async def _report_generation_stage(self) -> int:
        """Generate final reports for reviewed projects."""
        
        # Get reviewed projects needing final reports
        projects_df = self.registry_manager.load_registry('project_index')
        reviewed_projects = projects_df[projects_df['maturity'] == 'Reviewed']
        
        if len(reviewed_projects) == 0:
            return 0
        
        self.logger.info(f"Generating reports for {len(reviewed_projects)} reviewed projects")
        
        try:
            # Create reporter agent
            reporter = create_agent('reporter', str(self.config_dir))
            
            report_count = 0
            for _, project_row in reviewed_projects.iterrows():
                project = project_row.to_dict()
                
                self.logger.info(f"Generating final report for project {project['idea_id']}")
                
                # Execute reporter
                from ..agents.common import AgentExecutionContext
                context = AgentExecutionContext(
                    agent_name='reporter',
                    state_name='continuous_reporting',
                    input_data={'idea_id': project['idea_id']}
                )
                
                result = reporter.run(context)
                
                if result.success:
                    # Apply registry updates from reporter
                    if hasattr(result, 'registry_updates') and result.registry_updates:
                        from .graph import apply_registry_updates
                        apply_registry_updates(result.registry_updates, self.registry_manager)
                    
                    # Update to Complete status (project moves to Library)
                    library_path = result.output_data.get('library_path', '')
                    self.registry_manager.update_registry_row(
                        'project_index',
                        {'idea_id': project['idea_id']},
                        {
                            'maturity': 'Complete', 
                            'path': library_path,
                            'completed_at': datetime.now(timezone.utc).isoformat()
                        }
                    )
                    
                    self.logger.info(f"Project {project['idea_id']} completed and moved to Library")
                    report_count += 1
                else:
                    self.logger.warning(f"Report generation failed for {project['idea_id']}: {result.error_message}")
            
            return report_count
            
        except Exception as e:
            self.logger.error(f"Report generation stage failed: {e}")
            return 0
    
    async def _archive_completed_stage(self) -> int:
        """Archive completed projects that have been moved to Library folder."""
        
        # Get completed projects (those that have finished the full pipeline AND been moved to Library)
        projects_df = self.registry_manager.load_registry('project_index')
        completed_projects = projects_df[
            (projects_df['maturity'] == 'Complete') & 
            (projects_df['path'].str.contains('Library', na=False))  # Only count if moved to Library
        ]
        
        if len(completed_projects) == 0:
            return 0
        
        # Check if already archived to avoid double-counting
        existing_completed_df = self.registry_manager.load_registry('completed_index')
        existing_ids = set(existing_completed_df['idea_id'].tolist()) if not existing_completed_df.empty else set()
        
        # Move to completed registry (only new ones)
        completed_count = 0
        for _, project_row in completed_projects.iterrows():
            project = project_row.to_dict()
            
            # Skip if already archived
            if project['idea_id'] in existing_ids:
                continue
            
            # Add to completed registry
            completed_entry = {
                'idea_id': project['idea_id'],
                'title': f"Project {project['idea_id']}",
                'moved_to_library_at': datetime.now(timezone.utc).isoformat(),
                'project_path': project['path'],
                'has_paper': True if 'paper.html' in str(project.get('path', '')) else False
            }
            
            self.registry_manager.append_to_registry('completed_index', completed_entry)
            completed_count += 1
        
        self.logger.info(f"Archived {completed_count} completed ideas")
        return completed_count
    
    def _get_current_completed_count(self) -> int:
        """Get the current total count of completed projects that have been fully archived."""
        try:
            # Only count projects that have been moved to the Library folder (completed_index)
            completed_df = self.registry_manager.load_registry('completed_index')
            return len(completed_df)
        except Exception as e:
            self.logger.error(f"Error getting completed count: {e}")
            return 0
    
    async def _check_completion_criteria(self, 
                                       mode: str, 
                                       target: int, 
                                       max_duration: Optional[int]) -> bool:
        """Check if completion criteria are met."""
        
        # Check time limit (if specified)
        if max_duration:
            runtime_minutes = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
            if runtime_minutes >= max_duration:
                self.logger.info(f"⏰ Time limit reached: {runtime_minutes:.1f} minutes")
                self.logger.info("🏆 Auto-pausing pipeline due to time limit")
                self.pause()
                return True
        
        # Check mode-specific criteria
        if mode == "ideas":
            if self.completed_ideas >= target:
                self.logger.info(f"🎯 Goal achieved! Completed {self.completed_ideas}/{target} full research projects (moved to Library)")
                self.logger.info("🏆 Auto-pausing pipeline after successful completion")
                self.pause()
                return True
        elif mode == "time":
            runtime_minutes = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
            if runtime_minutes >= target:
                self.logger.info(f"⏰ Time target reached: {runtime_minutes:.1f} minutes")
                self.logger.info("🏆 Auto-pausing pipeline after time completion")
                self.pause()
                return True
        
        return False
    
    async def _send_status_update(self):
        """Send comprehensive status update to callbacks."""
        runtime = datetime.now(timezone.utc) - self.start_time
        
        # Get current pipeline statistics
        ideas_df = self.registry_manager.load_registry('ideas_register')
        projects_df = self.registry_manager.load_registry('project_index')
        
        # Calculate stage distribution
        stage_summary = {
            'proposed': len(ideas_df[ideas_df['status'] == 'Proposed']) if not ideas_df.empty else 0,
            'under_review': len(ideas_df[ideas_df['status'] == 'Under Review']) if not ideas_df.empty else 0,
            'needs_revision': len(ideas_df[ideas_df['status'] == 'Needs Revision']) if not ideas_df.empty else 0,
            'approved': len(ideas_df[ideas_df['status'] == 'Approved']) if not ideas_df.empty else 0,
            'projects_preparing': len(projects_df[projects_df['maturity'] == 'Prepared']) if not projects_df.empty else 0,
            'projects_ready': len(projects_df[projects_df['maturity'] == 'Ready']) if not projects_df.empty else 0,
            'projects_executing': len(projects_df[projects_df['maturity'] == 'Executed']) if not projects_df.empty else 0,
            'projects_reviewing': len(projects_df[projects_df['maturity'] == 'Reviewed']) if not projects_df.empty else 0
        }
        
        status = {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'runtime_seconds': runtime.total_seconds(),
            'completed_ideas': self.completed_ideas,
            'total_cycles': self.total_cycles,
            'stage_summary': stage_summary,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self._broadcast_status(status)
    
    async def _lab_technician_stage(self) -> Dict[str, Any]:
        """Execute lab technician analysis and improvement cycle (periodic)."""
        
        # Check if lab technician is enabled
        if not self.lab_technician_enabled:
            return {'status': 'disabled'}
        
        # Check if it's time to run lab technician
        current_time = datetime.now(timezone.utc)
        
        if self.last_lab_technician_run:
            time_since_last = (current_time - self.last_lab_technician_run).total_seconds()
            if time_since_last < self.lab_technician_interval:
                return {
                    'status': 'skipped',
                    'reason': f'Run too recent: {time_since_last:.0f}s ago, interval: {self.lab_technician_interval}s'
                }
        
        self.logger.info("🔧 Running Lab Technician analysis...")
        self._broadcast_stage_update('lab_technician', {'status': 'starting'})
        
        try:
            # Create lab technician agent
            lab_technician = create_agent('lab_technician', str(self.config_dir))
            
            # Create execution context
            from ..agents.common import AgentExecutionContext
            context = AgentExecutionContext(
                agent_name='lab_technician',
                state_name='continuous_improvement',
                input_data={
                    'analysis_trigger': 'periodic',
                    'pipeline_status': self.get_status()
                }
            )
            
            # Execute lab technician analysis
            result = lab_technician.run(context)
            
            # Update last run time
            self.last_lab_technician_run = current_time
            
            if result.success:
                analysis_results = result.output_data.get('analysis_results', {})
                improvements_made = result.output_data.get('improvements_made', [])
                
                # Log key findings
                if analysis_results.get('execution_analysis'):
                    exec_stats = analysis_results['execution_analysis']
                    success_rate = exec_stats.get('success_rate', 0)
                    self.logger.info(f"   📊 Experimenter success rate: {success_rate:.1%}")
                    
                    if success_rate < 0.8:
                        self.logger.warning(f"   ⚠️  Low success rate detected: {success_rate:.1%}")
                
                # Log improvements made
                if improvements_made:
                    successful_improvements = [imp for imp in improvements_made if imp.get('status') == 'success']
                    self.logger.info(f"   🛠️  Applied {len(successful_improvements)} code improvements")
                    
                    for improvement in successful_improvements:
                        improvement_type = improvement['opportunity'].get('type', 'unknown')
                        self.logger.info(f"      - Fixed: {improvement_type}")
                else:
                    self.logger.info("   ✅ No improvements needed - system performing well")
                
                self._broadcast_stage_update('lab_technician', {
                    'status': 'completed',
                    'improvements_made': len(improvements_made),
                    'success_rate': analysis_results.get('execution_analysis', {}).get('success_rate', 'N/A')
                })
                
                return {
                    'status': 'completed',
                    'execution_time': result.execution_time_seconds,
                    'analysis_results': analysis_results,
                    'improvements_made': improvements_made,
                    'analysis_report': result.output_data.get('report', 'No report generated')
                }
            else:
                self.logger.error(f"   ❌ Lab Technician analysis failed: {result.error_message}")
                self._broadcast_stage_update('lab_technician', {
                    'status': 'error',
                    'error': result.error_message
                })
                
                return {
                    'status': 'error',
                    'error_message': result.error_message,
                    'error_type': result.error_type
                }
                
        except Exception as e:
            self.logger.error(f"Lab Technician stage failed: {str(e)}")
            self._broadcast_stage_update('lab_technician', {
                'status': 'error',
                'error': str(e)
            })
            
            return {
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__
            }
    
    def pause(self):
        """Pause the continuous pipeline."""
        self.logger.info("⏸️  Pausing continuous pipeline...")
        self.is_paused = True
    
    def resume(self):
        """Resume the continuous pipeline."""
        self.logger.info("▶️  Resuming continuous pipeline...")
        self.is_paused = False
    
    def stop(self):
        """Stop the continuous pipeline."""
        self.logger.info("🛑 Stopping continuous pipeline...")
        self.should_stop = True
        self.is_running = False
        self.is_paused = False
    
    def enable_lab_technician(self):
        """Enable the lab technician for continuous improvement."""
        self.logger.info("🔧 Enabling Lab Technician continuous improvement")
        self.lab_technician_enabled = True
    
    def disable_lab_technician(self):
        """Disable the lab technician."""
        self.logger.info("🔧 Disabling Lab Technician continuous improvement")
        self.lab_technician_enabled = False
    
    def set_lab_technician_interval(self, interval_seconds: int):
        """Set the lab technician execution interval.
        
        Args:
            interval_seconds: Time between lab technician runs in seconds
        """
        self.lab_technician_interval = interval_seconds
        self.logger.info(f"🔧 Set Lab Technician interval to {interval_seconds} seconds")
    
    async def run_lab_technician_now(self) -> Dict[str, Any]:
        """Force run lab technician analysis immediately.
        
        Returns:
            Lab technician analysis results
        """
        self.logger.info("🔧 Force running Lab Technician analysis...")
        
        # Temporarily reset last run time to force execution
        original_last_run = self.last_lab_technician_run
        self.last_lab_technician_run = None
        
        try:
            result = await self._lab_technician_stage()
            return result
        finally:
            # Restore original last run time if the analysis failed
            if not result.get('status') == 'completed':
                self.last_lab_technician_run = original_last_run
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        if not self.start_time:
            return {
                'is_running': False,
                'is_paused': False,
                'status': 'idle'
            }
        
        runtime = datetime.now(timezone.utc) - self.start_time
        
        if self.should_stop or not self.is_running:
            status = 'stopped'
        elif self.is_paused:
            status = 'paused'
        elif self.is_running:
            status = 'running'
        else:
            status = 'idle'
        
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'should_stop': self.should_stop,
            'status': status,
            'runtime_seconds': runtime.total_seconds(),
            'completed_ideas': self.completed_ideas,
            'total_cycles': self.total_cycles,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
