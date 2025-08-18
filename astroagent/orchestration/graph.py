"""
AstroAgent Pipeline Orchestration System.

Implements the research pipeline state machine using LangGraph for
coordinated agent execution and state management.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..agents import (
    create_agent, 
    AgentExecutionContext, 
    AgentResult,
    AGENT_REGISTRY
)
from .tools import RegistryManager, StateValidator
from .registry import ProjectRegistry


# ============================================================================
# State Definition
# ============================================================================

class PipelineState(TypedDict):
    """State object passed between agents in the pipeline."""
    
    # Current processing context
    current_idea_id: Optional[str]
    current_state: str
    previous_state: Optional[str]
    
    # Agent inputs and outputs
    agent_inputs: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    
    # Pipeline metadata
    pipeline_id: str
    execution_start: datetime
    total_agents_run: int
    
    # Error handling
    errors: List[Dict[str, Any]]
    retry_count: int
    
    # Registry state
    ideas_updated: List[str]
    projects_updated: List[str]


# ============================================================================
# Utility Functions 
# ============================================================================

def apply_registry_updates(registry_updates: List[Dict[str, Any]], registry_manager: RegistryManager):
    """Apply registry updates from agent results."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    
    for update in registry_updates:
        try:
            registry_name = update.get('registry')
            action = update.get('action', 'update')
            filter_criteria = update.get('filter', {})
            data = update.get('data', {})
            
            if action == 'update':
                registry_manager.update_registry_row(registry_name, filter_criteria, data)
                logger.info(f"Applied registry update: {registry_name}")
            elif action == 'append':
                registry_manager.append_to_registry(registry_name, data)
                logger.info(f"Applied registry append: {registry_name}")
            else:
                logger.warning(f"Unknown registry action: {action}")
                
        except Exception as e:
            logger.error(f"Failed to apply registry update: {e}")


# ============================================================================
# Agent Node Functions
# ============================================================================

def hypothesis_maker_node(state: PipelineState) -> PipelineState:
    """Execute Hypothesis Maker agent."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.info("Executing Hypothesis Maker")
    
    try:
        # Get config directory from state or use default
        config_dir = state.get('config_dir', 'astroagent/config')
        
        # Create agent
        agent = create_agent('hypothesis_maker', config_dir)
        
        # Prepare context
        context = AgentExecutionContext(
            agent_name='hypothesis_maker',
            state_name='hypothesis_generation',
            previous_state=state.get('previous_state'),
            input_data=state.get('agent_inputs', {}),
            retry_count=state.get('retry_count', 0)
        )
        
        # Execute agent
        result = agent.run(context)
        
        # Update state
        state['agent_outputs']['hypothesis_maker'] = result.output_data
        state['total_agents_run'] += 1
        state['previous_state'] = state['current_state']
        
        if result.success:
            state['current_state'] = 'initial_review'
            
            # Track updated ideas
            hypotheses = result.output_data.get('hypotheses', [])
            for hypothesis in hypotheses:
                state['ideas_updated'].append(hypothesis['idea_id'])
        else:
            state['current_state'] = 'error'
            state['errors'].append({
                'agent': 'hypothesis_maker',
                'error': result.error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return state
        
    except Exception as e:
        logger.error(f"Hypothesis Maker execution failed: {str(e)}")
        state['current_state'] = 'error'
        state['errors'].append({
            'agent': 'hypothesis_maker',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        return state


def reviewer_node(state: PipelineState) -> PipelineState:
    """Execute Reviewer agent."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.info("Executing Reviewer")
    
    try:
        # Get config directory from state or use default
        config_dir = state.get('config_dir', 'astroagent/config')
        
        # Create agent
        agent = create_agent('reviewer', config_dir)
        
        # Prepare context
        context = AgentExecutionContext(
            agent_name='reviewer',
            state_name='initial_review',
            previous_state=state.get('previous_state'),
            input_data=state.get('agent_inputs', {}),
            retry_count=state.get('retry_count', 0)
        )
        
        # Execute agent
        result = agent.run(context)
        
        # Update state
        state['agent_outputs']['reviewer'] = result.output_data
        state['total_agents_run'] += 1
        state['previous_state'] = state['current_state']
        
        if result.success:
            # Determine next state based on review results
            reviewed_ideas = result.output_data.get('reviewed_ideas', [])
            approved_count = sum(1 for idea in reviewed_ideas if idea.get('status') == 'Approved')
            
            if approved_count > 0:
                state['current_state'] = 'experiment_design'
            else:
                state['current_state'] = 'archive'  # No ideas approved
        else:
            state['current_state'] = 'error'
            state['errors'].append({
                'agent': 'reviewer',
                'error': result.error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return state
        
    except Exception as e:
        logger.error(f"Reviewer execution failed: {str(e)}")
        state['current_state'] = 'error'
        state['errors'].append({
            'agent': 'reviewer',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        return state


def experiment_designer_node(state: PipelineState) -> PipelineState:
    """Execute Experiment Designer agent."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.info("Executing Experiment Designer")
    
    try:
        # Get config directory from state or use default
        config_dir = state.get('config_dir', 'astroagent/config')
        
        # Create agent
        agent = create_agent('experiment_designer', config_dir)
        
        # Get approved idea to design experiment for
        idea_id = state.get('current_idea_id')
        
        context = AgentExecutionContext(
            agent_name='experiment_designer',
            state_name='experiment_design',
            previous_state=state.get('previous_state'),
            input_data={'idea_id': idea_id} if idea_id else {},
            retry_count=state.get('retry_count', 0)
        )
        
        # Execute agent
        result = agent.run(context)
        
        # Update state
        state['agent_outputs']['experiment_designer'] = result.output_data
        state['total_agents_run'] += 1
        state['previous_state'] = state['current_state']
        
        if result.success:
            ready_checks_passed = result.output_data.get('ready_checks_passed', False)
            
            if ready_checks_passed:
                state['current_state'] = 'experiment_execution'
            else:
                state['current_state'] = 'ready_check_failed'
        else:
            state['current_state'] = 'error'
            state['errors'].append({
                'agent': 'experiment_designer',
                'error': result.error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return state
        
    except Exception as e:
        logger.error(f"Experiment Designer execution failed: {str(e)}")
        state['current_state'] = 'error'
        state['errors'].append({
            'agent': 'experiment_designer',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        return state


def experimenter_node(state: PipelineState) -> PipelineState:
    """Execute Experimenter agent."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.info("Executing Experimenter")
    
    try:
        # Get config directory from state or use default
        config_dir = state.get('config_dir', 'astroagent/config')
        
        # Create agent
        agent = create_agent('experimenter', config_dir)
        
        # Get ready project to execute
        idea_id = state.get('current_idea_id')
        
        context = AgentExecutionContext(
            agent_name='experimenter',
            state_name='experiment_execution',
            previous_state=state.get('previous_state'),
            input_data={'idea_id': idea_id} if idea_id else {},
            retry_count=state.get('retry_count', 0)
        )
        
        # Execute agent
        result = agent.run(context)
        
        # Apply registry updates from agent result
        if result.success and result.registry_updates:
            config_dir = state.get('config_dir', 'astroagent/config')
            # Create registry manager instance (we need the data_dir from config)
            registry_manager = RegistryManager(data_dir="data")
            apply_registry_updates(result.registry_updates, registry_manager)
        
        # Update state
        state['agent_outputs']['experimenter'] = result.output_data
        state['total_agents_run'] += 1
        state['previous_state'] = state['current_state']
        
        if result.success:
            # Move to peer review after successful experiment
            state['current_state'] = 'peer_review'
        else:
            state['current_state'] = 'error'
            state['errors'].append({
                'agent': 'experimenter',
                'error': result.error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return state
        
    except Exception as e:
        logger.error(f"Experimenter execution failed: {str(e)}")
        state['current_state'] = 'error'
        state['errors'].append({
            'agent': 'experimenter',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        return state


def peer_reviewer_node(state: PipelineState) -> PipelineState:
    """Execute Peer Reviewer agent."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.info("Executing Peer Reviewer")
    
    try:
        # Get config directory from state or use default
        config_dir = state.get('config_dir', 'astroagent/config')
        
        # Create agent
        agent = create_agent('peer_reviewer', config_dir)
        
        # Get project to review
        idea_id = state.get('current_idea_id')
        
        context = AgentExecutionContext(
            agent_name='peer_reviewer',
            state_name='peer_review',
            previous_state=state.get('previous_state'),
            input_data={'idea_id': idea_id} if idea_id else {},
            retry_count=state.get('retry_count', 0)
        )
        
        # Execute agent
        result = agent.run(context)
        
        # Apply registry updates from agent result
        if result.success and result.registry_updates:
            registry_manager = RegistryManager(data_dir="data")
            apply_registry_updates(result.registry_updates, registry_manager)
        
        # Update state
        state['agent_outputs']['peer_reviewer'] = result.output_data
        state['total_agents_run'] += 1
        state['previous_state'] = state['current_state']
        
        if result.success:
            # Check if peer review approved the work
            approved = result.output_data.get('approved', False)
            if approved:
                state['current_state'] = 'report_generation'
            else:
                # Changes requested - back to experimenter
                state['current_state'] = 'experiment_execution'
        else:
            state['current_state'] = 'error'
            state['errors'].append({
                'agent': 'peer_reviewer',
                'error': result.error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return state
        
    except Exception as e:
        logger.error(f"Peer Reviewer execution failed: {str(e)}")
        state['current_state'] = 'error'
        state['errors'].append({
            'agent': 'peer_reviewer',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        return state


def reporter_node(state: PipelineState) -> PipelineState:
    """Execute Reporter agent."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.info("Executing Reporter")
    
    try:
        # Get config directory from state or use default
        config_dir = state.get('config_dir', 'astroagent/config')
        
        # Create agent
        agent = create_agent('reporter', config_dir)
        
        # Get approved project to report
        idea_id = state.get('current_idea_id')
        
        context = AgentExecutionContext(
            agent_name='reporter',
            state_name='report_generation',
            previous_state=state.get('previous_state'),
            input_data={'idea_id': idea_id} if idea_id else {},
            retry_count=state.get('retry_count', 0)
        )
        
        # Execute agent
        result = agent.run(context)
        
        # Apply registry updates from agent result
        if result.success and result.registry_updates:
            registry_manager = RegistryManager(data_dir="data")
            apply_registry_updates(result.registry_updates, registry_manager)
        
        # Update state
        state['agent_outputs']['reporter'] = result.output_data
        state['total_agents_run'] += 1
        state['previous_state'] = state['current_state']
        
        if result.success:
            # Project completed and moved to library
            state['current_state'] = 'library'
        else:
            state['current_state'] = 'error'
            state['errors'].append({
                'agent': 'reporter',
                'error': result.error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return state
        
    except Exception as e:
        logger.error(f"Reporter execution failed: {str(e)}")
        state['current_state'] = 'error'
        state['errors'].append({
            'agent': 'reporter',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        return state


def archive_node(state: PipelineState) -> PipelineState:
    """Archive rejected or completed work."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.info("Archiving work")
    
    # Mark pipeline as completed
    state['current_state'] = 'archived'
    
    return state


def library_node(state: PipelineState) -> PipelineState:
    """Final successful completion - project in library."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.info("Project completed and moved to library")
    
    # Mark pipeline as successfully completed
    state['current_state'] = 'completed'
    
    return state


def error_node(state: PipelineState) -> PipelineState:
    """Handle pipeline errors."""
    
    logger = logging.getLogger('astroagent.orchestration.graph')
    logger.error("Pipeline encountered errors")
    
    # Log all errors
    for error in state.get('errors', []):
        logger.error(f"Error in {error['agent']}: {error['error']}")
    
    state['current_state'] = 'failed'
    
    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_from_start(state: PipelineState) -> str:
    """Route from start state."""
    return 'hypothesis_maker'


def route_from_hypothesis_maker(state: PipelineState) -> str:
    """Route from hypothesis maker."""
    current_state = state.get('current_state', '')
    
    if current_state == 'initial_review':
        return 'reviewer'
    elif current_state == 'error':
        return 'error'
    else:
        return 'error'


def route_from_reviewer(state: PipelineState) -> str:
    """Route from reviewer."""
    current_state = state.get('current_state', '')
    
    if current_state == 'experiment_design':
        return 'experiment_designer'
    elif current_state == 'archive':
        return 'archive'
    elif current_state == 'error':
        return 'error'
    else:
        return 'error'


def route_from_experiment_designer(state: PipelineState) -> str:
    """Route from experiment designer."""
    current_state = state.get('current_state', '')
    
    if current_state == 'experiment_execution':
        return 'experimenter'  # Route to experimenter when ready
    elif current_state == 'ready_check_failed':
        return 'archive'  # Would normally loop back to design
    elif current_state == 'error':
        return 'error'
    else:
        return 'error'


def route_from_experimenter(state: PipelineState) -> str:
    """Route from experimenter."""
    current_state = state.get('current_state', '')
    
    if current_state == 'peer_review':
        return 'peer_reviewer'  # Move to peer review after successful experiment
    elif current_state == 'error':
        return 'error'
    
    else:
        return 'error'


def route_from_peer_reviewer(state: PipelineState) -> str:
    """Route from peer reviewer."""
    current_state = state.get('current_state', '')
    
    if current_state == 'report_generation':
        return 'reporter'  # Approved work goes to reporter
    elif current_state == 'experiment_execution':
        return 'experimenter'  # Changes requested - back to experimenter
    elif current_state == 'error':
        return 'error'
    
    else:
        return 'error'


def route_from_reporter(state: PipelineState) -> str:
    """Route from reporter."""
    current_state = state.get('current_state', '')
    
    if current_state == 'library':
        return 'library'  # Completed work goes to library
    elif current_state == 'error':
        return 'error'
    
    else:
        return 'error'


# ============================================================================
# Pipeline Graph Construction
# ============================================================================

def create_pipeline_graph() -> StateGraph:
    """Create the main pipeline state graph."""
    
    # Initialize graph
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("hypothesis_maker", hypothesis_maker_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("experiment_designer", experiment_designer_node)
    workflow.add_node("experimenter", experimenter_node)
    workflow.add_node("peer_reviewer", peer_reviewer_node)
    workflow.add_node("reporter", reporter_node)
    workflow.add_node("library", library_node)
    workflow.add_node("archive", archive_node)
    workflow.add_node("error", error_node)
    
    # Define entry point
    workflow.set_entry_point("hypothesis_maker")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "hypothesis_maker",
        route_from_hypothesis_maker,
        {
            "reviewer": "reviewer",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "reviewer", 
        route_from_reviewer,
        {
            "experiment_designer": "experiment_designer",
            "archive": "archive",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "experiment_designer",
        route_from_experiment_designer,
        {
            "experimenter": "experimenter",
            "archive": "archive",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "experimenter",
        route_from_experimenter,
        {
            "peer_reviewer": "peer_reviewer",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "peer_reviewer",
        route_from_peer_reviewer,
        {
            "reporter": "reporter",
            "experimenter": "experimenter",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "reporter",
        route_from_reporter,
        {
            "library": "library",
            "error": "error"
        }
    )
    
    # Terminal states
    workflow.add_edge("library", END)
    workflow.add_edge("archive", END)
    workflow.add_edge("error", END)
    
    return workflow


# ============================================================================
# Pipeline Execution
# ============================================================================

class AstroAgentPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_dir: str = "config", data_dir: str = "data"):
        """Initialize pipeline.
        
        Args:
            config_dir: Configuration directory path
            data_dir: Data directory path
        """
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        
        self.logger = self._setup_logger()
        
        # Initialize components
        self.registry_manager = RegistryManager(data_dir)
        self.state_validator = StateValidator()
        
        # Create graph
        self.workflow = create_pipeline_graph()
        
        # Setup checkpointing
        checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=checkpointer)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup pipeline logger."""
        logger = logging.getLogger('astroagent.orchestration')
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def create_initial_state(self, agent_inputs: Dict[str, Any]) -> PipelineState:
        """Create initial pipeline state.
        
        Args:
            agent_inputs: Inputs for the first agent
            
        Returns:
            Initial pipeline state
        """
        
        return PipelineState(
            current_idea_id=None,
            current_state='start',
            previous_state=None,
            agent_inputs=agent_inputs,
            agent_outputs={},
            pipeline_id=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            execution_start=datetime.now(timezone.utc),
            total_agents_run=0,
            errors=[],
            retry_count=0,
            ideas_updated=[],
            projects_updated=[],
            config_dir=str(self.config_dir)  # Add config directory to state
        )
    
    def run_pipeline(self, agent_inputs: Dict[str, Any], 
                    config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Args:
            agent_inputs: Initial inputs for the pipeline
            config: Optional configuration overrides
            
        Returns:
            Pipeline execution results
        """
        
        self.logger.info("Starting AstroAgent pipeline execution")
        
        # Create initial state
        initial_state = self.create_initial_state(agent_inputs)
        
        try:
            # Run the pipeline
            final_state = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": initial_state["pipeline_id"]}}
            )
            
            # Extract results
            results = {
                'success': final_state['current_state'] not in ['failed', 'error'],
                'final_state': final_state['current_state'],
                'pipeline_id': final_state['pipeline_id'],
                'agents_run': final_state['total_agents_run'],
                'ideas_updated': final_state['ideas_updated'],
                'projects_updated': final_state['projects_updated'],
                'agent_outputs': final_state['agent_outputs'],
                'errors': final_state['errors'],
                'execution_time': (datetime.now(timezone.utc) - final_state['execution_start']).total_seconds()
            }
            
            if results['success']:
                self.logger.info(f"Pipeline completed successfully in {results['execution_time']:.1f}s")
            else:
                self.logger.error(f"Pipeline failed after {results['execution_time']:.1f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pipeline_id': initial_state['pipeline_id']
            }


# ============================================================================
# CLI Interface  
# ============================================================================

def main():
    """Command line interface for pipeline execution."""
    
    parser = argparse.ArgumentParser(description='AstroAgent Pipeline Orchestrator')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a specific agent or full pipeline')
    run_parser.add_argument('agent', choices=['HM', 'RV', 'ED', 'EX', 'PR', 'RP', 'pipeline'],
                           help='Agent to run or "pipeline" for full pipeline')
    run_parser.add_argument('--tags', type=str, help='Domain tags (comma-separated)')
    run_parser.add_argument('--n', type=int, default=5, help='Number of hypotheses to generate')
    run_parser.add_argument('--idea', type=str, help='Specific idea ID to process')
    run_parser.add_argument('--config-dir', type=str, default='config', help='Configuration directory')
    run_parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    status_parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        if args.command == 'run':
            pipeline = AstroAgentPipeline(args.config_dir, args.data_dir)
            
            if args.agent == 'pipeline':
                # Run full pipeline
                agent_inputs = {}
                if args.tags:
                    agent_inputs['domain_tags'] = [tag.strip() for tag in args.tags.split(',')]
                if args.n:
                    agent_inputs['n_hypotheses'] = args.n
                
                results = pipeline.run_pipeline(agent_inputs)
                
                if results['success']:
                    print(f"Pipeline completed successfully!")
                    print(f"Final state: {results['final_state']}")
                    print(f"Agents run: {results['agents_run']}")
                    print(f"Ideas updated: {len(results['ideas_updated'])}")
                else:
                    print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
                    if results.get('errors'):
                        for error in results['errors']:
                            print(f"  {error['agent']}: {error['error']}")
            
            else:
                # Run individual agent
                print(f"Individual agent execution not yet implemented for {args.agent}")
        
        elif args.command == 'status':
            # Show status
            registry_manager = RegistryManager(args.data_dir)
            
            # Load registries and show status
            print("Pipeline Status:")
            print("=" * 50)
            
            # Ideas register status
            try:
                ideas_df = registry_manager.load_registry('ideas_register')
                print(f"Ideas Register: {len(ideas_df)} total ideas")
                if not ideas_df.empty:
                    status_counts = ideas_df['status'].value_counts()
                    for status, count in status_counts.items():
                        print(f"  {status}: {count}")
            except Exception as e:
                print(f"  Error loading ideas register: {e}")
            
            print()
            
            # Project index status  
            try:
                projects_df = registry_manager.load_registry('project_index')
                print(f"Project Index: {len(projects_df)} total projects")
                if not projects_df.empty:
                    maturity_counts = projects_df['maturity'].value_counts()
                    for maturity, count in maturity_counts.items():
                        print(f"  {maturity}: {count}")
            except Exception as e:
                print(f"  Error loading project index: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
