"""
AstroAgent Pipeline - Agent Package

This package contains all the AI agents that form the research pipeline:
- HypothesisMaker: Generates novel research hypotheses
- Reviewer: Evaluates and scores hypotheses
- ExperimentDesigner: Creates detailed experimental plans
- Experimenter: Executes experiments and analyses
- PeerReviewer: Reviews experimental results
- Reporter: Generates final research reports
"""

from .common import (
    BaseAgent, 
    AgentExecutionContext, 
    AgentResult,
    IdeaSchema,
    ProjectSchema,
    CompletedProjectSchema,
    load_agent_config,
    create_project_folder,
    generate_ulid,
    setup_logging
)

from .hypothesis_maker import HypothesisMaker
from .reviewer import Reviewer
from .experiment_designer import ExperimentDesigner

# TODO: Import other agents once implemented
# from .experimenter import Experimenter
# from .peer_reviewer import PeerReviewer
# from .reporter import Reporter

__all__ = [
    # Base classes and schemas
    'BaseAgent',
    'AgentExecutionContext', 
    'AgentResult',
    'IdeaSchema',
    'ProjectSchema',
    'CompletedProjectSchema',
    
    # Utility functions
    'load_agent_config',
    'create_project_folder',
    'generate_ulid',
    'setup_logging',
    
    # Agent implementations
    'HypothesisMaker',
    'Reviewer',
    'ExperimentDesigner',
    
    # TODO: Add other agents
    # 'Experimenter',
    # 'PeerReviewer', 
    # 'Reporter'
]

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    'hypothesis_maker': HypothesisMaker,
    'HM': HypothesisMaker,
    
    'reviewer': Reviewer,
    'RV': Reviewer,
    
    'experiment_designer': ExperimentDesigner,
    'ED': ExperimentDesigner,
    
    # TODO: Add other agents
    # 'experimenter': Experimenter,
    # 'EX': Experimenter,
    
    # 'peer_reviewer': PeerReviewer,
    # 'PR': PeerReviewer,
    
    # 'reporter': Reporter,
    # 'RP': Reporter,
}


def get_agent_class(agent_name: str):
    """Get agent class by name."""
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(AGENT_REGISTRY.keys())}")
    
    return AGENT_REGISTRY[agent_name]


def create_agent(agent_name: str, config_dir: str = "config"):
    """Create and configure an agent instance."""
    agent_class = get_agent_class(agent_name)
    
    # Load configuration
    config = load_agent_config(agent_name, config_dir)
    
    # Create agent instance
    agent = agent_class(config)
    
    return agent
