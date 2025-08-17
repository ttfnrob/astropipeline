"""
AstroAgent Pipeline - AI-Powered Astrophysics Research System

A modular, auditable system of AI agents and tools to generate and test 
astrophysics hypotheses, then produce original research papers.

Main Components:
- agents: AI agents for hypothesis generation, review, and experimentation
- services: External data access and analysis services  
- orchestration: Pipeline workflow and state management
- templates: Document templates for research artifacts

Usage:
    from astroagent import create_agent, AstroAgentPipeline
    
    # Create individual agents
    hypothesis_maker = create_agent('hypothesis_maker')
    
    # Or run the full pipeline
    pipeline = AstroAgentPipeline()
    results = pipeline.run_pipeline({'domain_tags': ['stellar dynamics']})
"""

from .agents import create_agent, get_agent_class, AGENT_REGISTRY
from .orchestration.graph import AstroAgentPipeline
from .orchestration.registry import ProjectRegistry

__version__ = "0.1.0"
__author__ = "AstroAgent Team"

__all__ = [
    # Core functionality
    'AstroAgentPipeline',
    'ProjectRegistry',
    
    # Agent creation
    'create_agent',
    'get_agent_class', 
    'AGENT_REGISTRY',
    
    # Version info
    '__version__',
    '__author__'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'astroagent',
    'version': __version__,
    'description': 'AI-powered astrophysics research pipeline',
    'author': __author__,
    'license': 'MIT',
    'python_requires': '>=3.9',
    'keywords': ['astrophysics', 'AI', 'research', 'pipeline', 'automation'],
    'classifiers': [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
}
