"""
AstroAgent Pipeline - Orchestration Package

Provides pipeline workflow orchestration, state management, and registry systems.

Components:
- graph: LangGraph-based state machine for agent coordination
- registry: High-level registry management for research artifacts
- tools: Utilities for state validation and configuration management
"""

from .graph import AstroAgentPipeline, PipelineState
from .registry import ProjectRegistry
from .tools import RegistryManager, StateValidator, ConfigManager

__all__ = [
    'AstroAgentPipeline',
    'PipelineState', 
    'ProjectRegistry',
    'RegistryManager',
    'StateValidator',
    'ConfigManager'
]
