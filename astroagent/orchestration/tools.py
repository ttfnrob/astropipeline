"""
Orchestration tools for AstroAgent Pipeline.

Provides utilities for registry management, state validation,
and shared tools used by the orchestration system.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml


class RegistryManager:
    """Manages CSV-based data registries for the pipeline."""
    
    def __init__(self, data_dir: str = "data", logger: Optional[logging.Logger] = None):
        """Initialize registry manager.
        
        Args:
            data_dir: Base data directory path
            logger: Optional logger instance
        """
        self.data_dir = Path(data_dir)
        self.registry_dir = self.data_dir / "registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or self._setup_logger()
        
        # Registry file paths
        self.registries = {
            'ideas_register': self.registry_dir / "ideas_register.csv",
            'project_index': self.registry_dir / "project_index.csv", 
            'completed_index': self.registry_dir / "completed_index.csv"
        }
        
        # Ensure all registries exist
        self._ensure_registries_exist()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for registry manager."""
        logger = logging.getLogger('astroagent.orchestration.registry')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _ensure_registries_exist(self):
        """Ensure all registry files exist with proper headers."""
        
        headers = {
            'ideas_register': [
                'idea_id', 'title', 'hypothesis', 'rationale', 'domain_tags',
                'novelty_refs', 'required_data', 'methods', 'est_effort_days',
                'feasibility_score', 'impact_score', 'testability_score', 
                'novelty_score', 'total_score', 'version', 'parent_idea_id',
                'status', 'reviewer_notes', 'risk_notes', 'ethics_notes',
                'created_at', 'updated_at'
            ],
            'project_index': [
                'idea_id', 'slug', 'path', 'ready_checklist_passed',
                'data_requirements_met', 'analysis_plan_preregistered',
                'maturity', 'execution_start', 'execution_end',
                'compute_hours_used', 'storage_gb_used', 'created_at', 'updated_at'
            ],
            'completed_index': [
                'idea_id', 'title', 'abstract', 'key_findings', 'data_doi',
                'code_repo', 'paper_preprint', 'confidence', 'reviewer_signoff',
                'reviewer_name', 'moved_to_library_at', 'artefacts_manifest'
            ]
        }
        
        for registry_name, header_list in headers.items():
            registry_path = self.registries[registry_name]
            
            if not registry_path.exists():
                self.logger.info(f"Creating registry: {registry_name}")
                with open(registry_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header_list)
    
    def load_registry(self, registry_name: str) -> pd.DataFrame:
        """Load a registry as a pandas DataFrame.
        
        Args:
            registry_name: Name of registry to load
            
        Returns:
            DataFrame containing registry data
        """
        
        if registry_name not in self.registries:
            raise ValueError(f"Unknown registry: {registry_name}")
        
        registry_path = self.registries[registry_name]
        
        try:
            df = pd.read_csv(registry_path)
            self.logger.debug(f"Loaded {len(df)} rows from {registry_name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load registry {registry_name}: {str(e)}")
            raise
    
    def save_registry(self, registry_name: str, df: pd.DataFrame):
        """Save a DataFrame to a registry file.
        
        Args:
            registry_name: Name of registry to save
            df: DataFrame to save
        """
        
        if registry_name not in self.registries:
            raise ValueError(f"Unknown registry: {registry_name}")
        
        registry_path = self.registries[registry_name]
        
        try:
            df.to_csv(registry_path, index=False)
            self.logger.info(f"Saved {len(df)} rows to {registry_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save registry {registry_name}: {str(e)}")
            raise
    
    def append_to_registry(self, registry_name: str, data: Dict[str, Any]):
        """Append a single row to a registry.
        
        Args:
            registry_name: Name of registry
            data: Dictionary of data to append
        """
        
        # Load existing data
        df = self.load_registry(registry_name)
        
        # Add timestamps if not provided
        if 'created_at' not in data and 'created_at' in df.columns:
            data['created_at'] = datetime.now(timezone.utc).isoformat()
        
        if 'updated_at' not in data and 'updated_at' in df.columns:
            data['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Convert lists to JSON strings for CSV storage
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = json.dumps(value)
        
        # Append row
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save back
        self.save_registry(registry_name, df)
        
        self.logger.info(f"Appended row to {registry_name}")
    
    def update_registry_row(self, registry_name: str, 
                           filter_criteria: Dict[str, Any],
                           update_data: Dict[str, Any]):
        """Update rows in a registry matching filter criteria.
        
        Args:
            registry_name: Name of registry
            filter_criteria: Dictionary of column:value pairs to match
            update_data: Dictionary of updates to apply
        """
        
        # Load registry
        df = self.load_registry(registry_name)
        
        if df.empty:
            self.logger.warning(f"Registry {registry_name} is empty, cannot update")
            return
        
        # Find matching rows
        mask = pd.Series([True] * len(df))
        
        for col, value in filter_criteria.items():
            if col in df.columns:
                mask &= (df[col] == value)
            else:
                self.logger.warning(f"Filter column {col} not found in registry")
                return
        
        matching_rows = mask.sum()
        
        if matching_rows == 0:
            self.logger.warning(f"No rows found matching criteria: {filter_criteria}")
            return
        
        # Apply updates
        for col, value in update_data.items():
            if col in df.columns:
                if isinstance(value, list):
                    value = json.dumps(value)
                df.loc[mask, col] = value
            else:
                self.logger.warning(f"Update column {col} not found in registry")
        
        # Update timestamp
        if 'updated_at' in df.columns:
            df.loc[mask, 'updated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Save back
        self.save_registry(registry_name, df)
        
        self.logger.info(f"Updated {matching_rows} rows in {registry_name}")
    
    def query_registry(self, registry_name: str, 
                      filter_criteria: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Query a registry with optional filtering.
        
        Args:
            registry_name: Name of registry
            filter_criteria: Optional filter criteria
            
        Returns:
            Filtered DataFrame
        """
        
        df = self.load_registry(registry_name)
        
        if filter_criteria is None:
            return df
        
        # Apply filters
        mask = pd.Series([True] * len(df))
        
        for col, value in filter_criteria.items():
            if col in df.columns:
                mask &= (df[col] == value)
        
        return df[mask]


class StateValidator:
    """Validates pipeline state transitions and data consistency."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize state validator."""
        self.logger = logger or self._setup_logger()
        
        # Valid state transitions
        self.valid_transitions = {
            'start': ['hypothesis_generation'],
            'hypothesis_generation': ['initial_review', 'error'],
            'initial_review': ['experiment_design', 'archive', 'error'],
            'experiment_design': ['ready_check', 'error'],
            'ready_check': ['experiment_execution', 'experiment_design', 'error'],
            'experiment_execution': ['peer_review', 'error'],
            'peer_review': ['report_generation', 'experiment_execution', 'archive', 'error'],
            'report_generation': ['library', 'error'],
            'library': [],  # terminal state
            'archive': [],  # terminal state
            'error': [],    # terminal state
            'failed': []    # terminal state
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for state validator."""
        logger = logging.getLogger('astroagent.orchestration.validator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def validate_transition(self, from_state: str, to_state: str) -> bool:
        """Validate a state transition.
        
        Args:
            from_state: Current state
            to_state: Target state
            
        Returns:
            True if transition is valid
        """
        
        if from_state not in self.valid_transitions:
            self.logger.error(f"Unknown from_state: {from_state}")
            return False
        
        valid_next_states = self.valid_transitions[from_state]
        
        if to_state not in valid_next_states:
            self.logger.error(f"Invalid transition: {from_state} -> {to_state}")
            self.logger.error(f"Valid transitions from {from_state}: {valid_next_states}")
            return False
        
        return True
    
    def validate_pipeline_state(self, state: Dict[str, Any]) -> bool:
        """Validate overall pipeline state consistency.
        
        Args:
            state: Pipeline state dictionary
            
        Returns:
            True if state is valid
        """
        
        required_fields = [
            'current_state', 'pipeline_id', 'execution_start',
            'agent_inputs', 'agent_outputs', 'errors'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in state:
                self.logger.error(f"Missing required state field: {field}")
                return False
        
        # Validate state transition if previous_state exists
        if state.get('previous_state'):
            if not self.validate_transition(state['previous_state'], state['current_state']):
                return False
        
        # Check that errors list is valid
        errors = state.get('errors', [])
        if not isinstance(errors, list):
            self.logger.error("Errors field must be a list")
            return False
        
        # Validate error structure
        for error in errors:
            if not isinstance(error, dict):
                self.logger.error("Each error must be a dictionary")
                return False
            
            required_error_fields = ['agent', 'error', 'timestamp']
            for field in required_error_fields:
                if field not in error:
                    self.logger.error(f"Error missing required field: {field}")
                    return False
        
        return True


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: str = "config", logger: Optional[logging.Logger] = None):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.logger = logger or self._setup_logger()
        
        self._config_cache = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for config manager."""
        logger = logging.getLogger('astroagent.orchestration.config')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a YAML configuration file.
        
        Args:
            config_name: Name of configuration file (without .yaml extension)
            
        Returns:
            Configuration dictionary
        """
        
        if config_name in self._config_cache:
            return self._config_cache[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self._config_cache[config_name] = config
            self.logger.info(f"Loaded configuration: {config_name}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration {config_name}: {str(e)}")
            raise
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent configuration dictionary
        """
        
        agents_config = self.load_config('agents')
        
        if agent_name not in agents_config:
            raise KeyError(f"Configuration for agent '{agent_name}' not found")
        
        agent_config = agents_config[agent_name].copy()
        
        # Merge with global settings if present
        if 'global' in agents_config:
            global_config = agents_config['global']
            # Global settings have lower priority
            merged_config = global_config.copy()
            merged_config.update(agent_config)
            agent_config = merged_config
        
        return agent_config
    
    def get_orchestration_config(self) -> Dict[str, Any]:
        """Get orchestration configuration."""
        return self.load_config('orchestration')
    
    def get_datasources_config(self) -> Dict[str, Any]:
        """Get data sources configuration."""
        return self.load_config('datasources')
