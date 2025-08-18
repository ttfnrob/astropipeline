"""
Common base classes, schemas, and utilities for AstroAgent Pipeline agents.

This module defines the core abstractions used by all agents in the pipeline,
including data models, base agent classes, and telemetry utilities.
"""

import abc
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import uuid
import time
import base64

from pydantic import BaseModel, Field, field_validator
import yaml


# ============================================================================
# Data Models and Schemas
# ============================================================================

class IdeaSchema(BaseModel):
    """Schema for research ideas in the Ideas Register."""
    
    idea_id: str = Field(..., pattern=r"^01[A-Z0-9]{24}$", description="ULID identifier")
    title: str = Field(..., max_length=100, description="Concise idea title")
    hypothesis: str = Field(..., min_length=50, max_length=350, description="Falsifiable hypothesis")
    rationale: str = Field(..., min_length=100, max_length=500, description="Scientific rationale")
    domain_tags: List[str] = Field(..., description="Research domain tags")
    novelty_refs: List[str] = Field(..., description="Bibcodes or arXiv IDs")
    required_data: List[str] = Field(..., description="Required datasets")
    methods: List[str] = Field(..., description="Analysis methods")
    est_effort_days: int = Field(..., ge=1, le=30, description="Estimated effort in days")
    
    # Review scores (populated by Reviewer agent)
    feasibility_score: Optional[int] = Field(None, ge=1, le=5)
    impact_score: Optional[int] = Field(None, ge=1, le=5)
    testability_score: Optional[int] = Field(None, ge=1, le=5)
    novelty_score: Optional[int] = Field(None, ge=1, le=5)
    total_score: Optional[int] = Field(None, ge=4, le=20)
    
    # Status tracking
    version: str = Field(default="v1", description="Idea version")
    parent_idea_id: Optional[str] = Field(None, description="Parent idea if revision")
    status: str = Field(
        default="Proposed", 
        description="Idea status",
        pattern=r"^(Proposed|Under Review|Needs Revision|Approved|Rejected|Parked)$"
    )
    reviewer_notes: Optional[str] = Field(None, max_length=1000)
    risk_notes: Optional[str] = Field(None, max_length=500)
    ethics_notes: Optional[str] = Field(None, max_length=500)
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('idea_id', mode='before')
    @classmethod
    def generate_idea_id(cls, v):
        if not v:
            return generate_ulid()
        return v


class ProjectSchema(BaseModel):
    """Schema for project entries in the Project Index."""
    
    idea_id: str = Field(..., description="Link to Ideas Register")
    slug: str = Field(..., pattern=r"^[a-z0-9-]+$", description="URL-safe project slug")
    path: str = Field(..., description="Project folder path")
    
    # Readiness checklist
    ready_checklist_passed: bool = Field(default=False)
    data_requirements_met: bool = Field(default=False)
    analysis_plan_preregistered: bool = Field(default=False)
    
    # Maturity tracking
    maturity: str = Field(
        default="Draft",
        pattern=r"^(Draft|Preparing|Ready|Running|Executed|AwaitingReview|Reviewed|Blocked|Complete|Failed)$"
    )
    
    # Execution metadata
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    compute_hours_used: Optional[float] = None
    storage_gb_used: Optional[float] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompletedProjectSchema(BaseModel):
    """Schema for completed projects in the Completed Index."""
    
    idea_id: str = Field(..., description="Link to original idea")
    title: str = Field(..., description="Project title")
    abstract: str = Field(..., max_length=2000, description="Research abstract")
    key_findings: str = Field(..., max_length=1000, description="Main results")
    
    # External links
    data_doi: Optional[str] = Field(None, description="Dataset DOI")
    code_repo: Optional[str] = Field(None, description="Code repository URL")
    paper_preprint: Optional[str] = Field(None, description="Preprint URL")
    
    # Quality metrics
    confidence: float = Field(..., ge=0.0, le=1.0, description="Result confidence")
    reviewer_signoff: bool = Field(default=False)
    reviewer_name: Optional[str] = Field(None)
    
    # Archive metadata
    moved_to_library_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artefacts_manifest: Dict[str, Any] = Field(default_factory=dict)


class AgentExecutionContext(BaseModel):
    """Context passed to agents during execution."""
    
    agent_name: str
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Input data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # State tracking
    state_name: str
    previous_state: Optional[str] = None
    retry_count: int = 0
    
    # Resources
    max_tokens: int = 2000
    temperature: float = 0.5
    timeout_seconds: int = 120


class AgentResult(BaseModel):
    """Result returned by agent execution."""
    
    success: bool
    agent_name: str
    execution_id: str
    
    # Output data
    output_data: Dict[str, Any] = Field(default_factory=dict)
    files_created: List[str] = Field(default_factory=list)
    registry_updates: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Execution metadata
    execution_time_seconds: float
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Provenance
    input_hash: str
    output_hash: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Base Agent Classes
# ============================================================================

class BaseAgent(abc.ABC):
    """Abstract base class for all AstroAgent Pipeline agents."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.name = self.__class__.__name__
        self.logger = logger or self._setup_logger()
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Set up agent-specific logger."""
        logger = logging.getLogger(f"astroagent.agents.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abc.abstractmethod
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Execute the agent's core functionality."""
        pass
    
    def validate_input(self, context: AgentExecutionContext) -> bool:
        """Validate input data before execution."""
        return True
    
    def validate_output(self, result: AgentResult) -> bool:
        """Validate output data after execution."""
        return result.success
    
    def run(self, context: AgentExecutionContext) -> AgentResult:
        """Main entry point for agent execution with error handling and telemetry."""
        start_time = time.time()
        self.execution_stats['total_executions'] += 1
        
        try:
            # Input validation
            if not self.validate_input(context):
                raise ValueError("Input validation failed")
            
            # Compute input hash for provenance
            input_hash = self._compute_hash(context.model_dump())
            
            self.logger.info(f"Starting execution {context.execution_id}")
            
            # Execute agent logic
            result = self.execute(context)
            result.execution_id = context.execution_id
            result.agent_name = self.name
            
            # Compute output hash
            result.input_hash = input_hash
            result.output_hash = self._compute_hash(result.output_data)
            
            # Output validation
            if not self.validate_output(result):
                result.success = False
                result.error_message = "Output validation failed"
                result.error_type = "ValidationError"
            
            # Update stats on success
            if result.success:
                self.execution_stats['successful_executions'] += 1
                self.logger.info(f"Execution {context.execution_id} completed successfully")
            else:
                self.logger.error(f"Execution {context.execution_id} failed: {result.error_message}")
            
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Execution {context.execution_id} failed with exception: {str(e)}")
            result = AgentResult(
                success=False,
                agent_name=self.name,
                execution_id=context.execution_id,
                error_message=str(e),
                error_type=type(e).__name__,
                execution_time_seconds=0,
                input_hash=self._compute_hash(context.model_dump()),
                output_hash=""
            )
        
        finally:
            # Update timing statistics
            execution_time = time.time() - start_time
            result.execution_time_seconds = execution_time
            
            self.execution_stats['total_execution_time'] += execution_time
            if self.execution_stats['total_executions'] > 0:
                self.execution_stats['average_execution_time'] = (
                    self.execution_stats['total_execution_time'] / 
                    self.execution_stats['total_executions']
                )
        
        return result
    
    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data for provenance tracking."""
        if isinstance(data, dict):
            # Sort keys for deterministic hashing
            sorted_data = json.dumps(data, sort_keys=True, default=str)
        else:
            sorted_data = str(data)
        
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this agent."""
        return {
            'agent_name': self.name,
            'stats': self.execution_stats.copy(),
            'success_rate': (
                self.execution_stats['successful_executions'] / 
                max(1, self.execution_stats['total_executions'])
            )
        }


# ============================================================================
# Utility Functions
# ============================================================================

def load_agent_config(agent_name: str, config_dir: str = "config") -> Dict[str, Any]:
    """Load agent configuration from YAML file."""
    config_path = Path(config_dir) / "agents.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    agent_config = config.get(agent_name)
    if not agent_config:
        raise KeyError(f"Configuration for agent '{agent_name}' not found")
    
    # Merge with global settings
    global_config = config.get('global', {})
    agent_config.update(global_config)
    
    return agent_config


def create_project_folder(idea_id: str, title: str, base_path: str = "projects") -> tuple[str, str]:
    """Create project folder with standardized naming."""
    # Generate slug from title
    slug = title.lower().replace(' ', '-').replace('_', '-')
    # Remove non-alphanumeric characters except hyphens
    slug = ''.join(c for c in slug if c.isalnum() or c == '-')
    # Limit length and clean up multiple hyphens
    slug = '-'.join(filter(None, slug.split('-')))[:50]
    
    folder_name = f"{idea_id}__{slug}"
    project_path = Path(base_path) / "Preparing" / folder_name
    
    # Create directory structure
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "research").mkdir(exist_ok=True)
    (project_path / "notebooks").mkdir(exist_ok=True)
    (project_path / "scripts").mkdir(exist_ok=True)
    (project_path / "artefacts").mkdir(exist_ok=True)
    
    return str(project_path), slug


def move_project_folder(idea_id: str, slug: str, from_maturity: str, to_maturity: str, base_path: str = "projects") -> str:
    """Move project folder between directories based on maturity status."""
    import shutil
    
    # Map maturity statuses to directory names
    maturity_dirs = {
        'Preparing': 'Preparing',
        'Prepared': 'Preparing', 
        'Ready': 'Ready for Execution',
        'Running': 'Ready for Execution',
        'Executed': 'Ready for Execution',
        'AwaitingReview': 'Ready for Execution',
        'Reviewed': 'Ready for Execution',
        'Complete': 'Library',
        'Failed': 'Preparing'  # Failed projects go back to Preparing
    }
    
    from_dir = maturity_dirs.get(from_maturity, 'Preparing')
    to_dir = maturity_dirs.get(to_maturity, 'Preparing')
    
    # Skip if no movement needed
    if from_dir == to_dir:
        folder_name = f"{idea_id}__{slug}"
        return str(Path(base_path) / to_dir / folder_name)
    
    folder_name = f"{idea_id}__{slug}"
    from_path = Path(base_path) / from_dir / folder_name
    to_path = Path(base_path) / to_dir / folder_name
    
    # Create target directory if it doesn't exist
    to_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Move folder if source exists
    if from_path.exists():
        if to_path.exists():
            shutil.rmtree(to_path)  # Remove existing target
        shutil.move(str(from_path), str(to_path))
        print(f"📁 Moved project from {from_dir}/ to {to_dir}/")
    
    return str(to_path)


def generate_ulid() -> str:
    """Generate a new ULID-like identifier for idea identification."""
    # Generate a ULID-like ID using timestamp + random UUID
    # ULID format: 01ARYZ6S410000000000000000 (26 chars)
    # Our format: 01 + timestamp (8 chars) + UUID suffix (16 chars)
    
    # Get milliseconds since epoch
    timestamp_ms = int(time.time() * 1000)
    
    # Convert to base32-like encoding (using uppercase letters and numbers)
    # Simple approach: use hex and convert some chars to make it ULID-like
    timestamp_hex = f"{timestamp_ms:x}".upper()[:8].ljust(8, '0')
    
    # Generate random part using UUID
    random_part = str(uuid.uuid4()).replace('-', '').upper()[:16]
    
    return f"01{timestamp_hex}{random_part}"


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Set up structured logging for agents."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, level.upper()))
    return logger


# ============================================================================
# Constants
# ============================================================================

IDEA_STATUSES = [
    "Proposed", "Under Review", "Needs Revision", 
    "Approved", "Rejected", "Parked"
]

PROJECT_MATURITIES = [
    "Draft", "Preparing", "Ready", "Running", "Executed", 
    "AwaitingReview", "Reviewed", "Blocked", "Complete", "Failed"
]

REVIEW_VERDICTS = ["Approved", "Changes requested", "Rejected"]

# File paths
DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_PROJECTS_DIR = "projects"
DEFAULT_TEMPLATES_DIR = "templates"
