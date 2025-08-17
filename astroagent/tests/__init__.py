"""
AstroAgent Pipeline Test Suite

Comprehensive tests for all components of the AstroAgent research pipeline.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest


# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
MOCK_CONFIG_DIR = Path(__file__).parent / "config"


class TestFixtures:
    """Common test fixtures and utilities."""
    
    @staticmethod
    def create_temp_workspace() -> Path:
        """Create a temporary workspace for testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="astroagent_test_"))
        
        # Create basic directory structure
        (temp_dir / "config").mkdir()
        (temp_dir / "data" / "registry").mkdir(parents=True)
        (temp_dir / "data" / "vectors").mkdir()
        (temp_dir / "projects" / "Preparing").mkdir(parents=True)
        (temp_dir / "projects" / "Ready for Execution").mkdir()
        (temp_dir / "projects" / "Library").mkdir()
        (temp_dir / "projects" / "Archive").mkdir()
        (temp_dir / "templates").mkdir()
        
        return temp_dir
    
    @staticmethod
    def cleanup_temp_workspace(workspace_path: Path):
        """Clean up temporary workspace."""
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
    
    @staticmethod
    def get_mock_idea() -> Dict[str, Any]:
        """Get a mock research idea for testing."""
        return {
            'idea_id': '01HZRXP1K2M3N4P5Q6R7S8T9U0',
            'title': 'Test Stellar Dynamics Hypothesis',
            'hypothesis': 'We hypothesize that stellar velocity dispersions in galaxy clusters exhibit previously unrecognized correlations with dark matter substructure that can be detected through systematic analysis of Gaia DR3 proper motion data combined with weak lensing measurements.',
            'rationale': 'Recent theoretical work suggests that dark matter subhalos should imprint subtle but detectable signatures on stellar kinematics. This hypothesis builds on advances in both observational capabilities (Gaia DR3) and theoretical understanding of structure formation to test predictions from lambda-CDM cosmology.',
            'domain_tags': ['stellar dynamics', 'galaxy clusters', 'dark matter'],
            'novelty_refs': ['2023ApJ...900...1A', '2023MNRAS.520.1234B', '2022A&A...650.L5C'],
            'required_data': ['Gaia DR3', 'SDSS clusters', 'HSC weak lensing'],
            'methods': ['Bayesian hierarchical modeling', 'Monte Carlo simulation', 'Bootstrap resampling'],
            'est_effort_days': 12,
            'status': 'Proposed'
        }
    
    @staticmethod
    def get_mock_paper() -> Dict[str, Any]:
        """Get a mock literature paper for testing."""
        return {
            'bibcode': '2023ApJ...900...1A',
            'title': 'Dark Matter Substructure and Stellar Dynamics in Galaxy Clusters',
            'author': ['Smith, J.', 'Johnson, K.', 'Williams, L.'],
            'year': 2023,
            'abstract': 'We present a comprehensive analysis of stellar kinematics in galaxy clusters and their relationship to dark matter substructure. Our results suggest novel correlations that have implications for structure formation models.',
            'keyword': ['dark matter', 'galaxy clusters', 'stellar dynamics'],
            'citation_count': 15
        }
    
    @staticmethod
    def get_mock_agent_config() -> Dict[str, Any]:
        """Get mock agent configuration."""
        return {
            'model': 'mock-gpt-4',
            'temperature': 0.7,
            'max_tokens': 1000,
            'system_prompt': 'You are a test agent.',
            'guardrails': {
                'min_hypothesis_words': 10,
                'max_hypothesis_words': 200,
                'required_fields': ['hypothesis', 'rationale']
            }
        }
