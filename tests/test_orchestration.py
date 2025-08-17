"""
Unit tests for AstroAgent Pipeline orchestration components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
import json

from ..orchestration.tools import RegistryManager, StateValidator, ConfigManager
from ..orchestration.registry import ProjectRegistry
from ..orchestration.graph import AstroAgentPipeline, PipelineState
from . import TestFixtures


class TestRegistryManager:
    """Test the registry manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="registry_test_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_registry_manager_initialization(self):
        """Test registry manager initializes correctly."""
        rm = RegistryManager(data_dir=str(self.temp_dir))
        
        assert rm.data_dir == self.temp_dir
        assert rm.registry_dir.exists()
        
        # Check that all registry files are created
        for registry_name in rm.registries:
            assert rm.registries[registry_name].exists()
    
    def test_load_empty_registry(self):
        """Test loading an empty registry."""
        rm = RegistryManager(data_dir=str(self.temp_dir))
        
        df = rm.load_registry('ideas_register')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert 'idea_id' in df.columns
        assert 'title' in df.columns
    
    def test_append_to_registry(self):
        """Test appending data to a registry."""
        rm = RegistryManager(data_dir=str(self.temp_dir))
        
        idea_data = TestFixtures.get_mock_idea()
        rm.append_to_registry('ideas_register', idea_data)
        
        # Verify data was added
        df = rm.load_registry('ideas_register')
        assert len(df) == 1
        assert df.iloc[0]['idea_id'] == idea_data['idea_id']
        assert df.iloc[0]['title'] == idea_data['title']
        
        # Check that lists are stored as JSON strings
        stored_tags = df.iloc[0]['domain_tags']
        if isinstance(stored_tags, str):
            parsed_tags = json.loads(stored_tags)
            assert parsed_tags == idea_data['domain_tags']
        else:
            # Already parsed by the registry system
            assert stored_tags == idea_data['domain_tags']
    
    def test_update_registry_row(self):
        """Test updating registry rows."""
        rm = RegistryManager(data_dir=str(self.temp_dir))
        
        # Add initial data
        idea_data = TestFixtures.get_mock_idea()
        rm.append_to_registry('ideas_register', idea_data)
        
        # Update data
        updates = {'status': 'Approved', 'feasibility_score': 4}
        rm.update_registry_row(
            'ideas_register',
            {'idea_id': idea_data['idea_id']},
            updates
        )
        
        # Verify updates
        df = rm.load_registry('ideas_register')
        assert df.iloc[0]['status'] == 'Approved'
        assert df.iloc[0]['feasibility_score'] == 4
        assert 'updated_at' in df.columns
    
    def test_query_registry(self):
        """Test querying registry with filters."""
        rm = RegistryManager(data_dir=str(self.temp_dir))
        
        # Add multiple ideas with different statuses
        idea1 = TestFixtures.get_mock_idea()
        idea1['status'] = 'Proposed'
        rm.append_to_registry('ideas_register', idea1)
        
        idea2 = TestFixtures.get_mock_idea()
        idea2['idea_id'] = '01DIFFERENT123456789'
        idea2['status'] = 'Approved'
        rm.append_to_registry('ideas_register', idea2)
        
        # Query for proposed ideas
        proposed_df = rm.query_registry('ideas_register', {'status': 'Proposed'})
        assert len(proposed_df) == 1
        assert proposed_df.iloc[0]['status'] == 'Proposed'
        
        # Query for approved ideas
        approved_df = rm.query_registry('ideas_register', {'status': 'Approved'})
        assert len(approved_df) == 1
        assert approved_df.iloc[0]['status'] == 'Approved'
        
        # Query with no filter
        all_df = rm.query_registry('ideas_register')
        assert len(all_df) == 2
    
    def test_save_and_load_registry(self):
        """Test saving and loading registry data."""
        rm = RegistryManager(data_dir=str(self.temp_dir))
        
        # Create test data
        df = pd.DataFrame([
            {'idea_id': '01TEST123', 'title': 'Test Idea', 'status': 'Proposed'},
            {'idea_id': '01TEST456', 'title': 'Another Idea', 'status': 'Approved'}
        ])
        
        # Save data
        rm.save_registry('ideas_register', df)
        
        # Load data back
        loaded_df = rm.load_registry('ideas_register')
        
        assert len(loaded_df) == 2
        assert loaded_df.iloc[0]['idea_id'] == '01TEST123'
        assert loaded_df.iloc[1]['title'] == 'Another Idea'


class TestStateValidator:
    """Test the state validator."""
    
    def test_state_validator_initialization(self):
        """Test state validator initializes correctly."""
        validator = StateValidator()
        
        assert 'start' in validator.valid_transitions
        assert 'hypothesis_generation' in validator.valid_transitions
    
    def test_validate_valid_transition(self):
        """Test validation of valid state transitions."""
        validator = StateValidator()
        
        # Test valid transitions
        assert validator.validate_transition('start', 'hypothesis_generation')
        assert validator.validate_transition('hypothesis_generation', 'initial_review')
        assert validator.validate_transition('initial_review', 'experiment_design')
    
    def test_validate_invalid_transition(self):
        """Test validation rejects invalid transitions."""
        validator = StateValidator()
        
        # Test invalid transitions
        assert not validator.validate_transition('start', 'library')  # Can't go directly to end
        assert not validator.validate_transition('library', 'hypothesis_generation')  # Can't leave terminal state
        assert not validator.validate_transition('unknown_state', 'start')  # Unknown from state
    
    def test_validate_pipeline_state_valid(self):
        """Test validation of valid pipeline state."""
        validator = StateValidator()
        
        valid_state = {
            'current_state': 'hypothesis_generation',
            'previous_state': 'start',
            'pipeline_id': 'test_pipeline_123',
            'execution_start': datetime.now(timezone.utc),
            'agent_inputs': {'domain_tags': ['test']},
            'agent_outputs': {},
            'errors': []
        }
        
        assert validator.validate_pipeline_state(valid_state)
    
    def test_validate_pipeline_state_invalid(self):
        """Test validation rejects invalid pipeline states."""
        validator = StateValidator()
        
        # Missing required fields
        invalid_state = {
            'current_state': 'hypothesis_generation',
            # Missing other required fields
        }
        
        assert not validator.validate_pipeline_state(invalid_state)
        
        # Invalid transition
        invalid_transition_state = {
            'current_state': 'library',
            'previous_state': 'start',  # Can't go directly from start to library
            'pipeline_id': 'test_pipeline_123',
            'execution_start': datetime.now(timezone.utc),
            'agent_inputs': {},
            'agent_outputs': {},
            'errors': []
        }
        
        assert not validator.validate_pipeline_state(invalid_transition_state)
        
        # Invalid error structure
        invalid_error_state = {
            'current_state': 'hypothesis_generation',
            'pipeline_id': 'test_pipeline_123',
            'execution_start': datetime.now(timezone.utc),
            'agent_inputs': {},
            'agent_outputs': {},
            'errors': ['not a dict']  # Should be list of dicts
        }
        
        assert not validator.validate_pipeline_state(invalid_error_state)


class TestConfigManager:
    """Test the configuration manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="config_test_"))
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """Test config manager initializes correctly."""
        cm = ConfigManager(config_dir=str(self.config_dir))
        
        assert cm.config_dir == self.config_dir
    
    def test_load_config_success(self):
        """Test successful config loading."""
        # Create test config file
        config_data = {
            'test_agent': {
                'model': 'gpt-4',
                'temperature': 0.5
            },
            'global': {
                'timeout': 60
            }
        }
        
        config_file = self.config_dir / "test.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        cm = ConfigManager(config_dir=str(self.config_dir))
        
        loaded_config = cm.load_config('test')
        
        assert loaded_config == config_data
        assert loaded_config['test_agent']['model'] == 'gpt-4'
    
    def test_load_config_file_not_found(self):
        """Test config loading with missing file."""
        cm = ConfigManager(config_dir=str(self.config_dir))
        
        with pytest.raises(FileNotFoundError):
            cm.load_config('nonexistent')
    
    def test_get_agent_config_with_global_merge(self):
        """Test getting agent config with global settings merged."""
        # Create agents config file
        config_data = {
            'hypothesis_maker': {
                'model': 'gpt-4',
                'temperature': 0.7
            },
            'global': {
                'timeout': 60,
                'log_level': 'INFO'
            }
        }
        
        config_file = self.config_dir / "agents.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        cm = ConfigManager(config_dir=str(self.config_dir))
        
        agent_config = cm.get_agent_config('hypothesis_maker')
        
        # Should have agent-specific settings
        assert agent_config['model'] == 'gpt-4'
        assert agent_config['temperature'] == 0.7
        
        # Should have global settings merged in
        assert agent_config['timeout'] == 60
        assert agent_config['log_level'] == 'INFO'
    
    def test_get_agent_config_not_found(self):
        """Test getting config for unknown agent."""
        # Create empty agents config
        config_file = self.config_dir / "agents.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump({}, f)
        
        cm = ConfigManager(config_dir=str(self.config_dir))
        
        with pytest.raises(KeyError, match="Configuration for agent 'unknown_agent' not found"):
            cm.get_agent_config('unknown_agent')


class TestProjectRegistry:
    """Test the project registry high-level interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="project_reg_test_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_project_registry_initialization(self):
        """Test project registry initializes correctly."""
        pr = ProjectRegistry(data_dir=str(self.temp_dir))
        
        assert pr.data_dir == self.temp_dir
        assert isinstance(pr.registry_manager, RegistryManager)
    
    def test_create_idea(self):
        """Test creating a new research idea."""
        pr = ProjectRegistry(data_dir=str(self.temp_dir))
        
        idea_data = {
            'title': 'Test Idea',
            'hypothesis': 'Test hypothesis that is long enough to pass validation',
            'rationale': 'Test rationale that provides sufficient context and explanation for the hypothesis',
            'domain_tags': ['test', 'stellar dynamics']
        }
        
        idea_id = pr.create_idea(idea_data)
        
        assert idea_id is not None
        assert len(idea_id) == 26  # ULID length
        
        # Verify idea was stored
        retrieved_idea = pr.get_idea(idea_id)
        assert retrieved_idea is not None
        assert retrieved_idea['title'] == 'Test Idea'
        assert retrieved_idea['status'] == 'Proposed'  # Default status
    
    def test_create_idea_missing_required_field(self):
        """Test creating idea with missing required field."""
        pr = ProjectRegistry(data_dir=str(self.temp_dir))
        
        incomplete_idea = {
            'title': 'Test Idea',
            # Missing hypothesis, rationale, domain_tags
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            pr.create_idea(incomplete_idea)
    
    def test_update_idea(self):
        """Test updating an existing idea."""
        pr = ProjectRegistry(data_dir=str(self.temp_dir))
        
        # Create initial idea
        idea_data = TestFixtures.get_mock_idea()
        idea_id = pr.create_idea(idea_data)
        
        # Update idea
        updates = {'status': 'Approved', 'feasibility_score': 4}
        pr.update_idea(idea_id, updates)
        
        # Verify updates
        updated_idea = pr.get_idea(idea_id)
        assert updated_idea['status'] == 'Approved'
        assert updated_idea['feasibility_score'] == 4
    
    def test_get_ideas_by_status(self):
        """Test getting ideas filtered by status."""
        pr = ProjectRegistry(data_dir=str(self.temp_dir))
        
        # Create ideas with different statuses
        idea1 = TestFixtures.get_mock_idea()
        idea1['title'] = 'Proposed Idea'
        id1 = pr.create_idea(idea1)
        
        idea2 = TestFixtures.get_mock_idea()
        idea2['title'] = 'Approved Idea'
        id2 = pr.create_idea(idea2)
        pr.update_idea(id2, {'status': 'Approved'})
        
        # Get proposed ideas
        proposed_ideas = pr.get_ideas_by_status('Proposed')
        assert len(proposed_ideas) == 1
        assert proposed_ideas[0]['title'] == 'Proposed Idea'
        
        # Get approved ideas
        approved_ideas = pr.get_ideas_by_status('Approved')
        assert len(approved_ideas) == 1
        assert approved_ideas[0]['title'] == 'Approved Idea'
    
    def test_create_project(self):
        """Test creating a project entry."""
        pr = ProjectRegistry(data_dir=str(self.temp_dir))
        
        idea_id = '01TEST123456789'
        slug = 'test-project'
        project_path = '/path/to/project'
        
        project_data = pr.create_project(idea_id, slug, project_path)
        
        assert project_data['idea_id'] == idea_id
        assert project_data['slug'] == slug
        assert project_data['path'] == project_path
        assert project_data['maturity'] == 'Preparing'
        
        # Verify project was stored
        retrieved_project = pr.get_project(idea_id)
        assert retrieved_project is not None
        assert retrieved_project['slug'] == slug
    
    def test_get_pipeline_statistics(self):
        """Test getting pipeline statistics."""
        pr = ProjectRegistry(data_dir=str(self.temp_dir))
        
        # Create some test data
        idea1 = TestFixtures.get_mock_idea()
        id1 = pr.create_idea(idea1)
        
        idea2 = TestFixtures.get_mock_idea()
        idea2['title'] = 'Another Idea'
        id2 = pr.create_idea(idea2)
        pr.update_idea(id2, {'status': 'Approved'})
        
        # Get statistics
        stats = pr.get_pipeline_statistics()
        
        assert stats['total_ideas'] == 2
        assert stats['idea_statuses']['Proposed'] == 1
        assert stats['idea_statuses']['Approved'] == 1
        assert 'recent_ideas_30d' in stats
    
    def test_search_ideas(self):
        """Test searching ideas by text content."""
        pr = ProjectRegistry(data_dir=str(self.temp_dir))
        
        # Create ideas with searchable content
        idea1 = TestFixtures.get_mock_idea()
        idea1['title'] = 'Dark Matter Research'
        idea1['hypothesis'] = 'Dark matter has unique properties'
        pr.create_idea(idea1)
        
        idea2 = TestFixtures.get_mock_idea()
        idea2['title'] = 'Stellar Evolution Study'
        idea2['hypothesis'] = 'Stars evolve in predictable patterns'
        pr.create_idea(idea2)
        
        # Search for dark matter
        results = pr.search_ideas('dark matter')
        assert len(results) == 1
        assert 'Dark Matter Research' in results[0]['title']
        
        # Search for evolution
        results = pr.search_ideas('evolution')
        assert len(results) == 1
        assert 'Stellar Evolution Study' in results[0]['title']
        
        # Search for non-existent term
        results = pr.search_ideas('quantum gravity')
        assert len(results) == 0


class TestAstroAgentPipeline:
    """Test the main pipeline orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
        self.config_dir = self.temp_dir / "config"
        self.data_dir = self.temp_dir / "data"
        
        # Create directory structure
        self.config_dir.mkdir()
        (self.data_dir / "registry").mkdir(parents=True)
        (self.data_dir / "checkpoints").mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        # Create minimal config files
        self._create_minimal_configs()
        
        pipeline = AstroAgentPipeline(
            config_dir=str(self.config_dir),
            data_dir=str(self.data_dir)
        )
        
        assert pipeline.config_dir == self.config_dir
        assert pipeline.data_dir == self.data_dir
        assert pipeline.registry_manager is not None
        assert pipeline.workflow is not None
    
    def test_create_initial_state(self):
        """Test creating initial pipeline state."""
        self._create_minimal_configs()
        
        pipeline = AstroAgentPipeline(
            config_dir=str(self.config_dir),
            data_dir=str(self.data_dir)
        )
        
        agent_inputs = {'domain_tags': ['test'], 'n_hypotheses': 3}
        initial_state = pipeline.create_initial_state(agent_inputs)
        
        assert initial_state['current_state'] == 'start'
        assert initial_state['agent_inputs'] == agent_inputs
        assert initial_state['total_agents_run'] == 0
        assert initial_state['errors'] == []
        assert 'pipeline_id' in initial_state
    
    def _create_minimal_configs(self):
        """Create minimal configuration files for testing."""
        import yaml
        
        # Create agents config
        agents_config = {
            'hypothesis_maker': TestFixtures.get_mock_agent_config(),
            'reviewer': TestFixtures.get_mock_agent_config(),
            'experiment_designer': TestFixtures.get_mock_agent_config()
        }
        
        with open(self.config_dir / "agents.yaml", 'w') as f:
            yaml.dump(agents_config, f)
        
        # Create orchestration config
        orchestration_config = {
            'orchestration': {
                'engine': 'langgraph',
                'checkpointing': True
            }
        }
        
        with open(self.config_dir / "orchestration.yaml", 'w') as f:
            yaml.dump(orchestration_config, f)
        
        # Create datasources config
        datasources_config = {
            'literature': {
                'ads': {'base_url': 'https://api.adsabs.harvard.edu/v1'}
            }
        }
        
        with open(self.config_dir / "datasources.yaml", 'w') as f:
            yaml.dump(datasources_config, f)
