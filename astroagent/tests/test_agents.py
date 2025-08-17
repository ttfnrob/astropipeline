"""
Unit tests for AstroAgent Pipeline agents.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from ..agents.common import (
    BaseAgent, AgentExecutionContext, AgentResult, 
    IdeaSchema, generate_ulid, create_project_folder
)
from ..agents.hypothesis_maker import HypothesisMaker
from ..agents.reviewer import Reviewer
from ..agents.experiment_designer import ExperimentDesigner
from . import TestFixtures


class TestBaseAgent:
    """Test the base agent class."""
    
    def test_base_agent_initialization(self):
        """Test base agent can be initialized with config."""
        config = {'model': 'test', 'temperature': 0.5}
        
        # Create concrete implementation for testing
        class TestAgent(BaseAgent):
            def execute(self, context):
                return AgentResult(
                    success=True,
                    agent_name=self.name,
                    execution_id=context.execution_id,
                    execution_time_seconds=0,
                    input_hash="test",
                    output_hash="test"
                )
        
        agent = TestAgent(config)
        assert agent.config == config
        assert agent.name == "TestAgent"
        assert 'total_executions' in agent.execution_stats
    
    def test_agent_execution_context_creation(self):
        """Test creation of agent execution context."""
        context = AgentExecutionContext(
            agent_name="test_agent",
            state_name="test_state",
            input_data={"key": "value"}
        )
        
        assert context.agent_name == "test_agent"
        assert context.state_name == "test_state"
        assert context.input_data["key"] == "value"
        assert context.retry_count == 0
    
    def test_idea_schema_validation(self):
        """Test IdeaSchema validates correctly."""
        valid_idea = TestFixtures.get_mock_idea()
        
        # Should validate without errors
        idea_schema = IdeaSchema(**valid_idea)
        assert idea_schema.idea_id == valid_idea['idea_id']
        assert idea_schema.title == valid_idea['title']
    
    def test_idea_schema_validation_fails_invalid_data(self):
        """Test IdeaSchema fails validation with invalid data."""
        invalid_idea = {
            'title': 'Test',
            'hypothesis': 'Too short',  # Too short
            # Missing required fields
        }
        
        with pytest.raises(Exception):  # Pydantic validation error
            IdeaSchema(**invalid_idea)
    
    def test_generate_ulid(self):
        """Test ULID generation."""
        ulid1 = generate_ulid()
        ulid2 = generate_ulid()
        
        assert len(ulid1) == 26
        assert len(ulid2) == 26
        assert ulid1 != ulid2
        assert ulid1.startswith('01')
    
    def test_create_project_folder(self):
        """Test project folder creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            project_path, slug = create_project_folder(
                idea_id="01TEST123456789",
                title="Test Project With Spaces",
                base_path=str(base_path)
            )
            
            assert Path(project_path).exists()
            assert "test-project-with-spaces" in slug
            assert Path(project_path, "research").exists()
            assert Path(project_path, "notebooks").exists()
            assert Path(project_path, "scripts").exists()


class TestHypothesisMaker:
    """Test the Hypothesis Maker agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = TestFixtures.get_mock_agent_config()
        self.config['system_prompt'] = 'Generate hypotheses for testing.'
        self.config['user_prompt_template'] = 'Generate {n_hypotheses} hypotheses about {domain_tags}'
    
    @patch.dict('os.environ', {'ADS_API_TOKEN': 'mock_token_123'})
    def test_hypothesis_maker_initialization(self):
        """Test HypothesisMaker initializes correctly."""
        hm = HypothesisMaker(self.config)
        
        assert hm.name == "HypothesisMaker"
        assert hm.config == self.config
        assert hm.model == 'mock-gpt-4'
        assert hm.temperature == 0.7
    
    @patch.dict('os.environ', {'ADS_API_TOKEN': 'mock_token_123'})
    @patch('astroagent.agents.hypothesis_maker.ADSSearchService')
    @patch('astroagent.agents.hypothesis_maker.LiteratureService')
    def test_hypothesis_maker_execute_success(self, mock_lit_service, mock_ads_service):
        """Test successful hypothesis generation."""
        # Mock the services
        mock_ads_service.return_value.search_recent.return_value = [
            TestFixtures.get_mock_paper()
        ]
        mock_lit_service.return_value.extract_trending_topics.return_value = [
            {'term': 'dark matter', 'score': 0.8}
        ]
        
        hm = HypothesisMaker(self.config)
        
        context = AgentExecutionContext(
            agent_name="hypothesis_maker",
            state_name="hypothesis_generation",
            input_data={
                'domain_tags': ['stellar dynamics', 'dark matter'],
                'n_hypotheses': 2,
                'recency_years': 3
            }
        )
        
        result = hm.run(context)
        
        assert result.success
        assert result.agent_name == "HypothesisMaker"
        assert 'hypotheses' in result.output_data
        assert 'count' in result.output_data
        assert len(result.registry_updates) > 0
    
    @patch.dict('os.environ', {'ADS_API_TOKEN': 'mock_token_123'})
    def test_hypothesis_maker_input_validation(self):
        """Test input validation for HypothesisMaker."""
        hm = HypothesisMaker(self.config)
        
        # Valid input
        valid_context = AgentExecutionContext(
            agent_name="hypothesis_maker",
            state_name="hypothesis_generation",
            input_data={'domain_tags': ['test'], 'n_hypotheses': 3}
        )
        assert hm.validate_input(valid_context)
        
        # Invalid input - missing domain_tags
        invalid_context = AgentExecutionContext(
            agent_name="hypothesis_maker",
            state_name="hypothesis_generation",
            input_data={'n_hypotheses': 3}
        )
        assert not hm.validate_input(invalid_context)
        
        # Invalid input - empty domain_tags
        invalid_context2 = AgentExecutionContext(
            agent_name="hypothesis_maker",
            state_name="hypothesis_generation",
            input_data={'domain_tags': [], 'n_hypotheses': 3}
        )
        assert not hm.validate_input(invalid_context2)
    
    @patch.dict('os.environ', {'ADS_API_TOKEN': 'mock_token_123'})
    def test_hypothesis_validation(self):
        """Test individual hypothesis validation."""
        hm = HypothesisMaker(self.config)
        
        valid_hypothesis = TestFixtures.get_mock_idea()
        assert hm._validate_hypothesis(valid_hypothesis)
        
        # Test invalid hypothesis - too short
        invalid_hypothesis = valid_hypothesis.copy()
        invalid_hypothesis['hypothesis'] = 'Too short'
        assert not hm._validate_hypothesis(invalid_hypothesis)
        
        # Test missing required field
        invalid_hypothesis2 = valid_hypothesis.copy()
        del invalid_hypothesis2['hypothesis']
        assert not hm._validate_hypothesis(invalid_hypothesis2)


class TestReviewer:
    """Test the Reviewer agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = TestFixtures.get_mock_agent_config()
        self.config['scoring_rubric'] = {
            'impact': {5: 'High impact', 4: 'Medium impact'},
            'feasibility': {5: 'Highly feasible', 4: 'Feasible'}
        }
        self.config['approval_thresholds'] = {
            'approved': {'total_min': 13, 'individual_min': 3},
            'revision': {'total_min': 9, 'total_max': 12},
            'rejected': {'total_max': 8}
        }
    
    def test_reviewer_initialization(self):
        """Test Reviewer initializes correctly."""
        reviewer = Reviewer(self.config)
        
        assert reviewer.name == "Reviewer"
        assert reviewer.config == self.config
        assert reviewer.temperature == 0.7  # Should use config default since not overridden
    
    @patch('astroagent.agents.reviewer.LiteratureService')
    def test_reviewer_execute_success(self, mock_lit_service):
        """Test successful review execution."""
        reviewer = Reviewer(self.config)
        
        context = AgentExecutionContext(
            agent_name="reviewer",
            state_name="initial_review",
            input_data={'filter': {'status': 'Proposed'}}
        )
        
        result = reviewer.run(context)
        
        assert result.success
        assert result.agent_name == "Reviewer"
        assert 'reviewed_count' in result.output_data
    
    def test_assess_novelty(self):
        """Test novelty assessment logic."""
        reviewer = Reviewer(self.config)
        
        # Test with typical values
        idea = TestFixtures.get_mock_idea()
        novelty_score = reviewer._assess_novelty(idea)
        
        assert isinstance(novelty_score, int)
        assert 1 <= novelty_score <= 5
    
    def test_assess_impact(self):
        """Test impact assessment logic."""
        reviewer = Reviewer(self.config)
        
        idea = TestFixtures.get_mock_idea()
        impact_score = reviewer._assess_impact(idea)
        
        assert isinstance(impact_score, int)
        assert 1 <= impact_score <= 5
        
        # Test with high-impact keywords
        high_impact_idea = idea.copy()
        high_impact_idea['hypothesis'] = 'This revolutionary discovery will fundamentally change our understanding of dark matter paradigms.'
        high_impact_score = reviewer._assess_impact(high_impact_idea)
        
        assert high_impact_score >= impact_score
    
    def test_assess_feasibility(self):
        """Test feasibility assessment logic."""
        reviewer = Reviewer(self.config)
        
        idea = TestFixtures.get_mock_idea()
        feasibility_score = reviewer._assess_feasibility(idea)
        
        assert isinstance(feasibility_score, int)
        assert 1 <= feasibility_score <= 5
    
    def test_assess_testability(self):
        """Test testability assessment logic."""
        reviewer = Reviewer(self.config)
        
        idea = TestFixtures.get_mock_idea()
        testability_score = reviewer._assess_testability(idea)
        
        assert isinstance(testability_score, int)
        assert 1 <= testability_score <= 5
    
    def test_determine_status(self):
        """Test status determination logic."""
        reviewer = Reviewer(self.config)
        
        # High scores should be approved
        high_scores = {'impact': 5, 'feasibility': 5, 'testability': 4, 'novelty': 4}
        status = reviewer._determine_status(18, high_scores)
        assert status == 'Approved'
        
        # Medium scores should need revision
        medium_scores = {'impact': 3, 'feasibility': 3, 'testability': 3, 'novelty': 3}
        status = reviewer._determine_status(12, medium_scores)
        assert status == 'Needs Revision'
        
        # Low scores should be rejected
        low_scores = {'impact': 2, 'feasibility': 2, 'testability': 2, 'novelty': 2}
        status = reviewer._determine_status(8, low_scores)
        assert status == 'Rejected'


class TestExperimentDesigner:
    """Test the Experiment Designer agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = TestFixtures.get_mock_agent_config()
        self.config['ready_checklist'] = [
            'experiment_plan_complete',
            'data_sources_verified',
            'methods_documented'
        ]
    
    def test_experiment_designer_initialization(self):
        """Test ExperimentDesigner initializes correctly."""
        ed = ExperimentDesigner(self.config)
        
        assert ed.name == "ExperimentDesigner"
        assert ed.config == self.config
    
    def test_experiment_designer_execute_success(self):
        """Test successful experiment design."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup temporary templates directory
            templates_dir = Path(temp_dir) / "templates"
            templates_dir.mkdir()
            
            # Create a mock template
            template_path = templates_dir / "experiment_plan_template.md"
            template_path.write_text("# Experiment Plan: <idea_id> <title>")
            
            ed = ExperimentDesigner(self.config)
            
            # Mock the project directory to use temp directory
            with patch('astroagent.agents.experiment_designer.create_project_folder') as mock_create:
                mock_project_path = Path(temp_dir) / "test_project"
                mock_project_path.mkdir()
                (mock_project_path / "research").mkdir()
                (mock_project_path / "notebooks").mkdir()
                (mock_project_path / "scripts").mkdir()
                (mock_project_path / "artefacts").mkdir()
                
                mock_create.return_value = (str(mock_project_path), "test-slug")
                
                context = AgentExecutionContext(
                    agent_name="experiment_designer",
                    state_name="experiment_design",
                    input_data={'idea_id': '01TEST123456789'}
                )
                
                result = ed.run(context)
                
                assert result.success
                assert result.agent_name == "ExperimentDesigner"
                assert 'project_path' in result.output_data
                assert 'experiment_plan' in result.output_data
    
    def test_generate_experiment_plan(self):
        """Test experiment plan generation."""
        ed = ExperimentDesigner(self.config)
        
        idea = TestFixtures.get_mock_idea()
        plan = ed._generate_experiment_plan(idea)
        
        assert isinstance(plan, str)
        assert idea['idea_id'] in plan
        assert idea['title'] in plan
        assert 'Objectives' in plan
        assert 'Methods' in plan
    
    def test_readiness_checks(self):
        """Test project readiness validation."""
        ed = ExperimentDesigner(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create required structure
            (project_path / "experiment_plan.md").write_text("# Plan")
            (project_path / "research").mkdir()
            (project_path / "notebooks").mkdir()
            (project_path / "scripts").mkdir()
            (project_path / "artefacts").mkdir()
            
            idea = TestFixtures.get_mock_idea()
            checks_passed = ed._run_readiness_checks(str(project_path), idea)
            
            assert checks_passed  # Should pass with proper structure
    
    def test_input_validation(self):
        """Test input validation for ExperimentDesigner."""
        ed = ExperimentDesigner(self.config)
        
        # Should accept empty input (processes all approved ideas)
        empty_context = AgentExecutionContext(
            agent_name="experiment_designer",
            state_name="experiment_design",
            input_data={}
        )
        assert ed.validate_input(empty_context)
        
        # Should accept specific idea_id
        specific_context = AgentExecutionContext(
            agent_name="experiment_designer",
            state_name="experiment_design",
            input_data={'idea_id': '01TEST123456789'}
        )
        assert ed.validate_input(specific_context)
