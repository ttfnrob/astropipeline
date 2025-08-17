"""
Integration tests for AstroAgent Pipeline.

Tests the full pipeline workflow from hypothesis generation to completion.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import yaml
import json

from ..orchestration.graph import AstroAgentPipeline
from ..orchestration.registry import ProjectRegistry
from ..agents import create_agent
from . import TestFixtures


@pytest.mark.integration
class TestFullPipeline:
    """Test the complete pipeline workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_workspace = TestFixtures.create_temp_workspace()
        self._create_full_test_configs()
        
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'ADS_API_TOKEN': 'mock_token_123',
            'OPENAI_API_KEY': 'mock_openai_key'
        })
        self.env_patcher.start()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        TestFixtures.cleanup_temp_workspace(self.temp_workspace)
    
    def _create_full_test_configs(self):
        """Create complete configuration files for testing."""
        config_dir = self.temp_workspace / "config"
        
        # Agents configuration
        agents_config = {
            'hypothesis_maker': {
                'model': 'mock-gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000,
                'system_prompt': 'Generate astrophysics hypotheses.',
                'user_prompt_template': 'Generate {n_hypotheses} hypotheses for {domain_tags}',
                'guardrails': {
                    'min_hypothesis_words': 20,
                    'max_hypothesis_words': 200,
                    'min_rationale_words': 50,
                    'max_rationale_words': 300,
                    'required_fields': ['hypothesis', 'rationale', 'required_data', 'methods'],
                    'forbidden_phrases': ['might be', 'could possibly']
                }
            },
            'reviewer': {
                'model': 'mock-gpt-4',
                'temperature': 0.3,
                'system_prompt': 'Review research hypotheses rigorously.',
                'scoring_rubric': {
                    'impact': {5: 'Revolutionary', 4: 'High impact', 3: 'Moderate', 2: 'Low', 1: 'Minimal'},
                    'feasibility': {5: 'Highly feasible', 4: 'Feasible', 3: 'Moderate', 2: 'Challenging', 1: 'Infeasible'},
                    'testability': {5: 'Highly testable', 4: 'Testable', 3: 'Moderate', 2: 'Difficult', 1: 'Untestable'},
                    'novelty': {5: 'Completely novel', 4: 'Novel', 3: 'Moderate', 2: 'Limited', 1: 'Not novel'}
                },
                'approval_thresholds': {
                    'approved': {'total_min': 13, 'individual_min': 3},
                    'revision': {'total_min': 9, 'total_max': 12},
                    'rejected': {'total_max': 8}
                }
            },
            'experiment_designer': {
                'model': 'mock-gpt-4',
                'temperature': 0.5,
                'system_prompt': 'Design rigorous experiments.',
                'ready_checklist': [
                    'experiment_plan_complete',
                    'data_sources_verified',
                    'methods_documented',
                    'risks_assessed'
                ]
            }
        }
        
        with open(config_dir / "agents.yaml", 'w') as f:
            yaml.dump(agents_config, f)
        
        # Orchestration configuration
        orchestration_config = {
            'orchestration': {
                'engine': 'langgraph',
                'checkpointing': True,
                'max_concurrent_states': 1,
                'state_timeout_minutes': 30
            },
            'routing_rules': {
                'hypothesis_generation_to_initial_review': {
                    'condition': 'always',
                    'action': 'transition',
                    'target': 'initial_review'
                }
            }
        }
        
        with open(config_dir / "orchestration.yaml", 'w') as f:
            yaml.dump(orchestration_config, f)
        
        # Data sources configuration
        datasources_config = {
            'literature': {
                'ads': {
                    'base_url': 'https://api.adsabs.harvard.edu/v1',
                    'rate_limit': 5000
                }
            }
        }
        
        with open(config_dir / "datasources.yaml", 'w') as f:
            yaml.dump(datasources_config, f)
        
        # Create template files
        templates_dir = self.temp_workspace / "templates"
        
        with open(templates_dir / "experiment_plan_template.md", 'w') as f:
            f.write("""# Experiment Plan: <idea_id> <title>

## Objectives
- Primary outcome: <metric and acceptance threshold>

## Data
- Sources: <catalogues, surveys, observatories>

## Methods
- Preprocessing
- Statistical analysis
- Validation

## Risks
- Data availability issues
- Selection bias concerns

## Resources
- Compute requirements
- Storage needs

## Timeline
- Milestone schedule
""")
    
    @patch('astroagent.services.search_ads.requests.Session')
    @patch('astroagent.services.literature.SentenceTransformer')
    def test_hypothesis_generation_and_review_workflow(self, mock_transformer, mock_session):
        """Test hypothesis generation followed by review."""
        
        # Mock ADS API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {
                'docs': [TestFixtures.get_mock_paper()]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_transformer.return_value = mock_model
        
        # Initialize pipeline
        pipeline = AstroAgentPipeline(
            config_dir=str(self.temp_workspace / "config"),
            data_dir=str(self.temp_workspace / "data")
        )
        
        # Test input
        agent_inputs = {
            'domain_tags': ['stellar dynamics', 'galaxy clusters'],
            'n_hypotheses': 2,
            'recency_years': 3
        }
        
        # Run hypothesis maker
        hm = create_agent('hypothesis_maker', str(self.temp_workspace / "config"))
        
        from ..agents.common import AgentExecutionContext
        hm_context = AgentExecutionContext(
            agent_name='hypothesis_maker',
            state_name='hypothesis_generation',
            input_data=agent_inputs
        )
        
        hm_result = hm.run(hm_context)
        
        # Verify hypothesis generation succeeded
        assert hm_result.success, f"Hypothesis generation failed: {hm_result.error_message}"
        assert 'hypotheses' in hm_result.output_data
        assert hm_result.output_data['count'] > 0
        
        # Simulate storing hypotheses in registry
        registry = ProjectRegistry(data_dir=str(self.temp_workspace / "data"))
        hypothesis_ids = []
        
        for hypothesis in hm_result.output_data['hypotheses']:
            idea_id = registry.create_idea(hypothesis)
            hypothesis_ids.append(idea_id)
        
        # Now test reviewer
        reviewer = create_agent('reviewer', str(self.temp_workspace / "config"))
        
        rv_context = AgentExecutionContext(
            agent_name='reviewer',
            state_name='initial_review',
            input_data={'filter': {'status': 'Proposed'}}
        )
        
        rv_result = reviewer.run(rv_context)
        
        # Verify review succeeded
        assert rv_result.success, f"Review failed: {rv_result.error_message}"
        assert 'reviewed_ideas' in rv_result.output_data
        assert rv_result.output_data['reviewed_count'] > 0
        
        # Check that reviews include required scoring fields
        for reviewed_idea in rv_result.output_data['reviewed_ideas']:
            assert 'impact_score' in reviewed_idea
            assert 'feasibility_score' in reviewed_idea
            assert 'testability_score' in reviewed_idea
            assert 'novelty_score' in reviewed_idea
            assert 'total_score' in reviewed_idea
            assert 'status' in reviewed_idea
            assert reviewed_idea['status'] in ['Approved', 'Needs Revision', 'Rejected']
    
    @patch('astroagent.agents.experiment_designer.create_project_folder')
    def test_experiment_design_workflow(self, mock_create_folder):
        """Test experiment design workflow."""
        
        # Setup mock project folder
        mock_project_path = self.temp_workspace / "mock_project"
        mock_project_path.mkdir()
        (mock_project_path / "research").mkdir()
        (mock_project_path / "notebooks").mkdir()
        (mock_project_path / "scripts").mkdir()
        (mock_project_path / "artefacts").mkdir()
        
        mock_create_folder.return_value = (str(mock_project_path), "test-project")
        
        # Create registry with an approved idea
        registry = ProjectRegistry(data_dir=str(self.temp_workspace / "data"))
        
        approved_idea = TestFixtures.get_mock_idea()
        approved_idea['status'] = 'Approved'
        idea_id = registry.create_idea(approved_idea)
        
        # Test experiment designer
        ed = create_agent('experiment_designer', str(self.temp_workspace / "config"))
        
        from ..agents.common import AgentExecutionContext
        ed_context = AgentExecutionContext(
            agent_name='experiment_designer',
            state_name='experiment_design',
            input_data={'idea_id': idea_id}
        )
        
        ed_result = ed.run(ed_context)
        
        # Verify experiment design succeeded
        assert ed_result.success, f"Experiment design failed: {ed_result.error_message}"
        assert 'project_path' in ed_result.output_data
        assert 'experiment_plan' in ed_result.output_data
        assert 'ready_checks_passed' in ed_result.output_data
        
        # Check that experiment plan contains expected sections
        plan = ed_result.output_data['experiment_plan']
        assert 'Objectives' in plan
        assert 'Data' in plan
        assert 'Methods' in plan
        assert 'Risks' in plan
        assert idea_id in plan
    
    def test_registry_integration(self):
        """Test registry integration across workflow steps."""
        
        # Initialize registry
        registry = ProjectRegistry(data_dir=str(self.temp_workspace / "data"))
        
        # Create and store an idea
        idea_data = TestFixtures.get_mock_idea()
        idea_id = registry.create_idea(idea_data)
        
        # Verify idea was stored
        stored_idea = registry.get_idea(idea_id)
        assert stored_idea is not None
        assert stored_idea['title'] == idea_data['title']
        assert stored_idea['status'] == 'Proposed'
        
        # Update idea status (simulating review)
        registry.update_idea(idea_id, {
            'status': 'Approved',
            'impact_score': 4,
            'feasibility_score': 5,
            'testability_score': 4,
            'novelty_score': 3,
            'total_score': 16
        })
        
        # Verify updates
        updated_idea = registry.get_idea(idea_id)
        assert updated_idea['status'] == 'Approved'
        assert updated_idea['total_score'] == 16
        
        # Create project (simulating experiment design)
        project_data = registry.create_project(
            idea_id=idea_id,
            slug='test-stellar-dynamics',
            project_path=f'projects/Preparing/{idea_id}__test-stellar-dynamics'
        )
        
        # Verify project was created
        stored_project = registry.get_project(idea_id)
        assert stored_project is not None
        assert stored_project['slug'] == 'test-stellar-dynamics'
        assert stored_project['maturity'] == 'Preparing'
        
        # Update project status
        registry.update_project(idea_id, {
            'maturity': 'Ready',
            'ready_checklist_passed': True
        })
        
        # Verify project updates
        updated_project = registry.get_project(idea_id)
        assert updated_project['maturity'] == 'Ready'
        assert updated_project['ready_checklist_passed'] is True
        
        # Test pipeline statistics
        stats = registry.get_pipeline_statistics()
        assert stats['total_ideas'] == 1
        assert stats['total_projects'] == 1
        assert 'Approved' in stats['idea_statuses']
        assert 'Ready' in stats['project_maturities']
    
    def test_error_handling_and_recovery(self):
        """Test pipeline error handling and recovery mechanisms."""
        
        # Test with invalid input
        registry = ProjectRegistry(data_dir=str(self.temp_workspace / "data"))
        
        # Try to create idea with missing required fields
        invalid_idea = {
            'title': 'Invalid Idea',
            # Missing required fields
        }
        
        with pytest.raises(ValueError):
            registry.create_idea(invalid_idea)
        
        # Try to get non-existent idea
        non_existent_idea = registry.get_idea('01NONEXISTENT123456')
        assert non_existent_idea is None
        
        # Try to update non-existent idea (should not raise error)
        registry.update_idea('01NONEXISTENT123456', {'status': 'Updated'})
        
        # Verify no changes were made
        still_none = registry.get_idea('01NONEXISTENT123456')
        assert still_none is None
    
    def test_search_and_query_functionality(self):
        """Test search and query functionality across the pipeline."""
        
        registry = ProjectRegistry(data_dir=str(self.temp_workspace / "data"))
        
        # Create multiple ideas with different content
        ideas = [
            {
                'title': 'Dark Matter Dynamics',
                'hypothesis': 'Dark matter exhibits novel dynamical properties in galaxy clusters that can be detected through gravitational lensing.',
                'rationale': 'Recent theoretical work suggests dark matter may have self-interaction properties.',
                'domain_tags': ['dark matter', 'galaxy clusters'],
                'novelty_refs': ['2023ApJ...900...1A'],
                'required_data': ['HSC', 'Subaru'],
                'methods': ['Weak lensing analysis'],
                'est_effort_days': 14
            },
            {
                'title': 'Stellar Evolution Models',
                'hypothesis': 'New stellar evolution models can better predict red giant branch morphology in globular clusters.',
                'rationale': 'Current models show systematic deviations from observations in metal-poor environments.',
                'domain_tags': ['stellar evolution', 'globular clusters'],
                'novelty_refs': ['2023MNRAS.520.1234B'],
                'required_data': ['Gaia DR3', 'HST'],
                'methods': ['Isochrone fitting', 'Bayesian inference'],
                'est_effort_days': 10
            },
            {
                'title': 'Exoplanet Atmospheric Composition',
                'hypothesis': 'Transit spectroscopy reveals systematic differences in atmospheric composition for hot Jupiters around different stellar types.',
                'rationale': 'Stellar irradiation should affect atmospheric chemistry in predictable ways.',
                'domain_tags': ['exoplanets', 'atmospheric composition'],
                'novelty_refs': ['2023A&A...650.L5C'],
                'required_data': ['JWST', 'HST'],
                'methods': ['Spectroscopic analysis', 'Atmospheric modeling'],
                'est_effort_days': 12
            }
        ]
        
        idea_ids = []
        for idea in ideas:
            idea_id = registry.create_idea(idea)
            idea_ids.append(idea_id)
        
        # Test searching by different terms
        dark_matter_results = registry.search_ideas('dark matter')
        assert len(dark_matter_results) == 1
        assert 'Dark Matter Dynamics' in dark_matter_results[0]['title']
        
        stellar_results = registry.search_ideas('stellar')
        assert len(stellar_results) == 1
        assert 'Stellar Evolution Models' in stellar_results[0]['title']
        
        cluster_results = registry.search_ideas('cluster')
        assert len(cluster_results) == 2  # Both dark matter and stellar evolution mention clusters
        
        # Test getting ideas by status
        proposed_ideas = registry.get_ideas_by_status('Proposed')
        assert len(proposed_ideas) == 3
        
        # Update one idea status and test filtering
        registry.update_idea(idea_ids[0], {'status': 'Approved'})
        
        approved_ideas = registry.get_ideas_by_status('Approved')
        assert len(approved_ideas) == 1
        
        remaining_proposed = registry.get_ideas_by_status('Proposed')
        assert len(remaining_proposed) == 2
        
        # Test idea history functionality
        # Create a revision of the first idea
        revised_idea = ideas[0].copy()
        revised_idea['title'] = 'Dark Matter Dynamics - Revised'
        revised_idea['parent_idea_id'] = idea_ids[0]
        revised_idea['version'] = 'v2'
        
        revised_id = registry.create_idea(revised_idea)
        
        # Get version history
        history = registry.get_idea_history(idea_ids[0])
        assert len(history) >= 2  # Original + revision
        
        # Check that both versions are present
        titles = [h['title'] for h in history]
        assert 'Dark Matter Dynamics' in titles
        assert 'Dark Matter Dynamics - Revised' in titles


@pytest.mark.integration
class TestServiceIntegration:
    """Test integration between services and agents."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_workspace = TestFixtures.create_temp_workspace()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        TestFixtures.cleanup_temp_workspace(self.temp_workspace)
    
    @patch('astroagent.services.search_ads.requests.Session')
    @patch('astroagent.services.literature.SentenceTransformer')
    def test_literature_service_integration(self, mock_transformer, mock_session):
        """Test literature service integration with agents."""
        
        # Mock external services
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {
                'docs': [
                    TestFixtures.get_mock_paper(),
                    {
                        'bibcode': '2023MNRAS.520.9999Z',
                        'title': 'Another Related Paper',
                        'author': ['Brown, A.', 'Davis, B.'],
                        'year': 2023,
                        'abstract': 'Additional research on dark matter and stellar dynamics.',
                        'citation_count': 25
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        # Mock sentence transformer for similarity computations
        mock_model = Mock()
        mock_model.encode.side_effect = [
            [1.0, 0.0, 0.0],  # Hypothesis embedding
            [0.8, 0.2, 0.0],  # Similar paper
            [0.2, 0.8, 0.0]   # Different paper
        ]
        mock_transformer.return_value = mock_model
        
        # Test ADS search integration
        with patch.dict('os.environ', {'ADS_API_TOKEN': 'mock_token'}):
            from ..services.search_ads import ADSSearchService
            from ..services.literature import LiteratureService
            
            ads_service = ADSSearchService()
            lit_service = LiteratureService(cache_dir=str(self.temp_workspace / "data" / "vectors"))
            
            # Test search functionality
            results = ads_service.search("dark matter", max_results=10)
            assert len(results) == 2
            
            # Test novelty assessment
            hypothesis = "We propose a new method for detecting dark matter through gravitational effects."
            assessment = lit_service.assess_novelty(hypothesis, results)
            
            assert 'novelty_score' in assessment
            assert 'similar_papers' in assessment
            assert assessment['novelty_rating'] >= 1
            assert assessment['novelty_rating'] <= 5


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
