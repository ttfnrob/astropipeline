#!/usr/bin/env python3
"""
Quick test script to demonstrate AstroAgent functionality.
"""

import os
import sys
from pathlib import Path


def test_hypothesis_generation():
    """Test hypothesis generation without full CLI."""
    
    print("🚀 Testing AstroAgent Hypothesis Generation")
    print("=" * 50)
    
    try:
        # Set mock environment
        os.environ['ADS_API_TOKEN'] = 'mock_token_for_testing'
        
        # Import and test core functionality
        from astroagent.agents.common import AgentExecutionContext, generate_ulid
        from astroagent.agents.hypothesis_maker import HypothesisMaker
        from astroagent.agents.reviewer import Reviewer
        
        # Test ULID generation
        test_id = generate_ulid()
        print(f"✅ Generated ULID: {test_id}")
        
        # Test hypothesis maker
        hm_config = {
            'model': 'gpt-4',
            'temperature': 0.7,
            'system_prompt': 'Generate astrophysics hypotheses.',
            'user_prompt_template': 'Generate {n_hypotheses} hypotheses for {domain_tags}',
            'guardrails': {
                'min_hypothesis_words': 20,
                'max_hypothesis_words': 200,
                'required_fields': ['hypothesis', 'rationale']
            }
        }
        
        hm = HypothesisMaker(hm_config)
        print(f"✅ Created Hypothesis Maker: {hm.name}")
        
        # Test hypothesis generation
        context = AgentExecutionContext(
            agent_name='hypothesis_maker',
            state_name='hypothesis_generation',
            input_data={
                'domain_tags': ['stellar evolution', 'galactic dynamics'],
                'n_hypotheses': 2,
                'recency_years': 3
            }
        )
        
        result = hm.run(context)
        
        if result.success:
            hypotheses = result.output_data.get('hypotheses', [])
            print(f"✅ Generated {len(hypotheses)} hypotheses:")
            for i, hypothesis in enumerate(hypotheses, 1):
                print(f"   {i}. {hypothesis['title']}")
                print(f"      Effort: {hypothesis['est_effort_days']} days")
        else:
            print(f"❌ Hypothesis generation failed: {result.error_message}")
            return False
        
        # Test reviewer
        rv_config = {
            'model': 'gpt-4',
            'temperature': 0.3,
            'approval_thresholds': {
                'approved': {'total_min': 13, 'individual_min': 3},
                'revision': {'total_min': 9, 'total_max': 12},
                'rejected': {'total_max': 8}
            }
        }
        
        reviewer = Reviewer(rv_config)
        print(f"✅ Created Reviewer: {reviewer.name}")
        
        # Test a mock idea review
        mock_idea = {
            'idea_id': test_id,
            'title': 'Test Stellar Evolution Hypothesis',
            'hypothesis': 'Stellar evolution models exhibit systematic deviations in metal-poor environments that correlate with galactic chemical evolution patterns.',
            'rationale': 'Recent observations show unexplained discrepancies between predicted and observed stellar populations in ancient galactic components.',
            'domain_tags': ['stellar evolution', 'galactic chemistry'],
            'required_data': ['Gaia DR3', 'SDSS'],
            'methods': ['Isochrone fitting', 'Chemical abundance analysis'],
            'est_effort_days': 14,
            'status': 'Proposed'
        }
        
        review_result = reviewer._review_idea(mock_idea)
        
        print(f"✅ Review completed:")
        print(f"   Status: {review_result['status']}")
        print(f"   Total Score: {review_result['total_score']}/20")
        print(f"   Impact: {review_result['impact_score']}/5")
        print(f"   Feasibility: {review_result['feasibility_score']}/5")
        print(f"   Testability: {review_result['testability_score']}/5")
        print(f"   Novelty: {review_result['novelty_score']}/5")
        
        # Test registry
        from astroagent.orchestration.tools import RegistryManager
        
        registry = RegistryManager('data')
        print(f"✅ Registry system: {len(registry.registries)} registries ready")
        
        # Add the reviewed idea to registry
        registry.append_to_registry('ideas_register', review_result)
        print("✅ Added idea to registry")
        
        # Query it back
        ideas_df = registry.load_registry('ideas_register')
        print(f"✅ Registry now contains {len(ideas_df)} ideas")
        
        print("\n🎉 AstroAgent Pipeline Core Functionality VERIFIED!")
        print("\n🔧 To use with real research:")
        print("1. Get ADS API token: https://ui.adsabs.harvard.edu/user/settings/token")
        print("2. Get OpenAI or Anthropic API key")
        print("3. Create .env file with your tokens")
        print("4. The system will then generate real hypotheses from literature!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hypothesis_generation()
    sys.exit(0 if success else 1)
