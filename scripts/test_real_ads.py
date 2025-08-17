#!/usr/bin/env python3
"""
Test AstroAgent with real ADS API key and literature data.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def test_real_ads_integration():
    """Test hypothesis generation with real ADS literature data."""
    
    print("🚀 Testing AstroAgent with Real ADS Data")
    print("=" * 50)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if ADS API token is available
    ads_token = os.getenv('ADS_API_TOKEN')
    if not ads_token or ads_token == 'your_actual_ads_token_here':
        print("❌ Please update your .env file with a real ADS_API_TOKEN")
        print("   Get one from: https://ui.adsabs.harvard.edu/user/settings/token")
        return False
    
    print(f"✅ ADS API Token loaded: {ads_token[:10]}...{ads_token[-4:]}")
    
    try:
        # Import and test core functionality
        from astroagent.agents.common import AgentExecutionContext, generate_ulid
        from astroagent.agents.hypothesis_maker import HypothesisMaker
        from astroagent.services.search_ads import ADSSearchService
        
        # Test ADS service directly first
        print("\n📚 Testing ADS Literature Search...")
        ads_service = ADSSearchService(ads_token)
        
        # Search for a few recent papers on a specific topic
        papers = ads_service.search_recent(
            query="exoplanet atmosphere", 
            years=1, 
            max_results=5
        )
        
        if papers:
            print(f"✅ Found {len(papers)} recent papers on exoplanet atmospheres:")
            for i, paper in enumerate(papers[:3], 1):
                title = paper.get('title', ['Unknown'])[0] if isinstance(paper.get('title'), list) else paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                print(f"   {i}. {title} ({year})")
        else:
            print("❌ No papers found - check your ADS API token")
            return False
        
        # Test hypothesis maker with real literature
        print("\n🧠 Testing Hypothesis Generation with Real Data...")
        
        test_id = generate_ulid()
        print(f"✅ Generated ULID: {test_id}")
        
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
        
        # Test hypothesis generation with real literature context
        context = AgentExecutionContext(
            agent_name='hypothesis_maker',
            state_name='hypothesis_generation',
            input_data={
                'domain_tags': ['exoplanet atmosphere', 'spectroscopy'],
                'n_hypotheses': 2,
                'recency_years': 1
            }
        )
        
        result = hm.run(context)
        
        if result.success:
            hypotheses = result.output_data.get('hypotheses', [])
            print(f"✅ Generated {len(hypotheses)} hypotheses with real literature context:")
            for i, hypothesis in enumerate(hypotheses, 1):
                print(f"\n   {i}. {hypothesis['title']}")
                print(f"      Hypothesis: {hypothesis['hypothesis'][:100]}...")
                print(f"      Effort: {hypothesis['est_effort_days']} days")
                print(f"      Data needed: {', '.join(hypothesis.get('required_data', []))}")
        else:
            print(f"❌ Hypothesis generation failed: {result.error_message}")
            return False
        
        print("\n🎉 REAL ADS INTEGRATION WORKING!")
        print("✨ Your AstroAgent system is now connected to live literature data!")
        print("\n🔬 Next steps:")
        print("1. Try different research domains (e.g., 'stellar evolution', 'galaxy formation')")
        print("2. Add OpenAI/Anthropic API key for AI-powered hypothesis generation")
        print("3. Run full research pipeline with experiment design!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_real_ads_integration()
    sys.exit(0 if success else 1)
