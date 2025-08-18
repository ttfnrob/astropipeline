#!/usr/bin/env python3
"""
Test the revision feedback loop in HypothesisMaker.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_revision_loop():
    """Test that HypothesisMaker processes 'Needs Revision' ideas."""
    
    print("🔄 Testing Revision Feedback Loop")
    print("=" * 50)
    
    try:
        from astroagent.orchestration.tools import RegistryManager
        from astroagent.agents import create_agent, AgentExecutionContext
        
        # Check current state
        registry = RegistryManager('data')
        ideas_df = registry.load_registry('ideas_register')
        
        print(f"📊 Current ideas in registry:")
        for _, idea in ideas_df.iterrows():
            print(f"   - {idea['idea_id']}: {idea['title'][:50]}... (Status: {idea['status']})")
        
        # Count ideas needing revision
        revision_ideas = ideas_df[ideas_df['status'] == 'Needs Revision']
        print(f"\n🔍 Found {len(revision_ideas)} ideas needing revision")
        
        if len(revision_ideas) == 0:
            print("✅ No ideas need revision at this time")
            return True
        
        # Test HypothesisMaker revision processing
        print(f"\n🧠 Testing HypothesisMaker revision processing...")
        
        try:
            hm = create_agent('hypothesis_maker', 'astroagent/config')
            print("✅ HypothesisMaker created successfully")
            
            # Create context for revision processing
            context = AgentExecutionContext(
                agent_name='hypothesis_maker',
                state_name='revision_processing',
                input_data={'generate_new': False}  # Only process revisions
            )
            
            # Execute revision processing
            result = hm.execute(context)
            
            if result.success:
                revised_count = result.output_data.get('revised_count', 0)
                new_count = result.output_data.get('new_count', 0)
                
                print(f"✅ Revision processing successful!")
                print(f"   Revised ideas: {revised_count}")
                print(f"   New ideas: {new_count}")
                
                if revised_count > 0:
                    print(f"\n📝 Revision details:")
                    revised_hypotheses = result.output_data.get('hypotheses', [])
                    for i, hypothesis in enumerate(revised_hypotheses[:revised_count]):
                        print(f"   {i+1}. {hypothesis['title']}")
                        print(f"      Parent ID: {hypothesis.get('parent_idea_id', 'None')}")
                        print(f"      Version: {hypothesis.get('version', 'Unknown')}")
                
                # Check registry was updated
                updated_ideas_df = registry.load_registry('ideas_register')
                under_revision = updated_ideas_df[updated_ideas_df['status'] == 'Under Revision']
                print(f"\n   📋 Registry updated: {len(under_revision)} ideas marked 'Under Revision'")
                
            else:
                print(f"❌ Revision processing failed: {result.error_message}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing revision processing: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n🎉 REVISION FEEDBACK LOOP WORKING!")
        print(f"✨ HypothesisMaker can now improve ideas based on reviewer feedback!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_revision_loop()
    sys.exit(0 if success else 1)
