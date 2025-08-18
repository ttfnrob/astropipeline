#!/usr/bin/env python3
"""
Test script to continue the pipeline for Ready for Execution projects.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pipeline_continuation():
    """Test continuing the pipeline for ready projects."""
    
    print("🚀 Testing Pipeline Continuation")
    print("=" * 50)
    
    try:
        from astroagent.orchestration.tools import RegistryManager
        from astroagent.agents import create_agent, AgentExecutionContext
        
        # Load registry
        registry = RegistryManager('data')
        
        # Check Ready for Execution projects
        projects_df = registry.load_registry('project_index')
        ready_projects = projects_df[projects_df['maturity'] == 'Ready for Execution']
        
        print(f"📊 Found {len(ready_projects)} projects ready for execution:")
        for _, project in ready_projects.iterrows():
            print(f"   - {project['idea_id']}: {project['slug']}")
        
        if len(ready_projects) == 0:
            print("⚠️  No projects ready for execution")
            return True
        
        # Test Experimenter agent on first ready project
        first_project = ready_projects.iloc[0]
        project_id = first_project['idea_id']
        
        print(f"\n🧪 Testing Experimenter agent with project: {project_id}")
        
        try:
            # Create Experimenter agent
            experimenter = create_agent('experimenter', 'astroagent/config')
            print("✅ Experimenter agent created successfully")
            
            # Create execution context
            context = AgentExecutionContext(
                agent_name='experimenter',
                state_name='experiment_execution',
                input_data={'idea_id': project_id}
            )
            
            # Execute experiment
            print(f"⚗️  Executing experiment for project {project_id}...")
            result = experimenter.run(context)
            
            if result.success:
                print(f"✅ Experiment execution successful!")
                print(f"   Results: {result.output_data.keys()}")
                
                # Check if project status was updated
                updated_projects_df = registry.load_registry('project_index')
                updated_project = updated_projects_df[updated_projects_df['idea_id'] == project_id].iloc[0]
                print(f"   Updated maturity: {updated_project.get('maturity', 'Unknown')}")
                
            else:
                print(f"❌ Experiment execution failed: {result.error_message}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Experimenter: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n🎉 PIPELINE CONTINUATION TEST SUCCESSFUL!")
        print(f"✨ Ready projects are now being processed by Experimenter!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_continuation()
    sys.exit(0 if success else 1)
