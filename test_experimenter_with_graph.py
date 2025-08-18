#!/usr/bin/env python3
"""
Test experimenter agent using the orchestration graph to ensure registry updates are applied.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_experimenter_with_graph():
    """Test experimenter using the orchestration graph."""
    
    print("🚀 Testing Experimenter with Graph Orchestration")
    print("=" * 60)
    
    try:
        from astroagent.orchestration.tools import RegistryManager
        from astroagent.orchestration.graph import experimenter_node, PipelineState
        from astroagent.agents.common import generate_ulid
        
        # Load registry to get ready projects
        registry = RegistryManager('data')
        projects_df = registry.load_registry('project_index')
        ready_projects = projects_df[projects_df['maturity'] == 'Ready for Execution']
        
        if len(ready_projects) == 0:
            print("⚠️  No projects ready for execution")
            return True
        
        first_project = ready_projects.iloc[0]
        project_id = first_project['idea_id']
        
        print(f"📊 Testing with project: {project_id}")
        print(f"   Current maturity: {first_project['maturity']}")
        
        # Create pipeline state for experimenter node
        state = PipelineState(
            current_idea_id=project_id,
            current_state='experiment_execution',
            previous_state='experiment_design',
            agent_inputs={},
            agent_outputs={},
            pipeline_id=generate_ulid(),
            execution_start=datetime.now(timezone.utc),
            total_agents_run=0,
            errors=[],
            retry_count=0,
            ideas_updated=[],
            projects_updated=[],
            config_dir='astroagent/config'
        )
        
        print(f"🧪 Executing experimenter_node...")
        
        # Execute experimenter node (this should apply registry updates)
        result_state = experimenter_node(state)
        
        print(f"✅ Node execution completed")
        print(f"   Final state: {result_state.get('current_state')}")
        print(f"   Errors: {len(result_state.get('errors', []))}")
        
        # Check if project status was updated
        updated_projects_df = registry.load_registry('project_index')
        updated_project = updated_projects_df[updated_projects_df['idea_id'] == project_id]
        
        if not updated_project.empty:
            updated_maturity = updated_project.iloc[0]['maturity']
            print(f"   Updated maturity: {updated_maturity}")
            
            if updated_maturity == 'Complete':
                print(f"🎉 SUCCESS: Project maturity updated to Complete!")
                return True
            else:
                print(f"❌ ISSUE: Project maturity still {updated_maturity}")
                return False
        else:
            print(f"❌ ERROR: Project not found after update")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_experimenter_with_graph()
    sys.exit(0 if success else 1)
