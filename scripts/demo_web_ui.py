#!/usr/bin/env python3
"""
Demo script for AstroAgent Pipeline Web UI

This script demonstrates how to use the web interface with the existing pipeline system.
It runs a sample pipeline execution and shows the web UI monitoring it.
"""

import asyncio
import json
import os
import sys
import time
import subprocess
from pathlib import Path

# Add root project to path 
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_demo_environment():
    """Set up demo environment with mock API keys."""
    print("🔧 Setting up demo environment...")
    
    # Set mock environment variables for testing
    os.environ['ADS_API_TOKEN'] = 'mock_token_for_demo'
    os.environ['OPENAI_API_KEY'] = 'mock_openai_key_for_demo'
    
    # Ensure data directories exist
    data_dir = Path(__file__).parent.parent / "data"
    registry_dir = data_dir / "registry"
    
    data_dir.mkdir(exist_ok=True)
    registry_dir.mkdir(exist_ok=True)
    
    print("✅ Demo environment ready")

def check_web_ui_dependencies():
    """Check if web UI dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        return True
    except ImportError:
        return False

def install_web_ui_dependencies():
    """Install web UI dependencies."""
    print("📦 Installing web UI dependencies...")
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_file)
        ])
        print("✅ Web UI dependencies installed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def generate_demo_data():
    """Generate some demo data for the web UI to display."""
    print("📊 Generating demo data...")
    
    try:
        # Import after ensuring environment is set
        from astroagent.agents.common import AgentExecutionContext
        from astroagent.agents.hypothesis_maker import HypothesisMaker
        from astroagent.agents.reviewer import Reviewer
        from astroagent.orchestration.registry import ProjectRegistry
        
        # Create some demo hypotheses
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
        
        # Generate hypotheses
        context = AgentExecutionContext(
            agent_name='hypothesis_maker',
            state_name='hypothesis_generation',
            input_data={
                'domain_tags': ['stellar evolution', 'galactic dynamics', 'exoplanets'],
                'n_hypotheses': 3,
                'recency_years': 3
            }
        )
        
        result = hm.run(context)
        
        if result.success:
            hypotheses = result.output_data.get('hypotheses', [])
            print(f"✅ Generated {len(hypotheses)} demo hypotheses")
            
            # Review them
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
            
            # Review each hypothesis and add to registry
            registry = ProjectRegistry()
            
            for hypothesis in hypotheses:
                review_result = reviewer._review_idea(hypothesis)
                registry.create_idea(review_result)
                print(f"   • {hypothesis['title']} - Status: {review_result['status']}")
            
            print("✅ Demo data generated and added to registry")
            return True
        else:
            print(f"❌ Failed to generate demo hypotheses: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating demo data: {e}")
        return False

async def start_web_ui_server():
    """Start the web UI server."""
    print("🌐 Starting Web UI server...")
    
    try:
        # Change to root directory
        root_dir = Path(__file__).parent.parent
        os.chdir(str(root_dir))
        
        # Import and run the app
        import uvicorn
        from web_ui.app import app
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0", 
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        print("✅ Web UI started at http://localhost:8000")
        print("\n" + "=" * 60)
        print("🎉 DEMO READY!")
        print("=" * 60)
        print("Open your browser to http://localhost:8000 to see:")
        print("• Real-time pipeline monitoring dashboard")
        print("• Generated research ideas with scores")
        print("• Agent status and activity")
        print("• Interactive pipeline controls")
        print("\nTo test pipeline execution:")
        print("1. Go to the 'Pipeline Control' tab")
        print("2. Enter domain tags (e.g., 'stellar evolution, binary systems')")
        print("3. Set number of hypotheses (e.g., 2)")
        print("4. Click 'Start Pipeline'")
        print("5. Watch the real-time updates!")
        print("\nPress Ctrl+C to stop the demo")
        print("=" * 60)
        
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")

def main():
    """Main demo function."""
    print("🚀 AstroAgent Pipeline Web UI Demo")
    print("=" * 60)
    
    # Setup environment
    setup_demo_environment()
    
    # Check web UI dependencies
    if not check_web_ui_dependencies():
        print("❌ Web UI dependencies not found")
        response = input("Install them now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            if not install_web_ui_dependencies():
                print("Please install manually: pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("Please install dependencies first")
            sys.exit(1)
    
    # Generate demo data
    if not generate_demo_data():
        print("⚠️  Continuing without demo data (web UI will work but show empty state)")
    
    # Start web UI
    try:
        asyncio.run(start_web_ui_server())
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
