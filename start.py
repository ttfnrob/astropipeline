#!/usr/bin/env python3
"""
AstroAgent Pipeline - Unified Startup Script

This script provides a simple way to start the entire AstroAgent Pipeline system.
Users can run web UI, agent pipelines, or both from this single command.

Usage:
    python start.py web                    # Start web UI only
    python start.py pipeline              # Run agent pipeline only  
    python start.py all                   # Start both web UI and pipeline
    python start.py demo                  # Run demo with sample data
    python start.py test                  # Run quick functionality test
"""

import argparse
import asyncio
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).parent))

def check_environment():
    """Check if required environment variables and dependencies are set up."""
    print("🔧 Checking environment...")
    
    # Check for .env file
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("⚠️  No .env file found. Creating one from template...")
        create_env_file()
    
    # Check for data directory
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print("📁 Creating data directories...")
        setup_data_directories()
    
    print("✅ Environment ready")

def create_env_file():
    """Create .env file from template."""
    env_template = Path(__file__).parent / "environment_template.txt"
    env_file = Path(__file__).parent / ".env"
    
    if env_template.exists():
        # Copy template to .env file
        env_content = env_template.read_text()
        env_file.write_text(env_content)
        print(f"📝 Created .env file at {env_file}")
        print("⚠️  Please edit .env file and add your API keys before running with real data!")
    else:
        # Create basic .env file
        basic_env = """# AstroAgent Pipeline Environment Variables
# Edit these values with your actual API keys

# Required for hypothesis generation
ADS_API_TOKEN=your_ads_api_token_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Data directory
DATA_DIR=./data

# Environment
ENV=development
DEBUG=true
"""
        env_file.write_text(basic_env)
        print(f"📝 Created basic .env file at {env_file}")
        print("⚠️  Please add your API keys to the .env file!")

def setup_data_directories():
    """Set up the data directory structure."""
    data_dir = Path(__file__).parent / "data"
    subdirs = [
        "external", "interim", "processed", "raw", 
        "registry", "vectors"
    ]
    
    for subdir in subdirs:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create empty registry CSV files with headers
    registry_dir = data_dir / "registry"
    
    csv_files = {
        "ideas_register.csv": "idea_id,title,hypothesis,status,created_at,total_score,domain_tags,est_effort_days",
        "project_index.csv": "idea_id,slug,path,maturity,created_at",
        "completed_index.csv": "idea_id,title,moved_to_library_at"
    }
    
    for filename, header in csv_files.items():
        csv_file = registry_dir / filename
        if not csv_file.exists():
            csv_file.write_text(header + "\n")
    
    print("✅ Data directories created")

def check_dependencies():
    """Check if required Python packages are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'astropy', 'pandas', 'requests', 'pydantic', 
        'fastapi', 'uvicorn', 'rich', 'typer'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        response = input("Install missing dependencies? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            install_dependencies()
        else:
            print("Please install dependencies: pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies satisfied")
    
    return True

def install_dependencies():
    """Install missing dependencies."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        print("Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def load_env_file():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent / ".env"
    
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if not os.environ.get(key):  # Don't override existing env vars
                        os.environ[key] = value

async def start_web_ui():
    """Start the web UI server."""
    print("🌐 Starting AstroAgent Pipeline Web UI...")
    
    try:
        import uvicorn
        from web_ui.app import app
        
        print("✅ Web UI starting at http://localhost:8000")
        print("   - Dashboard: http://localhost:8000")
        print("   - API docs: http://localhost:8000/docs")
        print("   Press Ctrl+C to stop\n")
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except ImportError as e:
        print(f"❌ Error importing web UI components: {e}")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 Web UI stopped by user")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")

def run_pipeline(domain_tags=None, n_hypotheses=3):
    """Run the agent pipeline."""
    print("🤖 Starting AstroAgent Pipeline...")
    
    try:
        from astroagent.orchestration.graph import AstroAgentPipeline
        
        # Set up pipeline
        config_dir = str(Path(__file__).parent / "astroagent" / "config")
        data_dir = str(Path(__file__).parent / "data")
        
        pipeline = AstroAgentPipeline(
            config_dir=config_dir,
            data_dir=data_dir
        )
        
        # Default inputs
        if domain_tags is None:
            domain_tags = ['stellar evolution', 'galactic dynamics']
        
        agent_inputs = {
            'domain_tags': domain_tags,
            'n_hypotheses': n_hypotheses,
            'recency_years': 3
        }
        
        print(f"🎯 Generating {n_hypotheses} hypotheses for: {', '.join(domain_tags)}")
        print("⏳ This may take a few minutes...\n")
        
        # Run pipeline
        results = pipeline.run_pipeline(agent_inputs)
        
        if results['success']:
            print("✅ Pipeline completed successfully!")
            print(f"📊 Results: {results.get('summary', 'Check data/registry/ for outputs')}")
        else:
            print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        
        return results['success']
        
    except ImportError as e:
        print(f"❌ Error importing pipeline components: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False

def run_demo():
    """Run a demo with sample data."""
    print("🎮 Running AstroAgent Pipeline Demo")
    print("=" * 50)
    
    # Set demo environment variables
    os.environ['ADS_API_TOKEN'] = 'demo_token'
    os.environ['OPENAI_API_KEY'] = 'demo_key'
    
    print("📊 Demo mode enabled - using mock data")
    
    # Run a small pipeline demo
    success = run_pipeline(['stellar evolution'], 2)
    
    if success:
        print("\n🌐 Starting web UI to view results...")
        asyncio.run(start_web_ui())
    else:
        print("❌ Demo pipeline failed")

def run_test():
    """Run a quick functionality test."""
    print("🧪 Running AstroAgent Functionality Test")
    print("=" * 50)
    
    # Import and run the quick test script
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    
    try:
        from quick_test import test_hypothesis_generation
        success = test_hypothesis_generation()
        
        if success:
            print("\n✅ All tests passed! System is ready to use.")
        else:
            print("\n❌ Some tests failed. Check the error messages above.")
        
        return success
        
    except ImportError as e:
        print(f"❌ Error importing test script: {e}")
        return False

async def run_all(domain_tags=None, n_hypotheses=3):
    """Run both pipeline and web UI."""
    print("🚀 Starting Full AstroAgent Pipeline System")
    print("=" * 50)
    
    # Run pipeline first
    print("Phase 1: Running agent pipeline...")
    pipeline_success = run_pipeline(domain_tags, n_hypotheses)
    
    if pipeline_success:
        print("\nPhase 2: Starting web UI...")
        await start_web_ui()
    else:
        print("❌ Pipeline failed, not starting web UI")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="AstroAgent Pipeline - AI-Powered Astrophysics Research System",
        epilog="""
Examples:
  python start.py web                    # Start web UI only
  python start.py pipeline               # Run pipeline with defaults  
  python start.py pipeline --domains "stellar evolution,exoplanets" --count 5
  python start.py all                    # Run pipeline then start web UI
  python start.py demo                   # Demo mode with sample data
  python start.py test                   # Quick functionality test
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'action',
        choices=['web', 'pipeline', 'all', 'demo', 'test'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--domains',
        type=str,
        help='Comma-separated research domains (e.g., "stellar evolution,exoplanets")'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=3,
        help='Number of hypotheses to generate (default: 3)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip environment and dependency checks'
    )
    
    args = parser.parse_args()
    
    # Header
    print("🚀 AstroAgent Pipeline")
    print("=" * 50)
    
    # Environment setup (unless skipped)
    if not args.skip_checks:
        check_environment()
        load_env_file()
        
        if not check_dependencies():
            sys.exit(1)
    
    # Parse domains if provided
    domain_tags = None
    if args.domains:
        domain_tags = [d.strip() for d in args.domains.split(',')]
    
    # Execute the requested action
    try:
        if args.action == 'web':
            asyncio.run(start_web_ui())
            
        elif args.action == 'pipeline':
            success = run_pipeline(domain_tags, args.count)
            sys.exit(0 if success else 1)
            
        elif args.action == 'all':
            asyncio.run(run_all(domain_tags, args.count))
            
        elif args.action == 'demo':
            run_demo()
            
        elif args.action == 'test':
            success = run_test()
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
