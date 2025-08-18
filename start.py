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
import signal
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set manually
    pass

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

def clean_projects(force=False):
    """Clean all projects and registry data for fresh start."""
    import shutil
    
    print("🧹 Cleaning all projects and starting fresh...")
    
    # Safety confirmation unless forced
    if not force:
        print("⚠️  WARNING: This will permanently delete:")
        print("   - All projects in Library, Preparing, Ready for Execution, and Archive")
        print("   - All registry data (ideas_register.csv, project_index.csv, etc.)")
        print("   - All processed data and vectors")
        
        if not force:
            try:
                response = input("\nAre you sure you want to continue? (y/N): ").lower().strip()
                if response not in ['y', 'yes']:
                    print("❌ Operation cancelled.")
                    return False
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled by user.")
                return False
    
    base_dir = Path(__file__).parent
    projects_dir = base_dir / "projects"
    data_dir = base_dir / "data"
    
    # Project directories to clean
    project_subdirs = [
        "Library",
        "Preparing", 
        "Ready for Execution",
        "Archive"
    ]
    
    # Remove all project contents
    for subdir in project_subdirs:
        project_path = projects_dir / subdir
        if project_path.exists():
            print(f"  🗑️  Removing {subdir} projects...")
            shutil.rmtree(project_path)
            project_path.mkdir(exist_ok=True)  # Recreate empty directory
    
    # Reset registry CSV files to headers only
    registry_dir = data_dir / "registry"
    csv_files = {
        "ideas_register.csv": "idea_id,title,hypothesis,status,created_at,total_score,domain_tags,est_effort_days",
        "project_index.csv": "idea_id,slug,path,maturity,created_at", 
        "completed_index.csv": "idea_id,title,moved_to_library_at"
    }
    
    if registry_dir.exists():
        print("  📊 Resetting registry files...")
        for filename, header in csv_files.items():
            csv_file = registry_dir / filename
            csv_file.write_text(header + "\n")
    
    # Clean data processing directories (but keep structure)
    data_subdirs = ["interim", "processed", "vectors"]
    for subdir in data_subdirs:
        data_path = data_dir / subdir
        if data_path.exists():
            print(f"  🗑️  Cleaning {subdir} data...")
            shutil.rmtree(data_path)
            data_path.mkdir(exist_ok=True)
    
    print("✅ All projects and data cleaned! Starting with fresh slate.")
    print("   - All project directories emptied")
    print("   - Registry files reset to headers only") 
    print("   - Processing data cleared")
    print()
    
    return True

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
        try:
            response = input("Install missing dependencies? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                install_dependencies()
            else:
                print("Please install dependencies: pip install -r requirements.txt")
                return False
        except KeyboardInterrupt:
            print("\n❌ Installation cancelled by user.")
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

def run_pipeline(domain_tags=None, n_hypotheses=3, mode='discrete'):
    """Run the agent pipeline."""
    
    if mode == 'continuous':
        print("🔄 Starting Continuous AstroAgent Pipeline...")
        return run_continuous_pipeline(domain_tags, n_hypotheses)
    else:
        print("🤖 Starting Discrete AstroAgent Pipeline...")
        return run_discrete_pipeline(domain_tags, n_hypotheses)

def run_discrete_pipeline(domain_tags=None, n_hypotheses=3):
    """Run a single discrete pipeline execution."""
    
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
            'ambitious_mode': True  # Generate bold, paradigm-shifting hypotheses
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

def run_continuous_pipeline(domain_tags=None, n_hypotheses=3, complete_ideas=3, max_time_minutes=None):
    """Run continuous pipeline until completion criteria are met."""
    
    try:
        from astroagent.orchestration.continuous_pipeline import ContinuousPipeline
        
        # Set up pipeline
        config_dir = str(Path(__file__).parent / "astroagent" / "config")
        data_dir = str(Path(__file__).parent / "data")
        
        pipeline = ContinuousPipeline(
            config_dir=config_dir,
            data_dir=data_dir
        )
        
        # Default inputs
        if domain_tags is None:
            domain_tags = ['stellar evolution', 'galactic dynamics']
        
        initial_inputs = {
            'domain_tags': domain_tags,
            'min_pipeline_size': min(n_hypotheses, 3),  # Keep minimal pipeline
            'max_total_ideas': 15,  # Hard limit on total ideas
            'max_active_projects': 8,  # Hard limit on active projects  
            'ambitious_mode': True  # Focus on paradigm-shifting research
        }
        
        print(f"🎯 Domains: {', '.join(domain_tags)}")
        print(f"📊 Target: Complete {complete_ideas} ideas through full workflow")
        if max_time_minutes:
            print(f"⏰ Max time: {max_time_minutes} minutes")
        print("🔄 Running continuous pipeline...\n")
        
        # Run continuous pipeline
        import asyncio
        results = asyncio.run(pipeline.run_continuous(
            initial_inputs=initial_inputs,
            completion_mode="ideas",
            completion_target=complete_ideas,
            max_duration_minutes=max_time_minutes
        ))
        
        if results['success']:
            print(f"\n✅ Continuous pipeline completed!")
            print(f"📊 Completed {results['completed_ideas']} ideas in {results['runtime_seconds']:.1f} seconds")
            print(f"🔄 Total cycles: {results['total_cycles']}")
            print(f"⚡ Rate: {results['ideas_per_hour']:.1f} ideas/hour")
        
        return results['success']
        
    except ImportError as e:
        print(f"❌ Error importing continuous pipeline: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running continuous pipeline: {e}")
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
    """Run both pipeline and web UI sequentially."""
    print("🚀 Starting Full AstroAgent Pipeline System (Sequential Mode)")
    print("=" * 50)
    
    # Run pipeline first
    print("Phase 1: Running agent pipeline...")
    pipeline_success = run_pipeline(domain_tags, n_hypotheses)
    
    if pipeline_success:
        print("\nPhase 2: Starting web UI...")
        await start_web_ui()
    else:
        print("❌ Pipeline failed, not starting web UI")

async def run_concurrent(domain_tags=None, n_hypotheses=3):
    """Run both discrete pipeline and web UI concurrently."""
    print("🚀 Starting Full AstroAgent Pipeline System (Concurrent Discrete Mode)")
    print("=" * 50)
    
    # Start web UI in background
    print("🌐 Starting web UI server...")
    web_ui_task = asyncio.create_task(start_web_ui_background())
    
    # Give web UI time to start
    await asyncio.sleep(2)
    
    print("🤖 Starting discrete pipeline execution...")
    print("📊 Monitor progress at: http://localhost:8000")
    print("=" * 50)
    
    # Run pipeline in background while web UI serves
    try:
        pipeline_task = asyncio.create_task(run_pipeline_async(domain_tags, n_hypotheses))
        
        # Wait for both to complete (web UI runs indefinitely until interrupted)
        await asyncio.gather(pipeline_task, web_ui_task)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down system...")
        web_ui_task.cancel()
        try:
            await web_ui_task
        except asyncio.CancelledError:
            pass

async def run_concurrent_continuous(domain_tags=None, n_hypotheses=3, complete_ideas=3, max_time_minutes=None):
    """Run both continuous pipeline and web UI concurrently."""
    print("🚀 Starting Full AstroAgent Pipeline System (Concurrent Continuous Mode)")
    print("=" * 50)
    
    # Setup shutdown event but NO signal handlers here (they're set up in main)
    shutdown_event = asyncio.Event()
    tasks = []
    
    try:
        # Start web UI in background
        print("🌐 Starting web UI server...")
        web_ui_task = asyncio.create_task(start_web_ui_background())
        tasks.append(web_ui_task)
        
        # Give web UI time to start
        await asyncio.sleep(2)
        
        print("🔄 Starting continuous pipeline execution...")
        print(f"📊 Target: Complete {complete_ideas} ideas through full workflow")
        if max_time_minutes:
            print(f"⏰ Max time: {max_time_minutes} minutes")
        print("📊 Monitor progress at: http://localhost:8000")
        print("📋 Press Ctrl+C to stop gracefully")
        print("=" * 50)
        
        # Run continuous pipeline in background while web UI serves
        pipeline_task = asyncio.create_task(run_continuous_pipeline_async(
            domain_tags, n_hypotheses, complete_ideas, max_time_minutes))
        tasks.append(pipeline_task)
        
        # Wait for shutdown signal with much faster response
        while not shutdown_event.is_set():
            # Check if any task has completed/failed
            done_tasks = [t for t in tasks if t.done()]
            if done_tasks:
                # If any critical task failed, trigger shutdown
                for task in done_tasks:
                    if task.exception():
                        print(f"Task failed: {task.exception()}")
                        shutdown_event.set()
                        break
            
            # Very brief sleep to allow signal handling - much more responsive
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
                
    except KeyboardInterrupt:
        print("\n👋 Shutting down system (KeyboardInterrupt)...")
    finally:
        # Graceful shutdown
        print("🛑 Cancelling all tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait briefly for tasks to cancel with shorter timeout
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
            except asyncio.TimeoutError:
                print("⚠️  Some tasks took too long to cancel, forcing shutdown...")
            except Exception as e:
                print(f"⚠️  Error during shutdown: {e}")
        
        print("✅ Shutdown complete")

async def start_web_ui_background():
    """Start web UI in background mode."""
    try:
        import uvicorn
        from web_ui.app import app
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")

async def run_pipeline_async(domain_tags=None, n_hypotheses=3):
    """Run discrete pipeline asynchronously."""
    def run_in_thread():
        return run_discrete_pipeline(domain_tags, n_hypotheses)
    
    # Run the synchronous pipeline in a thread pool with proper cancellation
    import concurrent.futures
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = loop.run_in_executor(executor, run_in_thread)
        try:
            success = await future
        except asyncio.CancelledError:
            print("⚠️  Pipeline execution cancelled")
            # Cancel the future if possible
            future.cancel()
            return False
    
    if success:
        print("\n✅ Discrete pipeline completed successfully!")
        print("📊 View results at: http://localhost:8000")
    else:
        print("\n❌ Pipeline execution failed")
    
    return success

async def run_continuous_pipeline_async(domain_tags=None, n_hypotheses=3, complete_ideas=3, max_time_minutes=None):
    """Run continuous pipeline asynchronously."""
    def run_in_thread():
        return run_continuous_pipeline(domain_tags, n_hypotheses, complete_ideas, max_time_minutes)
    
    # Run the synchronous pipeline in a thread pool with proper cancellation
    import concurrent.futures
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = loop.run_in_executor(executor, run_in_thread)
        try:
            success = await future
        except asyncio.CancelledError:
            print("⚠️  Pipeline execution cancelled")
            # Cancel the future if possible
            future.cancel()
            return False
    
    if success:
        print("\n✅ Continuous pipeline completed successfully!")
        print("📊 View results at: http://localhost:8000")
    else:
        print("\n❌ Continuous pipeline execution failed")
    
    return success

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="AstroAgent Pipeline - AI-Powered Astrophysics Research System",
        epilog="""
Examples:
  python start.py                        # Run pipeline and web UI together (default, auto-pauses after 1 completed idea)
  python start.py --domains "exoplanets" --complete-ideas 2  
                                         # Run until 2 exoplanet ideas are completed, then auto-pause
  python start.py web                    # Start web UI only
  python start.py pipeline               # Run continuous pipeline only (auto-pauses after 1 idea completed)
  python start.py pipeline --mode discrete --count 5
                                         # Run discrete mode: generate 5 hypotheses once
  python start.py pipeline --complete-ideas 3 --max-time 60
                                         # Run until 3 ideas completed or 60 min max, whichever comes first
  python start.py all                    # Run pipeline then start web UI (sequential)
  python start.py demo                   # Demo mode with sample data
  python start.py test                   # Quick functionality test
  python start.py clean                  # Clean all projects and start fresh (with confirmation)
  python start.py clean --force          # Clean without confirmation prompt
  python start.py --fresh pipeline       # Clean first, then run pipeline
  python start.py --fresh --force pipeline # Clean without prompt, then run pipeline
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'action',
        choices=['web', 'pipeline', 'all', 'concurrent', 'demo', 'test', 'clean'],
        nargs='?',
        default='concurrent',
        help='Action to perform (default: concurrent - runs pipeline and web UI together)'
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
        '--mode',
        choices=['discrete', 'continuous'],
        default='continuous',
        help='Pipeline mode: continuous (until completion) or discrete (run once, default: continuous)'
    )
    
    parser.add_argument(
        '--complete-ideas',
        type=int,
        default=1,
        help='Number of ideas to complete through full workflow (continuous mode, default: 1 - auto-pauses after first completion)'
    )
    
    parser.add_argument(
        '--max-time',
        type=int,
        help='Maximum runtime in minutes (optional safety limit)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip environment and dependency checks'
    )
    
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Clean all projects and start fresh (useful for testing)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force clean without confirmation prompt (use with --fresh or clean action)'
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
    
    # Clean projects if requested
    if args.fresh or args.action == 'clean':
        success = clean_projects(force=args.force)
        if args.action == 'clean':
            if success:
                print("🎉 Projects cleaned successfully!")
                sys.exit(0)
            else:
                print("❌ Clean operation cancelled or failed.")
                sys.exit(1)
    
    # Setup signal handling for main process
    def signal_handler(sig, frame):
        print(f"\n⚠️  Received signal {sig}, shutting down immediately...")
        print("🔥 Force shutdown!")
        # Immediate exit - no cleanup
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Execute the requested action with proper signal handling
    def run_with_signal_handling(coro_func):
        """Run async function with immediate Ctrl+C handling."""
        try:
            return asyncio.run(coro_func)
        except KeyboardInterrupt:
            print("\n👋 Stopped by user (Ctrl+C)")
            os._exit(0)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            os._exit(1)
    
    try:
        if args.action == 'web':
            run_with_signal_handling(start_web_ui())
            
        elif args.action == 'pipeline':
            if args.mode == 'continuous':
                complete_ideas = getattr(args, 'complete_ideas', 3)
                max_time = getattr(args, 'max_time', None)
                success = run_continuous_pipeline(
                    domain_tags=domain_tags, 
                    n_hypotheses=args.count,
                    complete_ideas=complete_ideas,
                    max_time_minutes=max_time
                )
            else:
                success = run_discrete_pipeline(domain_tags, args.count)
            sys.exit(0 if success else 1)
            
        elif args.action == 'all':
            run_with_signal_handling(run_all(domain_tags, args.count))
            
        elif args.action == 'concurrent':
            if args.mode == 'continuous':
                # Set up IMMEDIATE signal handler - no asyncio interference
                def immediate_signal_handler(sig, frame):
                    print(f"\n⚠️  Received signal {sig}, shutting down immediately...")
                    print("🔥 Force shutdown - killing all processes...")
                    os._exit(0)
                
                # Register signal handlers at main thread level
                signal.signal(signal.SIGINT, immediate_signal_handler)
                signal.signal(signal.SIGTERM, immediate_signal_handler)
                
                # Don't use run_with_signal_handling - use plain asyncio.run
                try:
                    asyncio.run(run_concurrent_continuous(domain_tags, args.count, 
                                                        getattr(args, 'complete_ideas', 3),
                                                        getattr(args, 'max_time', None)))
                except KeyboardInterrupt:
                    print("\n👋 Stopped by user (Ctrl+C)")
                    os._exit(0)
            else:
                run_with_signal_handling(run_concurrent(domain_tags, args.count))
            
        elif args.action == 'demo':
            run_demo()
            
        elif args.action == 'test':
            success = run_test()
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n👋 Stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
