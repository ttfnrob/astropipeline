#!/usr/bin/env python3
"""
Start script for AstroAgent Pipeline Web UI

This script handles dependency checking and starts the web interface
with proper error handling and helpful messages.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'websockets', 
        'pydantic',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if requirements_file.exists():
        print("📦 Installing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("❌ requirements.txt not found")
        return False

def check_data_directory():
    """Check if the data directory exists."""
    data_dir = Path(__file__).parent.parent / "data"
    
    if not data_dir.exists():
        print(f"⚠️  Warning: Data directory not found at {data_dir}")
        print("   The web UI will work but may show empty data until you run the pipeline.")
        return False
    
    registry_dir = data_dir / "registry"
    if not registry_dir.exists():
        print(f"⚠️  Warning: Registry directory not found at {registry_dir}")
        print("   Creating registry directory...")
        registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Create empty CSV files with headers
        ideas_csv = registry_dir / "ideas_register.csv"
        projects_csv = registry_dir / "project_index.csv" 
        completed_csv = registry_dir / "completed_index.csv"
        
        if not ideas_csv.exists():
            ideas_csv.write_text("idea_id,title,hypothesis,status,created_at\n")
        if not projects_csv.exists():
            projects_csv.write_text("idea_id,slug,path,maturity,created_at\n")
        if not completed_csv.exists():
            completed_csv.write_text("idea_id,title,moved_to_library_at\n")
        
        print("✅ Registry directory created with empty CSV files")
    
    return True

def main():
    """Main startup function."""
    print("🚀 AstroAgent Pipeline Web UI")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        
        # Ask user if they want to install
        response = input("Would you like to install them now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if not install_dependencies():
                print("Failed to install dependencies. Please install manually:")
                print("pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("Please install dependencies manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    # Check data directory
    check_data_directory()
    
    print("\n🌐 Starting Web UI Server...")
    print("   URL: http://localhost:8000")
    print("   Press Ctrl+C to stop")
    print("-" * 50)
    
    # Start the server
    try:
        import uvicorn
        from app import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False  # Disable reload for production-like usage
        )
        
    except KeyboardInterrupt:
        print("\n👋 Web UI stopped by user")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
