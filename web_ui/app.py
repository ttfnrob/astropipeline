"""
AstroAgent Pipeline Web UI Backend

FastAPI application that provides a web interface for monitoring
the AstroAgent Pipeline execution status and agent activities.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add root directory to path to import astroagent
sys.path.insert(0, str(Path(__file__).parent.parent))

from astroagent.orchestration.registry import ProjectRegistry
from astroagent.orchestration.tools import RegistryManager
from astroagent.orchestration.graph import AstroAgentPipeline
from astroagent.orchestration.continuous_pipeline import ContinuousPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Application Setup
# ============================================================================

def clean_nan_values(obj: Any) -> Any:
    """Recursively clean NaN values from nested dictionaries and lists for JSON serialization."""
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, np.floating)):
        if pd.isna(obj) or not np.isfinite(obj):
            return None
        return obj.item()  # Convert numpy types to Python native types
    elif isinstance(obj, (float, int)) and (pd.isna(obj) or obj != obj):  # Check for NaN
        return None
    return obj

app = FastAPI(
    title="AstroAgent Pipeline Monitor",
    description="Real-time monitoring interface for AstroAgent Pipeline execution",
    version="0.1.0"
)

# Global variables for tracking active pipelines
active_pipelines: Dict[str, Dict[str, Any]] = {}
websocket_connections: List[WebSocket] = []

# Global continuous pipeline instance
continuous_pipeline: Optional[ContinuousPipeline] = None
pipeline_task: Optional[asyncio.Task] = None

# Background task tracking for cleanup
background_tasks: List[asyncio.Task] = []

# Global pipeline control state
pipeline_control_state = {
    "continuous_mode": False,
    "pipeline_state": "idle",  # idle, running, paused, stopping
    "current_cycle": 0,
    "completed_ideas": 0,
    "target_ideas": None,
    "start_time": None,
    "pause_time": None
}

# Initialize components
DATA_DIR = str(Path(__file__).parent.parent / "data")
registry_manager = RegistryManager(DATA_DIR)
project_registry = ProjectRegistry(DATA_DIR)

# ============================================================================
# Data Models
# ============================================================================

class PipelineStatus(BaseModel):
    """Current pipeline status information."""
    active_pipelines: int
    total_ideas: int
    total_projects: int
    recent_activity: List[Dict[str, Any]]
    agent_status: Dict[str, str]
    continuous_mode: bool = False
    pipeline_state: str = "idle"  # idle, running, paused, stopping
    current_cycle: int = 0
    completed_ideas: int = 0
    target_ideas: Optional[int] = None
    runtime_seconds: float = 0.0

class PipelineControlRequest(BaseModel):
    """Request for pipeline control actions."""
    action: str  # pause, resume, stop

class IdeaInfo(BaseModel):
    """Detailed idea information for display."""
    idea_id: str
    title: str
    status: str
    total_score: Optional[int] = None
    domain_tags: List[str]
    created_at: str

class AgentActivity(BaseModel):
    """Agent activity information."""
    agent_name: str
    status: str
    current_task: Optional[str] = None
    last_activity: str
    execution_time: Optional[float] = None

# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# ============================================================================
# Helper Functions
# ============================================================================

def get_pipeline_statistics() -> Dict[str, Any]:
    """Get current pipeline statistics."""
    try:
        stats = project_registry.get_pipeline_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting pipeline stats: {e}")
        return {
            'total_ideas': 0,
            'total_projects': 0,
            'total_completed': 0,
            'idea_statuses': {},
            'project_maturities': {},
            'recent_ideas_30d': 0
        }

def get_recent_ideas(limit: int = 10) -> List[Dict[str, Any]]:
    """Get most recent ideas from the registry."""
    try:
        ideas_df = registry_manager.load_registry('ideas_register')
        if ideas_df.empty:
            return []
        
        # Sort by created_at if available, otherwise by row order
        if 'created_at' in ideas_df.columns:
            ideas_df = ideas_df.sort_values('created_at', ascending=False)
        
        recent_ideas = []
        for _, row in ideas_df.head(limit).iterrows():
            idea = row.to_dict()
            
            # Handle NaN values and convert to JSON-serializable format
            for key, value in idea.items():
                if pd.isna(value):
                    idea[key] = None
                elif isinstance(value, (int, float)) and (pd.isna(value) or value != value):  # Check for NaN
                    idea[key] = None
            
            # Parse JSON fields
            for field in ['domain_tags', 'novelty_refs', 'required_data', 'methods']:
                if field in idea and isinstance(idea[field], str):
                    try:
                        idea[field] = json.loads(idea[field])
                    except (json.JSONDecodeError, TypeError):
                        idea[field] = []
                elif field in idea and idea[field] is None:
                    idea[field] = []
            
            recent_ideas.append(idea)
        
        return recent_ideas
    except Exception as e:
        logger.error(f"Error getting recent ideas: {e}")
        return []

def get_active_projects(limit: int = 10) -> List[Dict[str, Any]]:
    """Get active projects from the registry - projects that are still being worked on."""
    try:
        projects_df = registry_manager.load_registry('project_index')
        if projects_df.empty:
            return []
        
        # Filter for truly active projects - not Complete, Failed, or moved to Library
        active_df = projects_df[
            (~projects_df['maturity'].isin(['Complete', 'Failed'])) &
            (~projects_df['maturity'].isna())  # Exclude NaN maturity
        ]
        
        # Sort by created_at or updated_at to show most recent first
        if 'updated_at' in active_df.columns:
            active_df = active_df.sort_values('updated_at', ascending=False, na_position='last')
        elif 'created_at' in active_df.columns:
            active_df = active_df.sort_values('created_at', ascending=False, na_position='last')
        
        # Convert to dict and clean NaN values
        projects = active_df.head(limit).to_dict('records')
        return clean_nan_values(projects)
    except Exception as e:
        logger.error(f"Error getting active projects: {e}")
        return []

def get_current_agent_for_idea(idea: Dict[str, Any]) -> str:
    """Determine which agent should currently handle an idea based on its status."""
    status = idea.get('status', 'Unknown')
    
    # Map idea status to the agent responsible for the next step
    agent_mapping = {
        'Proposed': 'reviewer',           # Needs review
        'Under Review': 'reviewer',       # Being reviewed
        'Approved': 'experiment_designer',# Needs experiment design
        'Needs Revision': 'hypothesis_maker', # Needs revision by original maker
        'Rejected': 'none',               # No further action
    }
    
    # Check if there's a project for this idea
    try:
        projects_df = registry_manager.load_registry('project_index')
        if not projects_df.empty:
            project_exists = idea['idea_id'] in projects_df['idea_id'].values
            if project_exists:
                project = projects_df[projects_df['idea_id'] == idea['idea_id']].iloc[0]
                maturity = project.get('maturity', 'Unknown')
                
                # Map project maturity to agent
                if maturity == 'Preparing':
                    return 'experiment_designer'  # Still designing
                elif maturity == 'Prepared':
                    return 'experimenter'         # Ready for execution
                elif maturity == 'Ready':
                    return 'experimenter'         # Ready to execute
                elif maturity == 'Running':
                    return 'experimenter'         # Currently executing
                elif maturity == 'Executed':
                    return 'peer_reviewer'        # Needs peer review
                elif maturity == 'AwaitingReview':
                    return 'peer_reviewer'        # Currently under review
                elif maturity == 'Reviewed':
                    return 'reporter'             # Needs final report
                elif maturity == 'Complete':
                    return 'none'                 # Fully complete
                elif maturity == 'Published':
                    return 'none'                 # Complete
                elif maturity == 'Failed':
                    return 'experimenter'         # Retry failed experiments
    except Exception:
        pass  # Fall back to status-based mapping
    
    return agent_mapping.get(status, 'none')

def format_agent_status() -> Dict[str, Dict[str, Any]]:
    """Get detailed current agent status including what they're working on."""
    
    # Get all ideas and projects to determine current workloads
    try:
        ideas_df = registry_manager.load_registry('ideas_register')
        projects_df = registry_manager.load_registry('project_index')
    except Exception:
        ideas_df = pd.DataFrame()
        projects_df = pd.DataFrame()
    
    # Initialize agent status
    agents = ['hypothesis_maker', 'reviewer', 'experiment_designer', 'experimenter', 'peer_reviewer', 'reporter']
    agent_status = {}
    
    for agent in agents:
        agent_status[agent] = {
            'status': 'idle',
            'current_projects': [],
            'last_activity': 'Never',
            'total_executions': 0,
            'current_idea_id': None,
            'current_idea_title': None
        }
    
    # Analyze current workload for each agent
    if not ideas_df.empty:
        for _, idea in ideas_df.iterrows():
            idea_dict = idea.to_dict()
            current_agent = get_current_agent_for_idea(idea_dict)
            
            if current_agent != 'none' and current_agent in agent_status:
                agent_status[current_agent]['current_projects'].append({
                    'idea_id': idea_dict.get('idea_id', 'Unknown'),
                    'title': idea_dict.get('title', 'Unknown'),
                    'status': idea_dict.get('status', 'Unknown')
                })
                
                # Set the first project as current
                if agent_status[current_agent]['current_idea_id'] is None:
                    agent_status[current_agent]['current_idea_id'] = idea_dict.get('idea_id')
                    agent_status[current_agent]['current_idea_title'] = idea_dict.get('title', 'Unknown')
    
    # Update agent status based on workload
    for agent in agent_status:
        if len(agent_status[agent]['current_projects']) > 0:
            if pipeline_control_state["pipeline_state"] == "running":
                agent_status[agent]['status'] = 'active'
            elif pipeline_control_state["pipeline_state"] == "paused":
                agent_status[agent]['status'] = 'paused'
            else:
                agent_status[agent]['status'] = 'ready'  # Has work but pipeline not running
        
        # Update execution count (simulated for now)
        agent_status[agent]['total_executions'] = len(agent_status[agent]['current_projects'])
        
        # Update last activity based on recent data
        if agent_status[agent]['total_executions'] > 0:
            agent_status[agent]['last_activity'] = 'Recent'  # TODO: Get real timestamps
    
    return agent_status

async def broadcast_update(update_type: str, data: Dict[str, Any]):
    """Broadcast updates to all connected WebSocket clients."""
    message = json.dumps({
        'type': update_type,
        'data': data,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
    await manager.broadcast(message)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def read_root():
    """Serve the main dashboard page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>AstroAgent Pipeline Monitor</title></head>
            <body>
                <h1>AstroAgent Pipeline Monitor</h1>
                <p>Dashboard loading... Please ensure static files are available.</p>
            </body>
        </html>
        """, status_code=200)

@app.get("/api/status")
async def get_status():
    """Get current pipeline status with real-time agent activity."""
    try:
        stats = get_pipeline_statistics()
        recent_ideas = get_recent_ideas(5)
        agent_status = format_agent_status()
        
        # Calculate runtime
        runtime_seconds = 0.0
        if pipeline_control_state["start_time"]:
            current_time = datetime.now()
            if pipeline_control_state["pause_time"]:
                runtime_seconds = (pipeline_control_state["pause_time"] - pipeline_control_state["start_time"]).total_seconds()
            else:
                runtime_seconds = (current_time - pipeline_control_state["start_time"]).total_seconds()
        
        # Add current pipeline activity details
        pipeline_activity = {
            'current_stage': 'idle',
            'stage_progress': None,
            'estimated_completion': None
        }
        
        if continuous_pipeline and pipeline_control_state["pipeline_state"] == "running":
            # Try to determine current activity from continuous pipeline
            pipeline_activity['current_stage'] = 'executing_cycle'
            if pipeline_control_state["target_ideas"] and pipeline_control_state["target_ideas"] > 0:
                progress_pct = (pipeline_control_state["completed_ideas"] / pipeline_control_state["target_ideas"]) * 100
                pipeline_activity['stage_progress'] = min(100, progress_pct)
        
        status_data = {
            "active_pipelines": len(active_pipelines),
            "total_ideas": stats.get('total_ideas', 0),
            "total_projects": stats.get('total_projects', 0),
            "recent_activity": recent_ideas,
            "agent_status": agent_status,
            "continuous_mode": pipeline_control_state["continuous_mode"],
            "pipeline_state": pipeline_control_state["pipeline_state"],
            "current_cycle": pipeline_control_state["current_cycle"],
            "completed_ideas": pipeline_control_state["completed_ideas"],
            "target_ideas": pipeline_control_state["target_ideas"],
            "runtime_seconds": runtime_seconds,
            "pipeline_activity": pipeline_activity,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Clean NaN values before returning
        cleaned_data = clean_nan_values(status_data)
        return JSONResponse(content=cleaned_data)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/control")
async def control_pipeline(request: PipelineControlRequest):
    """Control the continuous pipeline (pause, resume, stop)."""
    global continuous_pipeline, pipeline_task
    
    try:
        action = request.action.lower()
        
        if action == "pause":
            if pipeline_control_state["pipeline_state"] == "running":
                pipeline_control_state["pipeline_state"] = "paused"
                pipeline_control_state["pause_time"] = datetime.now()
                
                # Pause the continuous pipeline
                if continuous_pipeline:
                    continuous_pipeline.is_paused = True
                
                await broadcast_update("pipeline_paused", {"message": "Pipeline paused"})
                return {"success": True, "message": "Pipeline paused"}
            else:
                return {"success": False, "message": "Pipeline is not running"}
                
        elif action == "resume":
            if pipeline_control_state["pipeline_state"] == "paused":
                pipeline_control_state["pipeline_state"] = "running"
                if pipeline_control_state["pause_time"]:
                    # Adjust start time to account for pause duration
                    pause_duration = datetime.now() - pipeline_control_state["pause_time"]
                    if pipeline_control_state["start_time"]:
                        pipeline_control_state["start_time"] += pause_duration
                pipeline_control_state["pause_time"] = None
                
                # Resume the continuous pipeline
                if continuous_pipeline:
                    continuous_pipeline.is_paused = False
                
                await broadcast_update("pipeline_resumed", {"message": "Pipeline resumed"})
                return {"success": True, "message": "Pipeline resumed"}
            else:
                return {"success": False, "message": "Pipeline is not paused"}
                
        elif action == "stop":
            pipeline_control_state["pipeline_state"] = "stopping"
            await broadcast_update("pipeline_stopping", {"message": "Pipeline stopping..."})
            
            # Stop the continuous pipeline
            if continuous_pipeline:
                continuous_pipeline.should_stop = True
                continuous_pipeline.is_running = False
            
            # Cancel the pipeline task
            if pipeline_task and not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await asyncio.wait_for(pipeline_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Cancel all background tasks
            if background_tasks:
                logger.info(f"Cancelling {len(background_tasks)} background tasks...")
                for task in background_tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to finish cancelling
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*[t for t in background_tasks if not t.done()], return_exceptions=True),
                        timeout=3.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some background tasks did not cancel in time")
                
                background_tasks.clear()
            
            # Reset state
            pipeline_control_state.update({
                "continuous_mode": False,
                "pipeline_state": "idle",
                "current_cycle": 0,
                "completed_ideas": 0,
                "target_ideas": None,
                "start_time": None,
                "pause_time": None
            })
            
            continuous_pipeline = None
            pipeline_task = None
            
            await broadcast_update("pipeline_stopped", {"message": "Pipeline stopped"})
            return {"success": True, "message": "Pipeline stopped"}
            
        else:
            return {"success": False, "message": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error controlling pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ideas")
async def get_ideas(
    limit: int = 20, 
    status: Optional[str] = None,
    sort_by: Optional[str] = "created_at",
    sort_order: Optional[str] = "desc",
    search: Optional[str] = None
):
    """Get ideas from the registry with sorting, filtering, and search."""
    try:
        ideas_df = registry_manager.load_registry('ideas_register')
        
        if ideas_df.empty:
            return JSONResponse(content={"ideas": [], "total": 0})
        
        # Filter by status if provided
        if status and status != "all":
            ideas_df = ideas_df[ideas_df['status'] == status]
        
        # Search functionality
        if search:
            search_lower = search.lower()
            mask = (
                ideas_df['title'].str.lower().str.contains(search_lower, na=False) |
                ideas_df['hypothesis'].str.lower().str.contains(search_lower, na=False) |
                ideas_df['domain_tags'].str.lower().str.contains(search_lower, na=False)
            )
            ideas_df = ideas_df[mask]
        
        # Sort the dataframe
        if sort_by in ideas_df.columns:
            ascending = sort_order.lower() == "asc"
            ideas_df = ideas_df.sort_values(sort_by, ascending=ascending, na_position='last')
        
        # Apply limit
        limited_df = ideas_df.head(limit)
        
        # Convert to dict and clean NaN values
        ideas = []
        for _, row in limited_df.iterrows():
            idea = row.to_dict()
            # Clean NaN values
            cleaned_idea = clean_nan_values(idea)
            
            # Parse JSON fields
            for field in ['domain_tags', 'novelty_refs', 'required_data', 'methods']:
                if field in cleaned_idea and isinstance(cleaned_idea[field], str):
                    try:
                        cleaned_idea[field] = json.loads(cleaned_idea[field])
                    except (json.JSONDecodeError, TypeError):
                        cleaned_idea[field] = []
            
            # Add current agent information
            cleaned_idea['current_agent'] = get_current_agent_for_idea(cleaned_idea)
            
            ideas.append(cleaned_idea)
        
        return JSONResponse(content={
            "ideas": ideas, 
            "total": len(ideas_df),
            "filtered": len(limited_df)
        })
        
    except Exception as e:
        logger.error(f"Error getting ideas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects")
async def get_projects(limit: int = 20, maturity: Optional[str] = None):
    """Get projects from the registry."""
    try:
        if maturity:
            projects = project_registry.get_projects_by_maturity(maturity)
        else:
            projects = get_active_projects(limit)
        
        # Clean NaN values for JSON serialization
        cleaned_projects = clean_nan_values(projects)
        
        return JSONResponse(content={"projects": cleaned_projects})
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/completed-projects") 
async def get_completed_projects():
    """Get completed projects from Library with metadata about papers and peer reviews."""
    try:
        completed_projects = []
        
        # Check projects in Library directory
        library_path = Path("projects/Library")
        if library_path.exists():
            for project_dir in library_path.iterdir():
                if project_dir.is_dir():
                    # Extract idea_id from directory name
                    dir_name = project_dir.name
                    idea_id = dir_name.split('__')[0] if '__' in dir_name else dir_name
                    
                    # Check for key files
                    has_paper = (project_dir / "paper.md").exists()
                    has_peer_review = (project_dir / "reviewer_report.md").exists()
                    
                    # Generate HTML if paper exists but HTML doesn't
                    html_path = project_dir / "paper.html"
                    
                    if has_paper and not html_path.exists():
                        await generate_interactive_paper(project_dir, idea_id)
                    
                    has_html = html_path.exists()
                    
                    # Get title from idea.md if available
                    title = idea_id  # fallback
                    idea_file = project_dir / "idea.md"
                    if idea_file.exists():
                        try:
                            idea_content = idea_file.read_text()
                            # Look for title in first line
                            for line in idea_content.split('\n'):
                                if line.startswith('# Idea:'):
                                    title = line.replace('# Idea:', '').strip()
                                    break
                                elif line.startswith('#'):
                                    title = line.replace('#', '').strip()
                                    break
                        except:
                            pass
                    
                    # Get completion date from directory stats
                    moved_to_library_at = None
                    try:
                        moved_to_library_at = project_dir.stat().st_ctime
                        moved_to_library_at = datetime.fromtimestamp(moved_to_library_at, tz=timezone.utc).isoformat()
                    except:
                        pass
                    
                    completed_projects.append({
                        'idea_id': idea_id,
                        'title': title,
                        'project_path': str(project_dir),
                        'has_paper': has_paper,
                        'has_peer_review': has_peer_review,
                        'has_html': has_html,
                        'paper_url': f"/api/view-paper/{idea_id}" if has_html else None,
                        'moved_to_library_at': moved_to_library_at
                    })
        
        # Sort by completion date (newest first)
        completed_projects.sort(key=lambda x: x['moved_to_library_at'] or '', reverse=True)
        
        return JSONResponse(content={"completed_projects": completed_projects})
        
    except Exception as e:
        logger.error(f"Error getting completed projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """Get detailed pipeline statistics."""
    try:
        stats = get_pipeline_statistics()
        cleaned_stats = clean_nan_values(stats)
        return JSONResponse(content=cleaned_stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents")
async def get_agent_status():
    """Get detailed agent status including current workloads."""
    try:
        agent_status = format_agent_status()
        cleaned_status = clean_nan_values(agent_status)
        return JSONResponse(content={"agents": cleaned_status})
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/kill-rogue")
async def kill_rogue_processes():
    """Kill any rogue pipeline processes that are generating garbage data."""
    try:
        import subprocess
        
        # Kill any python processes running start.py or pipeline-related scripts
        try:
            result = subprocess.run(['pkill', '-f', 'python.*start.py'], capture_output=True, text=True)
            killed_start = result.returncode == 0
        except:
            killed_start = False
            
        try:
            result = subprocess.run(['pkill', '-f', 'python.*pipeline'], capture_output=True, text=True)
            killed_pipeline = result.returncode == 0
        except:
            killed_pipeline = False
        
        # Reset any corrupted registries
        project_index_path = Path(DATA_DIR) / "registry" / "project_index.csv"
        if project_index_path.exists():
            lines = project_index_path.read_text().count('\n')
            if lines > 10:  # If more than 10 lines (header + some legitimate projects)
                # Reset to header only
                header = "idea_id,slug,path,ready_checklist_passed,data_requirements_met,analysis_plan_preregistered,maturity,execution_start,execution_end,compute_hours_used,storage_gb_used,created_at,updated_at,project_path,ready_checks_passed,experiment_plan\n"
                project_index_path.write_text(header)
                reset_registry = True
            else:
                reset_registry = False
        
        return {
            "success": True, 
            "killed_start_processes": killed_start,
            "killed_pipeline_processes": killed_pipeline,
            "reset_corrupted_registry": reset_registry,
            "message": "Rogue processes cleaned up"
        }
    except Exception as e:
        logger.error(f"Error killing rogue processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/start")
async def start_pipeline(agent_inputs: Dict[str, Any]):
    """Start the continuous pipeline."""
    global continuous_pipeline, pipeline_task
    
    try:
        # Safety check: kill any existing rogue processes first
        await kill_rogue_processes()
        
        # Check if pipeline is already running
        if pipeline_control_state["pipeline_state"] == "running":
            return {"success": False, "message": "Pipeline is already running"}
        
        # Create continuous pipeline instance
        continuous_pipeline = ContinuousPipeline(
            config_dir=str(Path(__file__).parent.parent / "astroagent" / "config"),
            data_dir=DATA_DIR
        )
        
        # Add enhanced status callback for real-time updates with proper task tracking
        def safe_status_callback(status):
            try:
                # Add current pipeline control state to status updates
                enhanced_status = {
                    **status,
                    'continuous_mode': pipeline_control_state["continuous_mode"],
                    'pipeline_state': pipeline_control_state["pipeline_state"],
                    'current_cycle': pipeline_control_state["current_cycle"],
                    'completed_ideas': pipeline_control_state["completed_ideas"],
                    'target_ideas': pipeline_control_state["target_ideas"]
                }
                
                task = asyncio.create_task(broadcast_update('pipeline_status', enhanced_status))
                # Clean up completed tasks
                global background_tasks
                background_tasks.append(task)
                background_tasks = [t for t in background_tasks if not t.done()]
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
        
        continuous_pipeline.status_callbacks.append(safe_status_callback)
        
        # Update control state
        pipeline_control_state.update({
            "continuous_mode": True,
            "pipeline_state": "running",
            "start_time": datetime.now(),
            "target_ideas": agent_inputs.get("n_hypotheses", 3),
            "current_cycle": 0,
            "completed_ideas": 0
        })
        
        # Start continuous pipeline asynchronously
        pipeline_task = asyncio.create_task(run_continuous_pipeline(agent_inputs))
        
        await broadcast_update('pipeline_started', {
            'message': 'Continuous pipeline started',
            'target_ideas': agent_inputs.get("n_hypotheses", 3)
        })
        
        return {"success": True, "message": "Continuous pipeline started"}
        
    except Exception as e:
        logger.error(f"Error starting continuous pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_continuous_pipeline(agent_inputs: Dict[str, Any]):
    """Run the continuous pipeline in background."""
    global continuous_pipeline, pipeline_control_state
    
    try:
        if not continuous_pipeline:
            return
            
        # Run continuous pipeline with specified completion criteria
        results = await continuous_pipeline.run_continuous(
            initial_inputs=agent_inputs,
            completion_mode="ideas",
            completion_target=agent_inputs.get("n_hypotheses", 3),
            max_duration_minutes=agent_inputs.get("max_duration_minutes", 60)
        )
        
        # Update final state
        pipeline_control_state.update({
            "pipeline_state": "completed",
            "completed_ideas": results.get("completed_ideas", 0),
            "total_cycles": results.get("total_cycles", 0)
        })
        
        await broadcast_update('pipeline_completed', {
            'message': 'Continuous pipeline completed',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Continuous pipeline error: {e}")
        pipeline_control_state["pipeline_state"] = "error"
        await broadcast_update('pipeline_error', {
            'message': f'Pipeline error: {str(e)}'
        })

async def run_pipeline_with_updates(pipeline: AstroAgentPipeline, pipeline_id: str, agent_inputs: Dict[str, Any]):
    """Run pipeline and broadcast updates."""
    try:
        # Run the pipeline
        results = pipeline.run_pipeline(agent_inputs)
        
        # Update status
        if pipeline_id in active_pipelines:
            active_pipelines[pipeline_id].update({
                'status': 'completed' if results['success'] else 'failed',
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'results': results
            })
        
        # Broadcast completion
        await broadcast_update('pipeline_completed', {
            'pipeline_id': pipeline_id,
            'success': results['success'],
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Pipeline {pipeline_id} failed: {e}")
        if pipeline_id in active_pipelines:
            active_pipelines[pipeline_id].update({
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now(timezone.utc).isoformat()
            })
        
        await broadcast_update('pipeline_failed', {
            'pipeline_id': pipeline_id,
            'error': str(e)
        })

@app.get("/api/pipelines")
async def get_active_pipelines():
    """Get information about active/recent pipelines."""
    return {"pipelines": list(active_pipelines.values())}

# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        # Send initial status
        stats = get_pipeline_statistics()
        await websocket.send_text(json.dumps({
            'type': 'initial_status',
            'data': stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }))
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            # Echo back for now - could handle client commands here
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ============================================================================
# Background Tasks
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks."""
    logger.info("AstroAgent Pipeline Monitor started")
    
    # Start periodic status updates with proper tracking
    task = asyncio.create_task(periodic_status_updates())
    background_tasks.append(task)

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean up background tasks."""
    logger.info("AstroAgent Pipeline Monitor shutting down...")
    
    # Cancel continuous pipeline
    if continuous_pipeline:
        continuous_pipeline.should_stop = True
        continuous_pipeline.is_running = False
    
    # Cancel pipeline task
    if pipeline_task and not pipeline_task.done():
        pipeline_task.cancel()
        try:
            await asyncio.wait_for(pipeline_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    
    # Cancel all background tasks
    if background_tasks:
        logger.info(f"Cancelling {len(background_tasks)} background tasks...")
        for task in background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        try:
            await asyncio.wait_for(
                asyncio.gather(*[t for t in background_tasks if not t.done()], return_exceptions=True),
                timeout=3.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some background tasks did not cancel in time")
    
    logger.info("Pipeline Monitor shutdown complete")

async def periodic_status_updates():
    """Send periodic status updates to connected clients."""
    try:
        while True:
            try:
                if manager.active_connections:
                    # Get more comprehensive status including agent details
                    stats = get_pipeline_statistics()
                    agent_status = format_agent_status()
                    
                    # Calculate runtime if pipeline is active
                    runtime_seconds = 0.0
                    if pipeline_control_state["start_time"]:
                        current_time = datetime.now()
                        if pipeline_control_state["pause_time"]:
                            runtime_seconds = (pipeline_control_state["pause_time"] - pipeline_control_state["start_time"]).total_seconds()
                        else:
                            runtime_seconds = (current_time - pipeline_control_state["start_time"]).total_seconds()
                    
                    # Create comprehensive status update
                    comprehensive_status = {
                        **stats,
                        'agent_status': agent_status,
                        'continuous_mode': pipeline_control_state["continuous_mode"],
                        'pipeline_state': pipeline_control_state["pipeline_state"],
                        'current_cycle': pipeline_control_state["current_cycle"],
                        'completed_ideas': pipeline_control_state["completed_ideas"],
                        'target_ideas': pipeline_control_state["target_ideas"],
                        'runtime_seconds': runtime_seconds,
                        'active_pipelines': len(active_pipelines)
                    }
                    
                    await broadcast_update('status_update', comprehensive_status)
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
            
            # Dynamic update frequency: faster when pipeline is running
            is_active = pipeline_control_state["pipeline_state"] in ["running", "paused"]
            update_interval = 2 if is_active else 8  # 2s when active, 8s when idle
            
            try:
                await asyncio.wait_for(asyncio.sleep(update_interval), timeout=update_interval + 0.1)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        logger.info("Periodic status updates cancelled")
        raise

# ============================================================================
# Static Files
# ============================================================================

# Mount static files directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

async def generate_interactive_paper(project_path: Path, idea_id: str):
    """Generate enhanced interactive paper for completed projects."""
    try:
        # Import the enhanced generator
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from astroagent.agents.enhanced_paper_generator import generate_enhanced_paper
        from astroagent.orchestration.tools import RegistryManager
        
        # Load idea data
        registry_manager = RegistryManager(str(Path(__file__).parent.parent / "data"))
        ideas_df = registry_manager.load_registry('ideas_register')
        idea_data = ideas_df[ideas_df['idea_id'] == idea_id].iloc[0].to_dict()
        
        # Generate enhanced interactive paper
        paper_html = generate_enhanced_paper(project_path, idea_data)
        
        # Save to project directory
        html_path = project_path / "paper.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(paper_html)
        
        logger.info(f"Generated enhanced interactive paper: {html_path}")
        
    except Exception as e:
        logger.error(f"Error generating interactive paper: {e}")
        # Fallback to simple conversion
        await generate_html_from_markdown_simple(project_path / "paper.md", project_path / "paper.html")

async def generate_html_from_markdown_simple(md_path: Path, html_path: Path):
    """Simple fallback HTML generation."""
    try:
        cmd = [
            'pandoc', str(md_path), 
            '-o', str(html_path),
            '-s', '--self-contained'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Generated basic HTML: {html_path}")
        else:
            logger.error(f"HTML generation failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error generating HTML from {md_path}: {e}")



@app.get("/api/view-paper/{idea_id}")
async def view_paper(idea_id: str):
    """View HTML paper for a completed project."""
    try:
        # Find the project directory
        library_path = Path("projects/Library")
        for project_dir in library_path.iterdir():
            if project_dir.is_dir() and project_dir.name.startswith(idea_id):
                html_path = project_dir / "paper.html"
                if html_path.exists():
                    return FileResponse(
                        path=str(html_path),
                        filename=f"{idea_id}_paper.html", 
                        media_type="text/html"
                    )
        
        raise HTTPException(status_code=404, detail="Paper not found")
    except Exception as e:
        logger.error(f"Error serving paper for {idea_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
