"""
AstroAgent Pipeline Web UI Backend

FastAPI application that provides a web interface for monitoring
the AstroAgent Pipeline execution status and agent activities.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add root directory to path to import astroagent
sys.path.insert(0, str(Path(__file__).parent.parent))

from astroagent.orchestration.registry import ProjectRegistry
from astroagent.orchestration.tools import RegistryManager
from astroagent.orchestration.graph import AstroAgentPipeline

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
    est_effort_days: Optional[int] = None

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
    """Get active projects from the registry."""
    try:
        projects_df = registry_manager.load_registry('project_index')
        if projects_df.empty:
            return []
        
        # Filter active projects (not Complete/Failed)
        active_df = projects_df[~projects_df['maturity'].isin(['Complete', 'Failed'])]
        
        return active_df.head(limit).to_dict('records')
    except Exception as e:
        logger.error(f"Error getting active projects: {e}")
        return []

def format_agent_status() -> Dict[str, str]:
    """Get current agent status based on pipeline state and recent activity."""
    
    # Base status - all agents idle by default
    status = {
        'hypothesis_maker': 'idle',
        'reviewer': 'idle', 
        'experiment_designer': 'idle',
        'experimenter': 'idle'
    }
    
    # If continuous mode is running, show agents as active based on pipeline state
    if pipeline_control_state["continuous_mode"] and pipeline_control_state["pipeline_state"] == "running":
        # In continuous mode, cycle through agents
        current_cycle = pipeline_control_state["current_cycle"]
        cycle_stage = current_cycle % 4  # 4 stages in the pipeline
        
        if cycle_stage == 0:
            status['hypothesis_maker'] = 'active'
        elif cycle_stage == 1:
            status['reviewer'] = 'active'
        elif cycle_stage == 2:
            status['experiment_designer'] = 'active'
        elif cycle_stage == 3:
            status['experimenter'] = 'active'
    
    # Check for any active pipelines from the legacy tracking
    elif len(active_pipelines) > 0:
        # Simple heuristic: if pipelines are running, hypothesis maker is likely active
        status['hypothesis_maker'] = 'active'
    
    # If paused, show all as paused
    if pipeline_control_state["pipeline_state"] == "paused":
        for agent in status:
            if status[agent] == 'active':
                status[agent] = 'paused'
    
    # Fallback to data-based heuristics if not in continuous mode
    if not pipeline_control_state["continuous_mode"]:
        stats = get_pipeline_statistics()
        if stats.get('recent_ideas_30d', 0) > 0:
            status['hypothesis_maker'] = 'recently_active'
        
        idea_statuses = stats.get('idea_statuses', {})
        if idea_statuses.get('Under Review', 0) > 0:
            status['reviewer'] = 'recently_active'
        if idea_statuses.get('Approved', 0) > 0:
            status['experiment_designer'] = 'recently_active'
    
    return status

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
    """Get current pipeline status."""
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
            "runtime_seconds": runtime_seconds
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
    try:
        action = request.action.lower()
        
        if action == "pause":
            if pipeline_control_state["pipeline_state"] == "running":
                pipeline_control_state["pipeline_state"] = "paused"
                pipeline_control_state["pause_time"] = datetime.now()
                await broadcast_to_websockets({"type": "pipeline_paused"})
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
                await broadcast_to_websockets({"type": "pipeline_resumed"})
                return {"success": True, "message": "Pipeline resumed"}
            else:
                return {"success": False, "message": "Pipeline is not paused"}
                
        elif action == "stop":
            pipeline_control_state["pipeline_state"] = "stopping"
            await broadcast_to_websockets({"type": "pipeline_stopping"})
            # Reset after a short delay to allow graceful shutdown
            await asyncio.sleep(2)
            pipeline_control_state.update({
                "continuous_mode": False,
                "pipeline_state": "idle",
                "current_cycle": 0,
                "completed_ideas": 0,
                "target_ideas": None,
                "start_time": None,
                "pause_time": None
            })
            await broadcast_to_websockets({"type": "pipeline_stopped"})
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
        
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """Get detailed pipeline statistics."""
    try:
        stats = get_pipeline_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/start")
async def start_pipeline(agent_inputs: Dict[str, Any]):
    """Start a new pipeline execution."""
    try:
        # Create pipeline instance
        pipeline = AstroAgentPipeline(
            config_dir=str(Path(__file__).parent.parent / "astroagent" / "config"),
            data_dir=DATA_DIR
        )
        
        # Start pipeline in background
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store pipeline info
        active_pipelines[pipeline_id] = {
            'id': pipeline_id,
            'status': 'running',
            'started_at': datetime.now(timezone.utc).isoformat(),
            'agent_inputs': agent_inputs
        }
        
        # Run pipeline asynchronously
        asyncio.create_task(run_pipeline_with_updates(pipeline, pipeline_id, agent_inputs))
        
        await broadcast_update('pipeline_started', {'pipeline_id': pipeline_id})
        
        return {"pipeline_id": pipeline_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # Start periodic status updates
    asyncio.create_task(periodic_status_updates())

async def periodic_status_updates():
    """Send periodic status updates to connected clients."""
    while True:
        try:
            if manager.active_connections:
                stats = get_pipeline_statistics()
                await broadcast_update('status_update', stats)
        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
        
        # Update every 10 seconds
        await asyncio.sleep(10)

# ============================================================================
# Static Files
# ============================================================================

# Mount static files directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
