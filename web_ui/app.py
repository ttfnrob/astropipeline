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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

app = FastAPI(
    title="AstroAgent Pipeline Monitor",
    description="Real-time monitoring interface for AstroAgent Pipeline execution",
    version="0.1.0"
)

# Global variables for tracking active pipelines
active_pipelines: Dict[str, Dict[str, Any]] = {}
websocket_connections: List[WebSocket] = []

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
            
            # Parse JSON fields
            for field in ['domain_tags', 'novelty_refs', 'required_data', 'methods']:
                if field in idea and isinstance(idea[field], str):
                    try:
                        idea[field] = json.loads(idea[field])
                    except (json.JSONDecodeError, TypeError):
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
    """Get current agent status based on recent activity."""
    # This is a simplified version - in production you'd track actual agent execution
    stats = get_pipeline_statistics()
    
    status = {
        'hypothesis_maker': 'idle',
        'reviewer': 'idle', 
        'experiment_designer': 'idle',
        'experimenter': 'idle'
    }
    
    # Basic heuristics based on data
    if stats.get('recent_ideas_30d', 0) > 0:
        status['hypothesis_maker'] = 'recently_active'
    
    idea_statuses = stats.get('idea_statuses', {})
    if idea_statuses.get('Under Review', 0) > 0:
        status['reviewer'] = 'active'
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

@app.get("/api/status", response_model=PipelineStatus)
async def get_status():
    """Get current pipeline status."""
    stats = get_pipeline_statistics()
    recent_ideas = get_recent_ideas(5)
    agent_status = format_agent_status()
    
    return PipelineStatus(
        active_pipelines=len(active_pipelines),
        total_ideas=stats.get('total_ideas', 0),
        total_projects=stats.get('total_projects', 0),
        recent_activity=recent_ideas,
        agent_status=agent_status
    )

@app.get("/api/ideas")
async def get_ideas(limit: int = 20, status: Optional[str] = None):
    """Get ideas from the registry."""
    try:
        if status:
            ideas = project_registry.get_ideas_by_status(status)
        else:
            ideas = get_recent_ideas(limit)
        
        return {"ideas": ideas}
    except Exception as e:
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
