# AstroAgent Pipeline Web UI 🌐

A modern web interface for monitoring and controlling the AstroAgent Pipeline execution in real-time.

## Features

- **Real-time Monitoring**: Live updates via WebSocket connection
- **Pipeline Control**: Start new research pipelines directly from the web interface
- **Agent Status**: Monitor the current state of all AI agents
- **Data Visualization**: Charts and graphs showing pipeline statistics
- **Responsive Design**: Clean, modern interface that works on all devices

## Quick Start

1. **Install Dependencies**
   ```bash
   cd web_ui
   pip install -r requirements.txt
   ```

2. **Start the Web Server**
   ```bash
   python app.py
   ```

3. **Open Browser**
   Navigate to [http://localhost:8000](http://localhost:8000)

## Interface Overview

### Dashboard
- **Status Cards**: Overview of active pipelines, total ideas, projects, and recent activity
- **Agent Status**: Real-time status of each AI agent (Hypothesis Maker, Reviewer, etc.)
- **Activity Log**: Live feed of pipeline events and system activities

### Tabs
- **Recent Ideas**: Browse generated research hypotheses with scores and status
- **Active Projects**: Monitor project maturity and execution progress  
- **Statistics**: Visual charts showing idea status and project distribution
- **Pipeline Control**: Start new pipeline executions with custom parameters

## API Endpoints

The web UI provides a REST API for programmatic access:

- `GET /api/status` - Current pipeline status
- `GET /api/ideas` - List of research ideas
- `GET /api/projects` - List of active projects
- `GET /api/statistics` - Detailed statistics
- `POST /api/pipeline/start` - Start a new pipeline
- `WebSocket /ws` - Real-time updates

## Configuration

The web UI automatically connects to the AstroAgent Pipeline data directory:
- Ideas Registry: `../astroagent/data/registry/ideas_register.csv`
- Project Index: `../astroagent/data/registry/project_index.csv`
- Configuration: `../astroagent/config/`

## Development

To extend or modify the web UI:

1. **Backend**: Edit `app.py` (FastAPI application)
2. **Frontend**: Edit files in `static/` directory
   - `index.html` - HTML structure
   - `styles.css` - Styling and layout
   - `app.js` - JavaScript functionality

The interface uses:
- FastAPI for the backend API
- WebSocket for real-time updates
- Chart.js for data visualization
- Modern CSS Grid/Flexbox for responsive layout

## Screenshots

The interface provides:
- Clean, professional dashboard design
- Real-time status indicators for all agents
- Interactive data tables with filtering
- Live activity log showing system events
- Form-based pipeline control with validation

Perfect for research teams wanting to monitor their AstroAgent Pipeline executions visually!
