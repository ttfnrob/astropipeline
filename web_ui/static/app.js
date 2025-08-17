/**
 * AstroAgent Pipeline Monitor - Frontend JavaScript
 * Handles real-time updates, API calls, and UI interactions
 */

class PipelineMonitor {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 5000;
        this.charts = {};
        this.isConnected = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadInitialData();
        this.setupPeriodicUpdates();
    }
    
    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });
        
        // Pipeline form submission
        const pipelineForm = document.getElementById('pipeline-form');
        if (pipelineForm) {
            pipelineForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.startPipeline();
            });
        }
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.handlePageHidden();
            } else {
                this.handlePageVisible();
            }
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.updateConnectionStatus('connected');
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('disconnected');
                
                // Attempt to reconnect
                setTimeout(() => {
                    if (!this.isConnected) {
                        this.connectWebSocket();
                    }
                }, this.reconnectInterval);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus('disconnected');
        }
    }
    
    handleWebSocketMessage(message) {
        console.log('WebSocket message:', message);
        
        switch (message.type) {
            case 'initial_status':
                this.updateDashboard(message.data);
                break;
                
            case 'status_update':
                this.updateDashboard(message.data);
                break;
                
            case 'pipeline_started':
                this.handlePipelineStarted(message.data);
                break;
                
            case 'pipeline_completed':
                this.handlePipelineCompleted(message.data);
                break;
                
            case 'pipeline_failed':
                this.handlePipelineFailed(message.data);
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
        
        // Add to activity log
        this.addActivityLogEntry(message);
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        statusElement.className = `status-badge ${status}`;
        
        const statusText = {
            'connecting': 'Connecting...',
            'connected': 'Connected',
            'disconnected': 'Disconnected'
        };
        
        statusElement.innerHTML = `<i class="fas fa-circle"></i> ${statusText[status]}`;
    }
    
    async loadInitialData() {
        try {
            // Load status
            const status = await this.fetchAPI('/api/status');
            this.updateStatusCards(status);
            this.updateAgentStatus(status.agent_status);
            
            // Load ideas
            await this.loadIdeas();
            
            // Load projects
            await this.loadProjects();
            
            // Load statistics
            await this.loadStatistics();
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.addActivityLogEntry({
                type: 'error',
                data: { message: 'Failed to load initial data' },
                timestamp: new Date().toISOString()
            });
        }
    }
    
    async fetchAPI(endpoint, options = {}) {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    updateDashboard(data) {
        // Update status cards based on received data
        if (data.total_ideas !== undefined) {
            document.getElementById('total-ideas').textContent = data.total_ideas;
        }
        
        if (data.total_projects !== undefined) {
            document.getElementById('total-projects').textContent = data.total_projects;
        }
        
        if (data.recent_ideas_30d !== undefined) {
            document.getElementById('recent-activity').textContent = data.recent_ideas_30d;
        }
        
        // Update charts if statistics changed
        if (data.idea_statuses || data.project_maturities) {
            this.updateStatisticsCharts(data);
        }
    }
    
    updateStatusCards(status) {
        document.getElementById('active-pipelines').textContent = status.active_pipelines;
        document.getElementById('total-ideas').textContent = status.total_ideas;
        document.getElementById('total-projects').textContent = status.total_projects;
        
        // Calculate recent activity from the received data
        if (status.recent_activity) {
            document.getElementById('recent-activity').textContent = status.recent_activity.length;
        }
    }
    
    updateAgentStatus(agentStatus) {
        Object.entries(agentStatus).forEach(([agentName, status]) => {
            const agentElement = document.getElementById(agentName);
            if (agentElement) {
                const statusElement = agentElement.querySelector('.agent-status');
                const detailsElement = agentElement.querySelector('.last-activity');
                
                // Update status
                statusElement.className = `agent-status ${status}`;
                statusElement.textContent = status.replace('_', ' ').toUpperCase();
                
                // Update last activity (simplified)
                const lastActivity = status === 'idle' ? 'Never' : 'Recently';
                detailsElement.textContent = `Last active: ${lastActivity}`;
            }
        });
    }
    
    async loadIdeas() {
        try {
            const response = await this.fetchAPI('/api/ideas');
            const ideas = response.ideas;
            
            const tbody = document.getElementById('ideas-table-body');
            tbody.innerHTML = '';
            
            if (ideas.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="loading">No ideas found</td></tr>';
                return;
            }
            
            ideas.forEach(idea => {
                const row = this.createIdeaRow(idea);
                tbody.appendChild(row);
            });
            
        } catch (error) {
            console.error('Error loading ideas:', error);
            document.getElementById('ideas-table-body').innerHTML = 
                '<tr><td colspan="6" class="loading">Error loading ideas</td></tr>';
        }
    }
    
    createIdeaRow(idea) {
        const row = document.createElement('tr');
        
        // Format data
        const title = idea.title || 'Untitled';
        const status = idea.status || 'Unknown';
        const score = idea.total_score ? `${idea.total_score}/20` : 'N/A';
        const tags = Array.isArray(idea.domain_tags) ? idea.domain_tags : [];
        const effort = idea.est_effort_days ? `${idea.est_effort_days} days` : 'N/A';
        const created = idea.created_at ? 
            new Date(idea.created_at).toLocaleDateString() : 'N/A';
        
        row.innerHTML = `
            <td title="${title}">${title.length > 50 ? title.substring(0, 47) + '...' : title}</td>
            <td><span class="status-badge-table ${status.toLowerCase().replace(' ', '-')}">${status}</span></td>
            <td>${score}</td>
            <td>
                <div class="tags">
                    ${tags.slice(0, 3).map(tag => `<span class="tag">${tag}</span>`).join('')}
                    ${tags.length > 3 ? `<span class="tag">+${tags.length - 3}</span>` : ''}
                </div>
            </td>
            <td>${effort}</td>
            <td>${created}</td>
        `;
        
        return row;
    }
    
    async loadProjects() {
        try {
            const response = await this.fetchAPI('/api/projects');
            const projects = response.projects;
            
            const tbody = document.getElementById('projects-table-body');
            tbody.innerHTML = '';
            
            if (projects.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="loading">No projects found</td></tr>';
                return;
            }
            
            projects.forEach(project => {
                const row = this.createProjectRow(project);
                tbody.appendChild(row);
            });
            
        } catch (error) {
            console.error('Error loading projects:', error);
            document.getElementById('projects-table-body').innerHTML = 
                '<tr><td colspan="5" class="loading">Error loading projects</td></tr>';
        }
    }
    
    createProjectRow(project) {
        const row = document.createElement('tr');
        
        const path = project.path || 'N/A';
        const maturity = project.maturity || 'Unknown';
        const readyChecks = project.ready_checklist_passed ? '✅' : '❌';
        const executionStart = project.execution_start ? 
            new Date(project.execution_start).toLocaleDateString() : 'N/A';
        const created = project.created_at ? 
            new Date(project.created_at).toLocaleDateString() : 'N/A';
        
        row.innerHTML = `
            <td title="${path}">${path.length > 40 ? path.substring(0, 37) + '...' : path}</td>
            <td><span class="status-badge-table ${maturity.toLowerCase()}">${maturity}</span></td>
            <td>${readyChecks}</td>
            <td>${executionStart}</td>
            <td>${created}</td>
        `;
        
        return row;
    }
    
    async loadStatistics() {
        try {
            const stats = await this.fetchAPI('/api/statistics');
            this.updateStatisticsCharts(stats);
        } catch (error) {
            console.error('Error loading statistics:', error);
        }
    }
    
    updateStatisticsCharts(stats) {
        // Idea Status Chart
        if (stats.idea_statuses) {
            this.updateChart('idea-status-chart', {
                type: 'doughnut',
                data: {
                    labels: Object.keys(stats.idea_statuses),
                    datasets: [{
                        data: Object.values(stats.idea_statuses),
                        backgroundColor: [
                            '#3b82f6', '#10b981', '#f59e0b', 
                            '#ef4444', '#8b5cf6', '#06b6d4'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
        
        // Project Maturity Chart
        if (stats.project_maturities) {
            this.updateChart('project-maturity-chart', {
                type: 'doughnut',
                data: {
                    labels: Object.keys(stats.project_maturities),
                    datasets: [{
                        data: Object.values(stats.project_maturities),
                        backgroundColor: [
                            '#3b82f6', '#10b981', '#f59e0b', 
                            '#ef4444', '#8b5cf6', '#06b6d4'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
    }
    
    updateChart(canvasId, config) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        const ctx = canvas.getContext('2d');
        this.charts[canvasId] = new Chart(ctx, config);
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        // Load data if needed
        switch (tabName) {
            case 'ideas':
                this.loadIdeas();
                break;
            case 'projects':
                this.loadProjects();
                break;
            case 'statistics':
                this.loadStatistics();
                break;
        }
    }
    
    async startPipeline() {
        const form = document.getElementById('pipeline-form');
        const formData = new FormData(form);
        
        const agentInputs = {
            domain_tags: formData.get('domain-tags').split(',').map(tag => tag.trim()),
            n_hypotheses: parseInt(formData.get('n-hypotheses')),
            recency_years: parseInt(formData.get('recency-years'))
        };
        
        try {
            const submitButton = form.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
            
            const response = await this.fetchAPI('/api/pipeline/start', {
                method: 'POST',
                body: JSON.stringify(agentInputs)
            });
            
            this.addActivityLogEntry({
                type: 'pipeline_started',
                data: response,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            console.error('Error starting pipeline:', error);
            this.addActivityLogEntry({
                type: 'error',
                data: { message: `Failed to start pipeline: ${error.message}` },
                timestamp: new Date().toISOString()
            });
        } finally {
            const submitButton = form.querySelector('button[type="submit"]');
            submitButton.disabled = false;
            submitButton.innerHTML = '<i class="fas fa-play"></i> Start Pipeline';
        }
    }
    
    handlePipelineStarted(data) {
        const pipelineStatus = document.getElementById('pipeline-status');
        const pipelineInfo = document.getElementById('pipeline-info');
        
        pipelineStatus.style.display = 'block';
        pipelineInfo.innerHTML = `
            <p><strong>Pipeline ID:</strong> ${data.pipeline_id}</p>
            <p><strong>Status:</strong> <span class="status-badge-table active">Running</span></p>
            <p><strong>Started:</strong> ${new Date().toLocaleString()}</p>
        `;
        
        // Update active pipelines count
        const currentCount = parseInt(document.getElementById('active-pipelines').textContent);
        document.getElementById('active-pipelines').textContent = currentCount + 1;
    }
    
    handlePipelineCompleted(data) {
        const pipelineInfo = document.getElementById('pipeline-info');
        const statusBadge = pipelineInfo.querySelector('.status-badge-table');
        
        if (statusBadge) {
            statusBadge.className = 'status-badge-table approved';
            statusBadge.textContent = data.success ? 'Completed' : 'Failed';
        }
        
        // Update active pipelines count
        const currentCount = parseInt(document.getElementById('active-pipelines').textContent);
        document.getElementById('active-pipelines').textContent = Math.max(0, currentCount - 1);
        
        // Refresh data
        setTimeout(() => {
            this.loadIdeas();
            this.loadProjects();
            this.loadStatistics();
        }, 1000);
    }
    
    handlePipelineFailed(data) {
        this.handlePipelineCompleted({ ...data, success: false });
    }
    
    addActivityLogEntry(message) {
        const activityList = document.getElementById('activity-list');
        const entry = document.createElement('div');
        entry.className = 'activity-item';
        
        const time = new Date(message.timestamp).toLocaleTimeString();
        let messageText = '';
        
        switch (message.type) {
            case 'pipeline_started':
                messageText = `Pipeline started: ${message.data.pipeline_id}`;
                break;
            case 'pipeline_completed':
                messageText = `Pipeline completed: ${message.data.pipeline_id}`;
                break;
            case 'pipeline_failed':
                messageText = `Pipeline failed: ${message.data.pipeline_id}`;
                break;
            case 'status_update':
                messageText = 'Status updated';
                break;
            case 'error':
                messageText = message.data.message;
                break;
            default:
                messageText = `${message.type} event`;
        }
        
        entry.innerHTML = `
            <div class="activity-time">${time}</div>
            <div class="activity-message">${messageText}</div>
        `;
        
        // Add to top of list
        activityList.insertBefore(entry, activityList.firstChild);
        
        // Keep only last 20 entries
        while (activityList.children.length > 20) {
            activityList.removeChild(activityList.lastChild);
        }
    }
    
    setupPeriodicUpdates() {
        // Refresh data every 30 seconds
        setInterval(() => {
            if (this.isConnected) {
                this.loadInitialData();
            }
        }, 30000);
    }
    
    handlePageHidden() {
        // Reduce update frequency when page is hidden
        if (this.ws) {
            this.ws.close();
        }
    }
    
    handlePageVisible() {
        // Resume normal operations when page becomes visible
        if (!this.isConnected) {
            this.connectWebSocket();
            this.loadInitialData();
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PipelineMonitor();
});
