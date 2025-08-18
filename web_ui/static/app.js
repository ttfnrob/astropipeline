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
        this.setupPipelineControls();
        this.setupTableControls();
        this.setupModal();
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
        
        // Continuous form submission
        const continuousForm = document.getElementById('continuous-form');
        if (continuousForm) {
            continuousForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.startContinuousResearch();
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
            this.updatePipelineStatus(status);
            
            // Load ideas using standard table
            await this.loadIdeas();
            
            // Load projects
            await this.loadProjects();
            
            // Load agent status
            await this.loadAgentStatus();
            
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
        
        if (data.completed_projects !== undefined) {
            document.getElementById('completed-projects').textContent = data.completed_projects;
        }
        
        // Update charts if statistics changed
        if (data.idea_statuses || data.project_maturities) {
            this.updateStatisticsCharts(data);
        }
    }
    
    updateStatusCards(status) {
        // Update system status
        const systemStatusEl = document.getElementById('system-status');
        if (systemStatusEl && status.pipeline_state) {
            systemStatusEl.textContent = status.pipeline_state;
        }
        
        // Update total ideas
        const totalIdeasEl = document.getElementById('total-ideas');
        if (totalIdeasEl && status.total_ideas !== undefined) {
            totalIdeasEl.textContent = status.total_ideas;
        }
        
        // Update total projects
        const totalProjectsEl = document.getElementById('total-projects');
        if (totalProjectsEl && status.total_projects !== undefined) {
            totalProjectsEl.textContent = status.total_projects;
        }
        
        // Update completed projects
        const completedProjectsEl = document.getElementById('completed-projects');
        if (completedProjectsEl && status.completed_projects !== undefined) {
            completedProjectsEl.textContent = status.completed_projects;
        }
    }
    

    
    async loadIdeas() {
        try {
            // Get current filter/sort values from UI controls
            const searchInput = document.getElementById('ideas-search');
            const statusFilter = document.getElementById('ideas-status-filter');
            const sortSelect = document.getElementById('ideas-sort');
            const sortOrderSelect = document.getElementById('ideas-sort-order');
            const limitSelect = document.getElementById('ideas-limit');
            
            // Build query parameters
            const params = new URLSearchParams();
            
            if (limitSelect && limitSelect.value) {
                params.append('limit', limitSelect.value);
            }
            
            if (statusFilter && statusFilter.value && statusFilter.value !== 'all') {
                params.append('status', statusFilter.value);
            }
            
            if (sortSelect && sortSelect.value) {
                params.append('sort_by', sortSelect.value);
            }
            
            if (sortOrderSelect && sortOrderSelect.value) {
                params.append('sort_order', sortOrderSelect.value);
            }
            
            if (searchInput && searchInput.value.trim()) {
                params.append('search', searchInput.value.trim());
            }
            
            // Make API call with parameters
            const url = '/api/ideas' + (params.toString() ? '?' + params.toString() : '');
            const response = await this.fetchAPI(url);
            const ideas = response.ideas || [];
            const total = response.total || 0;
            
            const tbody = document.getElementById('ideas-table-body');
            tbody.innerHTML = '';
            
            if (ideas.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="loading">No ideas found</td></tr>';
            } else {
                ideas.forEach(idea => {
                    const row = this.createIdeaRow(idea);
                    tbody.appendChild(row);
                });
            }
            
            // Update table info
            const tableInfo = document.getElementById('ideas-table-info');
            if (tableInfo) {
                const showing = ideas.length;
                const limit = limitSelect ? parseInt(limitSelect.value) : 20;
                tableInfo.textContent = `Showing ${showing} of ${total} ideas (limit: ${limit})`;
            }
            
            // Update sort header indicators
            this.updateSortHeaders();
            
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
        const currentAgent = idea.current_agent || 'none';
        const agentDisplayName = this.formatAgentName(currentAgent);
        const score = idea.total_score ? `${idea.total_score}/20` : 'N/A';
        const tags = Array.isArray(idea.domain_tags) ? idea.domain_tags : [];

        const created = idea.created_at ? 
            new Date(idea.created_at).toLocaleDateString() : 'N/A';
        
        row.innerHTML = `
            <td title="${title}">${title.length > 50 ? title.substring(0, 47) + '...' : title}</td>
            <td><span class="status-badge-table ${status ? status.toLowerCase().replace(' ', '-') : 'unknown'}">${status || 'Unknown'}</span></td>
            <td>
                <span class="agent-badge ${currentAgent}" title="Current agent handling this idea">
                    ${agentDisplayName}
                </span>
            </td>
            <td>${score}</td>
            <td>
                <div class="tags">
                    ${tags.slice(0, 3).map(tag => `<span class="tag">${tag}</span>`).join('')}
                    ${tags.length > 3 ? `<span class="tag">+${tags.length - 3}</span>` : ''}
                </div>
            </td>
            <td>${created}</td>
        `;
        
        // Add click handler for modal
        row.style.cursor = 'pointer';
        row.className = 'table-row clickable';
        row.addEventListener('click', () => {
            this.showIdeaModal(idea);
        });
        
        return row;
    }
    
    formatAgentName(agentKey) {
        const agentNames = {
            'hypothesis_maker': 'Hypothesis Maker',
            'reviewer': 'Reviewer', 
            'experiment_designer': 'Experiment Designer',
            'experimenter': 'Experimenter',
            'peer_reviewer': 'Peer Reviewer',
            'reporter': 'Reporter',
            'none': 'None'
        };
        return agentNames[agentKey] || agentKey.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            return String(unsafe || '');
        }
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
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
    
    async loadCompletedProjects() {
        try {
            const response = await this.fetchAPI('/api/completed-projects');
            const completedProjects = response.completed_projects || [];
            
            const tbody = document.getElementById('completed-table-body');
            if (!tbody) return;
            
            if (completedProjects.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="no-data">No completed projects found</td></tr>';
                return;
            }
            
            tbody.innerHTML = completedProjects.map(project => {
                // Check if project has required files
                const hasPaper = project.has_paper || false;
                const hasPeerReview = project.has_peer_review || false;
                
                return `
                    <tr class="table-row">
                        <td class="project-title">
                            <div class="project-title-main">${this.escapeHtml(project.title || project.idea_id)}</div>
                            <div class="project-subtitle">${this.escapeHtml(project.idea_id)}</div>
                        </td>
                        <td class="project-path">
                            <code>${this.escapeHtml(project.project_path || 'N/A')}</code>
                        </td>
                        <td class="status-cell">
                            <span class="status-badge ${hasPaper ? 'complete' : 'missing'}">
                                ${hasPaper ? '✓ Yes' : '✗ No'}
                            </span>
                        </td>
                        <td class="status-cell">
                            <span class="status-badge ${hasPeerReview ? 'complete' : 'missing'}">
                                ${hasPeerReview ? '✓ Yes' : '✗ No'}
                            </span>
                        </td>
                        <td class="status-cell">
                            ${project.has_html ? 
                                `<a href="${project.paper_url}" target="_blank" class="paper-link">📄 View Paper</a>` : 
                                '<span class="status-badge missing">❌ No Paper</span>'}
                        </td>
                        <td class="date-cell">
                            ${project.moved_to_library_at ? new Date(project.moved_to_library_at).toLocaleDateString() : 'N/A'}
                        </td>
                    </tr>
                `;
            }).join('');
            
        } catch (error) {
            console.error('Error loading completed projects:', error);
            const tbody = document.getElementById('completed-table-body');
            if (tbody) {
                tbody.innerHTML = '<tr><td colspan="5" class="error">Error loading completed projects</td></tr>';
            }
        }
    }
    
    async loadAgentStatus() {
        try {
            const response = await this.fetchAPI('/api/agents');
            const agents = response.agents || {};
            this.updateAgentStatus(agents);
        } catch (error) {
            console.error('Error loading agent status:', error);
        }
    }
    
    updateAgentStatus(agents) {
        // Update each agent card
        if (!agents || typeof agents !== 'object') {
            console.warn('Invalid agents data received:', agents);
            return;
        }
        
        Object.keys(agents).forEach(agentKey => {
            const agent = agents[agentKey];
            const agentCard = document.getElementById(agentKey);
            
            if (agentCard && agent) {
                // Update status
                const statusEl = agentCard.querySelector('.agent-status');
                if (statusEl && agent.status) {
                    const status = typeof agent.status === 'string' ? agent.status : 'idle';
                    statusEl.className = `agent-status ${status}`;
                    statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
                }
                
                // Update current project
                const projectEl = agentCard.querySelector('.current-project');
                if (projectEl) {
                    if (agent.current_idea_title) {
                        const title = this.escapeHtml(agent.current_idea_title.substring(0, 30));
                        projectEl.innerHTML = `Project: <span title="${agent.current_idea_id || ''}">${title}...</span>`;
                    } else {
                        projectEl.textContent = 'Project: None';
                    }
                }
                
                // Update last activity
                const activityEl = agentCard.querySelector('.last-activity');
                if (activityEl) {
                    const lastActivity = agent.last_activity || 'Never';
                    activityEl.textContent = `Last active: ${lastActivity}`;
                }
                
                // Update execution stats
                const statsEl = agentCard.querySelector('.execution-stats');
                if (statsEl) {
                    const projectCount = agent.current_projects ? agent.current_projects.length : 0;
                    statsEl.textContent = `${projectCount} projects queued`;
                }
            }
        });
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
                this.loadAgentStatus(); // Also load agent status for current agent column
                break;
            case 'projects':
                this.loadProjects();
                break;
            case 'completed':
                this.loadCompletedProjects();
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
            ambition_level: formData.get('ambition-level') || 'significant'
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
            
            // Immediately refresh the UI to show active state
            setTimeout(() => {
                this.loadInitialData();
            }, 1000);
            
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
            this.loadAgentStatus();
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
    
    // Pipeline Control Methods
    setupPipelineControls() {
        // Setup dropdown functionality
        const controlDropdownBtn = document.getElementById('control-dropdown-btn');
        const controlDropdown = document.getElementById('control-dropdown');
        
        if (controlDropdownBtn) {
            controlDropdownBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const dropdown = controlDropdownBtn.parentElement;
                dropdown.classList.toggle('active');
            });
        }
        
        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            const dropdown = document.querySelector('.dropdown');
            if (dropdown) {
                dropdown.classList.remove('active');
            }
        });
        
        const pauseResumeBtn = document.getElementById('pause-resume-btn');
        const stopBtn = document.getElementById('stop-btn');
        const killRogueBtn = document.getElementById('kill-rogue-btn');
        
        if (pauseResumeBtn) {
            pauseResumeBtn.addEventListener('click', () => {
                const action = pauseResumeBtn.classList.contains('pause-btn') ? 'pause' : 'resume';
                this.controlPipeline(action);
                // Close dropdown
                document.querySelector('.dropdown').classList.remove('active');
            });
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to stop the continuous pipeline?')) {
                    this.controlPipeline('stop');
                    // Close dropdown
                    document.querySelector('.dropdown').classList.remove('active');
                }
            });
        }
        
        if (killRogueBtn) {
            killRogueBtn.addEventListener('click', () => {
                if (confirm('Kill all rogue pipeline processes and reset corrupted data?')) {
                    this.killRogueProcesses();
                    // Close dropdown
                    document.querySelector('.dropdown').classList.remove('active');
                }
            });
        }
    }
    
    async controlPipeline(action) {
        try {
            const response = await fetch('/api/pipeline/control', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action })
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`Pipeline ${action} successful:`, result.message);
                // Update UI immediately
                this.updatePipelineControls();
            } else {
                console.error(`Pipeline ${action} failed:`, result.message);
                alert(`Failed to ${action} pipeline: ${result.message}`);
            }
        } catch (error) {
            console.error(`Error ${action} pipeline:`, error);
            alert(`Error ${action} pipeline. Please try again.`);
        }
    }
    
    updatePipelineControls() {
        // This will be called when status updates to refresh control states
        setTimeout(() => {
            this.loadInitialData();
        }, 500);
    }
    
    async killRogueProcesses() {
        try {
            const killBtn = document.getElementById('kill-rogue-btn');
            const originalText = killBtn.innerHTML;
            
            // Show loading state
            killBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Killing...';
            killBtn.disabled = true;
            
            const response = await fetch('/api/pipeline/kill-rogue', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addActivityLogEntry({
                    type: 'success',
                    data: { message: '🔥 Rogue processes eliminated!' }
                });
                
                // Show detailed results
                let details = [];
                if (result.killed_start_processes) details.push('start.py processes killed');
                if (result.killed_pipeline_processes) details.push('pipeline processes killed');
                if (result.reset_corrupted_registry) details.push('corrupted registry reset');
                
                if (details.length > 0) {
                    this.addActivityLogEntry({
                        type: 'info',
                        data: { message: `Details: ${details.join(', ')}` }
                    });
                }
                
                // Refresh the data after cleanup
                setTimeout(() => {
                    this.loadInitialData();
                }, 1000);
                
            } else {
                this.addActivityLogEntry({
                    type: 'error',
                    data: { message: 'Failed to kill rogue processes' }
                });
            }
        } catch (error) {
            console.error('Error killing rogue processes:', error);
            this.addActivityLogEntry({
                type: 'error',
                data: { message: `Error killing rogue processes: ${error.message}` }
            });
        } finally {
            // Restore button
            const killBtn = document.getElementById('kill-rogue-btn');
            killBtn.innerHTML = '<i class="fas fa-skull"></i> Kill Rogue Processes';
            killBtn.disabled = false;
        }
    }
    
    // Table Functionality
    setupTableControls() {
        // Ideas table controls
        this.setupIdeasTableControls();
    }
    
    setupIdeasTableControls() {
        const searchInput = document.getElementById('ideas-search');
        const statusFilter = document.getElementById('ideas-status-filter');
        const sortSelect = document.getElementById('ideas-sort');
        const sortOrderSelect = document.getElementById('ideas-sort-order');
        const limitSelect = document.getElementById('ideas-limit');
        
        // Debounced search function
        let searchTimeout;
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.loadIdeas();
                }, 300);
            });
        }
        
        // Filter and sort controls
        [statusFilter, sortSelect, sortOrderSelect, limitSelect].forEach(control => {
            if (control) {
                control.addEventListener('change', () => {
                    this.loadIdeas();
                });
            }
        });
        
        // Sortable headers
        document.querySelectorAll('.sortable-header').forEach(header => {
            header.addEventListener('click', () => {
                const column = header.dataset.column;
                this.sortTable(column);
            });
        });
    }
    
    sortTable(column) {
        const sortSelect = document.getElementById('ideas-sort');
        const sortOrderSelect = document.getElementById('ideas-sort-order');
        
        if (sortSelect && sortOrderSelect) {
            // If clicking same column, toggle order
            if (sortSelect.value === column) {
                sortOrderSelect.value = sortOrderSelect.value === 'asc' ? 'desc' : 'asc';
            } else {
                sortSelect.value = column;
                sortOrderSelect.value = 'desc';
            }
            
            this.loadIdeas();
        }
    }
    

    

    
    updateSortHeaders() {
        const sortSelect = document.getElementById('ideas-sort');
        const sortOrderSelect = document.getElementById('ideas-sort-order');
        
        if (!sortSelect || !sortOrderSelect) return;
        
        const currentSort = sortSelect.value;
        const currentOrder = sortOrderSelect.value;
        
        // Reset all headers
        document.querySelectorAll('.sortable-header').forEach(header => {
            header.classList.remove('sorting-asc', 'sorting-desc');
            const icon = header.querySelector('i');
            if (icon) {
                icon.className = 'fas fa-sort';
            }
        });
        
        // Update current sort header
        const currentHeader = document.querySelector(`[data-column="${currentSort}"]`);
        if (currentHeader) {
            currentHeader.classList.add(currentOrder === 'asc' ? 'sorting-asc' : 'sorting-desc');
            const icon = currentHeader.querySelector('i');
            if (icon) {
                icon.className = currentOrder === 'asc' ? 'fas fa-sort-up' : 'fas fa-sort-down';
            }
        }
    }

    
    truncateText(text, maxLength) {
        if (!text) return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }
    
    // Enhanced Status Updates
    updatePipelineStatus(status) {
        // Update pipeline control visibility and state
        const controlsSection = document.getElementById('pipeline-controls');
        const statusBadge = document.getElementById('pipeline-status-badge');
        const statusText = document.getElementById('pipeline-status-text');
        const pauseResumeBtn = document.getElementById('pause-resume-btn');
        const stopBtn = document.getElementById('stop-btn');
        const runtime = document.getElementById('pipeline-runtime');
        const cycle = document.getElementById('pipeline-cycle');
        const completed = document.getElementById('pipeline-completed');
        const target = document.getElementById('pipeline-target');
        const progressFill = document.getElementById('progress-fill');
        
        if (status.continuous_mode) {
            if (controlsSection) controlsSection.style.display = 'block';
            
            // Update status badge
            if (statusBadge && statusText) {
                const pipelineState = status.pipeline_state || 'idle';
                statusBadge.className = `pipeline-status-badge ${pipelineState}`;
                statusText.textContent = typeof pipelineState === 'string' ? 
                    pipelineState.replace('_', ' ').toUpperCase() : 
                    'IDLE';
            }
            
            // Update metrics
            if (runtime) {
                const hours = Math.floor(status.runtime_seconds / 3600);
                const minutes = Math.floor((status.runtime_seconds % 3600) / 60);
                const seconds = Math.floor(status.runtime_seconds % 60);
                runtime.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
            if (cycle) cycle.textContent = status.current_cycle || 0;
            if (completed) completed.textContent = status.completed_ideas || 0;
            if (target) target.textContent = `/${status.target_ideas || 0}`;
            
            // Update progress bar
            if (progressFill && status.target_ideas) {
                const progress = Math.min(100, (status.completed_ideas / status.target_ideas) * 100);
                progressFill.style.width = `${progress}%`;
            }
            
            // Update control buttons in header
            const headerPauseResumeBtn = document.getElementById('pause-resume-btn');
            const headerStopBtn = document.getElementById('stop-btn');
            
            if (headerPauseResumeBtn && headerStopBtn) {
                if (status.pipeline_state === 'running') {
                    headerPauseResumeBtn.disabled = false;
                    headerPauseResumeBtn.className = 'control-btn pause-btn';
                    headerPauseResumeBtn.innerHTML = '<i class="fas fa-pause"></i>';
                    headerPauseResumeBtn.title = 'Pause Pipeline';
                    headerStopBtn.disabled = false;
                } else if (status.pipeline_state === 'paused') {
                    headerPauseResumeBtn.disabled = false;
                    headerPauseResumeBtn.className = 'control-btn resume-btn';
                    headerPauseResumeBtn.innerHTML = '<i class="fas fa-play"></i>';
                    headerPauseResumeBtn.title = 'Resume Pipeline';
                    headerStopBtn.disabled = false;
                } else {
                    headerPauseResumeBtn.disabled = true;
                    headerPauseResumeBtn.title = 'Pipeline Not Active';
                    headerStopBtn.disabled = true;
                }
            }
            
            // Update legacy control buttons (if they exist)
            if (pauseResumeBtn && stopBtn) {
                if (status.pipeline_state === 'running') {
                    pauseResumeBtn.disabled = false;
                    pauseResumeBtn.className = 'control-btn pause-btn';
                    pauseResumeBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
                    stopBtn.disabled = false;
                } else if (status.pipeline_state === 'paused') {
                    pauseResumeBtn.disabled = false;
                    pauseResumeBtn.className = 'control-btn resume-btn';
                    pauseResumeBtn.innerHTML = '<i class="fas fa-play"></i> Resume';
                    stopBtn.disabled = false;
                } else {
                    pauseResumeBtn.disabled = true;
                    stopBtn.disabled = true;
                }
            }
        } else {
            if (controlsSection) controlsSection.style.display = 'none';
        }
        
        // Update agent status with new indicators - use the detailed agent status
        if (status.agent_status && typeof status.agent_status === 'object') {
            // The agent_status from /api/status is already in the detailed format we need
            // No conversion needed - just pass it directly to updateAgentStatus
            this.updateAgentStatus(status.agent_status);
        }
    }
    

    
    // Modal functionality
    setupModal() {
        const modal = document.getElementById('idea-modal');
        const closeBtn = document.getElementById('modal-close-btn');
        const closeFooterBtn = document.getElementById('modal-close-footer-btn');
        
        // Close modal handlers
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hideModal());
        }
        
        if (closeFooterBtn) {
            closeFooterBtn.addEventListener('click', () => this.hideModal());
        }
        
        // Close modal when clicking outside
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideModal();
                }
            });
        }
        
        // Close modal with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal && modal.classList.contains('show')) {
                this.hideModal();
            }
        });
    }
    
    showIdeaModal(idea) {
        if (!idea) return;
        
        const modal = document.getElementById('idea-modal');
        if (!modal) return;
        
        // Populate modal with idea data
        this.populateModal(idea);
        
        // Show modal
        modal.classList.add('show');
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
    }
    
    hideModal() {
        const modal = document.getElementById('idea-modal');
        if (modal) {
            modal.classList.remove('show');
            document.body.style.overflow = ''; // Restore scrolling
        }
    }
    
    populateModal(idea) {
        // Basic Information
        this.setModalElement('modal-title', idea.title || 'Untitled Idea');
        this.setModalElement('modal-idea-id', idea.idea_id || '-');
        this.setModalElement('modal-status', idea.status || 'Unknown');
        this.setModalElement('modal-created', idea.created_at ? 
            new Date(idea.created_at).toLocaleDateString() : '-');
        
        // Domain tags
        const domains = Array.isArray(idea.domain_tags) ? idea.domain_tags : [];
        this.setModalElement('modal-domains', domains.length > 0 ? domains.join(', ') : 'None specified');
        
        // Scores & Metrics
        this.setModalElement('modal-total-score', idea.total_score || '-');
        this.setModalElement('modal-novelty-score', idea.novelty_score || '-');
        this.setModalElement('modal-impact-score', idea.impact_score || '-');
        this.setModalElement('modal-feasibility-score', idea.feasibility_score || '-');
        

        
        // Hypothesis and Rationale
        this.setModalElement('modal-hypothesis', idea.hypothesis || 'No hypothesis provided');
        this.setModalElement('modal-rationale', idea.rationale || 'No rationale provided');
        
        // Required Data
        const requiredData = Array.isArray(idea.required_data) ? idea.required_data : [];
        this.setModalList('modal-required-data', requiredData, 'No data requirements specified');
        
        // Methods
        const methods = Array.isArray(idea.methods) ? idea.methods : [];
        this.setModalList('modal-methods', methods, 'No methods specified');
        
        // Reviewer Notes (show section only if notes exist)
        const notesSection = document.getElementById('modal-notes-section');
        const reviewerNotes = idea.reviewer_notes;
        
        if (reviewerNotes && reviewerNotes.trim()) {
            this.setModalElement('modal-notes', reviewerNotes);
            if (notesSection) notesSection.style.display = 'block';
        } else {
            if (notesSection) notesSection.style.display = 'none';
        }
    }
    
    setModalElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value || '-';
        }
    }
    
    setModalList(elementId, items, emptyMessage) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        if (items.length === 0) {
            element.innerHTML = `<li>${emptyMessage}</li>`;
        } else {
            element.innerHTML = items.map(item => `<li>${item}</li>`).join('');
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PipelineMonitor();
});
