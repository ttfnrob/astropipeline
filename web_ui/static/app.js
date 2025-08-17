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
            this.updatePipelineStatus(status);
            
            // Load ideas using enhanced table
            await this.loadIdeasTable();
            
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
    
    // Pipeline Control Methods
    setupPipelineControls() {
        const pauseResumeBtn = document.getElementById('pause-resume-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        if (pauseResumeBtn) {
            pauseResumeBtn.addEventListener('click', () => {
                const action = pauseResumeBtn.classList.contains('pause-btn') ? 'pause' : 'resume';
                this.controlPipeline(action);
            });
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to stop the continuous pipeline?')) {
                    this.controlPipeline('stop');
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
                    this.loadIdeasTable();
                }, 300);
            });
        }
        
        // Filter and sort controls
        [statusFilter, sortSelect, sortOrderSelect, limitSelect].forEach(control => {
            if (control) {
                control.addEventListener('change', () => {
                    this.loadIdeasTable();
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
            
            this.loadIdeasTable();
        }
    }
    
    async loadIdeasTable() {
        const searchInput = document.getElementById('ideas-search');
        const statusFilter = document.getElementById('ideas-status-filter');
        const sortSelect = document.getElementById('ideas-sort');
        const sortOrderSelect = document.getElementById('ideas-sort-order');
        const limitSelect = document.getElementById('ideas-limit');
        
        const params = new URLSearchParams();
        
        if (limitSelect) params.append('limit', limitSelect.value);
        if (statusFilter && statusFilter.value !== 'all') params.append('status', statusFilter.value);
        if (sortSelect) params.append('sort_by', sortSelect.value);
        if (sortOrderSelect) params.append('sort_order', sortOrderSelect.value);
        if (searchInput && searchInput.value.trim()) params.append('search', searchInput.value.trim());
        
        try {
            const response = await fetch(`/api/ideas?${params}`);
            const data = await response.json();
            
            this.updateIdeasTable(data.ideas || []);
            this.updateTableInfo(data.total || 0, data.filtered || 0);
            this.updateSortHeaders();
            
        } catch (error) {
            console.error('Error loading ideas table:', error);
            document.getElementById('ideas-table-body').innerHTML = 
                '<tr><td colspan="6" class="error">Error loading ideas</td></tr>';
        }
    }
    
    updateIdeasTable(ideas) {
        const tbody = document.getElementById('ideas-table-body');
        if (!tbody) return;
        
        if (ideas.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty">No ideas found</td></tr>';
            return;
        }
        
        tbody.innerHTML = ideas.map((idea, index) => {
            const domains = Array.isArray(idea.domain_tags) ? idea.domain_tags : [];
            const effort = idea.est_effort_days || idea.effort_estimate_days || '-';
            const score = idea.total_score || '-';
            const created = idea.created_at ? new Date(idea.created_at).toLocaleDateString() : '-';
            
            const statusClass = idea.status ? idea.status.toLowerCase().replace(/\s+/g, '-') : 'unknown';
            
            return `
                <tr data-idea-index="${index}" data-idea-id="${idea.idea_id}">
                    <td title="${idea.hypothesis || ''}">${this.truncateText(idea.title || 'Untitled', 50)}</td>
                    <td><span class="status-badge ${statusClass}">${idea.status || 'Unknown'}</span></td>
                    <td>${score}</td>
                    <td title="${domains.join(', ')}">${this.truncateText(domains.join(', '), 30)}</td>
                    <td>${effort} days</td>
                    <td>${created}</td>
                </tr>
            `;
        }).join('');
        
        // Store ideas data for modal use
        this.currentIdeasData = ideas;
        
        // Add click handlers to table rows
        tbody.querySelectorAll('tr[data-idea-index]').forEach(row => {
            row.addEventListener('click', () => {
                const ideaIndex = parseInt(row.getAttribute('data-idea-index'));
                this.showIdeaModal(ideas[ideaIndex]);
            });
        });
    }
    
    updateTableInfo(total, filtered) {
        const info = document.getElementById('ideas-table-info');
        if (info) {
            if (total === filtered) {
                info.textContent = `Showing ${filtered} of ${total} ideas`;
            } else {
                info.textContent = `Showing ${filtered} of ${total} ideas (filtered)`;
            }
        }
    }
    
    updateSortHeaders() {
        const sortSelect = document.getElementById('ideas-sort');
        const sortOrderSelect = document.getElementById('ideas-sort-order');
        
        if (!sortSelect || !sortOrderSelect) return;
        
        // Clear all sort indicators
        document.querySelectorAll('.sortable-header').forEach(header => {
            header.classList.remove('sorting-asc', 'sorting-desc');
        });
        
        // Add sort indicator to active column
        const activeHeader = document.querySelector(`[data-column="${sortSelect.value}"]`);
        if (activeHeader) {
            activeHeader.classList.add(sortOrderSelect.value === 'asc' ? 'sorting-asc' : 'sorting-desc');
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
                statusBadge.className = `pipeline-status-badge ${status.pipeline_state}`;
                statusText.textContent = status.pipeline_state.replace('_', ' ').toUpperCase();
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
            
            // Update control buttons
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
        
        // Update agent status with new indicators
        this.updateAgentStatus(status.agent_status);
    }
    
    updateAgentStatus(agentStatus) {
        Object.entries(agentStatus).forEach(([agentName, status]) => {
            const agentCard = document.getElementById(agentName.replace('_', '-'));
            if (agentCard) {
                const statusElement = agentCard.querySelector('.agent-status');
                if (statusElement) {
                    statusElement.className = `agent-status ${status}`;
                    statusElement.textContent = status.replace('_', ' ').toUpperCase();
                }
            }
        });
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
        
        const effort = idea.est_effort_days || idea.effort_estimate_days;
        this.setModalElement('modal-effort', effort ? `${effort} days` : '-');
        
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
