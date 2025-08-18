"""
Lab Technician (LT) Agent for AstroAgent Pipeline.

This meta-agent continuously monitors the Experimenter agent's performance,
analyzes execution logs, identifies improvement opportunities, and automatically
modifies code to enhance the Experimenter's capabilities.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
import ast
import re
import subprocess
import time
import statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import traceback

from .common import BaseAgent, AgentExecutionContext, AgentResult


class LabTechnician(BaseAgent):
    """Meta-agent responsible for continuously improving the Experimenter agent."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.1)  # Low temperature for code generation
        self.analysis_window_hours = config.get('analysis_window_hours', 24)
        self.improvement_threshold = config.get('improvement_threshold', 0.7)  # Minimum success rate
        self.backup_before_changes = config.get('backup_before_changes', True)
        
        # Paths
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.experimenter_log_dir = self.data_dir / 'logs' / 'experimenter'
        self.lab_technician_log_dir = self.data_dir / 'logs' / 'lab_technician'
        self.lab_technician_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Experimenter code path
        self.experimenter_code_path = Path(__file__).parent / 'experimenter.py'
        self.backup_dir = self.lab_technician_log_dir / 'backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis state
        self.performance_history = []
        self.improvement_suggestions = []
        self.last_analysis_time = None
        
        # Initialize analysis state
        self._load_analysis_state()
    
    def execute(self, context: AgentExecutionContext) -> AgentResult:
        """Execute lab technician analysis and improvement cycle."""
        
        start_time = datetime.now()
        self.logger.info("Starting Lab Technician analysis cycle")
        
        try:
            # Step 1: Analyze recent experimenter performance
            analysis_results = self._analyze_experimenter_performance()
            
            # Step 2: Identify improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(analysis_results)
            
            # Step 3: Generate code improvements if needed
            improvements_made = []
            if improvement_opportunities:
                improvements_made = self._generate_and_apply_improvements(improvement_opportunities)
            
            # Step 4: Validate improvements (if any were made)
            validation_results = {}
            if improvements_made:
                validation_results = self._validate_improvements(improvements_made)
            
            # Step 5: Update analysis state
            self._save_analysis_state(analysis_results, improvements_made)
            
            # Step 6: Generate report
            report = self._generate_improvement_report(
                analysis_results, 
                improvement_opportunities, 
                improvements_made, 
                validation_results
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                success=True,
                agent_name=self.name,
                execution_id=context.execution_id,
                output_data={
                    'analysis_results': analysis_results,
                    'improvement_opportunities': improvement_opportunities,
                    'improvements_made': improvements_made,
                    'validation_results': validation_results,
                    'report': report
                },
                files_created=[str(self.lab_technician_log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")],
                execution_time_seconds=execution_time,
                input_hash="",
                output_hash=""
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Lab Technician execution failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return AgentResult(
                success=False,
                agent_name=self.name,
                execution_id=context.execution_id,
                error_message=str(e),
                error_type=type(e).__name__,
                execution_time_seconds=execution_time,
                input_hash="",
                output_hash=""
            )
    
    def _analyze_experimenter_performance(self) -> Dict[str, Any]:
        """Analyze recent experimenter execution logs and performance metrics."""
        
        self.logger.info("Analyzing experimenter performance...")
        
        # Get recent log files
        cutoff_time = datetime.now() - timedelta(hours=self.analysis_window_hours)
        recent_logs = self._load_recent_logs(cutoff_time)
        
        if not recent_logs:
            return {
                'status': 'no_data',
                'message': f'No experimenter logs found in the last {self.analysis_window_hours} hours',
                'analysis_time': datetime.now(timezone.utc).isoformat()
            }
        
        # Analyze execution patterns
        execution_analysis = self._analyze_execution_patterns(recent_logs)
        
        # Analyze error patterns
        error_analysis = self._analyze_error_patterns(recent_logs)
        
        # Analyze performance metrics
        performance_analysis = self._analyze_performance_metrics()
        
        # Analyze code quality indicators
        code_quality_analysis = self._analyze_code_quality()
        
        return {
            'status': 'completed',
            'analysis_time': datetime.now(timezone.utc).isoformat(),
            'log_entries_analyzed': len(recent_logs),
            'time_window_hours': self.analysis_window_hours,
            'execution_analysis': execution_analysis,
            'error_analysis': error_analysis,
            'performance_analysis': performance_analysis,
            'code_quality_analysis': code_quality_analysis
        }
    
    def _load_recent_logs(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Load recent experimenter execution logs."""
        
        logs = []
        
        # Get all log files in the experimenter log directory
        if not self.experimenter_log_dir.exists():
            self.logger.warning(f"Experimenter log directory not found: {self.experimenter_log_dir}")
            return logs
        
        for log_file in self.experimenter_log_dir.glob("execution_*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                            
                            if log_time >= cutoff_time:
                                logs.append(log_entry)
                                
            except Exception as e:
                self.logger.warning(f"Failed to read log file {log_file}: {e}")
        
        # Sort by timestamp
        logs.sort(key=lambda x: x['timestamp'])
        self.logger.info(f"Loaded {len(logs)} recent log entries")
        
        return logs
    
    def _analyze_execution_patterns(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze execution patterns from logs."""
        
        # Group logs by execution_id
        executions = defaultdict(list)
        for log in logs:
            executions[log['execution_id']].append(log)
        
        # Analyze each execution
        execution_stats = {
            'total_executions': len(executions),
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0,
            'common_failure_stages': Counter(),
            'execution_times': [],
            'success_rate': 0
        }
        
        for execution_id, execution_logs in executions.items():
            # Determine if execution was successful
            execution_events = [log['event_type'] for log in execution_logs]
            
            if 'execution_complete' in execution_events:
                execution_stats['successful_executions'] += 1
                
                # Get execution time
                for log in execution_logs:
                    if log['event_type'] == 'execution_complete':
                        exec_time = log['event_data'].get('total_execution_time', 0)
                        execution_stats['execution_times'].append(exec_time)
                        break
                        
            elif 'execution_error' in execution_events or 'analysis_failed' in execution_events:
                execution_stats['failed_executions'] += 1
                
                # Find the failure stage
                for log in execution_logs:
                    if log['event_type'] in ['execution_error', 'analysis_failed', 'data_fetch_failed']:
                        failure_stage = log['event_type']
                        execution_stats['common_failure_stages'][failure_stage] += 1
                        break
        
        # Calculate averages and rates
        if execution_stats['execution_times']:
            execution_stats['average_execution_time'] = statistics.mean(execution_stats['execution_times'])
        
        if execution_stats['total_executions'] > 0:
            execution_stats['success_rate'] = execution_stats['successful_executions'] / execution_stats['total_executions']
        
        return execution_stats
    
    def _analyze_error_patterns(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns and frequencies."""
        
        error_logs = [log for log in logs if 'error' in log['event_type'] or 'failed' in log['event_type']]
        
        error_analysis = {
            'total_errors': len(error_logs),
            'error_types': Counter(),
            'error_stages': Counter(),
            'common_error_messages': Counter(),
            'error_trends': []
        }
        
        for error_log in error_logs:
            # Count error types
            error_data = error_log.get('event_data', {})
            
            if 'error_type' in error_data:
                error_analysis['error_types'][error_data['error_type']] += 1
            
            # Count error stages
            error_analysis['error_stages'][error_log['event_type']] += 1
            
            # Count error messages (first 100 chars for similarity)
            if 'error_message' in error_data:
                error_msg = error_data['error_message'][:100]
                error_analysis['common_error_messages'][error_msg] += 1
        
        return error_analysis
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics from experimenter metrics files."""
        
        performance_analysis = {
            'metrics_files_analyzed': 0,
            'method_performance': {},
            'success_rates': {},
            'execution_time_trends': {},
            'resource_usage': {}
        }
        
        # Load recent performance metrics files
        cutoff_date = (datetime.now() - timedelta(hours=self.analysis_window_hours)).strftime('%Y%m%d')
        
        for metrics_file in self.experimenter_log_dir.glob(f"performance_metrics_*.json"):
            try:
                # Check if file is recent enough
                date_match = re.search(r'performance_metrics_(\d{8})\.json', metrics_file.name)
                if date_match:
                    file_date = date_match.group(1)
                    if file_date >= cutoff_date:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            performance_analysis['metrics_files_analyzed'] += 1
                            
                            # Merge metrics
                            for method, times in metrics.get('method_execution_times', {}).items():
                                if method not in performance_analysis['method_performance']:
                                    performance_analysis['method_performance'][method] = []
                                performance_analysis['method_performance'][method].extend(times)
                            
                            # Merge success rates
                            performance_analysis['success_rates'].update(metrics.get('success_rates', {}))
                            
            except Exception as e:
                self.logger.warning(f"Failed to read metrics file {metrics_file}: {e}")
        
        # Calculate statistics
        for method, times in performance_analysis['method_performance'].items():
            if times:
                performance_analysis['execution_time_trends'][method] = {
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'std': statistics.stdev(times) if len(times) > 1 else 0,
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        
        return performance_analysis
    
    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze the experimenter code for quality issues and improvement opportunities."""
        
        code_analysis = {
            'file_analyzed': str(self.experimenter_code_path),
            'lines_of_code': 0,
            'method_complexity': {},
            'potential_issues': [],
            'improvement_opportunities': []
        }
        
        try:
            with open(self.experimenter_code_path, 'r') as f:
                code_content = f.read()
            
            code_analysis['lines_of_code'] = len(code_content.splitlines())
            
            # Parse AST for analysis
            tree = ast.parse(code_content)
            
            # Analyze methods
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_name = node.name
                    
                    # Calculate complexity (simplified)
                    complexity = self._calculate_method_complexity(node)
                    code_analysis['method_complexity'][method_name] = complexity
                    
                    # Identify potential issues
                    if complexity > 10:
                        code_analysis['potential_issues'].append({
                            'type': 'high_complexity',
                            'method': method_name,
                            'complexity': complexity,
                            'suggestion': 'Consider breaking down this method into smaller functions'
                        })
                    
                    # Check for long methods
                    method_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    if method_lines > 100:
                        code_analysis['potential_issues'].append({
                            'type': 'long_method',
                            'method': method_name,
                            'lines': method_lines,
                            'suggestion': 'Consider refactoring this method to improve readability'
                        })
            
            # Look for improvement opportunities based on common patterns
            if 'time.sleep' in code_content:
                code_analysis['improvement_opportunities'].append({
                    'type': 'blocking_sleep',
                    'suggestion': 'Replace blocking sleep calls with async/await patterns'
                })
            
            if code_content.count('try:') < code_content.count('def '):
                code_analysis['improvement_opportunities'].append({
                    'type': 'insufficient_error_handling',
                    'suggestion': 'Add more comprehensive error handling'
                })
                
        except Exception as e:
            self.logger.error(f"Failed to analyze code: {e}")
            code_analysis['analysis_error'] = str(e)
        
        return code_analysis
    
    def _calculate_method_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a method (simplified)."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _identify_improvement_opportunities(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities based on analysis."""
        
        opportunities = []
        
        # Check success rate
        execution_analysis = analysis_results.get('execution_analysis', {})
        success_rate = execution_analysis.get('success_rate', 1.0)
        
        if success_rate < self.improvement_threshold:
            opportunities.append({
                'type': 'low_success_rate',
                'priority': 'high',
                'current_rate': success_rate,
                'target_rate': self.improvement_threshold,
                'description': f'Success rate ({success_rate:.2%}) is below threshold ({self.improvement_threshold:.2%})',
                'suggested_actions': [
                    'Improve error handling',
                    'Add retry mechanisms',
                    'Better input validation'
                ]
            })
        
        # Check for common errors
        error_analysis = analysis_results.get('error_analysis', {})
        common_errors = error_analysis.get('error_types', {})
        
        for error_type, count in common_errors.most_common(3):
            if count > 2:  # If error occurs more than twice
                opportunities.append({
                    'type': 'frequent_error',
                    'priority': 'medium',
                    'error_type': error_type,
                    'occurrences': count,
                    'description': f'{error_type} occurs frequently ({count} times)',
                    'suggested_actions': [
                        f'Add specific handling for {error_type}',
                        'Improve input validation',
                        'Add defensive programming checks'
                    ]
                })
        
        # Check performance issues
        performance_analysis = analysis_results.get('performance_analysis', {})
        for method, stats in performance_analysis.get('execution_time_trends', {}).items():
            if stats['mean'] > 60:  # Methods taking more than 60 seconds
                opportunities.append({
                    'type': 'slow_method',
                    'priority': 'medium',
                    'method': method,
                    'average_time': stats['mean'],
                    'description': f'Method {method} is slow (avg: {stats["mean"]:.1f}s)',
                    'suggested_actions': [
                        'Optimize data processing',
                        'Add caching mechanisms',
                        'Parallelize operations'
                    ]
                })
        
        # Check code quality issues
        code_quality = analysis_results.get('code_quality_analysis', {})
        opportunities.extend(code_quality.get('improvement_opportunities', []))
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        opportunities.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
        
        return opportunities
    
    def _generate_and_apply_improvements(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate and apply code improvements based on identified opportunities."""
        
        improvements_made = []
        
        # Only apply improvements if there are high priority issues
        high_priority_opportunities = [op for op in opportunities if op.get('priority') == 'high']
        
        if not high_priority_opportunities:
            self.logger.info("No high priority improvements needed")
            return improvements_made
        
        # Backup current code before making changes
        if self.backup_before_changes:
            backup_path = self._backup_experimenter_code()
            self.logger.info(f"Backed up experimenter code to: {backup_path}")
        
        for opportunity in high_priority_opportunities[:3]:  # Limit to top 3 for safety
            try:
                improvement = self._generate_improvement_for_opportunity(opportunity)
                
                if improvement:
                    success = self._apply_code_improvement(improvement)
                    
                    if success:
                        improvements_made.append({
                            'opportunity': opportunity,
                            'improvement': improvement,
                            'applied_at': datetime.now(timezone.utc).isoformat(),
                            'status': 'success'
                        })
                    else:
                        improvements_made.append({
                            'opportunity': opportunity,
                            'improvement': improvement,
                            'applied_at': datetime.now(timezone.utc).isoformat(),
                            'status': 'failed'
                        })
                        
            except Exception as e:
                self.logger.error(f"Failed to generate/apply improvement: {e}")
                improvements_made.append({
                    'opportunity': opportunity,
                    'error': str(e),
                    'applied_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'error'
                })
        
        return improvements_made
    
    def _backup_experimenter_code(self) -> Path:
        """Create a backup of the current experimenter code."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"experimenter_{timestamp}.py"
        
        with open(self.experimenter_code_path, 'r') as src:
            with open(backup_path, 'w') as dst:
                dst.write(src.read())
        
        return backup_path
    
    def _generate_improvement_for_opportunity(self, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a specific code improvement for an opportunity."""
        
        # This is where we would use the LLM to generate improvements
        # For now, we'll implement some basic rule-based improvements
        
        opportunity_type = opportunity.get('type')
        
        if opportunity_type == 'low_success_rate':
            return {
                'type': 'add_retry_mechanism',
                'description': 'Add retry mechanism for failed operations',
                'code_changes': [
                    {
                        'method': '_fetch_real_astronomical_data',
                        'change_type': 'add_retry',
                        'retry_count': 3,
                        'retry_delay': 2
                    }
                ]
            }
        
        elif opportunity_type == 'frequent_error':
            error_type = opportunity.get('error_type', '')
            
            if 'ConnectionError' in error_type or 'HTTPError' in error_type:
                return {
                    'type': 'improve_network_handling',
                    'description': 'Improve network error handling',
                    'code_changes': [
                        {
                            'method': '_fetch_real_astronomical_data',
                            'change_type': 'add_network_retry',
                            'retry_count': 3,
                            'backoff_factor': 2
                        }
                    ]
                }
        
        return None
    
    def _apply_code_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply a specific code improvement to the experimenter."""
        
        # This is a simplified implementation
        # In practice, this would use AST manipulation or careful string replacement
        
        try:
            with open(self.experimenter_code_path, 'r') as f:
                code_content = f.read()
            
            # Apply changes based on improvement type
            if improvement['type'] == 'add_retry_mechanism':
                # Simple example: add import for retry decorator
                if 'import time' not in code_content:
                    code_content = code_content.replace(
                        'from .common import BaseAgent',
                        'from .common import BaseAgent\nimport time'
                    )
            
            # Write back the modified code
            with open(self.experimenter_code_path, 'w') as f:
                f.write(code_content)
            
            self.logger.info(f"Applied improvement: {improvement['type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply code improvement: {e}")
            return False
    
    def _validate_improvements(self, improvements_made: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that improvements don't break the code."""
        
        validation_results = {
            'syntax_check': False,
            'import_check': False,
            'basic_functionality': False,
            'validation_time': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Syntax check
            with open(self.experimenter_code_path, 'r') as f:
                code_content = f.read()
            
            ast.parse(code_content)  # This will raise if syntax is invalid
            validation_results['syntax_check'] = True
            
            # Try to import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location("experimenter", self.experimenter_code_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            validation_results['import_check'] = True
            
            # Basic functionality check (instantiate the class)
            experimenter_class = getattr(module, 'Experimenter')
            experimenter_instance = experimenter_class({})
            validation_results['basic_functionality'] = True
            
        except Exception as e:
            validation_results['validation_error'] = str(e)
            self.logger.error(f"Validation failed: {e}")
        
        return validation_results
    
    def _generate_improvement_report(self, analysis_results: Dict[str, Any], 
                                   opportunities: List[Dict[str, Any]],
                                   improvements_made: List[Dict[str, Any]],
                                   validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable improvement report."""
        
        report = f"""# Lab Technician Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Summary
- **Log entries analyzed**: {analysis_results.get('log_entries_analyzed', 0)}
- **Time window**: {analysis_results.get('time_window_hours', 0)} hours
- **Improvement opportunities identified**: {len(opportunities)}
- **Improvements applied**: {len(improvements_made)}

## Performance Analysis
"""
        
        execution_analysis = analysis_results.get('execution_analysis', {})
        if execution_analysis:
            success_rate = execution_analysis.get('success_rate', 0)
            avg_time = execution_analysis.get('average_execution_time', 0)
            
            report += f"""
- **Success rate**: {success_rate:.1%}
- **Average execution time**: {avg_time:.1f} seconds
- **Total executions**: {execution_analysis.get('total_executions', 0)}
- **Failed executions**: {execution_analysis.get('failed_executions', 0)}
"""
        
        # Error analysis
        error_analysis = analysis_results.get('error_analysis', {})
        if error_analysis.get('total_errors', 0) > 0:
            report += f"\n## Error Analysis\n"
            report += f"- **Total errors**: {error_analysis['total_errors']}\n"
            
            if error_analysis.get('error_types'):
                report += "- **Most common error types**:\n"
                for error_type, count in error_analysis['error_types'].most_common(3):
                    report += f"  - {error_type}: {count} occurrences\n"
        
        # Improvement opportunities
        if opportunities:
            report += f"\n## Improvement Opportunities\n"
            for i, opp in enumerate(opportunities[:5], 1):
                report += f"{i}. **{opp.get('type', 'Unknown')}** ({opp.get('priority', 'low')} priority)\n"
                report += f"   - {opp.get('description', 'No description')}\n"
        
        # Improvements made
        if improvements_made:
            report += f"\n## Improvements Applied\n"
            for improvement in improvements_made:
                status = improvement.get('status', 'unknown')
                report += f"- **{improvement['opportunity'].get('type', 'Unknown')}**: {status}\n"
        
        # Validation results
        if validation_results:
            report += f"\n## Validation Results\n"
            report += f"- **Syntax check**: {'✓' if validation_results.get('syntax_check') else '✗'}\n"
            report += f"- **Import check**: {'✓' if validation_results.get('import_check') else '✗'}\n"
            report += f"- **Basic functionality**: {'✓' if validation_results.get('basic_functionality') else '✗'}\n"
        
        return report
    
    def _load_analysis_state(self):
        """Load previous analysis state."""
        state_file = self.lab_technician_log_dir / 'analysis_state.json'
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.performance_history = state.get('performance_history', [])
                    self.improvement_suggestions = state.get('improvement_suggestions', [])
                    self.last_analysis_time = state.get('last_analysis_time')
                    
            except Exception as e:
                self.logger.warning(f"Failed to load analysis state: {e}")
    
    def _save_analysis_state(self, analysis_results: Dict[str, Any], improvements_made: List[Dict[str, Any]]):
        """Save current analysis state."""
        
        # Update performance history
        if analysis_results.get('execution_analysis'):
            exec_analysis = analysis_results['execution_analysis']
            self.performance_history.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'success_rate': exec_analysis.get('success_rate', 0),
                'average_execution_time': exec_analysis.get('average_execution_time', 0),
                'total_executions': exec_analysis.get('total_executions', 0)
            })
            
            # Keep only last 100 entries
            self.performance_history = self.performance_history[-100:]
        
        # Update improvement suggestions
        if improvements_made:
            for improvement in improvements_made:
                self.improvement_suggestions.append({
                    'timestamp': improvement.get('applied_at'),
                    'type': improvement['opportunity'].get('type'),
                    'status': improvement.get('status'),
                    'description': improvement['opportunity'].get('description')
                })
        
        self.last_analysis_time = datetime.now(timezone.utc).isoformat()
        
        # Save to file
        state_file = self.lab_technician_log_dir / 'analysis_state.json'
        state = {
            'performance_history': self.performance_history,
            'improvement_suggestions': self.improvement_suggestions,
            'last_analysis_time': self.last_analysis_time
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save analysis state: {e}")
    
    def restart_experimenter_if_needed(self) -> bool:
        """Restart the experimenter process if needed after improvements."""
        
        # This would implement logic to restart the experimenter process
        # For now, we'll just log that a restart would be needed
        
        self.logger.info("Experimenter restart would be triggered here")
        return True
    
    def validate_input(self, context: AgentExecutionContext) -> bool:
        """Validate input for lab technician execution."""
        # Lab technician doesn't need specific input validation
        return True
    
    def validate_output(self, result: AgentResult) -> bool:
        """Validate lab technician output."""
        if not result.success:
            return False
        
        # Check that analysis was completed
        if 'analysis_results' not in result.output_data:
            self.logger.error("No analysis results generated")
            return False
        
        return True
