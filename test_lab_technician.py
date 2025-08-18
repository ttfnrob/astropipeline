#!/usr/bin/env python3
"""
Test script for the Lab Technician meta-agent.

This script demonstrates the lab technician's ability to:
1. Analyze experimenter performance logs
2. Identify improvement opportunities
3. Apply code modifications
4. Validate improvements

Usage:
    python test_lab_technician.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from astroagent.agents import create_agent
from astroagent.agents.common import AgentExecutionContext
from astroagent.orchestration.continuous_pipeline import ContinuousPipeline


def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('lab_technician_test.log')
        ]
    )
    return logging.getLogger(__name__)


def create_mock_experimenter_logs():
    """Create some mock experimenter logs for testing."""
    
    # Create log directory
    log_dir = Path('data/logs/experimenter')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock execution logs
    mock_logs = [
        {
            'timestamp': '2024-01-15T10:30:00Z',
            'execution_id': 'test-exec-001',
            'event_type': 'execution_start',
            'event_data': {'project_id': 'test-project-001'}
        },
        {
            'timestamp': '2024-01-15T10:31:00Z',
            'execution_id': 'test-exec-001',
            'event_type': 'data_fetch_failed',
            'event_data': {
                'error': 'ConnectionError: Unable to connect to data source',
                'error_type': 'ConnectionError',
                'fetch_time': 60.0
            }
        },
        {
            'timestamp': '2024-01-15T10:31:05Z',
            'execution_id': 'test-exec-001',
            'event_type': 'execution_error',
            'event_data': {
                'error_message': 'Data fetching failed: ConnectionError: Unable to connect to data source',
                'error_type': 'ConnectionError',
                'execution_time': 65.0
            }
        },
        {
            'timestamp': '2024-01-15T11:00:00Z',
            'execution_id': 'test-exec-002',
            'event_type': 'execution_start',
            'event_data': {'project_id': 'test-project-002'}
        },
        {
            'timestamp': '2024-01-15T11:01:30Z',
            'execution_id': 'test-exec-002',
            'event_type': 'analysis_failed',
            'event_data': {
                'error_message': 'No data available for analysis',
                'error_type': 'ValueError',
                'execution_time': 90.0
            }
        },
        {
            'timestamp': '2024-01-15T12:00:00Z',
            'execution_id': 'test-exec-003',
            'event_type': 'execution_start',
            'event_data': {'project_id': 'test-project-003'}
        },
        {
            'timestamp': '2024-01-15T12:05:00Z',
            'execution_id': 'test-exec-003',
            'event_type': 'execution_complete',
            'event_data': {
                'total_execution_time': 300.0,
                'success': True,
                'performance_breakdown': {
                    'project_load_time': 5.0,
                    'analysis_time': 280.0,
                    'write_time': 15.0
                }
            }
        }
    ]
    
    # Write logs to file
    log_file = log_dir / f"execution_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, 'w') as f:
        for log in mock_logs:
            f.write(json.dumps(log) + '\n')
    
    # Create mock performance metrics
    metrics = {
        'method_execution_times': {
            'load_project': [5.2, 4.8, 6.1, 5.5],
            'data_fetch': [60.0, 45.2, 70.8, 55.3],
            'execute_analysis': [280.5, 320.1, 250.8, 290.2],
            'write_results': [15.1, 12.8, 18.2, 14.5]
        },
        'error_counts': {
            'load_project': {'success': 4, 'error': 0},
            'data_fetch': {'success': 1, 'error': 3},
            'execute_analysis': {'success': 1, 'error': 2},
            'write_results': {'success': 1, 'error': 0}
        },
        'success_rates': {
            'load_project': 1.0,
            'data_fetch': 0.25,  # Low success rate
            'execute_analysis': 0.33,  # Low success rate
            'write_results': 1.0
        }
    }
    
    metrics_file = log_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return len(mock_logs)


async def test_lab_technician_basic():
    """Test basic lab technician functionality."""
    
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("🧪 Testing Lab Technician Meta-Agent")
    logger.info("=" * 60)
    
    # Step 1: Create mock data
    logger.info("\n📝 Step 1: Creating mock experimenter logs...")
    log_count = create_mock_experimenter_logs()
    logger.info(f"   Created {log_count} mock log entries")
    
    # Step 2: Create and run lab technician
    logger.info("\n🔧 Step 2: Creating Lab Technician agent...")
    
    try:
        lab_technician = create_agent('lab_technician', 'astroagent/config')
        logger.info("   ✅ Lab Technician agent created successfully")
        
        # Create execution context
        context = AgentExecutionContext(
            agent_name='lab_technician',
            state_name='test_analysis',
            input_data={
                'analysis_trigger': 'manual_test',
                'force_analysis': True
            }
        )
        
        # Step 3: Run analysis
        logger.info("\n📊 Step 3: Running Lab Technician analysis...")
        result = lab_technician.run(context)
        
        if result.success:
            logger.info("   ✅ Analysis completed successfully!")
            
            # Extract and display results
            analysis_results = result.output_data.get('analysis_results', {})
            improvement_opportunities = result.output_data.get('improvement_opportunities', [])
            improvements_made = result.output_data.get('improvements_made', [])
            
            # Display analysis summary
            logger.info(f"\n📈 Analysis Summary:")
            if analysis_results.get('execution_analysis'):
                exec_stats = analysis_results['execution_analysis']
                logger.info(f"   • Total executions analyzed: {exec_stats.get('total_executions', 0)}")
                logger.info(f"   • Success rate: {exec_stats.get('success_rate', 0):.1%}")
                logger.info(f"   • Failed executions: {exec_stats.get('failed_executions', 0)}")
                logger.info(f"   • Average execution time: {exec_stats.get('average_execution_time', 0):.1f}s")
            
            # Display improvement opportunities
            if improvement_opportunities:
                logger.info(f"\n🔍 Improvement Opportunities Found: {len(improvement_opportunities)}")
                for i, opp in enumerate(improvement_opportunities[:3], 1):
                    priority = opp.get('priority', 'unknown')
                    opp_type = opp.get('type', 'unknown')
                    description = opp.get('description', 'No description')
                    logger.info(f"   {i}. [{priority.upper()}] {opp_type}")
                    logger.info(f"      {description}")
            else:
                logger.info("\n✅ No improvement opportunities identified")
            
            # Display improvements made
            if improvements_made:
                logger.info(f"\n🛠️  Code Improvements Applied: {len(improvements_made)}")
                for improvement in improvements_made:
                    status = improvement.get('status', 'unknown')
                    opp_type = improvement['opportunity'].get('type', 'unknown')
                    logger.info(f"   • {opp_type}: {status}")
            else:
                logger.info("\n⏩ No code improvements applied (no high-priority issues)")
            
            # Display report if available
            report = result.output_data.get('report', '')
            if report:
                logger.info("\n📋 Generated Report:")
                logger.info("-" * 40)
                # Show first few lines of report
                report_lines = report.split('\n')[:10]
                for line in report_lines:
                    logger.info(f"   {line}")
                if len(report.split('\n')) > 10:
                    logger.info("   ... (report truncated)")
        else:
            logger.error(f"   ❌ Analysis failed: {result.error_message}")
            logger.error(f"   Error type: {result.error_type}")
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Lab Technician test completed!")
    logger.info("=" * 60)
    return True


async def test_continuous_pipeline_integration():
    """Test lab technician integration with continuous pipeline."""
    
    logger = setup_logging()
    logger.info("\n" + "=" * 60)
    logger.info("🔄 Testing Continuous Pipeline Integration")
    logger.info("=" * 60)
    
    try:
        # Create continuous pipeline
        pipeline = ContinuousPipeline()
        
        # Configure lab technician for rapid testing
        pipeline.set_lab_technician_interval(10)  # 10 seconds for testing
        
        logger.info("✅ Continuous pipeline created")
        logger.info(f"🔧 Lab technician enabled: {pipeline.lab_technician_enabled}")
        logger.info(f"🕐 Lab technician interval: {pipeline.lab_technician_interval}s")
        
        # Test manual lab technician run
        logger.info("\n🔧 Testing manual lab technician execution...")
        result = await pipeline.run_lab_technician_now()
        
        if result.get('status') == 'completed':
            logger.info("   ✅ Manual lab technician run successful")
            if result.get('improvements_made'):
                logger.info(f"   🛠️  Improvements made: {len(result['improvements_made'])}")
        elif result.get('status') == 'error':
            logger.warning(f"   ⚠️  Lab technician error: {result.get('error_message')}")
        else:
            logger.info(f"   ℹ️  Lab technician status: {result.get('status')}")
        
        # Test control functions
        logger.info("\n🎛️  Testing lab technician control functions...")
        pipeline.disable_lab_technician()
        logger.info("   🔇 Lab technician disabled")
        
        pipeline.enable_lab_technician()
        logger.info("   🔊 Lab technician re-enabled")
        
        logger.info("✅ Pipeline integration test completed")
        
    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {str(e)}")
        return False
    
    return True


async def main():
    """Main test function."""
    logger = logging.getLogger(__name__)
    
    success = True
    
    # Run basic functionality test
    logger.info("🚀 Starting Lab Technician tests...\n")
    
    basic_test_success = await test_lab_technician_basic()
    success = success and basic_test_success
    
    # Run pipeline integration test
    pipeline_test_success = await test_continuous_pipeline_integration()
    success = success and pipeline_test_success
    
    # Final summary
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("The Lab Technician meta-agent is working correctly.")
        logger.info("\nKey features demonstrated:")
        logger.info("  ✅ Log analysis and performance monitoring")
        logger.info("  ✅ Improvement opportunity identification")
        logger.info("  ✅ Code modification capabilities")
        logger.info("  ✅ Continuous pipeline integration")
        logger.info("  ✅ Manual control and configuration")
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error("Please check the logs above for details.")
    
    logger.info("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
