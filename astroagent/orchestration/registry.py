"""
Registry management for AstroAgent Pipeline.

Provides high-level interfaces for managing research ideas, projects,
and completed work through the CSV-based registry system.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .tools import RegistryManager


class ProjectRegistry:
    """High-level interface for managing project lifecycle through registries."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize project registry.
        
        Args:
            data_dir: Base data directory path
        """
        self.data_dir = Path(data_dir)
        self.registry_manager = RegistryManager(data_dir)
    
    def create_idea(self, idea_data: Dict[str, Any]) -> str:
        """Create a new research idea in the registry.
        
        Args:
            idea_data: Dictionary containing idea details
            
        Returns:
            Generated idea ID
        """
        
        # Ensure idea has required fields
        required_fields = ['title', 'hypothesis', 'rationale', 'domain_tags']
        for field in required_fields:
            if field not in idea_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Generate idea ID if not provided
        if 'idea_id' not in idea_data:
            from ..agents.common import generate_ulid
            idea_data['idea_id'] = generate_ulid()
        
        # Set default values
        idea_data.setdefault('status', 'Proposed')
        idea_data.setdefault('version', 'v1')
        
        # Add to registry
        self.registry_manager.append_to_registry('ideas_register', idea_data)
        
        return idea_data['idea_id']
    
    def update_idea(self, idea_id: str, updates: Dict[str, Any]):
        """Update an existing idea in the registry.
        
        Args:
            idea_id: ID of idea to update
            updates: Dictionary of fields to update
        """
        
        self.registry_manager.update_registry_row(
            'ideas_register',
            {'idea_id': idea_id},
            updates
        )
    
    def get_idea(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific idea by ID.
        
        Args:
            idea_id: ID of idea to retrieve
            
        Returns:
            Idea data dictionary or None if not found
        """
        
        df = self.registry_manager.query_registry(
            'ideas_register',
            {'idea_id': idea_id}
        )
        
        if df.empty:
            return None
        
        # Convert to dictionary and parse JSON fields
        idea = df.iloc[0].to_dict()
        
        # Parse JSON fields back to lists
        json_fields = ['domain_tags', 'novelty_refs', 'required_data', 'methods']
        for field in json_fields:
            if field in idea and isinstance(idea[field], str):
                try:
                    idea[field] = json.loads(idea[field])
                except (json.JSONDecodeError, TypeError):
                    # Keep as string if parsing fails
                    pass
        
        return idea
    
    def get_ideas_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all ideas with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of idea dictionaries
        """
        
        df = self.registry_manager.query_registry(
            'ideas_register',
            {'status': status}
        )
        
        ideas = []
        for _, row in df.iterrows():
            idea = row.to_dict()
            
            # Parse JSON fields
            json_fields = ['domain_tags', 'novelty_refs', 'required_data', 'methods']
            for field in json_fields:
                if field in idea and isinstance(idea[field], str):
                    try:
                        idea[field] = json.loads(idea[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            ideas.append(idea)
        
        return ideas
    
    def create_project(self, idea_id: str, slug: str, project_path: str) -> Dict[str, Any]:
        """Create a new project entry for an approved idea.
        
        Args:
            idea_id: ID of the associated idea
            slug: URL-safe project identifier
            project_path: Path to project directory
            
        Returns:
            Created project data
        """
        
        # Check if project already exists for this idea
        existing_projects = self.registry_manager.query_registry('project_index', {'idea_id': idea_id})
        if not existing_projects.empty:
            raise ValueError(f"Project already exists for idea {idea_id}")
        
        project_data = {
            'idea_id': idea_id,
            'slug': slug,
            'path': project_path,
            'maturity': 'Preparing',
            'ready_checklist_passed': False,
            'data_requirements_met': False,
            'analysis_plan_preregistered': False
        }
        
        self.registry_manager.append_to_registry('project_index', project_data)
        
        return project_data
    
    def update_project(self, idea_id: str, updates: Dict[str, Any]):
        """Update a project entry.
        
        Args:
            idea_id: ID of the associated idea
            updates: Dictionary of fields to update
        """
        
        self.registry_manager.update_registry_row(
            'project_index',
            {'idea_id': idea_id},
            updates
        )
    
    def get_project(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """Get project by idea ID.
        
        Args:
            idea_id: ID of the associated idea
            
        Returns:
            Project data dictionary or None if not found
        """
        
        df = self.registry_manager.query_registry(
            'project_index',
            {'idea_id': idea_id}
        )
        
        if df.empty:
            return None
        
        return df.iloc[0].to_dict()
    
    def get_projects_by_maturity(self, maturity: str) -> List[Dict[str, Any]]:
        """Get all projects with specific maturity level.
        
        Args:
            maturity: Maturity level to filter by
            
        Returns:
            List of project dictionaries
        """
        
        df = self.registry_manager.query_registry(
            'project_index',
            {'maturity': maturity}
        )
        
        return [row.to_dict() for _, row in df.iterrows()]
    
    def complete_project(self, idea_id: str, completion_data: Dict[str, Any]) -> str:
        """Move a project to completed status.
        
        Args:
            idea_id: ID of the completed project
            completion_data: Data for completed index
            
        Returns:
            Completion record ID
        """
        
        # Get original idea for title
        idea = self.get_idea(idea_id)
        if not idea:
            raise ValueError(f"Idea {idea_id} not found")
        
        # Prepare completion record
        completion_record = {
            'idea_id': idea_id,
            'title': idea.get('title', 'Unknown'),
            'moved_to_library_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Add provided completion data
        completion_record.update(completion_data)
        
        # Add to completed index
        self.registry_manager.append_to_registry('completed_index', completion_record)
        
        # Update project maturity
        self.update_project(idea_id, {'maturity': 'Complete'})
        
        return idea_id
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get overall pipeline statistics.
        
        Returns:
            Statistics dictionary
        """
        
        # Load all registries
        ideas_df = self.registry_manager.load_registry('ideas_register')
        projects_df = self.registry_manager.load_registry('project_index')
        completed_df = self.registry_manager.load_registry('completed_index')
        
        stats = {
            'total_ideas': len(ideas_df),
            'total_projects': len(projects_df),
            'total_completed': len(completed_df)
        }
        
        # Idea status breakdown
        if not ideas_df.empty:
            status_counts = ideas_df['status'].value_counts().to_dict()
            stats['idea_statuses'] = status_counts
        else:
            stats['idea_statuses'] = {}
        
        # Project maturity breakdown
        if not projects_df.empty:
            maturity_counts = projects_df['maturity'].value_counts().to_dict()
            stats['project_maturities'] = maturity_counts
        else:
            stats['project_maturities'] = {}
        
        # Recent activity (last 30 days)
        current_time = datetime.now(timezone.utc)
        thirty_days_ago = current_time.replace(day=current_time.day-30) if current_time.day > 30 else current_time.replace(month=current_time.month-1)
        
        if not ideas_df.empty and 'created_at' in ideas_df.columns:
            # Convert created_at to datetime for filtering
            ideas_df['created_at'] = pd.to_datetime(ideas_df['created_at'], errors='coerce')
            recent_ideas = ideas_df[ideas_df['created_at'] >= thirty_days_ago]
            stats['recent_ideas_30d'] = len(recent_ideas)
        else:
            stats['recent_ideas_30d'] = 0
        
        return stats
    
    def search_ideas(self, query: str, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search ideas by text content.
        
        Args:
            query: Search query string
            fields: Fields to search in (default: title, hypothesis, rationale)
            
        Returns:
            List of matching ideas
        """
        
        if fields is None:
            fields = ['title', 'hypothesis', 'rationale']
        
        ideas_df = self.registry_manager.load_registry('ideas_register')
        
        if ideas_df.empty:
            return []
        
        # Simple text search across specified fields
        query_lower = query.lower()
        matches = []
        
        for _, row in ideas_df.iterrows():
            idea = row.to_dict()
            
            # Check if query appears in any of the specified fields
            for field in fields:
                if field in idea and idea[field]:
                    if query_lower in str(idea[field]).lower():
                        # Parse JSON fields
                        json_fields = ['domain_tags', 'novelty_refs', 'required_data', 'methods']
                        for json_field in json_fields:
                            if json_field in idea and isinstance(idea[json_field], str):
                                try:
                                    idea[json_field] = json.loads(idea[json_field])
                                except (json.JSONDecodeError, TypeError):
                                    pass
                        
                        matches.append(idea)
                        break
        
        return matches
    
    def get_idea_history(self, base_idea_id: str) -> List[Dict[str, Any]]:
        """Get version history for an idea.
        
        Args:
            base_idea_id: ID of the original idea
            
        Returns:
            List of idea versions in chronological order
        """
        
        ideas_df = self.registry_manager.load_registry('ideas_register')
        
        if ideas_df.empty:
            return []
        
        # Find all versions of this idea
        versions = []
        
        # Add the original idea
        original = ideas_df[ideas_df['idea_id'] == base_idea_id]
        if not original.empty:
            versions.extend(original.to_dict('records'))
        
        # Find all ideas that have this as parent
        children = ideas_df[ideas_df['parent_idea_id'] == base_idea_id]
        if not children.empty:
            versions.extend(children.to_dict('records'))
        
        # Sort by created_at if available
        if versions and 'created_at' in versions[0]:
            versions.sort(key=lambda x: x.get('created_at', ''))
        
        # Parse JSON fields for all versions
        for idea in versions:
            json_fields = ['domain_tags', 'novelty_refs', 'required_data', 'methods']
            for field in json_fields:
                if field in idea and isinstance(idea[field], str):
                    try:
                        idea[field] = json.loads(idea[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        return versions
