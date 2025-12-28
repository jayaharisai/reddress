import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from click import get_app_dir
import hashlib

class RAGDatabase:
    """Track indexed projects and their metadata"""
    
    def __init__(self):
        self.db_path = os.path.join(get_app_dir("reddress"), "rag_projects.json")
        self._ensure_db()
    
    def _ensure_db(self):
        """Create database if doesn't exist"""
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump({"projects": []}, f)
    
    def add_project(self, project_data: Dict):
        """Add or update indexed project"""
        db = self._load_db()
        
        # Check if project exists
        existing_idx = next((i for i, p in enumerate(db['projects']) 
                           if p['name'] == project_data['name']), None)
        
        if existing_idx is not None:
            # Merge file_hashes from old and new
            old_hashes = db['projects'][existing_idx].get('file_hashes', {})
            new_hashes = project_data.get('file_hashes', {})
            project_data['file_hashes'] = {**old_hashes, **new_hashes}
            
            db['projects'][existing_idx] = project_data
        else:
            db['projects'].append(project_data)
        
        self._save_db(db)
    
    def get_project(self, name: str) -> Optional[Dict]:
        """Get project by name"""
        db = self._load_db()
        return next((p for p in db['projects'] if p['name'] == name), None)
    
    def list_projects(self) -> List[Dict]:
        """List all indexed projects"""
        db = self._load_db()
        return db['projects']
    
    def delete_project(self, name: str) -> bool:
        """Delete project"""
        db = self._load_db()
        initial_count = len(db['projects'])
        db['projects'] = [p for p in db['projects'] if p['name'] != name]
        
        if len(db['projects']) < initial_count:
            self._save_db(db)
            return True
        return False
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except:
            return ""
    
    def get_indexed_files(self, project_name: str) -> Dict[str, str]:
        """Get dictionary of indexed files and their hashes"""
        project = self.get_project(project_name)
        if project:
            return project.get('file_hashes', {})
        return {}
    
    def update_file_hashes(self, project_name: str, file_hashes: Dict[str, str]):
        """Update file hashes for a project"""
        project = self.get_project(project_name)
        if project:
            project['file_hashes'] = file_hashes
            self.add_project(project)
    
    def _load_db(self) -> Dict:
        """Load database"""
        with open(self.db_path, 'r') as f:
            return json.load(f)
    
    def _save_db(self, db: Dict):
        """Save database"""
        with open(self.db_path, 'w') as f:
            json.dump(db, f, indent=2)
