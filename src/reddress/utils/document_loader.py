import os
import PyPDF2
import docx
from pathlib import Path
from typing import List, Dict
from rich.console import Console

console = Console()

class DocumentLoader:
    """Load and extract text from various file types"""
    
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
        '.go', '.rs', '.rb', '.php', '.html', '.css', '.scss', '.json',
        '.yaml', '.yml', '.md', '.txt', '.sql', '.sh', '.bash',
        '.pdf', '.docx', '.doc'
    }
    
    def __init__(self):
        self.loaded_files = []
    
    def load_directory(self, path: str, recursive: bool = True) -> List[Dict]:
        """Load all supported files from directory"""
        documents = []
        path_obj = Path(path)
        
        if path_obj.is_file():
            doc = self.load_file(str(path_obj))
            if doc:
                documents.append(doc)
        else:
            pattern = "**/*" if recursive else "*"
            for file_path in path_obj.glob(pattern):
                if file_path.is_file() and file_path.suffix in self.SUPPORTED_EXTENSIONS:
                    # Skip hidden files and common ignore patterns
                    if self._should_skip(file_path):
                        continue
                    
                    doc = self.load_file(str(file_path))
                    if doc:
                        documents.append(doc)
        
        console.print(f"[green]✅ Loaded {len(documents)} files[/green]")
        return documents
    
    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_dirs = {
            'node_modules', '__pycache__', '.git', 'venv', 'env',
            'build', 'dist', '.next', 'target', '.idea'
        }
        
        # Check if any parent is in skip_dirs
        return any(part in skip_dirs for part in file_path.parts)
    
    def load_file(self, file_path: str) -> Dict:
        """Load single file and extract text"""
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            # Text-based files
            if extension in {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', 
                           '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php',
                           '.html', '.css', '.scss', '.json', '.yaml', '.yml',
                           '.md', '.txt', '.sql', '.sh', '.bash'}:
                content = self._load_text_file(file_path)
            
            # PDF files
            elif extension == '.pdf':
                content = self._load_pdf(file_path)
            
            # Word documents
            elif extension in {'.docx', '.doc'}:
                content = self._load_docx(file_path)
            
            else:
                return None
            
            if not content or len(content.strip()) == 0:
                return None
            
            return {
                'content': content,
                'metadata': {
                    'source': file_path,
                    'filename': path.name,
                    'extension': extension,
                    'size': os.path.getsize(file_path),
                    'type': self._get_file_type(extension)
                }
            }
        
        except Exception as e:
            console.print(f"[yellow]⚠️  Error loading {file_path}: {str(e)}[/yellow]")
            return None
    
    def _load_text_file(self, file_path: str) -> str:
        """Load text-based file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _load_pdf(self, file_path: str) -> str:
        """Load PDF file"""
        text = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text.append(f"[Page {page_num + 1}]\n{page_text}")
        return "\n\n".join(text)
    
    def _load_docx(self, file_path: str) -> str:
        """Load Word document"""
        doc = docx.Document(file_path)
        return "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _get_file_type(self, extension: str) -> str:
        """Categorize file type"""
        code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', 
                          '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php'}
        markup_extensions = {'.html', '.css', '.scss', '.xml'}
        data_extensions = {'.json', '.yaml', '.yml', '.sql'}
        doc_extensions = {'.md', '.txt', '.pdf', '.docx', '.doc'}
        
        if extension in code_extensions:
            return 'code'
        elif extension in markup_extensions:
            return 'markup'
        elif extension in data_extensions:
            return 'data'
        elif extension in doc_extensions:
            return 'documentation'
        else:
            return 'other'
