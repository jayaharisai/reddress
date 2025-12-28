from typing import List
from rich.console import Console
import os

console = Console()

class EmbeddingGenerator:
    """Generate embeddings using cost-effective models"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.total_tokens = 0
        self.use_local = False
        self.client = None
        
        # Try to get API key from reddress config
        api_key = self._get_api_key()
        
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                console.print("[green]✅ Using OpenAI embeddings[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  OpenAI not available: {str(e)}[/yellow]")
                self._init_local_embeddings()
        else:
            console.print("[yellow]⚠️  No OpenAI API key found, using local embeddings[/yellow]")
            self._init_local_embeddings()
    
    def _get_api_key(self) -> str:
        """Get API key from reddress config"""
        try:
            from ..utils import get_openai_key
            return get_openai_key()
        except:
            # Try environment variable
            return os.getenv('OPENAI_API_KEY')
    
    def _init_local_embeddings(self):
        """Initialize local embedding model (free, no API needed)"""
        try:
            from sentence_transformers import SentenceTransformer
            console.print("[cyan]📥 Loading local embedding model (one-time download)...[/cyan]")
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
            self.use_local = True
            console.print("[green]✅ Using local embeddings (FREE)[/green]")
        except ImportError:
            console.print("[red]❌ Please install: pip install sentence-transformers[/red]")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings in batches"""
        
        if self.use_local:
            return self._generate_local_embeddings(texts)
        else:
            return self._generate_openai_embeddings(texts, batch_size)
    
    def _generate_openai_embeddings(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            with console.status(f"[cyan]Generating embeddings {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}..."):
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
            
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            
            self.total_tokens += response.usage.total_tokens
        
        cost = self.calculate_cost(self.total_tokens)
        console.print(f"[green]✅ Generated {len(all_embeddings)} embeddings[/green]")
        console.print(f"[dim]Tokens used: {self.total_tokens:,} | Cost: ${cost:.4f}[/dim]")
        
        return all_embeddings
    
    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model (FREE)"""
        console.print(f"[cyan]🔄 Generating {len(texts)} embeddings locally (FREE)...[/cyan]")
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with console.status(f"[cyan]Processing {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}..."):
                embeddings = self.local_model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(embeddings.tolist())
        
        console.print(f"[green]✅ Generated {len(all_embeddings)} embeddings (FREE, no API cost)[/green]")
        return all_embeddings
    
    def calculate_cost(self, tokens: int) -> float:
        """Calculate embedding cost (only for OpenAI)"""
        if self.use_local:
            return 0.0
        
        if self.model == "text-embedding-3-small":
            return (tokens / 1_000_000) * 0.02
        elif self.model == "text-embedding-3-large":
            return (tokens / 1_000_000) * 0.13
        else:  # ada-002
            return (tokens / 1_000_000) * 0.10
