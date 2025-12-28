import os
import subprocess
import docker
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional, Dict

console = Console()

# Vector store configurations [web:82][web:84][web:86]
VECTOR_STORES = {
    "chromadb": {
        "name": "ChromaDB",
        "image": "chromadb/chroma:latest",
        "port": 8000,
        "description": "Fast, Python-native, Great for development",
        "performance": "⭐⭐⭐",
        "best_for": "Development & Prototyping",
        "docker_compose": """version: '3.8'
services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: reddress-chromadb
    ports:
      - "8000:8000"
    volumes:
      - chromadb-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    restart: unless-stopped

volumes:
  chromadb-data:
""",
        "health_check": "http://localhost:8000/api/v1/heartbeat",
        "docs": "https://docs.trychroma.com/"
    },
    
    "qdrant": {
        "name": "Qdrant",
        "image": "qdrant/qdrant:latest",
        "port": 6333,
        "description": "Best performance, Production-ready, Rust-based",
        "performance": "⭐⭐⭐⭐⭐",
        "best_for": "Production & High Performance",
        "docker_compose": """version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: reddress-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant-storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

volumes:
  qdrant-storage:
""",
        "health_check": "http://localhost:6333/",
        "docs": "https://qdrant.tech/documentation/"
    },
    
    "milvus": {
        "name": "Milvus",
        "image": "milvusdb/milvus:latest",
        "port": 19530,
        "description": "Billion-scale, GPU support, Cloud-native",
        "performance": "⭐⭐⭐⭐",
        "best_for": "Large Scale & GPU Acceleration",
        "docker_compose": """version: '3.8'
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd-data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    restart: unless-stopped

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio-data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    restart: unless-stopped

  milvus:
    container_name: reddress-milvus
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus-data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
    restart: unless-stopped

volumes:
  etcd-data:
  minio-data:
  milvus-data:
""",
        "health_check": "http://localhost:9091/healthz",
        "docs": "https://milvus.io/docs"
    },
    
    "weaviate": {
        "name": "Weaviate",
        "image": "semitechnologies/weaviate:latest",
        "port": 8080,
        "description": "GraphQL API, Multi-modal, Enterprise features",
        "performance": "⭐⭐⭐⭐",
        "best_for": "Enterprise & Multi-modal Search",
        "docker_compose": """version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    container_name: reddress-weaviate
    ports:
      - "8080:8080"
    volumes:
      - weaviate-data:/var/lib/weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'node1'
    restart: unless-stopped

volumes:
  weaviate-data:
""",
        "health_check": "http://localhost:8080/v1/.well-known/ready",
        "docs": "https://weaviate.io/developers/weaviate"
    },
    
    "pgvector": {
        "name": "pgvector",
        "image": "pgvector/pgvector:pg17",
        "port": 5432,
        "description": "PostgreSQL extension, SQL-based, Familiar",
        "performance": "⭐⭐⭐",
        "best_for": "Existing PostgreSQL Users",
        "docker_compose": """version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg17
    container_name: reddress-pgvector
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: vectordb
    ports:
      - "5432:5432"
    volumes:
      - pgvector-data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  pgvector-data:
""",
        "health_check": "pg_isready",
        "docs": "https://github.com/pgvector/pgvector"
    }
}


class VectorStoreManager:
    """Manage vector stores with Docker"""
    
    def __init__(self):
        self.base_dir = os.path.join(os.path.expanduser("~"), ".reddress", "vector-stores")
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Check if Docker is available
        self.docker_available = self._check_docker()
        
        if not self.docker_available:
            console.print("[yellow]⚠️  Docker is not running. Please start Docker Desktop.[/yellow]")
            console.print("[dim]Tip: Open Docker Desktop app and wait for it to start[/dim]\n")
    
    def _check_docker(self) -> bool:
        """Check if Docker is running"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _ensure_docker(self):
        """Ensure Docker is available before operations"""
        if not self.docker_available:
            console.print("[red]❌ Docker is not running![/red]")
            console.print("[yellow]Please start Docker Desktop and try again.[/yellow]\n")
            console.print("Steps:")
            console.print("1. Open Docker Desktop application")
            console.print("2. Wait for 'Docker is running' status")
            console.print("3. Run your command again\n")
            raise SystemExit(1)
    
    def list_vector_stores(self):
        """Display all available vector stores"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Performance")
        table.add_column("Best For")
        
        for key, config in VECTOR_STORES.items():
            # Only check status if Docker is available
            if self.docker_available:
                status = self._check_status(key)
                status_icon = "🟢 Up" if status else "⚪ Off"
            else:
                status_icon = "❓ Unknown"
            
            table.add_row(
                config["name"],
                status_icon,
                config["performance"],
                config["best_for"]
            )
        
        console.print(Panel(table, title="[bold]📦 Available Vector Stores[/bold]", border_style="cyan"))
        
        if not self.docker_available:
            console.print("\n[yellow]💡 Start Docker Desktop to manage vector stores[/yellow]")
    
    def _check_status(self, store_name: str) -> bool:
        """Check if vector store is running"""
        try:
            container_name = f"reddress-{store_name}"
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "Up" in result.stdout
        except:
            return False
    
    def start_vector_store(self, store_name: Optional[str] = None):
        """Start a vector store with Docker"""
        self._ensure_docker()  # Check Docker before proceeding
        
        if not store_name:
            # Interactive selection
            import questionary
            choices = [
                f"{config['name']} - {config['description']}"
                for config in VECTOR_STORES.values()
            ]
            
            selection = questionary.select(
                "Select vector store:",
                choices=choices
            ).ask()
            
            if not selection:
                console.print("[yellow]❌ Cancelled[/yellow]")
                return
            
            # Extract store name
            store_name = list(VECTOR_STORES.keys())[choices.index(selection)]
        
        if store_name not in VECTOR_STORES:
            console.print(f"[red]❌ Unknown vector store: {store_name}[/red]")
            return
        
        config = VECTOR_STORES[store_name]
        
        # Check if already running
        if self._check_status(store_name):
            console.print(f"[yellow]⚠️  {config['name']} is already running[/yellow]")
            return
        
        console.print(f"\n🚀 Starting {config['name']}...\n")
        
        # Create docker-compose file
        store_dir = os.path.join(self.base_dir, store_name)
        os.makedirs(store_dir, exist_ok=True)
        
        compose_file = os.path.join(store_dir, "docker-compose.yml")
        with open(compose_file, 'w') as f:
            f.write(config["docker_compose"])
        
        # Start with docker compose
        try:
            with console.status(f"[bold cyan]Pulling {config['image']}..."):
                subprocess.run(
                    ["docker", "compose", "up", "-d"],
                    cwd=store_dir,
                    check=True,
                    capture_output=True
                )
            
            console.print(f"✅ {config['name']} started successfully!")
            console.print(f"🌐 Access at: [cyan]http://localhost:{config['port']}[/cyan]")
            console.print(f"📖 Documentation: [cyan]{config['docs']}[/cyan]\n")
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Failed to start {config['name']}[/red]")
            console.print(f"[red]{e.stderr.decode()}[/red]")
    
    def stop_vector_store(self, store_name: str):
        """Stop a running vector store"""
        self._ensure_docker()
        
        if store_name not in VECTOR_STORES:
            console.print(f"[red]❌ Unknown vector store: {store_name}[/red]")
            return
        
        config = VECTOR_STORES[store_name]
        store_dir = os.path.join(self.base_dir, store_name)
        
        if not os.path.exists(store_dir):
            console.print(f"[yellow]⚠️  {config['name']} is not running[/yellow]")
            return
        
        console.print(f"🛑 Stopping {config['name']}...")
        
        try:
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=store_dir,
                check=True,
                capture_output=True
            )
            console.print(f"✅ {config['name']} stopped\n")
        except subprocess.CalledProcessError:
            console.print(f"[red]❌ Failed to stop {config['name']}[/red]")
    
    def status(self):
        """Show status of all vector stores"""
        if not self.docker_available:
            console.print("[yellow]⚠️  Docker is not running[/yellow]")
            console.print("[dim]Start Docker Desktop to see vector store status[/dim]\n")
            return
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Status", justify="center")
        table.add_column("Port")
        table.add_column("Container")
        
        for key, config in VECTOR_STORES.items():
            is_running = self._check_status(key)
            status_icon = "🟢 Up" if is_running else "⚪ Off"
            port = str(config["port"]) if is_running else "-"
            container = f"reddress-{key}" if is_running else "-"
            
            table.add_row(
                config["name"],
                status_icon,
                port,
                container
            )
        
        console.print(Panel(table, title="[bold]📊 Vector Store Status[/bold]", border_style="cyan"))