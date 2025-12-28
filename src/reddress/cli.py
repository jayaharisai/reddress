import click
from .core import (
    greet_user,
    configuration,
    reddress_start,
)
from .templates.main import server
from .utils import show_usage_stats
from .utils.vector_manager import VectorStoreManager, VECTOR_STORES
from .utils.prompt_manager import (
    interactive_create_prompt,
    interactive_view_prompts,
    PromptManager
)
from .utils.rag_engine import RAGEngine
from .utils.rag_database import RAGDatabase
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich.syntax import Syntax
import os
import json
import time
import re

console = Console()

@click.group()
@click.version_option()
def main():
    """Reddress - AI Chat CLI Tool with RAG"""
    pass

# ============= EXISTING COMMANDS =============

@main.command()
def welcome():
    """Display welcome message"""
    click.echo(click.style(greet_user(), fg='green', bold=True))

@main.command()
def config():
    """Configure OpenAI API key"""
    configuration()

@main.command()
def start():
    """Start interactive chat session"""
    reddress_start()

@main.command()
def run():
    """Start web server"""
    server()

@main.command()
@click.option('--recent', '-r', default=10, help='Show recent N messages')
def stats(recent):
    """Show usage statistics"""
    show_usage_stats(recent)

# ============= HELPER FUNCTION FOR RENDERING MARKDOWN + CODE =============

def render_answer(answer: str):
    """Render markdown with proper formatting and syntax-highlighted code blocks"""
    # Split by code fences (triple backticks)
    pattern = r'(``````)'
    parts = re.split(pattern, answer)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Check if it's a code block
        if part.startswith('``````'):
            # Extract language and code
            code_content = part[3:-3]  # Remove ```
            
            # Check if first line contains language identifier
            lines = code_content.split('\n', 1)
            if len(lines) > 1 and lines.strip() and ' ' not in lines.strip():
                language = lines.strip()
                code = lines if len(lines) > 1 else ''
            else:
                language = 'python'  # Default to python
                code = code_content
            
            # Render code with syntax highlighting
            if code.strip():
                try:
                    syntax = Syntax(code, language, theme="monokai", line_numbers=False, word_wrap=True)
                    console.print(syntax)
                except:
                    # Fallback if language not recognized
                    syntax = Syntax(code, "text", theme="monokai", line_numbers=False, word_wrap=True)
                    console.print(syntax)
        else:
            # Render as markdown
            try:
                md = Markdown(part)
                console.print(md)
            except:
                # Fallback to plain text
                console.print(part)

# ============= VECTOR STORE COMMANDS =============

@main.group()
def vector():
    """Manage vector stores for RAG"""
    pass

@vector.command('list')
def vector_list():
    """List all available vector stores"""
    manager = VectorStoreManager()
    manager.list_vector_stores()

@vector.command('start')
@click.argument('store_name', required=False)
def vector_start(store_name):
    """Start a vector store"""
    manager = VectorStoreManager()
    manager.start_vector_store(store_name)

@vector.command('stop')
@click.argument('store_name')
def vector_stop(store_name):
    """Stop a vector store"""
    manager = VectorStoreManager()
    manager.stop_vector_store(store_name)

@vector.command('status')
def vector_status():
    """Show status of all vector stores"""
    manager = VectorStoreManager()
    manager.status()

@vector.command('restart')
@click.argument('store_name')
def vector_restart(store_name):
    """Restart a vector store"""
    manager = VectorStoreManager()
    manager.stop_vector_store(store_name)
    manager.start_vector_store(store_name)

# ============= RAG COMMANDS =============

@main.group()
def rag():
    """RAG (Retrieval Augmented Generation) commands"""
    pass

@rag.command('index')
@click.argument('path', type=click.Path(exists=True))
@click.option('--name', '-n', help='Project name (default: folder name)')
@click.option('--vector', '-v', help='Vector store to use (leave empty for interactive)')
@click.option('--recursive/--no-recursive', default=True, help='Index subdirectories')
@click.option('--chunk-size', default=1000, help='Chunk size for splitting')
@click.option('--force', '-f', is_flag=True, help='Force re-index all files (ignore incremental)')
def rag_index(path, name, vector, recursive, chunk_size, force):
    """Index a codebase or document folder"""
    console.print(f"\n[bold cyan]📚 Indexing: {path}[/bold cyan]\n")
    
    if not vector:
        vector = select_vector_store()
        if not vector:
            console.print("[yellow]❌ Cancelled[/yellow]")
            return
    
    engine = RAGEngine(vector_store=vector)
    
    success = engine.index_project(
        path=path,
        project_name=name,
        recursive=recursive,
        chunk_size=chunk_size,
        force=force
    )
    
    if success:
        console.print(f"\n[green]✅ Successfully indexed {path}[/green]")
        console.print(f"[green]✅ Stored in: {vector}[/green]")
        console.print(f"[dim]Use 'reddress rag chat' to start querying[/dim]\n")
    else:
        console.print(f"\n[red]❌ Failed to index {path}[/red]\n")

@rag.command('chat')
@click.option('--project', '-p', help='Project name to chat with')
@click.option('--top-k', '-k', default=10, help='Number of sources to retrieve (default: 10)')
@click.option('--prompt', help='Prompt template name to use')
@click.option('--model', '-m', default='gpt-4o-mini', help='Model to use (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)')
def rag_chat(project, top_k, prompt, model):
    """Start interactive RAG chat with markdown and code rendering
    
    Special commands during chat:
        /switch   - Switch to different vector store
        /prompt   - Select/change prompt template
        /model    - Change AI model
        /info     - Show current project info
        /sources  - Toggle showing all sources
        /help     - Show available commands
        exit      - Quit chat
    """
    db = RAGDatabase()
    projects = db.list_projects()
    
    if not projects:
        console.print("[yellow]⚠️  No projects indexed yet![/yellow]")
        console.print("[dim]Use 'reddress rag index <path>' to index a project[/dim]\n")
        return
    
    # Select project
    if not project:
        if len(projects) == 1:
            project = projects[0]['name']  # Fixed: Access first element, then get 'name'
            console.print(f"[cyan]Using project: {project}[/cyan]")
        else:
            choices = []
            for p in projects:
                vector_store = p.get('vector_store', 'unknown')
                choices.append(f"{p['name']} [{vector_store}] - {p.get('total_files', 0)} files")
                selection = questionary.select(
                    "Select project:",
                    choices=choices
                ).ask()
            
            if not selection:
                console.print("[yellow]❌ Cancelled[/yellow]")
                return
            
            project = selection.split(' [')
    
    # Get project data
    project_data = db.get_project(project)
    if not project_data:
        console.print(f"[red]❌ Project '{project}' not found[/red]")
        return
    
    current_vector_store = project_data.get('vector_store', 'memory')
    show_all_sources = False
    current_model = model
    
    # Load prompt templates
    prompt_manager = PromptManager()
    current_prompt = None
    
    if prompt:
        current_prompt = prompt_manager.get_prompt_by_name(prompt)
        if not current_prompt:
            console.print(f"[yellow]⚠️  Prompt '{prompt}' not found. Using default.[/yellow]")
    
    # Initialize RAG engine
    engine = None
    
    # Session stats
    session_stats = {
        'total_queries': 0,
        'total_cost': 0.0,
        'total_tokens': 0,
        'start_time': time.time(),
        'model_usage': {}
    }
    
    def load_engine(vector_store):
        """Helper to load engine with specific vector store"""
        nonlocal engine
        try:
            engine = RAGEngine(vector_store=vector_store)
            engine.load_project(project)
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to load with {vector_store}: {str(e)}[/red]")
            return False
    
    def get_system_prompt():
        """Get system prompt (custom or default)"""
        if current_prompt:
            return current_prompt.get('system_prompt', get_default_system_prompt())
        return get_default_system_prompt()
    
    def get_default_system_prompt():
        """Default system prompt for RAG"""
        return """You are a helpful coding assistant. Answer questions about the codebase based on the provided context.

Rules:
- Only use information from the provided context
- Be concise but complete
- Include code snippets when relevant (use ```language for code blocks)
- Use markdown formatting for better readability
- If you're unsure, say so
- Reference specific files when mentioning code"""
    
    # Initial load
    console.print(f"\n[bold cyan]💬 RAG Chat: {project}[/bold cyan]")
    console.print(f"[dim]Vector Store: {current_vector_store} | Model: {current_model} | Top-K: {top_k}[/dim]")
    if current_prompt:
        console.print(f"[dim]Prompt Template: {current_prompt['name']}[/dim]")
    console.print(f"[dim]Files: {project_data.get('total_files', 0)} | Chunks: {project_data.get('total_chunks', 0)}[/dim]\n")
    
    if not load_engine(current_vector_store):
        return
    
    console.print("[dim]Ask questions about your codebase.[/dim]")
    console.print("[dim]Special commands: /switch /model /prompt /info /sources /help exit[/dim]\n")
    
    # Chat loop
    while True:
        try:
            query = input("▶ ").strip()
            
            if not query:
                continue
            
            # Handle special commands
            if query.lower() in ['exit', 'quit', '/exit', '/quit']:
                duration = time.time() - session_stats['start_time']
                console.print(f"\n[bold cyan]📊 Session Summary[/bold cyan]")
                console.print(f"[cyan]Total Queries:[/cyan] {session_stats['total_queries']}")
                console.print(f"[cyan]Total Tokens:[/cyan] {session_stats['total_tokens']:,}")
                console.print(f"[cyan]Total Cost:[/cyan] ${session_stats['total_cost']:.4f}")
                console.print(f"[cyan]Duration:[/cyan] {int(duration // 60)}m {int(duration % 60)}s")
                
                if session_stats['model_usage']:
                    console.print(f"\n[cyan]Model Usage:[/cyan]")
                    for m, stats in session_stats['model_usage'].items():
                        console.print(f"  • {m}: {stats['queries']} queries, ${stats['cost']:.4f}")
                
                console.print(f"\n[cyan]👋 Goodbye![/cyan]\n")
                break
            
            elif query.lower() in ['/switch', '/s']:
                console.print()
                new_vector_store = select_vector_store_chat()
                if new_vector_store and new_vector_store != current_vector_store:
                    console.print(f"\n[cyan]🔄 Switching from {current_vector_store} to {new_vector_store}...[/cyan]\n")
                    if load_engine(new_vector_store):
                        current_vector_store = new_vector_store
                        console.print(f"[green]✅ Switched to {new_vector_store}[/green]\n")
                    else:
                        console.print(f"[yellow]⚠️  Staying on {current_vector_store}[/yellow]\n")
                else:
                    console.print(f"[dim]Still using {current_vector_store}[/dim]\n")
                continue
            
            elif query.lower() in ['/model', '/m']:
                console.print()
                new_model = select_model()
                if new_model:
                    current_model = new_model
                    console.print(f"[green]✅ Switched to {current_model}[/green]\n")
                else:
                    console.print(f"[dim]Still using {current_model}[/dim]\n")
                continue
            
            elif query.lower() in ['/prompt', '/p']:
                console.print()
                new_prompt = select_prompt_template()
                if new_prompt == 'cancel':
                    console.print(f"[dim]Keeping current prompt[/dim]\n")
                elif new_prompt:
                    current_prompt = new_prompt
                    console.print(f"[green]✅ Using prompt: {current_prompt['name']}[/green]")
                    console.print(f"[dim]{current_prompt.get('description', 'No description')}[/dim]\n")
                else:
                    current_prompt = None
                    console.print(f"[cyan]Using default prompt[/cyan]\n")
                continue
            
            elif query.lower() in ['/sources', '/src']:
                show_all_sources = not show_all_sources
                console.print()
                if show_all_sources:
                    console.print(f"[green]✅ Now showing all {top_k} sources[/green]\n")
                else:
                    console.print(f"[yellow]📝 Now showing top 10 sources (use /sources to toggle)[/yellow]\n")
                continue
            
            elif query.lower() in ['/info', '/i']:
                console.print()
                console.print(f"[bold cyan]📊 Current Session Info[/bold cyan]")
                console.print(f"[cyan]Project:[/cyan] {project}")
                console.print(f"[cyan]Vector Store:[/cyan] {current_vector_store}")
                console.print(f"[cyan]AI Model:[/cyan] {current_model}")
                console.print(f"[cyan]Prompt Template:[/cyan] {current_prompt['name'] if current_prompt else 'Default'}")
                console.print(f"[cyan]Top-K Sources:[/cyan] {top_k}")
                console.print(f"[cyan]Show All Sources:[/cyan] {show_all_sources}")
                console.print(f"[cyan]Files:[/cyan] {project_data.get('total_files', 0)}")
                console.print(f"[cyan]Chunks:[/cyan] {project_data.get('total_chunks', 0)}")
                console.print(f"[cyan]Session Queries:[/cyan] {session_stats['total_queries']}")
                console.print(f"[cyan]Session Cost:[/cyan] ${session_stats['total_cost']:.4f}")
                console.print(f"[cyan]Session Tokens:[/cyan] {session_stats['total_tokens']:,}")
                console.print()
                continue
            
            elif query.lower() in ['/help', '/h', '/?']:
                console.print()
                console.print("[bold cyan]💡 Available Commands[/bold cyan]")
                console.print("[cyan]/switch[/cyan] or [cyan]/s[/cyan]     - Switch to different vector store")
                console.print("[cyan]/model[/cyan] or [cyan]/m[/cyan]      - Change AI model")
                console.print("[cyan]/prompt[/cyan] or [cyan]/p[/cyan]     - Select/change prompt template")
                console.print("[cyan]/sources[/cyan] or [cyan]/src[/cyan]  - Toggle showing all sources")
                console.print("[cyan]/info[/cyan] or [cyan]/i[/cyan]       - Show current project info")
                console.print("[cyan]/help[/cyan] or [cyan]/h[/cyan]       - Show this help message")
                console.print("[cyan]exit[/cyan] or [cyan]quit[/cyan]      - Exit chat")
                console.print()
                continue
            
            # Regular query with animation
            console.print()
            
            with Live(Spinner("dots", text="[cyan]🔍 Searching relevant code...[/cyan]"), console=console, refresh_per_second=10):
                time.sleep(0.3)
                response = engine.query_with_prompt(
                    query=query, 
                    top_k=top_k,
                    system_prompt=get_system_prompt(),
                    model=current_model,
                    show_spinner=False
                )
            
            console.print("\n[bold cyan]🤖 Answer:[/bold cyan]\n")
            
            # Render answer with markdown and code formatting
            render_answer(response['answer'])
            console.print()
            
            # Update session stats
            session_stats['total_queries'] += 1
            if response.get('cost'):
                session_stats['total_cost'] += response['cost']
                
                if current_model not in session_stats['model_usage']:
                    session_stats['model_usage'][current_model] = {'queries': 0, 'cost': 0.0}
                session_stats['model_usage'][current_model]['queries'] += 1
                session_stats['model_usage'][current_model]['cost'] += response['cost']
            
            if response.get('tokens'):
                session_stats['total_tokens'] += response['tokens']
            
            # Save usage to database
            try:
                if response.get('cost') and response.get('tokens'):
                    from .utils.stats import save_usage
                    save_usage(
                        prompt=query,
                        response=response['answer'],
                        model=current_model,
                        prompt_tokens=response.get('tokens', 0) // 2,
                        completion_tokens=response.get('tokens', 0) // 2,
                        total_cost=response['cost']
                    )
            except:
                pass
            
            # Show sources
            if response.get('sources'):
                total_sources = len(response['sources'])
                display_count = total_sources if show_all_sources else min(10, total_sources)
                
                console.print(f"[bold dim]📚 Top {display_count} Sources (of {total_sources}):[/bold dim]")
                
                sources_table = Table(show_header=True, header_style="bold cyan", box=None)
                sources_table.add_column("#", style="dim", width=3)
                sources_table.add_column("File", style="cyan")
                sources_table.add_column("Score", justify="right", style="green", width=8)
                
                for i, source in enumerate(response['sources'][:display_count], 1):
                    file_path = source['file']
                    if len(file_path) > 60:
                        file_path = "..." + file_path[-57:]
                    
                    sources_table.add_row(
                        str(i),
                        file_path,
                        f"{source['score']:.3f}"
                    )
                
                console.print(sources_table)
                
                if total_sources > 10 and not show_all_sources:
                    console.print(f"[dim]Use /sources to show all {total_sources} sources[/dim]")
                
                console.print()
            
            # Show cost
            if response.get('cost'):
                prompt_info = f" | Prompt: {current_prompt['name']}" if current_prompt else ""
                console.print(f"[dim]💰 This Query: ${response['cost']:.6f} | Tokens: {response.get('tokens', 0):,} | Model: {current_model}{prompt_info}[/dim]")
                console.print(f"[dim]💳 Session Total: ${session_stats['total_cost']:.4f} | {session_stats['total_queries']} queries[/dim]\n")
        
        except KeyboardInterrupt:
            console.print("\n\n[cyan]👋 Goodbye![/cyan]\n")
            break
        except EOFError:
            console.print("\n\n[cyan]👋 Goodbye![/cyan]\n")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error: {str(e)}[/red]\n")
            import traceback
            traceback.print_exc()

@rag.command('search')
@click.argument('query')
@click.option('--project', '-p', help='Project name')
@click.option('--top-k', '-k', default=5, help='Number of results')
def rag_search(query, project, top_k):
    """Search indexed codebase"""
    db = RAGDatabase()
    
    if not project:
        projects = db.list_projects()
        if len(projects) == 1:
            project = projects[0]['name']
        else:
            console.print("[red]❌ Please specify --project[/red]")
            console.print("[dim]Use 'reddress rag projects' to see available projects[/dim]")
            return
    
    project_data = db.get_project(project)
    if not project_data:
        console.print(f"[red]❌ Project '{project}' not found[/red]")
        return
    
    vector_store = project_data.get('vector_store', 'memory')
    
    console.print(f"[dim]Searching in: {project} [{vector_store}][/dim]\n")
    
    try:
        engine = RAGEngine(vector_store=vector_store)
        engine.load_project(project)
        
        with Live(Spinner("dots", text="[cyan]🔍 Searching...[/cyan]"), console=console):
            results = engine.search(query, top_k=top_k)
        
        console.print(f"\n[bold cyan]🔍 Search Results for: '{query}'[/bold cyan]\n")
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        for i, result in enumerate(results, 1):
            console.print(f"[cyan]{i}. {result['file']}[/cyan] (Score: {result['score']:.2f})")
            console.print(f"[dim]{result['content'][:200]}...[/dim]\n")
    
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")

@rag.command('projects')
def rag_projects():
    """List all indexed projects"""
    db = RAGDatabase()
    projects = db.list_projects()
    
    if not projects:
        console.print("[yellow]📭 No projects indexed yet[/yellow]")
        return
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Path")
    table.add_column("Vector Store", style="magenta")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Indexed")
    
    for project in projects:
        vector_store = project.get('vector_store', 'unknown')
        if vector_store == 'memory':
            vector_display = "💾 Memory"
        elif vector_store == 'chromadb':
            vector_display = "🔵 ChromaDB"
        elif vector_store == 'qdrant':
            vector_display = "🟣 Qdrant"
        elif vector_store == 'milvus':
            vector_display = "🟢 Milvus"
        else:
            vector_display = vector_store
        
        table.add_row(
            project['name'],
            project['path'][:30] + "..." if len(project['path']) > 30 else project['path'],
            vector_display,
            str(project.get('total_files', 0)),
            str(project.get('total_chunks', 0)),
            project.get('indexed_at', 'N/A')[:10]
        )
    
    console.print(Panel(table, title="[bold]📚 Indexed Projects[/bold]", border_style="cyan"))

@rag.command('delete')
@click.argument('project_name')
def rag_delete(project_name):
    """Delete an indexed project"""
    db = RAGDatabase()
    
    confirm = questionary.confirm(
        f"Delete project '{project_name}'?",
        default=False
    ).ask()
    
    if not confirm:
        console.print("[yellow]❌ Cancelled[/yellow]")
        return
    
    project_data = db.get_project(project_name)
    
    if db.delete_project(project_name):
        console.print(f"[green]✅ Deleted project '{project_name}'[/green]")
        
        if project_data:
            try:
                vector_store = project_data.get('vector_store', 'memory')
                engine = RAGEngine(vector_store=vector_store)
                engine.delete_collection(project_name)
            except:
                pass
    else:
        console.print(f"[red]❌ Project '{project_name}' not found[/red]")

@rag.command('update')
@click.argument('project_name')
def rag_update(project_name):
    """Update/re-index a project"""
    db = RAGDatabase()
    project_data = db.get_project(project_name)
    
    if not project_data:
        console.print(f"[red]❌ Project '{project_name}' not found[/red]")
        return
    
    console.print(f"[cyan]🔄 Re-indexing {project_name}...[/cyan]\n")
    
    vector_store = project_data.get('vector_store', 'memory')
    engine = RAGEngine(vector_store=vector_store)
    success = engine.index_project(
        path=project_data['path'],
        project_name=project_name,
        recursive=project_data.get('recursive', True)
    )
    
    if success:
        console.print(f"[green]✅ Updated {project_name}[/green]")
    else:
        console.print(f"[red]❌ Failed to update {project_name}[/red]")

@rag.command('info')
@click.argument('project_name')
def rag_info(project_name):
    """Show detailed project information"""
    db = RAGDatabase()
    project = db.get_project(project_name)
    
    if not project:
        console.print(f"[red]❌ Project '{project_name}' not found[/red]")
        return
    
    vector_store = project.get('vector_store', 'unknown')
    if vector_store == 'memory':
        vector_display = "💾 In-Memory (no Docker needed)"
    elif vector_store == 'chromadb':
        vector_display = "🔵 ChromaDB (localhost:8000)"
    elif vector_store == 'qdrant':
        vector_display = "🟣 Qdrant (localhost:6333)"
    elif vector_store == 'milvus':
        vector_display = "🟢 Milvus (localhost:19530)"
    else:
        vector_display = vector_store
    
    info_text = f"""
[bold]Name:[/bold] {project['name']}
[bold]Path:[/bold] {project['path']}
[bold]Vector Store:[/bold] {vector_display}
[bold]Total Files:[/bold] {project.get('total_files', 0):,}
[bold]Total Chunks:[/bold] {project.get('total_chunks', 0):,}
[bold]Embedding Model:[/bold] {project.get('embedding_model', 'N/A')}
[bold]Chunk Size:[/bold] {project.get('chunk_size', 'N/A')}
[bold]Embedding Cost:[/bold] ${project.get('embedding_cost', 0):.4f}
[bold]Indexed At:[/bold] {project.get('indexed_at', 'N/A')}
"""
    
    console.print(Panel(info_text, title=f"[bold]📊 Project Info: {project_name}[/bold]", border_style="cyan"))

# ============= PROMPT COMMANDS =============

@main.group()
def prompt():
    """Manage prompt templates"""
    pass

@prompt.command('create')
def prompt_create():
    """Create a new prompt template"""
    interactive_create_prompt()

@prompt.command('list')
def prompt_list():
    """View all prompt templates"""
    manager = PromptManager()
    manager.display_prompts_table()

@prompt.command('view')
def prompt_view():
    """View and manage prompts"""
    interactive_view_prompts()

@prompt.command('delete')
@click.argument('prompt_name')
def prompt_delete(prompt_name):
    """Delete a prompt by name"""
    manager = PromptManager()
    prompt = manager.get_prompt_by_name(prompt_name)
    if prompt:
        confirm = questionary.confirm(
            f"Delete '{prompt_name}'?",
            default=False
        ).ask()
        if confirm:
            manager.delete_prompt(prompt['id'])
    else:
        click.echo(f"❌ Prompt '{prompt_name}' not found")

# ============= HELPER FUNCTIONS =============

def select_vector_store() -> str:
    """Interactive vector store selection"""
    manager = VectorStoreManager()
    
    console.print("\n[bold cyan]📦 Select Vector Store[/bold cyan]\n")
    
    choices = []
    
    choices.append({
        'name': "💾 In-Memory [🟢 Always Available] - No Docker needed, works offline",
        'value': 'memory'
    })
    
    for key, config in VECTOR_STORES.items():
        is_running = manager._check_status(key)
        status = "🟢 Running" if is_running else "⚪ Offline"
        
        choice_text = f"{config['name']} [{status}] - {config['description']}"
        choices.append({
            'name': choice_text,
            'value': key
        })
    
    result = questionary.select(
        "Choose where to store embeddings:",
        choices=[choice['name'] for choice in choices]
    ).ask()
    
    if not result:
        return None
    
    selected_value = None
    for choice in choices:
        if choice['name'] == result:
            selected_value = choice['value']
            break
    
    if selected_value != 'memory' and not manager._check_status(selected_value):
        should_start = questionary.confirm(
            f"\n{VECTOR_STORES[selected_value]['name']} is not running. Start it now?",
            default=True
        ).ask()
        
        if should_start:
            console.print()
            manager.start_vector_store(selected_value)
            console.print()
        else:
            console.print("\n[yellow]⚠️  Using in-memory storage instead[/yellow]\n")
            return 'memory'
    
    return selected_value

def select_model():
    """Interactive model selection"""
    console.print("[bold cyan]🤖 Select AI Model[/bold cyan]\n")
    
    models = [
        {
            'name': '🚀 GPT-4o - Most capable, multimodal ($2.50/$10.00 per 1M tokens)',
            'value': 'gpt-4o'
        },
        {
            'name': '⚡ GPT-4o-mini - Fast and affordable ($0.15/$0.60 per 1M tokens) [Recommended]',
            'value': 'gpt-4o-mini'
        },
        {
            'name': '💨 GPT-3.5-turbo - Legacy, fast ($0.50/$1.50 per 1M tokens)',
            'value': 'gpt-3.5-turbo'
        },
        {
            'name': '❌ Cancel (keep current)',
            'value': None
        }
    ]
    
    result = questionary.select(
        "Choose AI model:",
        choices=[m['name'] for m in models]
    ).ask()
    
    if not result:
        return None
    
    for model in models:
        if model['name'] == result:
            return model['value']
    
    return None

def select_prompt_template():
    """Interactive prompt template selection"""
    prompt_manager = PromptManager()
    
    try:
        prompts = prompt_manager.prompts
    except AttributeError:
        try:
            prompts = prompt_manager.load_prompts()
        except AttributeError:
            from click import get_app_dir
            prompt_file = os.path.join(get_app_dir("reddress"), "prompts.json")
            try:
                with open(prompt_file, 'r') as f:
                    data = json.load(f)
                    prompts = data.get('prompts', [])
            except:
                prompts = []
    
    console.print("[bold cyan]📝 Select Prompt Template[/bold cyan]\n")
    
    if not prompts:
        console.print("[yellow]⚠️  No prompt templates found[/yellow]")
        console.print("[dim]Create one with: reddress prompt create[/dim]\n")
        return None
    
    choices = []
    
    choices.append({
        'name': "🔵 Default - Standard RAG assistant",
        'value': None
    })
    
    for prompt in prompts:
        choice_text = f"📋 {prompt['name']}"
        if prompt.get('description'):
            choice_text += f" - {prompt['description'][:50]}"
        
        choices.append({
            'name': choice_text,
            'value': prompt
        })
    
    choices.append({
        'name': "❌ Cancel (keep current)",
        'value': 'cancel'
    })
    
    result = questionary.select(
        "Choose prompt template:",
        choices=[choice['name'] for choice in choices]
    ).ask()
    
    if not result:
        return 'cancel'
    
    for choice in choices:
        if choice['name'] == result:
            if choice['value'] == 'cancel':
                return 'cancel'
            return choice['value']
    
    return None

def select_vector_store_chat() -> str:
    """Interactive vector store selection during chat"""
    manager = VectorStoreManager()
    
    console.print("[bold cyan]🔄 Switch Vector Store[/bold cyan]\n")
    
    choices = []
    
    choices.append({
        'name': "💾 In-Memory [🟢 Always Available] - No Docker, fast switching",
        'value': 'memory'
    })
    
    for key, config in VECTOR_STORES.items():
        is_running = manager._check_status(key)
        status = "🟢 Running" if is_running else "⚪ Offline"
        
        choice_text = f"{config['name']} [{status}] - {config['description']}"
        choices.append({
            'name': choice_text,
            'value': key
        })
    
    choices.append({
        'name': "❌ Cancel (keep current)",
        'value': None
    })
    
    result = questionary.select(
        "Select vector store:",
        choices=[choice['name'] for choice in choices]
    ).ask()
    
    if not result:
        return None
    
    selected_value = None
    for choice in choices:
        if choice['name'] == result:
            selected_value = choice['value']
            break
    
    if selected_value is None:
        return None
    
    if selected_value != 'memory' and not manager._check_status(selected_value):
        should_start = questionary.confirm(
            f"\n{VECTOR_STORES[selected_value]['name']} is not running. Start it now?",
            default=True
        ).ask()
        
        if should_start:
            console.print()
            manager.start_vector_store(selected_value)
            console.print()
        else:
            console.print("\n[yellow]⚠️  Cannot switch to offline vector store[/yellow]")
            return None
    
    return selected_value

@main.command('examples')
def examples():
    """Show usage examples"""
    examples_text = """
[bold cyan]🎯 Reddress Usage Examples[/bold cyan]

[bold]1. Setup & Configuration[/bold]
   reddress config                          # Configure OpenAI API key
   reddress vector start chromadb           # Start vector database

[bold]2. Index Your Codebase[/bold]
   reddress rag index ./my-project          # Interactive vector store selection
   reddress rag index ./docs --name "Docs"  # Index with custom name
   reddress rag index ./src --vector memory # Use in-memory (no Docker)
   reddress rag index ./src --force         # Force re-index all files

[bold]3. Query Your Code[/bold]
   reddress rag chat                        # Interactive chat
   reddress rag chat --model gpt-4o         # Use specific model
   reddress rag chat --prompt "Code Reviewer"  # Use custom prompt
   reddress rag search "auth logic"         # Quick search
   reddress rag chat --project backend      # Chat with specific project

[bold]4. Manage Projects[/bold]
   reddress rag projects                    # List all projects
   reddress rag info my-project             # Show project details
   reddress rag update my-project           # Re-index project
   reddress rag delete old-project          # Delete project

[bold]5. Prompts[/bold]
   reddress prompt create                   # Create prompt template
   reddress prompt list                     # View all prompts
   reddress start                           # Chat with prompts

[bold]6. Vector Stores[/bold]
   reddress vector list                     # See all vector stores
   reddress vector status                   # Check running stores
   reddress vector stop chromadb            # Stop vector store

[bold]7. Statistics[/bold]
   reddress stats                           # View usage statistics
   reddress stats --recent 20               # Show recent 20 queries

[bold]8. During Chat Commands[/bold]
   /switch  - Switch vector store
   /model   - Change AI model
   /prompt  - Change prompt template
   /sources - Toggle showing all sources
   /info    - Show session info
   /help    - Show help
   exit     - Quit chat
"""
    console.print(examples_text)

if __name__ == "__main__":
    main()
