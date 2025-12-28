import json
import os
from datetime import datetime
from click import get_app_dir
from rich.console import Console
from rich.table import Table

console = Console()


def get_usage_file():
    """Get path to usage stats file"""
    app_dir = get_app_dir("reddress")
    os.makedirs(app_dir, exist_ok=True)
    return os.path.join(app_dir, "usage_stats.json")


def save_usage(prompt, response, model, prompt_tokens, completion_tokens, total_cost):
    """Save usage statistics"""
    usage_file = get_usage_file()
    
    # Load existing data
    if os.path.exists(usage_file):
        with open(usage_file, 'r') as f:
            try:
                data = json.load(f)
            except:
                data = {"conversations": []}
    else:
        data = {"conversations": []}
    
    # Add new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt[:200],  # Truncate long prompts
        "response": response[:200],  # Truncate long responses
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost": total_cost
    }
    
    data["conversations"].append(entry)
    
    # Save
    with open(usage_file, 'w') as f:
        json.dump(data, f, indent=2)


def get_usage_data():
    """Get all usage data"""
    usage_file = get_usage_file()
    
    if os.path.exists(usage_file):
        with open(usage_file, 'r') as f:
            try:
                return json.load(f)
            except:
                return {"conversations": []}
    return {"conversations": []}


def show_usage_stats(recent=10):
    """Display usage statistics"""
    data = get_usage_data()
    conversations = data.get("conversations", [])
    
    if not conversations:
        console.print("[yellow]📭 No usage data yet[/yellow]")
        return
    
    # Calculate totals
    total_conversations = len(conversations)
    total_tokens = sum(c.get("total_tokens", 0) for c in conversations)
    total_cost = sum(c.get("cost", 0) for c in conversations)
    
    # Show summary
    console.print(f"\n[bold cyan]📊 Usage Statistics[/bold cyan]\n")
    console.print(f"[cyan]Total Conversations:[/cyan] {total_conversations:,}")
    console.print(f"[cyan]Total Tokens:[/cyan] {total_tokens:,}")
    console.print(f"[cyan]Total Cost:[/cyan] ${total_cost:.4f}\n")
    
    # Show recent conversations
    recent_conversations = conversations[-recent:]
    
    if recent_conversations:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Date", style="dim")
        table.add_column("Prompt", style="cyan", max_width=40)
        table.add_column("Model", style="magenta")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right", style="green")
        
        for conv in reversed(recent_conversations):
            timestamp = conv.get("timestamp", "")
            date = timestamp[:10] if timestamp else "N/A"
            prompt = conv.get("prompt", "")[:37] + "..." if len(conv.get("prompt", "")) > 40 else conv.get("prompt", "")
            model = conv.get("model", "unknown")
            tokens = conv.get("total_tokens", 0)
            cost = conv.get("cost", 0)
            
            table.add_row(
                date,
                prompt,
                model,
                f"{tokens:,}",
                f"${cost:.6f}"
            )
        
        console.print(table)
        console.print()


def clear_usage_stats():
    """Clear all usage statistics"""
    usage_file = get_usage_file()
    if os.path.exists(usage_file):
        os.remove(usage_file)
        console.print("[green]✅ Usage statistics cleared[/green]")
    else:
        console.print("[yellow]No usage data to clear[/yellow]")
