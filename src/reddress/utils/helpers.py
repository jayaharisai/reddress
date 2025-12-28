import os
import toml
from click import get_app_dir
import openai
from openai import OpenAI
from typing import Tuple
import json
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table


CONFIG_PATH = os.path.join(get_app_dir("reddress"), "config.toml")
USAGE_LOG_PATH = os.path.join(get_app_dir("reddress"), "usage_log.json")

def validate_openai_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate if an OpenAI API key is valid.
    
    Args:
        api_key: The OpenAI API key to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not api_key or not api_key.startswith('sk-'):
        return False, "Invalid API key format. OpenAI keys start with 'sk-'"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        # Make a lightweight API call to check validity
        client.models.list()
        return True, "✅ Valid OpenAI API key"
    
    except openai.AuthenticationError:
        return False, "❌ Authentication failed. Invalid API key"
    
    except openai.PermissionDeniedError:
        return False, "❌ Permission denied. Check your API key permissions"
    
    except openai.RateLimitError:
        # Key is valid but rate limited
        return True, "✅ Valid API key (rate limit reached)"
    
    except openai.APIConnectionError:
        return False, "❌ Network error. Could not connect to OpenAI API"
    
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"


def load_config():
    """Load configuration from TOML file"""
    if not os.path.exists(CONFIG_PATH):
        return {}
    return toml.load(CONFIG_PATH)


def save_config(config: dict):
    """Save configuration to TOML file"""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        toml.dump(config, f)


def has_openai_config():
    """Check if OpenAI API key is configured"""
    config = load_config()
    return "openai" in config and "api_key" in config["openai"]


def get_openai_key():
    """Get saved OpenAI API key"""
    config = load_config()
    return config.get("openai", {}).get("api_key")


def save_openai_key(api_key: str):
    """Save OpenAI API key to config"""
    config = load_config()
    config["openai"] = {"api_key": api_key}
    save_config(config)


def get_openai_client():
    """Get initialized OpenAI client"""
    config = load_config()
    api_key = config.get("openai", {}).get("api_key")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def fetch_openai_models():
    """Fetch available GPT models from OpenAI"""
    client = get_openai_client()
    if not client:
        return []

    try:
        models = client.models.list()
    except Exception as e:
        return []

    allowed_prefixes = ("gpt-",)

    model_ids = [
        m.id for m in models.data
        if m.id.startswith(allowed_prefixes)
    ]

    return sorted(model_ids)


def get_selected_model():
    """Get the currently selected model"""
    config = load_config()
    return config.get("chat", {}).get("model")


def save_selected_model(model: str):
    """Save selected model to config"""
    config = load_config()

    if "chat" not in config:
        config["chat"] = {}

    config["chat"]["model"] = model
    save_config(config)


def log_message_usage(model: str, input_tokens: int, output_tokens: int, total_tokens: int, cost: float, user_message: str, assistant_message: str):
    """Log token usage for each message exchange."""
    
    # Load existing logs or create new list
    if os.path.exists(USAGE_LOG_PATH):
        with open(USAGE_LOG_PATH, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens
        },
        "cost": cost,
        "messages": {
            "user": user_message[:100] + "..." if len(user_message) > 100 else user_message,  # Truncate long messages
            "assistant": assistant_message[:100] + "..." if len(assistant_message) > 100 else assistant_message
        }
    }
    
    # Append and save
    logs.append(log_entry)
    
    os.makedirs(os.path.dirname(USAGE_LOG_PATH), exist_ok=True)
    with open(USAGE_LOG_PATH, 'w') as f:
        json.dump(logs, f, indent=2)


def get_usage_stats():
    """Get aggregated usage statistics."""
    if not os.path.exists(USAGE_LOG_PATH):
        return None
    
    with open(USAGE_LOG_PATH, 'r') as f:
        logs = json.load(f)
    
    if not logs:
        return None
    
    total_cost = sum(log['cost'] for log in logs)
    total_tokens = sum(log['tokens']['total'] for log in logs)
    total_messages = len(logs)
    
    # Get stats by model
    model_stats = {}
    for log in logs:
        model = log['model']
        if model not in model_stats:
            model_stats[model] = {
                "messages": 0,
                "tokens": 0,
                "cost": 0.0
            }
        model_stats[model]["messages"] += 1
        model_stats[model]["tokens"] += log['tokens']['total']
        model_stats[model]["cost"] += log['cost']
    
    return {
        "total_messages": total_messages,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "by_model": model_stats,
        "first_use": logs[0]['timestamp'],
        "last_use": logs[-1]['timestamp']
    }


def get_recent_usage(limit=10):
    """Get recent usage logs."""
    if not os.path.exists(USAGE_LOG_PATH):
        return []
    
    with open(USAGE_LOG_PATH, 'r') as f:
        logs = json.load(f)
    
    return logs[-limit:]  


def show_usage_stats(recent_count=10):
    """Display usage statistics."""
    from datetime import datetime
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    import os
    import json
    
    # Create console for this function
    console = Console()
    
    stats = get_usage_stats()
    
    if not stats:
        console.print("[yellow]No usage data available yet.[/yellow]")
        return
    
    # Overall stats
    console.print(Panel.fit(
        f"[bold cyan]📊 Overall Usage Statistics[/bold cyan]\n\n"
        f"Total Messages: [bold]{stats['total_messages']}[/bold]\n"
        f"Total Tokens: [bold]{stats['total_tokens']:,}[/bold]\n"
        f"Total Cost: [bold green]${stats['total_cost']:.4f}[/bold green]\n"
        f"First Use: {datetime.fromisoformat(stats['first_use']).strftime('%Y-%m-%d %I:%M %p')}\n"
        f"Last Use: {datetime.fromisoformat(stats['last_use']).strftime('%Y-%m-%d %I:%M %p')}",
        border_style="cyan"
    ))
    console.print()
    
    # Stats by model
    console.print("[bold cyan]📈 Usage by Model:[/bold cyan]\n")
    model_table = Table(show_header=True)
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Messages", justify="right")
    model_table.add_column("Tokens", justify="right")
    model_table.add_column("Cost", justify="right", style="green")
    
    for model, data in stats['by_model'].items():
        model_table.add_row(
            model,
            str(data['messages']),
            f"{data['tokens']:,}",
            f"${data['cost']:.4f}"
        )
    
    console.print(model_table)
    console.print()
    
    # Token usage visualization
    console.print("[bold cyan]📊 Token Usage Distribution:[/bold cyan]\n")
    
    # Color palette for different models
    colors = [
        "magenta",
        "cyan",
        "green",
        "yellow",
        "blue",
        "red",
        "bright_magenta",
        "bright_cyan",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_red"
    ]
    
    # Calculate max tokens for scaling
    max_tokens = max(data['tokens'] for data in stats['by_model'].values())
    bar_width = 50
    
    # Assign colors to models
    for idx, (model, data) in enumerate(stats['by_model'].items()):
        tokens = data['tokens']
        bar_length = int((tokens / max_tokens) * bar_width)
        bar = "█" * bar_length
        
        # Cycle through colors
        color = colors[idx % len(colors)]
        
        percentage = (tokens / stats['total_tokens']) * 100
        console.print(
            f"[dim]{model:30}[/dim] [{color}]{bar}[/{color}] "
            f"[bold]{tokens:,}[/bold] ([dim]{percentage:.1f}%[/dim])"
        )
    
    console.print()

