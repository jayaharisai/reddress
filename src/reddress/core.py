import click
import questionary
from .utils import (
    validate_openai_key,
    load_config,
    save_config,
    has_openai_config,
    get_openai_key,
    save_openai_key,
    get_openai_client,
    fetch_openai_models,
    save_selected_model,
    get_selected_model,
    get_recent_usage,
    get_usage_stats,
    log_message_usage,
    show_usage_stats
)

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from datetime import datetime
import sys
from typing import Optional, Dict

# Initialize Rich console
console = Console()

# OpenAI pricing per 1M tokens (as of Dec 2024)
PRICING = {
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
    "gpt-5": {"input": 5.00, "output": 15.00},  # Estimated pricing
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 3.00, "output": 12.00},
}

# Models that don't support temperature and other parameters
REASONING_MODELS = ["gpt-5", "o1", "o1-mini", "o1-preview", "o3-mini", "o3"]

def is_reasoning_model(model_name: str) -> bool:
    """Check if model is a reasoning model that doesn't support temperature"""
    model_lower = model_name.lower()
    return any(reasoning_model in model_lower for reasoning_model in REASONING_MODELS)

def apply_template_variables(template: str, variables: Dict[str, str]) -> str:
    """
    Apply variable substitution to a template string [web:35][web:36][web:38]
    Supports {{variable}} and {variable} syntax
    """
    result = template
    for key, value in variables.items():
        # Support both {{key}} and {key} syntax
        result = result.replace(f"{{{{{key}}}}}", str(value))
        result = result.replace(f"{{{key}}}", str(value))
    return result

def get_model_pricing(model_name):
    """Get pricing for a model, with fallback to base model name."""
    # Try exact match first
    if model_name in PRICING:
        return PRICING[model_name]
    
    # Try matching base model (e.g., "gpt-4-turbo-2024-04-09" -> "gpt-4-turbo")
    for key in PRICING.keys():
        if model_name.startswith(key):
            return PRICING[key]
    
    # Default fallback
    return {"input": 0.50, "output": 1.50}

def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculate cost based on token usage."""
    pricing = get_model_pricing(model_name)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

def greet_user():
    message = """
👋 Welcome to Reddress!

Everything is set up and ready to go.
Use `reddress --help` to see available commands.
"""
    return message

def configuration():
    click.echo(click.style("\n🔐 Reddress Configuration\n", bold=True))

    provider = questionary.select(
        "Select a provider",
        choices=["openai"]
    ).ask()

    if provider is None:
        click.echo("❌ Cancelled")
        return

    if provider == "openai":
        if has_openai_config():
            existing_key = get_openai_key()
            masked = existing_key[:4] + "****" + existing_key[-2:]

            click.echo(f"\n🔎 OpenAI is already configured: {masked}")

            change = questionary.confirm(
                "Do you want to change the OpenAI API key?",
                default=False
            ).ask()

            if not change:
                click.echo("✅ Keeping existing OpenAI key")
                return

        api_key = questionary.password(
            "Enter your OpenAI API key"
        ).ask()

        confirm_key = questionary.password(
            "Re-enter your OpenAI API key"
        ).ask()

        if api_key != confirm_key:
            click.echo(click.style("❌ Keys do not match", fg="red"))
            return

        is_valid, message = validate_openai_key(api_key)
        if not is_valid:
            click.echo(click.style(f"❌ {message}", fg="red"))
            return

        save_openai_key(api_key)
        click.echo(click.style("✅ OpenAI API key saved successfully!", fg="green"))

def reddress_start():
    if not has_openai_config():
        click.echo("❌ OpenAI is not configured. Run `reddress config` first.")
        return

    chat_history = []
    
    # Fancy header
    console.print(Panel.fit(
        "[bold cyan]💬 Reddress Chat[/bold cyan]\n\n"
        "[yellow]/model[/yellow] → Select model | "
        "[yellow]/prompt[/yellow] → Select prompt | "
        "[yellow]/new[/yellow] → New chat | "
        "[yellow]/exit[/yellow] → Exit",
        border_style="cyan"
    ))
    console.print()
    
    # Check if model is already selected
    current_model = get_selected_model()
    
    # If no model selected, prompt user to select one
    if not current_model:
        console.print("⚠️  [yellow]No model selected yet.[/yellow]\n")
        
        with console.status("[bold cyan]🔄 Fetching available models...", spinner="dots"):
            models = fetch_openai_models()
        
        if not models:
            click.echo("❌ Failed to fetch models. Check your API key.")
            return
        
        result = questionary.select(
            "Select OpenAI Model:",
            choices=models
        ).ask()
        
        if not result:
            click.echo("❌ No model selected. Exiting.")
            return
        
        save_selected_model(result)
        current_model = result
        console.print(f"\n✅ Model set to: [bold green]{current_model}[/bold green]")
        
        # Show warning for reasoning models
        if is_reasoning_model(current_model):
            console.print(f"[yellow]ℹ️  Note: {current_model} is a reasoning model with fixed parameters[/yellow]\n")
    else:
        console.print(f"📊 Current model: [bold green]{current_model}[/bold green]")
        if is_reasoning_model(current_model):
            console.print(f"[dim]ℹ️  Reasoning model - using optimized settings[/dim]\n")
        else:
            console.print()
    
    # Prompt selection
    selected_prompt = None
    try:
        from .utils.prompt_manager import select_prompt_for_chat
        
        use_prompt = questionary.confirm(
            "Use a prompt template?",
            default=False
        ).ask()
        
        if use_prompt:
            selected_prompt = select_prompt_for_chat()
            if selected_prompt:
                console.print(f"\n✅ Using prompt: [bold green]{selected_prompt['name']}[/bold green]")
                console.print(f"[dim]{selected_prompt.get('description', '')}[/dim]")
                
                # Show template info if user prompt template exists
                if selected_prompt.get('user_prompt_template'):
                    console.print(f"[dim]💡 User input will be formatted with template[/dim]\n")
                else:
                    console.print()
                
                # Add system prompt to chat history
                if selected_prompt.get('system_prompt'):
                    chat_history.append({
                        "role": "system",
                        "content": selected_prompt['system_prompt']
                    })
    except ImportError:
        # Prompt manager not available, skip
        pass

    # Get OpenAI client
    client = get_openai_client()
    if not client:
        click.echo("❌ Failed to initialize OpenAI client")
        return

    # Main chat loop
    while True:
        try:
            user_input = input("▶ ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input == "/model":
                with console.status("[bold cyan]🔄 Fetching available models...", spinner="dots"):
                    models = fetch_openai_models()
                
                if models:
                    result = questionary.select(
                        "Select OpenAI Model:",
                        choices=models
                    ).ask()
                    
                    if result:
                        save_selected_model(result)
                        current_model = result
                        console.print(f"\n✅ Model set to: [bold green]{current_model}[/bold green]")
                        
                        if is_reasoning_model(current_model):
                            console.print(f"[yellow]ℹ️  Note: {current_model} is a reasoning model with fixed parameters[/yellow]\n")
                        else:
                            console.print()
                    else:
                        console.print("\n❌ [red]Model selection cancelled[/red]\n")
                else:
                    console.print("\n❌ [red]Failed to fetch models[/red]\n")
                continue
            
            if user_input == "/prompt":
                try:
                    from .utils.prompt_manager import select_prompt_for_chat
                    
                    new_prompt = select_prompt_for_chat()
                    if new_prompt:
                        selected_prompt = new_prompt
                        # Clear chat history and add new system prompt
                        chat_history.clear()
                        if selected_prompt.get('system_prompt'):
                            chat_history.append({
                                "role": "system",
                                "content": selected_prompt['system_prompt']
                            })
                        console.print(f"\n✅ Prompt changed to: [bold green]{selected_prompt['name']}[/bold green]\n")
                    else:
                        console.print("\n❌ [red]Prompt selection cancelled[/red]\n")
                except ImportError:
                    console.print("\n❌ [red]Prompt manager not available[/red]\n")
                continue
            
            if user_input == "/new":
                chat_history.clear()
                # Re-add system prompt if using a prompt template
                if selected_prompt and selected_prompt.get('system_prompt'):
                    chat_history.append({
                        "role": "system",
                        "content": selected_prompt['system_prompt']
                    })
                console.print("\n🆕 [green]New chat started[/green]\n")
                continue
            
            if user_input == "/exit":
                console.print("\n👋 [bold cyan]Goodbye![/bold cyan]")
                break

            # Apply user prompt template if exists [web:35][web:36][web:38]
            processed_input = user_input
            if selected_prompt and selected_prompt.get('user_prompt_template'):
                template = selected_prompt['user_prompt_template']
                # Apply variable substitution [web:35][web:36]
                processed_input = apply_template_variables(template, {
                    'input': user_input,
                    'query': user_input,
                    'question': user_input,
                    'user_input': user_input,
                    'message': user_input
                })
                
                # If no substitution happened (no placeholders), append user input
                if processed_input == template:
                    processed_input = f"{template}\n\nUser: {user_input}"

            # Normal message - send to OpenAI
            console.print()
            chat_history.append({"role": "user", "content": processed_input})
            assistant_response = send_chat_message(
                client, 
                current_model, 
                chat_history,
                prompt_config=selected_prompt
            )
            
            if assistant_response:
                chat_history.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            console.print("\n\n👋 [bold cyan]Goodbye![/bold cyan]")
            break
        except EOFError:
            console.print("\n\n👋 [bold cyan]Goodbye![/bold cyan]")
            break

def send_chat_message(client, model: str, messages: list, prompt_config: Optional[Dict] = None):
    """Send message to OpenAI and stream the response with progress indicator."""
    try:
        start_time = datetime.now()
        
        # Get the user's message (last message in the list)
        user_message = messages[-1]['content']
        
        # Prepare API parameters with prompt configuration
        api_params = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        # Check if this is a reasoning model
        is_reasoning = is_reasoning_model(model)
        
        # Apply prompt config parameters if provided
        # Reasoning models don't support temperature and other parameters
        if not is_reasoning:
            if prompt_config:
                api_params["temperature"] = prompt_config.get('temperature', 0.7)
                if prompt_config.get('max_tokens'):
                    api_params["max_tokens"] = prompt_config['max_tokens']
                api_params["top_p"] = prompt_config.get('top_p', 1.0)
                api_params["frequency_penalty"] = prompt_config.get('frequency_penalty', 0.0)
                api_params["presence_penalty"] = prompt_config.get('presence_penalty', 0.0)
            else:
                api_params["temperature"] = 0.7
        else:
            # Reasoning models: only set reasoning_effort if supported
            if "o1" in model.lower() and "mini" not in model.lower():
                # Only full o1 model supports reasoning_effort
                if prompt_config and prompt_config.get('reasoning_effort'):
                    api_params["reasoning_effort"] = prompt_config['reasoning_effort']
        
        stream = client.chat.completions.create(**api_params)
        
        full_response = ""
        chunk_count = 0
        usage_data = None
        
        # Show progress while streaming
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("🤖 Generating response...", total=None)
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        chunk_count += 1
                        
                        if chunk_count % 5 == 0:  # Update every 5 chunks
                            progress.update(
                                task, 
                                description=f"🤖 Generating response... [dim]({len(full_response)} chars)[/dim]"
                            )
                
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage_data = chunk.usage
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display the formatted response
        console.print("\n🤖 [bold cyan]Assistant:[/bold cyan]\n")
        md = Markdown(full_response)
        console.print(md)
        console.print()
        
        # Create and display statistics table
        if usage_data:
            input_tokens = usage_data.prompt_tokens
            output_tokens = usage_data.completion_tokens
            total_tokens = usage_data.total_tokens
            cost = calculate_cost(model, input_tokens, output_tokens)
            
            # LOG THE USAGE HERE
            log_message_usage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=cost,
                user_message=user_message,
                assistant_message=full_response
            )
            
            stats_table = Table(show_header=False, box=None, padding=(0, 1))
            stats_table.add_column(style="dim")
            stats_table.add_column(style="cyan")
            
            stats_table.add_row("📊 Input tokens:", f"{input_tokens:,}")
            stats_table.add_row("📤 Output tokens:", f"{output_tokens:,}")
            stats_table.add_row("🔢 Total tokens:", f"[bold]{total_tokens:,}[/bold]")
            stats_table.add_row("💰 Estimated cost:", f"[green]${cost:.6f}[/green]")
            stats_table.add_row("⏱️  Response time:", f"{duration:.2f}s")
            stats_table.add_row("🕐 Timestamp:", end_time.strftime("%I:%M:%S %p"))
            
            # Add model type indicator for reasoning models
            if is_reasoning:
                stats_table.add_row("🧠 Model type:", "[yellow]Reasoning Model[/yellow]")
            
            console.print(Panel(
                stats_table,
                title="[bold]📈 Response Stats[/bold]",
                border_style="dim",
                padding=(0, 2)
            ))
            console.print()
        
        return full_response
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Error:[/bold red] {str(e)}\n")
        return None
