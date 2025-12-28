import os
import json
from datetime import datetime
from typing import Optional, List, Dict
from click import get_app_dir
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import questionary

console = Console()

PROMPTS_PATH = os.path.join(get_app_dir("reddress"), "prompts.json")

# Default OpenAI chat completion parameters [web:17][web:20]
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "max_tokens": None,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "response_format": {"type": "text"}
}


class PromptManager:
    """Manages prompt templates with full configuration [web:12][web:15]"""
    
    def __init__(self):
        self.prompts_file = PROMPTS_PATH
        self._ensure_prompts_file()
    
    def _ensure_prompts_file(self):
        """Create prompts file if it doesn't exist"""
        if not os.path.exists(self.prompts_file):
            os.makedirs(os.path.dirname(self.prompts_file), exist_ok=True)
            with open(self.prompts_file, 'w') as f:
                json.dump([], f, indent=2)
    
    def load_prompts(self) -> List[Dict]:
        """Load all prompts from file"""
        try:
            with open(self.prompts_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def save_prompts(self, prompts: List[Dict]):
        """Save prompts to file"""
        with open(self.prompts_file, 'w') as f:
            json.dump(prompts, f, indent=2)
    
    def create_prompt(self, prompt_data: Dict) -> bool:
        """Create a new prompt template [web:13][web:14]"""
        prompts = self.load_prompts()
        
        # Check for duplicate names
        if any(p['name'] == prompt_data['name'] for p in prompts):
            console.print(f"[red]❌ Prompt with name '{prompt_data['name']}' already exists[/red]")
            return False
        
        # Add metadata
        prompt_data['id'] = self._generate_id()
        prompt_data['created_at'] = datetime.now().isoformat()
        prompt_data['updated_at'] = datetime.now().isoformat()
        prompt_data['usage_count'] = 0
        
        prompts.append(prompt_data)
        self.save_prompts(prompts)
        
        console.print(f"[green]✅ Prompt '{prompt_data['name']}' created successfully![/green]")
        return True
    
    def _generate_id(self) -> str:
        """Generate unique ID for prompt"""
        prompts = self.load_prompts()
        return f"prompt_{len(prompts) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def get_prompt(self, prompt_id: str) -> Optional[Dict]:
        """Get a specific prompt by ID"""
        prompts = self.load_prompts()
        return next((p for p in prompts if p['id'] == prompt_id), None)
    
    def get_prompt_by_name(self, name: str) -> Optional[Dict]:
        """Get a specific prompt by name"""
        prompts = self.load_prompts()
        return next((p for p in prompts if p['name'] == name), None)
    
    def update_prompt(self, prompt_id: str, updates: Dict) -> bool:
        """Update an existing prompt"""
        prompts = self.load_prompts()
        
        for i, prompt in enumerate(prompts):
            if prompt['id'] == prompt_id:
                prompts[i].update(updates)
                prompts[i]['updated_at'] = datetime.now().isoformat()
                self.save_prompts(prompts)
                console.print(f"[green]✅ Prompt updated successfully![/green]")
                return True
        
        console.print(f"[red]❌ Prompt not found[/red]")
        return False
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt"""
        prompts = self.load_prompts()
        original_length = len(prompts)
        prompts = [p for p in prompts if p['id'] != prompt_id]
        
        if len(prompts) < original_length:
            self.save_prompts(prompts)
            console.print(f"[green]✅ Prompt deleted successfully![/green]")
            return True
        
        console.print(f"[red]❌ Prompt not found[/red]")
        return False
    
    def increment_usage(self, prompt_id: str):
        """Increment usage counter for a prompt"""
        prompts = self.load_prompts()
        
        for i, prompt in enumerate(prompts):
            if prompt['id'] == prompt_id:
                prompts[i]['usage_count'] = prompts[i].get('usage_count', 0) + 1
                prompts[i]['last_used'] = datetime.now().isoformat()
                self.save_prompts(prompts)
                break
    
    def list_prompts(self) -> List[Dict]:
        """List all prompts with summary"""
        return self.load_prompts()
    
    def display_prompts_table(self):
        """Display prompts in a formatted table"""
        prompts = self.load_prompts()
        
        if not prompts:
            console.print("[yellow]📭 No prompts found. Create your first prompt![/yellow]")
            return
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan", no_wrap=False)
        table.add_column("Description", style="dim", max_width=40)
        table.add_column("Category", justify="center")
        table.add_column("Usage", justify="right")
        table.add_column("Temperature", justify="center")
        table.add_column("Max Tokens", justify="center")
        
        for prompt in prompts:
            table.add_row(
                prompt['name'],
                prompt.get('description', 'N/A')[:50] + "..." if len(prompt.get('description', '')) > 50 else prompt.get('description', 'N/A'),
                prompt.get('category', 'General'),
                str(prompt.get('usage_count', 0)),
                str(prompt.get('temperature', 0.7)),
                str(prompt.get('max_tokens', 'Auto'))
            )
        
        console.print(Panel(table, title="[bold]📚 Your Prompt Library[/bold]", border_style="cyan"))
    
    def display_prompt_detail(self, prompt: Dict):
        """Display detailed view of a prompt [web:13]"""
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]{prompt['name']}[/bold cyan]\n\n"
            f"[dim]ID:[/dim] {prompt['id']}\n"
            f"[dim]Category:[/dim] {prompt.get('category', 'General')}\n"
            f"[dim]Created:[/dim] {datetime.fromisoformat(prompt['created_at']).strftime('%Y-%m-%d %I:%M %p')}\n"
            f"[dim]Usage Count:[/dim] {prompt.get('usage_count', 0)}",
            border_style="cyan",
            title="[bold]Prompt Details[/bold]"
        ))
        
        console.print("\n[bold cyan]Description:[/bold cyan]")
        console.print(prompt.get('description', 'No description provided'))
        
        console.print("\n[bold cyan]System Prompt Template:[/bold cyan]")
        console.print(Panel(prompt['system_prompt'], border_style="green", padding=(1, 2)))
        
        if prompt.get('user_prompt_template'):
            console.print("\n[bold cyan]User Prompt Template:[/bold cyan]")
            console.print(Panel(prompt['user_prompt_template'], border_style="blue", padding=(1, 2)))
        
        console.print("\n[bold cyan]Configuration:[/bold cyan]")
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column(style="dim")
        config_table.add_column(style="cyan")
        
        config_table.add_row("Temperature:", str(prompt.get('temperature', 0.7)))
        config_table.add_row("Max Tokens:", str(prompt.get('max_tokens', 'Auto')))
        config_table.add_row("Top P:", str(prompt.get('top_p', 1.0)))
        config_table.add_row("Frequency Penalty:", str(prompt.get('frequency_penalty', 0.0)))
        config_table.add_row("Presence Penalty:", str(prompt.get('presence_penalty', 0.0)))
        config_table.add_row("Response Style:", prompt.get('response_style', 'Default'))
        
        console.print(config_table)
        console.print()


def interactive_create_prompt():
    """Interactive prompt creation wizard [web:16]"""
    console.print(Panel.fit(
        "[bold cyan]✨ Create New Prompt Template[/bold cyan]\n\n"
        "Define a reusable prompt with custom configuration",
        border_style="cyan"
    ))
    console.print()
    
    # Basic Information
    name = questionary.text(
        "Prompt Name:",
        validate=lambda text: len(text) > 0 or "Name cannot be empty"
    ).ask()
    
    if not name:
        console.print("[red]❌ Cancelled[/red]")
        return
    
    description = questionary.text(
        "Description:",
        validate=lambda text: len(text) > 0 or "Description cannot be empty"
    ).ask()
    
    category = questionary.select(
        "Category:",
        choices=[
            "Code Generation",
            "Code Review",
            "Documentation",
            "Debugging",
            "Explanation",
            "Creative Writing",
            "Analysis",
            "General",
            "Custom"
        ]
    ).ask()
    
    if category == "Custom":
        category = questionary.text("Enter custom category:").ask()
    
    # Prompt Templates [web:13][web:14]
    console.print("\n[bold cyan]📝 System Prompt[/bold cyan]")
    console.print("[dim]This sets the AI's role and behavior. Use {{variable}} for placeholders.[/dim]\n")
    
    system_prompt = questionary.text(
        "System Prompt:",
        multiline=True,
        validate=lambda text: len(text) > 0 or "System prompt cannot be empty"
    ).ask()
    
    has_user_template = questionary.confirm(
        "Add user prompt template? (for formatting user input)",
        default=False
    ).ask()
    
    user_prompt_template = None
    if has_user_template:
        console.print("\n[bold cyan]💬 User Prompt Template[/bold cyan]")
        console.print("[dim]Template for formatting user input. Use {{input}} for user message.[/dim]\n")
        user_prompt_template = questionary.text(
            "User Prompt Template:",
            multiline=True
        ).ask()
    
    # Response Configuration
    console.print("\n[bold cyan]⚙️  Response Configuration[/bold cyan]\n")
    
    response_style = questionary.select(
        "Response Style:",
        choices=[
            "Concise",
            "Detailed",
            "Technical",
            "Creative",
            "Conversational",
            "Professional"
        ]
    ).ask()
    
    # OpenAI Parameters [web:17][web:20]
    temperature = questionary.text(
        "Temperature (0.0-2.0, default 0.7):",
        default="0.7",
        validate=lambda x: 0.0 <= float(x) <= 2.0 or "Must be between 0.0 and 2.0"
    ).ask()
    
    use_max_tokens = questionary.confirm(
        "Set max tokens limit?",
        default=False
    ).ask()
    
    max_tokens = None
    if use_max_tokens:
        max_tokens = questionary.text(
            "Max Tokens:",
            validate=lambda x: x.isdigit() and int(x) > 0 or "Must be a positive number"
        ).ask()
        max_tokens = int(max_tokens) if max_tokens else None
    
    # Advanced Parameters
    show_advanced = questionary.confirm(
        "Configure advanced parameters?",
        default=False
    ).ask()
    
    top_p = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    
    if show_advanced:
        top_p = float(questionary.text(
            "Top P (0.0-1.0, default 1.0):",
            default="1.0"
        ).ask())
        
        frequency_penalty = float(questionary.text(
            "Frequency Penalty (-2.0 to 2.0, default 0.0):",
            default="0.0"
        ).ask())
        
        presence_penalty = float(questionary.text(
            "Presence Penalty (-2.0 to 2.0, default 0.0):",
            default="0.0"
        ).ask())
    
    # Create prompt data structure
    prompt_data = {
        "name": name,
        "description": description,
        "category": category,
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "response_style": response_style,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }
    
    # Confirm creation
    console.print("\n[bold cyan]📋 Prompt Summary:[/bold cyan]")
    console.print(f"Name: {name}")
    console.print(f"Category: {category}")
    console.print(f"Temperature: {temperature}")
    console.print(f"Max Tokens: {max_tokens or 'Auto'}")
    console.print()
    
    confirm = questionary.confirm(
        "Create this prompt?",
        default=True
    ).ask()
    
    if confirm:
        manager = PromptManager()
        manager.create_prompt(prompt_data)
    else:
        console.print("[yellow]❌ Cancelled[/yellow]")


def interactive_view_prompts():
    """Interactive prompt viewer and manager"""
    manager = PromptManager()
    prompts = manager.list_prompts()
    
    if not prompts:
        console.print("[yellow]📭 No prompts found. Create your first prompt![/yellow]")
        return None
    
    manager.display_prompts_table()
    console.print()
    
    # Create choices with prompt names
    prompt_choices = [f"{p['name']} ({p.get('category', 'General')})" for p in prompts]
    prompt_choices.append("↩️  Go Back")
    
    selection = questionary.select(
        "Select a prompt to view details:",
        choices=prompt_choices
    ).ask()
    
    if not selection or selection == "↩️  Go Back":
        return None
    
    # Extract prompt name (remove category part)
    prompt_name = selection.split(" (")[0]
    selected_prompt = manager.get_prompt_by_name(prompt_name)
    
    if selected_prompt:
        manager.display_prompt_detail(selected_prompt)
        
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "✅ Use this prompt",
                "✏️  Edit prompt",
                "🗑️  Delete prompt",
                "↩️  Go Back"
            ]
        ).ask()
        
        if action == "✅ Use this prompt":
            return selected_prompt
        elif action == "✏️  Edit prompt":
            edit_prompt(selected_prompt)
        elif action == "🗑️  Delete prompt":
            confirm = questionary.confirm(
                f"Are you sure you want to delete '{selected_prompt['name']}'?",
                default=False
            ).ask()
            if confirm:
                manager.delete_prompt(selected_prompt['id'])
        
    return None


def edit_prompt(prompt: Dict):
    """Edit an existing prompt"""
    manager = PromptManager()
    
    console.print(f"\n[bold cyan]✏️  Editing: {prompt['name']}[/bold cyan]\n")
    
    field = questionary.select(
        "What would you like to edit?",
        choices=[
            "Description",
            "System Prompt",
            "User Prompt Template",
            "Temperature",
            "Max Tokens",
            "Response Style",
            "Advanced Parameters",
            "↩️  Cancel"
        ]
    ).ask()
    
    if field == "↩️  Cancel":
        return
    
    updates = {}
    
    if field == "Description":
        new_desc = questionary.text("New description:", default=prompt.get('description', '')).ask()
        updates['description'] = new_desc
    
    elif field == "System Prompt":
        new_system = questionary.text("New system prompt:", multiline=True, default=prompt['system_prompt']).ask()
        updates['system_prompt'] = new_system
    
    elif field == "User Prompt Template":
        new_template = questionary.text("New user prompt template:", multiline=True, default=prompt.get('user_prompt_template', '')).ask()
        updates['user_prompt_template'] = new_template
    
    elif field == "Temperature":
        new_temp = questionary.text("New temperature:", default=str(prompt.get('temperature', 0.7))).ask()
        updates['temperature'] = float(new_temp)
    
    elif field == "Max Tokens":
        new_max = questionary.text("New max tokens (leave empty for auto):", default=str(prompt.get('max_tokens', ''))).ask()
        updates['max_tokens'] = int(new_max) if new_max else None
    
    elif field == "Response Style":
        new_style = questionary.select(
            "Response Style:",
            choices=["Concise", "Detailed", "Technical", "Creative", "Conversational", "Professional"]
        ).ask()
        updates['response_style'] = new_style
    
    if updates:
        manager.update_prompt(prompt['id'], updates)


def select_prompt_for_chat():
    """Select a prompt to use in chat session"""
    manager = PromptManager()
    prompts = manager.list_prompts()
    
    if not prompts:
        return None
    
    prompt_choices = [f"{p['name']} - {p.get('description', '')[:40]}" for p in prompts]
    prompt_choices.insert(0, "❌ No prompt (default)")
    
    selection = questionary.select(
        "Select a prompt template:",
        choices=prompt_choices
    ).ask()
    
    if not selection or selection == "❌ No prompt (default)":
        return None
    
    prompt_name = selection.split(" - ")[0]
    selected_prompt = manager.get_prompt_by_name(prompt_name)
    
    if selected_prompt:
        manager.increment_usage(selected_prompt['id'])
    
    return selected_prompt
