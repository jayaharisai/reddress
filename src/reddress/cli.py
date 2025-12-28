import click
from .core import (
    greet_user,
    configuration,
    reddress_start,
)

from .templates.main import server

from .utils import (
    show_usage_stats,
    interactive_create_prompt,
    interactive_view_prompts,
    PromptManager
    )


@click.group()
@click.version_option()
def main():
    """Reddress - AI Chat CLI Tool"""
    pass


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

@main.command()
@click.option('--recent', '-r', default=10, help='Show recent N messages')
def stats(recent):
    """Show usage statistics"""
    show_usage_stats(recent)




if __name__ == "__main__":
    main()
