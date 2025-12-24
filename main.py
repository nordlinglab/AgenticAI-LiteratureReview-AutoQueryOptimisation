import typer
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

# Import Core Components
from src.core.config import load_project_config
from src.core.models import Record
from src.adapters.databases.openalex_adapter import OpenAlexAdapter
from src.adapters.databases.scopus_adapter import ScopusAdapter
from src.adapters.databases.wos_adapter import WosAdapter
from src.adapters.llms.gemini_adapter import GeminiAdapter

# Initialize App
app = typer.Typer(help="Agentic AI Literature Review CLI")
console = Console()
load_dotenv()

def get_db_adapter(db_name: str):
    if db_name == 'scopus':
        return ScopusAdapter()
    elif db_name == 'wos':
        return WosAdapter()
    return OpenAlexAdapter()

def human_review(record: Record, llm_reason: str):
    console.print(f"\n[yellow]UNCERTAIN RECORD[/yellow]")
    console.print(f"[bold]{record.title}[/bold]")
    console.print(f"[italic]{record.abstract[:200]}...[/italic]")
    console.print(f"LLM Reasoning: {llm_reason}")
    return Prompt.ask("Classify", choices=["relevant", "irrelevant", "skip"])

@app.command()
def run(project: str = typer.Argument(..., help="Name of the project file (e.g., '01_attempts')")):
    """
    Start the search and optimisation loop for a specific project.
    """
    try:
        config = load_project_config(project)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    console.rule(f"[bold blue]Project: {config['name']}[/bold blue]")
    console.print(f"[dim]{config['description']}[/dim]")

    # Initialize Components
    db_name = config['search'].get('database', 'openalex')
    db = get_db_adapter(db_name)
    
    llm_model = config.get('system', {}).get('llm_model', 'gemini/gemini-1.5-pro-latest')
    llm = GeminiAdapter(model_name=llm_model)
    
    # Loop State
    current_query = config['search']['initial_query']
    max_iters = config['search']['max_iterations']
    precision_target = config['search']['precision_threshold']
    
    for iteration in range(1, max_iters + 1):
        console.rule(f"[bold red]Iteration {iteration}[/bold red]")
        console.print(f"Query: [green]{current_query}[/green]")
        
        # 1. Search
        records = db.search(current_query, limit=config['search']['max_results_per_iter'])
        if not records:
            console.print("No records found.")
            break
            
        relevant_count = 0
        irrelevant_records = []
        
        # 2. Classify
        with console.status(f"[bold green]Classifying {len(records)} papers..."):
            for record in records:
                try:
                    result = llm.classify(record, config['criteria'])
                    decision = result.relevance
                    
                    if decision == "uncertain":
                        decision = human_review(record, result.reasoning)
                    
                    if decision == "relevant":
                        relevant_count += 1
                        console.print(f"[blue]Relevant:[/blue] {record.title[:60]}...")
                    elif decision == "irrelevant":
                        irrelevant_records.append(record)
                        
                except Exception as e:
                    console.print(f"[red]Error:[/red] {e}")

        # 3. Assess & Optimise
        total = len(records)
        precision = relevant_count / total if total > 0 else 0
        console.print(f"\nIteration Precision: {precision:.1%}")
        
        if precision >= precision_target:
            console.print("[bold green]Target precision reached![/bold green]")
            break
            
        if irrelevant_records and iteration < max_iters:
            console.print("\n[bold purple]Optimising Query...[/bold purple]")
            suggestion = llm.optimize_query(current_query, irrelevant_records)
            console.print(f"Critique: {suggestion.critique}")
            console.print(f"New Query: {suggestion.new_query}")
            current_query = suggestion.new_query
        else:
            break

if __name__ == "__main__":
    app()
