import yaml
import os
from rich.console import Console
from rich.prompt import Prompt
from src.adapters.databases.openalex_adapter import OpenAlexAdapter
from src.adapters.llms.gemini_adapter import GeminiAdapter
from src.core.models import Record
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
console = Console()

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def human_review(record: Record, llm_reason: str):
    """Triggered when LLM is uncertain."""
    console.print(f"\n[yellow]UNCERTAIN RECORD[/yellow]")
    console.print(f"[bold]{record.title}[/bold]")
    console.print(f"[italic]{record.abstract[:200]}...[/italic]")
    console.print(f"LLM Reasoning: {llm_reason}")
    
    choice = Prompt.ask("Classify", choices=["relevant", "irrelevant", "skip"])
    return choice

def main():
    config = load_config()
    db = OpenAlexAdapter()
    llm = GeminiAdapter(model_name=config['llm']['model'])
    
    current_query = config['search']['initial_query']
    max_iters = config['search']['max_iterations']
    
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
        
        # 2. Classify Loop
        with console.status("[bold green]Classifying papers with Gemini..."):
            for record in records:
                try:
                    result = llm.classify(record, config['criteria'])
                    final_decision = result.relevance
                    
                    # 3. Human in the Loop for Uncertainty
                    if result.relevance == "uncertain":
                        final_decision = human_review(record, result.reasoning)
                    
                    if final_decision == "relevant":
                        relevant_count += 1
                        console.print(f"[blue]Relevant:[/blue] {record.title[:60]}... (Conf: {result.confidence})")
                    elif final_decision == "irrelevant":
                        irrelevant_records.append(record)
                        
                except Exception as e:
                    console.print(f"Error classifying {record.id}: {e}")

        # Summary
        precision = relevant_count / len(records) if records else 0
        console.print(f"\nIteration Summary: Precision: {precision:.1%}")
        
        if precision >= config['search']['precision_threshold']:
            console.print("[bold green]Precision goal met! Stopping optimisation.[/bold green]")
            break
            
        # 4. Optimise
        if irrelevant_records and iteration < max_iters:
            console.print("\n[bold purple]Optimising Query...[/bold purple]")
            suggestion = llm.optimize_query(current_query, irrelevant_records)
            console.print(f"Critique: {suggestion.critique}")
            console.print(f"New Query: {suggestion.new_query}")
            
            # Auto-update query for next loop
            current_query = suggestion.new_query
        else:
            console.print("No false positives to optimise against or max iterations reached.")
            break

if __name__ == "__main__":
    main()
