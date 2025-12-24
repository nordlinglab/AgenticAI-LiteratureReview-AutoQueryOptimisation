# AgenticAI Literature Review & Query Optimisation

**Repository:** [nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation](https://github.com/nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation)  

**Description:** AgenticAI for systematic literature search using an LLM to optimise the search query and analyse the search result.

This project implements a **closed-loop literature search agent**. It performs three key functions iteratively:
1.  **Search:** Queries academic databases (OpenAlex) for papers.
2.  **Classify:** Uses an LLM (Gemini via LiteLLM) to read titles/abstracts and classify papers as *Relevant*, *Irrelevant*, or *Uncertain*.
3.  **Optimise:** Analyses "Irrelevant" results (false positives) to refine the Boolean search string, reducing noise in the next iteration.

**Author:** Torbjörn E.M. Nordling (torbjorn.nordling@nordlinglab.org)

**License:** Apache License Version 2.0

**AI usage:** Created using Claude Opus 4.5 and Google Gemini 3.0 Pro

---

## 1. Quick Start Guide (by Gemini 3.0 Pro)

### Prerequisites
* **Python 3.11+**
* **Gemini API Key** (from Google AI Studio)
* **Git**

### Installation

# 1. Clone the repository
```bash
git clone https://github.com/nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation.git 
cd AgenticAI-LiteratureReview-AutoQueryOptimisation
```

# 2. Create virtual environment (using standard venv)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```
# 3. Install dependencies
```bash
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env

```


2. Edit `.env` and add your keys:
```ini
GEMINI_API_KEY=your_key_here
OPENALEX_EMAIL=your_email@university.edu

```


### Usage

Run the main orchestration loop:

```bash
python main.py

```

---

## 2. Project Structure & Setup (by Gemini 3.0 Pro)

To re-create this system from scratch, follow the file structure and code blocks below.

### Git Repository

Go to GitHub and create a git repository, like [nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation](https://github.com/nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation), then clone it locally and change to the folder containing it.

### Directory Layout

Run these commands to generate the folder structure:

```bash
mkdir -p config/projects
mkdir -p config/databases config/llms config/prompts
mkdir -p src/adapters/databases
mkdir -p src/adapters/llms
mkdir -p src/core
mkdir -p src/orchestration
mkdir -p src/ui src/data
touch src/__init__.py

```

### Dependency Management (`pyproject.toml`)

Create this file in the root directory.

```toml
[project]
name = "agentic-ai-lit-review"
version = "0.1.0"
description = "AgenticAI for systematic literature search using an LLM to optimise the search query and analyse the search result."
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "litellm>=1.0",      # LLM Abstraction
    "instructor>=1.0",   # Structured Output
    "pyalex>=0.13",      # OpenAlex Wrapper
    "rich>=13.0",        # Pretty terminal output
    "python-dotenv>=1.0"
]

[tool.setuptools]
packages = ["src"]

```

### Standard Git Ignore (`.gitignore`)

At the end of the standard Python GitHub `.gitignore` add:

```text
# Custom
.DS_Store
logs/
data/
.env.keys

```

---

## 3. Configuration Files (by Gemini 3.0 Pro)

### Global Settings (`config/settings.yaml`)

Defines the search scope and inclusion/exclusion criteria.

```yaml
project:
  name: "Reproducibility Evaluation Review"

llm:
  # Using Gemini 1.5 Pro for high-reasoning capability
  model: "gemini/gemini-1.5-pro-latest" 
  temperature: 0.1

search:
  initial_query: 'reproducibility AND ("replication study" OR "meta-research")'
  max_results_per_iter: 20
  max_iterations: 3
  precision_threshold: 0.9

criteria:
  inclusion:
    - "Reports of actual attempts to reproduce or replicate original studies."
    - "Methodologies or frameworks specifically for evaluating reproducibility from written descriptions."
    - "Assessments or audits of how well reproducibility can be evaluated."
    - "Systematic reviews or meta-research synthesizing reproducibility evaluations."
  exclusion:
    - "Studies that merely mention reproducibility as a keyword without it being the focus."
    - "Purely computational or software reproducibility (unless discussing general evaluation methods)."
    - "Statistical protocols without empirical results or synthesis."

```

### Environment Variables (`.env.example`)

```ini
GEMINI_API_KEY=
OPENALEX_EMAIL=

```

---

## 4. Source Code Implementation (by Gemini 3.0 Pro)

### A. Data Models (`src/core/models.py`)

Standardizes how data moves between the database, the LLM, and the user.

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Record(BaseModel):
    """Represents a single academic paper."""
    id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    year: Optional[int] = None
    doi: Optional[str] = None
    
    def to_text(self):
        """Formats the record for the LLM prompt."""
        return f"Title: {self.title}\nAbstract: {self.abstract or 'No Abstract'}\nYear: {self.year}"

class Classification(BaseModel):
    """Structured output for paper relevance."""
    relevance: Literal["relevant", "irrelevant", "uncertain"]
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Brief explanation of the decision based on criteria")

class QuerySuggestion(BaseModel):
    """Structured output for query optimisation."""
    critique: str = Field(..., description="Analysis of why false positives occurred")
    new_query: str = Field(..., description="The optimised Boolean query string for OpenAlex")
    expected_improvement: str = Field(..., description="Why this query is better")

```

### B. OpenAlex Database Adapter (`src/adapters/databases/openalex_adapter.py`)

Handles connection to the OpenAlex API using `pyalex`.

```python
import pyalex
from src.core.models import Record
import os

class OpenAlexAdapter:
    def __init__(self):
        # Setting email places us in the 'Polite Pool' for faster/better access
        email = os.getenv("OPENALEX_EMAIL")
        if email:
            pyalex.config.email = email

    def search(self, query: str, limit: int = 20) -> List[Record]:
        print(f"🔎 Searching OpenAlex for: {query}")
        try:
            # OpenAlex boolean search works best with search= parameter for keywords
            results = pyalex.Works().search(query).get(per_page=limit)
            
            records = []
            for w in results:
                # Handle missing abstracts (inverted index reconstruction)
                abstract = w.get("abstract_inverted_index")
                abstract_text = None
                if abstract:
                    try:
                        words = sorted([(pos, word) for word, positions in abstract.items() for pos in positions])
                        abstract_text = " ".join(word for _, word in words)
                    except:
                        abstract_text = "Error parsing abstract"

                records.append(Record(
                    id=w.get("id", ""),
                    title=w.get("display_name", "No Title"),
                    abstract=abstract_text,
                    authors=[a['author']['display_name'] for a in w.get('authorships', [])],
                    year=w.get("publication_year"),
                    doi=w.get("doi")
                ))
            return records
        except Exception as e:
            print(f"Error querying OpenAlex: {e}")
            return []

```

### C. Gemini LLM Adapter (`src/adapters/llms/gemini_adapter.py`)

Uses `instructor` + `litellm` to interface with Gemini.

```python
import instructor
import litellm
import os
from src.core.models import Record, Classification, QuerySuggestion

class GeminiAdapter:
    def __init__(self, model_name: str = "gemini/gemini-1.5-pro-latest"):
        self.model_name = model_name
        self.client = instructor.from_litellm(litellm.completion)
        self.api_key = os.getenv("GEMINI_API_KEY")

    def classify(self, record: Record, criteria: dict) -> Classification:
        prompt = f"""
        Analyze the following academic paper against the research criteria.
        
        PAPER:
        {record.to_text()}
        
        INCLUSION CRITERIA:
        {criteria['inclusion']}
        
        EXCLUSION CRITERIA:
        {criteria['exclusion']}
        
        Task: Classify as 'relevant', 'irrelevant', or 'uncertain'.
        Provide a confidence score (0.0 to 1.0) and brief reasoning.
        """
        
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_model=Classification,
            api_key=self.api_key
        )

    def optimize_query(self, current_query: str, false_positives: list[Record]) -> QuerySuggestion:
        fp_text = "\n".join([f"- {r.title}" for r in false_positives[:5]])
        
        prompt = f"""
        I am conducting a systematic review on reproducibility evaluation.
        
        CURRENT QUERY: {current_query}
        
        PROBLEM: The query returned these IRRELEVANT papers (False Positives):
        {fp_text}
        
        TASK: 
        1. Analyze why these papers were caught (e.g., polysemy, wrong context).
        2. Construct a new, boolean OpenAlex-compatible query string to exclude these types of papers while keeping relevant ones.
        3. Explain your logic.
        """
        
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_model=QuerySuggestion,
            api_key=self.api_key
        )

```

### D. Main Orchestration Loop (`main.py`)

Connects the components and handles the user interface.

```python
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

```




---

## Project Overview (Opus 4.5)

**Prompt:**
I am a researcher conducting a literature review and need help to optimise queries on Scopus and Web of Science based on automatic analysis of the results. Describe the best way to do this in a closed loop using a LLM to analyse the hits and classify each as relevant or irrelevant asking for human input when not able to determine if a hit is relevant or not, and then another LLM to optimse the query to reduce the number of false positive and false negative hits.

**Answer:**
This is a fascinating application of LLMs to systematic review methodology. Here's a robust approach to building this closed-loop system.

### Conceptual Architecture

The system operates in iterative cycles:

```
Query → Database Search → LLM Classifier → Human Review (uncertain cases) → LLM Query Optimiser → Refined Query → ...
```

### Component 1: The Classification LLM

This LLM receives each hit (title, abstract, keywords, and your inclusion/exclusion criteria) and outputs one of three classifications:

**Relevant** — clearly meets criteria
**Irrelevant** — clearly fails criteria  
**Uncertain** — requires human judgement

The key to making this work well is providing the classifier with a detailed prompt containing your research question, explicit inclusion/exclusion criteria, and ideally a few labelled examples (few-shot learning). The LLM should also provide a brief justification for each classification, which helps you audit decisions and refine criteria.

For calibration, I'd recommend starting conservatively—the classifier should flag items as "uncertain" liberally in early iterations, then become more confident as it learns from your feedback on edge cases.

### Component 2: Human-in-the-Loop Review

When the classifier returns "uncertain," you review the item and provide a definitive label. This serves two purposes: it ensures recall on borderline cases, and it generates training signal for both components. You might also periodically audit a random sample of "relevant" and "irrelevant" classifications to catch systematic errors.

### Component 3: The Query Optimisation LLM

After each iteration, this LLM analyses the classification results and proposes query modifications. It receives:

- The current query
- False positives (classified irrelevant but retrieved) with their abstracts
- False negatives (if discoverable—perhaps from citation chaining or known relevant papers not retrieved)
- Patterns in the uncertain cases

The optimiser then suggests specific changes: adding exclusion terms (via NOT operators or field restrictions), adjusting proximity operators, refining MeSH/subject headings, or restructuring Boolean logic. Importantly, it should explain *why* each change should improve precision or recall, so you can evaluate the reasoning before implementing.

### Practical Implementation Considerations

**Iteration strategy**: Run 3–5 cycles, tracking precision and recall estimates at each stage. Diminishing returns typically set in quickly.

**Database-specific syntax**: Scopus and Web of Science have different query syntaxes. The optimiser needs to output valid syntax for each, or you need a translation layer.

**Stopping criteria**: Define these upfront—perhaps when precision exceeds a threshold, or when two consecutive iterations yield no significant improvement.

**Documentation**: Log every query version, classification decision, and human override. This is essential for PRISMA reporting if this feeds into a systematic review.

### A Potential Pitfall

Be cautious about over-optimisation. If you aggressively tune against your current corpus of hits, you may inadvertently exclude relevant papers that use different terminology. Consider periodically introducing "exploration" searches with broader terms to check for blind spots.

Would you like me to sketch out specific prompts for the classifier and optimiser components, or discuss how to handle the practical mechanics of extracting results from Scopus/Web of Science for processing?




---

## Tools and Software for Implementation (Opus 4.5)

There are several approaches depending on your technical comfort level and infrastructure preferences.

### Database Access

**Scopus**: The Scopus API (via Elsevier Developer Portal) allows programmatic querying. The `pybliometrics` Python library provides a clean wrapper—you can execute searches, retrieve abstracts, and paginate through results. You'll need an API key, which requires institutional access.

**Web of Science**: The Web of Science API (Clarivate) is more restrictive. The Starter API offers limited functionality; the Expanded API requires a separate subscription. Alternatively, you can export results manually as CSV/RIS files and ingest those, which is less elegant but works without API access.

If API access is problematic, a pragmatic workaround is exporting results in batches (Scopus allows 2,000 per export, WoS around 1,000) and processing the files.

### LLM Integration

**OpenAI API** (GPT-4o or GPT-4 Turbo): Robust, well-documented, handles long abstracts well. Cost scales with volume but is manageable for typical review sizes (hundreds to low thousands of papers).

**Anthropic API** (Claude): Similarly capable, with a larger context window if you want to batch multiple abstracts per call.

**Local models** (via Ollama or llama.cpp): If you're processing sensitive data or want to avoid API costs, models like Llama 3 or Mistral can run locally. Classification performance is somewhat weaker than frontier models but often adequate for this task.

For the query optimisation component specifically, I'd lean toward a more capable model (GPT-4o, Claude Opus/Sonnet) since it requires nuanced reasoning about Boolean logic and domain terminology.

### Orchestration and Workflow

**LangChain** or **LlamaIndex**: These frameworks help structure the prompts, manage conversation history, and chain the classifier and optimiser together. LangChain's `RunnableSequence` abstraction fits this use case well. That said, for a relatively simple two-stage pipeline, you might find plain Python with direct API calls cleaner and easier to debug.

**Python scripting**: Core libraries you'll want include `pandas` for managing your corpus of hits, `pybliometrics` for Scopus, `requests` for API calls, and possibly `RIS` or `rispy` for parsing bibliographic exports.

### Human Review Interface

For the uncertain cases requiring your input, options range from simple to sophisticated:

**Simple**: A command-line script that displays the title/abstract and prompts you to type R/I/skip.

**Moderate**: A Streamlit or Gradio app that presents items in a web interface with buttons for classification. This is more pleasant for reviewing dozens of items.

**Sophisticated**: Tools like **ASReview** (open source, designed precisely for systematic review screening) can be integrated. It has active learning built in, though adapting it to your custom LLM classifier requires some work.

### A Minimal Stack

If I were building this tomorrow, I'd use:

| Component | Tool |
|-----------|------|
| Database queries | `pybliometrics` + manual WoS exports |
| Data management | `pandas` + SQLite for persistence |
| LLM calls | Direct API calls to Claude or OpenAI |
| Human review | Streamlit app (perhaps 50 lines of code) |
| Logging | Simple CSV/JSON logs per iteration |

This avoids heavy framework dependencies while remaining flexible.

### Existing Tools Worth Knowing

**ASReview**: Open-source active learning for systematic reviews. Not LLM-based by default, but extensible. Could serve as your human review interface or even replace part of the pipeline.

**Rayyan**: Commercial/freemium screening tool. No LLM integration, but useful if you want collaborative human review.

**Elicit** and **Consensus**: These are LLM-powered research tools that do related tasks (finding and summarising papers), though they don't offer the query-refinement loop you're designing.




---

## Scientific Reference Databases with API Access (Opus 4.5)

**Prompt:**
Which scientific reference databases offer API access? How much does it cost? How to gain access?

**Answer:**
I'll search for comprehensive information on scientific database APIs and their access models. Now let me compile this information into a comprehensive overview for you.

Based on my research, here's a comprehensive overview of the major databases, their API access models, costs, and how to gain access.

---

### Fully Free / Open Access APIs

#### OpenAlex
OpenAlex is a fully open catalog of the global research system. It's named after the ancient Library of Alexandria and made by the nonprofit OurResearch.

- **Cost**: Completely free, no authentication required
- **Coverage**: 240+ million works, updated daily
- **Rate limits**: The daily limit for API calls is 100,000 requests per user per day.
- **Access**: Just start using it at `api.openalex.org`—add your email for faster "polite pool" access
- **Data**: CC0 license (public domain)
- **Python wrapper**: `pyalex`

**Best for**: Researchers who need large-scale bibliometric data without cost or bureaucratic barriers.

---

#### PubMed / NCBI E-utilities
The E-utilities are the public API to the NCBI Entrez system and allow access to all Entrez databases including PubMed, PMC, Gene, Nuccore and Protein.

- **Cost**: Free
- **Rate limits**: The E-utilities API is a free service open to the public and will continue to be. Without an API key: 3 requests/second. With API key: 10 requests/second
- **Access**: Create a free NCBI account, generate an API key from Settings
- **Coverage**: Biomedical literature (36+ million citations)
- **Python wrapper**: `biopython`, `metapub`

**Best for**: Biomedical and life sciences research.

---

#### Crossref
Our publicly available REST API exposes the scholarly metadata that members and trusted sources deposit with Crossref.

- **Cost**: Free
- **Coverage**: 176+ million DOI records from 23,600+ publishers
- **Access**: No sign-up required; include your email in requests to be a "polite" user
- **Rate limits**: Variable; polite users get better service
- **Python wrapper**: `habanero`

**Best for**: DOI metadata, reference matching, publisher information.

---

#### Semantic Scholar
Most Semantic Scholar endpoints are available to the public without authentication, but they are rate-limited to 1000 requests per second shared among all unauthenticated users.

- **Cost**: Free (non-profit, Allen Institute for AI)
- **Coverage**: 200+ million papers across all disciplines
- **Rate limits**: Unauthenticated: shared pool. The API allows up to 100 requests per 5 minutes. To access a higher rate limit, complete the form to request authentication for your project.
- **Access**: Request an API key for dedicated rate limits
- **Python wrapper**: `semanticscholar`
- **Extras**: TLDRs, citation context, influence scores

**Best for**: AI-enhanced discovery, citation analysis, computer science coverage.

---

### Freemium / Conditional Free Access

#### Dimensions
The Metrics API is available for free for non-commercial purposes so that academic institutions, independent researchers and non-profit organizations can leverage high-quality citation metrics without budget constraints.

- **Free tier**: Metrics API for non-commercial use; limited web interface (Dimensions Free)
- **Academic access**: We offer free access to Dimensions for non-commercial scientometric research projects, through which you will get access to Dimensions Analytics, the Dimensions API and/or Dimensions data on Google BigQuery.
- **Requirements for free research access**: Time-limited bibliometrics/scientometrics project, non-commercial purpose, intent to publish results
- **Paid**: Full institutional subscription required for comprehensive access
- **Coverage**: 140+ million publications, grants, patents, clinical trials, datasets
- **Python wrapper**: `dimcli`

**Best for**: Funding/grant analysis, linking publications to grants and patents.

---

#### The Lens
The Lens database is a free platform for enabling discovery of both scholarly and patent literature.

- **Free tier**: Web search, exports up to certain limits, basic analysis
- **API access**: As part of [institutional] Toolkit subscription, all [institution] users are eligible to make use of the Lens's Scholarly and Patent APIs.
- **Institutional plans**: Available for universities
- **Coverage**: 200+ million scholarly works, 136+ million patent records
- **Unique feature**: Cross-mapping between scholarly citations and patents

**Best for**: Patent-literature linkage, innovation tracking.

---

### Subscription-Based / Institutional APIs

#### Scopus (Elsevier)
- **Cost**: API access is included with institutional Scopus subscription (not always—must be negotiated)
- **Free for academics**: You can request an API Key on Elsevier's Developer Portal. If you do not already have an Elsevier user ID, you will have to register. Non-commercial academic use is free if your institution subscribes
- **Rate limits**: ~5,000 retrieval requests or 20,000 search requests per week per key
- **Access**: Register at dev.elsevier.com with institutional email; may need institutional token from library
- **Coverage**: 92+ million records, 27,000+ journals
- **Python wrapper**: `pybliometrics`

**Best for**: Comprehensive multidisciplinary coverage, citation analysis, author profiles.

---

#### Web of Science (Clarivate)
- **Cost**: Access to this API requires a paid license.
- **Starter API (limited)**: This plan is available to anyone, even if your organization does not subscribe to Web of Science. It does not return times cited and is limited to 50 requests per day.
- **Expanded API**: Requires institutional subscription + additional licensing
- **Access**: Register at developer.clarivate.com; approval process can be slow
- **Coverage**: 85+ million records, highly curated
- **Python wrapper**: `wos` (community), official clients available

**Best for**: Traditional bibliometrics, journal impact factors, highly curated content.

---

### Summary Comparison Table

| Database | Cost | Auth Required | Rate Limits | Best Python Library |
|----------|------|---------------|-------------|---------------------|
| **OpenAlex** | Free | No | 100k/day | `pyalex` |
| **PubMed** | Free | Optional | 3-10 req/sec | `biopython` |
| **Crossref** | Free | No | Polite pool | `habanero` |
| **Semantic Scholar** | Free | Optional | 100/5min free | `semanticscholar` |
| **Dimensions** | Conditional | Yes | By application | `dimcli` |
| **The Lens** | Conditional | Yes | Institutional | Custom |
| **Scopus** | Institutional | Yes | Weekly quotas | `pybliometrics` |
| **Web of Science** | Paid | Yes | 50/day (Starter) | `wos` |

---

### Recommendation for Your Literature Review Optimisation Project

For your closed-loop query optimisation system, I'd suggest:

1. **Start with OpenAlex** — it's free, has excellent coverage, and no barriers to entry. You can prototype your entire system without worrying about rate limits or access issues.

2. **Add PubMed** if you're in biomedical sciences — the E-utilities are robust and free.

3. **Use Crossref** for DOI metadata enrichment and reference matching.

4. **Scopus/WoS** for final validation — once your system is working, validate results against Scopus or WoS if your institution has API access (worth asking NCKU Library).

Most organizations offer free access to APIs for non-commercial purposes, but some have limitations on metadata retrieval. However, CrossRef, OpenCitations, or OpenAlex have no restrictions on the metadata retrieval.





## Scientific Reference Databases with API Access--In depth (Opus 4.5)

**Prompt:**
Describe the coverage and unique content of each of these. Add this description to the summary table. Which do you recommend for the following search:
1. reports of attempts to reproduce original studies,
2. ways to evaluate reproducibility of research from written descriptions
3. assessments of ways to evaluate reproducibility of research from written descriptions, and 
4. review and meta-research articles that synthesise ways to evaluate reproducibility or attempts to reproduce studies.

**Answer:**
I'll search for more specific information about the coverage and unique content of each database to give you a more detailed comparison.Now I have enough information to compile a comprehensive comparison table and make recommendations for your specific research topics. Let me create this for you.

### Comprehensive Database Comparison with Coverage Details

| Database | Cost | Coverage & Unique Content | Records | Disciplines | Unique Strengths | Best For |
|----------|------|---------------------------|---------|-------------|------------------|----------|
| **OpenAlex** | Free, no auth | Aggregates Crossref, PubMed, ORCID, Unpaywall, institutional repositories. Among the 62,701 active OA journals listed in ROAD, OpenAlex indexes 34,217 vs WoS 6,157 and Scopus 7,351. Strong preprint coverage (arXiv, bioRxiv, medRxiv). | 240M+ works | All disciplines; especially strong for open access, Global South, preprints | Largest free index; best OA coverage; CC0 license; topic classification | Broad discovery, preprint tracking, bibliometrics without barriers |
| **PubMed/NCBI** | Free | Life sciences and biomedical focus. MeSH controlled vocabulary. Links to PMC full text. Clinical trials via ClinicalTrials.gov. Genetic/molecular data via Entrez system. | 36M+ citations | Biomedical, life sciences | MeSH indexing; clinical content; links to sequences/genes; daily updates | Biomedical systematic reviews, clinical research |
| **Crossref** | Free | DOI metadata from 23,600+ publisher members. Reference lists, funder information, license data. No abstracts. | 176M+ DOIs | All disciplines | Authoritative DOI registry; reference matching; funder metadata | DOI lookup, reference validation, publisher data |
| **Semantic Scholar** | Free | AI-curated corpus with TLDRs, citation intent, influence scores. Strong CS/AI coverage. Bulk datasets available (S2ORC). | 200M+ papers | All; strongest in CS, biomedicine | AI features (TLDRs, citation context); free bulk data; influence metrics | AI-enhanced discovery, citation context analysis |
| **Dimensions** | Freemium | The Dimensions database is a scholarly database that goes beyond research articles and their citations by including not only books, chapters, and conference proceedings, but also awarded grants, patents, clinical trials, policy documents. Links publications to funding. | 140M+ pubs; 3.7M grants; 34M patents | All disciplines | Grants-to-publications linkage; policy documents; Altmetric integration; funding data | Grant tracking, research impact assessment, policy analysis |
| **The Lens** | Freemium | The Lens database is a free platform for enabling discovery of both scholarly and patent literature. Cross-maps citations between scholarly works and patents. 136M+ patents. Biological sequence data (PatSeq). | 200M+ scholarly; 136M patents | All; strong patent-science linkage | Patent-literature citation mapping; sequence data; patent analytics | Innovation tracking, patent landscape analysis, translational research |
| **Scopus** | Institutional | 92M+ records from 7,000+ publishers. Its coverage of conference proceedings is also generally better than WoS. Author/affiliation profiles. CiteScore metrics. For the humanities, Scopus covers 43% of this literature. | 92M+ records | All; broader social sciences, engineering, emerging fields | Conference proceedings; author IDs; broader journal coverage; non-English content | Comprehensive reviews, author tracking, emerging fields |
| **Web of Science** | Institutional | Highly selective; Arts & Humanities Citation Index (AHCI) covers more than 1,800 journals. Journal Impact Factor (JIF). Coverage is from 1900 to present. Conference Proceedings Citation Index. | 85M+ records | All; curated for quality | Impact Factor; historical depth to 1900; rigorous selection; A&HCI | High-impact citation analysis, historical research, established fields |

---

### Recommendations for Your Reproducibility Research

Your four research topics are interconnected and fall within **meta-research / metascience**, a relatively new interdisciplinary field. Here's my analysis:

#### Topic Characteristics

| Topic | Nature | Key Terminology | Likely Sources |
|-------|--------|-----------------|----------------|
| 1. Reproduction attempts | Empirical studies | "replication study", "reproducibility", "failed to replicate", "direct replication" | Psychology, biomedicine, social sciences journals; preprint servers |
| 2. Methods to evaluate reproducibility | Methodological papers | "reproducibility assessment", "computational reproducibility", "replicability criteria", "reporting guidelines" | Methods journals, meta-research venues |
| 3. Assessments of evaluation methods | Meta-methodological | "meta-research", "research on research", "methodological review" | Specialized meta-research journals (e.g., *Research Integrity and Peer Review*, *Meta-Psychology*) |
| 4. Reviews & meta-research syntheses | Systematic reviews, meta-analyses | "systematic review", "meta-analysis", "scoping review", "umbrella review" | Cochrane, Campbell, methods journals |

#### Recommended Database Strategy

**Primary databases (use all three):**

1. **PubMed** — Essential because:
   - MeSH term `"Reproducibility of Results"` is a controlled vocabulary term specifically for this topic
   - Strong coverage of biomedical replication studies
   - Links to PMC full text for methods extraction
   - Clinical trials registry linkage

2. **OpenAlex** — Essential because:
   - Captures preprints where many replication studies first appear (OSF Preprints, PsyArXiv, MetaArXiv)
   - OpenAlex indexes nearly all the journals covered by Scopus and Web of Science, plus additional sources
   - Free bulk access for systematic processing
   - Good coverage of meta-research which is often in newer/smaller journals

3. **Scopus** (if NCKU has API access) — Important because:
   - Better conference proceedings coverage (meta-research is often presented at conferences like AIMOS, SIPS)
   - Broader social science coverage where replication crisis discourse is prominent
   - Author profiles help identify key researchers in the field

**Secondary databases (recommended for completeness):**

4. **Semantic Scholar** — Useful for:
   - Citation intent classification (can help identify papers that *discuss* vs *report* replications)
   - AI-generated summaries speed screening
   - Good psychology and CS coverage (fields with active replication movements)

5. **Dimensions** — Valuable for:
   - Linking to grants funding replication research (e.g., from foundations promoting open science)
   - Policy documents citing reproducibility research
   - Altmetric data showing public attention to replication studies

#### Suggested Search Strategy

**Core query concepts to combine:**

```
Concept 1 (Reproducibility):
reproducib* OR replicab* OR "replication study" OR "replication attempt" 
OR "failed to replicate" OR "successful replication"

Concept 2 (Evaluation/Assessment):
evaluat* OR assess* OR measur* OR criteria OR indicator* OR metric* 
OR checklist OR guideline* OR framework*

Concept 3 (Meta-research):
"meta-research" OR metascience OR "research on research" 
OR "methodological review" OR "reporting quality"

Concept 4 (Synthesis):
"systematic review" OR "meta-analysis" OR "scoping review" 
OR "umbrella review" OR synthesis
```

**For each of your four topics:**

| Topic | Query Logic | Primary Databases |
|-------|-------------|-------------------|
| 1. Reproduction attempts | Concept 1 AND (empirical OR study OR experiment) NOT (review OR meta-analysis) | PubMed, OpenAlex, PsyArXiv |
| 2. Methods to evaluate | Concept 1 AND Concept 2 AND (method* OR tool OR approach) | PubMed, Scopus, OpenAlex |
| 3. Assessments of methods | Concept 2 AND Concept 3 | OpenAlex, Dimensions, Scopus |
| 4. Reviews & syntheses | Concept 1 AND Concept 4 | PubMed, Cochrane, OpenAlex |

#### Why Not Rely on Just One Database?

The study found that Clarivate Analytics and Elsevier offer highly versatile APIs, while non-profit organizations, such as OpenCitations and OurResearch promote the Open Science philosophy. Most organizations offer free access to APIs for non-commercial purposes, but some have limitations on metadata retrieval. However, CrossRef, OpenCitations, or OpenAlex have no restrictions on the metadata retrieval.

For meta-research specifically:
- Many key papers are in newer journals not yet in WoS
- Preprints are essential (many replication studies appear first on OSF/PsyArXiv)
- The field spans psychology, medicine, statistics, philosophy of science — no single database covers all well
- A 2025 systematic review of 23 comparative studies found that multidisciplinary databases like OpenAlex achieve broader coverage than traditional subscription databases

#### Practical Implementation

Given your goal of building an LLM-assisted query optimisation system, I'd recommend:

1. **Start with OpenAlex + PubMed** — both have free, robust APIs
2. **Add Scopus** if NCKU provides API access (check with library)
3. **Use Semantic Scholar** for citation context analysis in your classifier
4. **Supplement with manual exports** from Cochrane Library (for existing systematic reviews on reproducibility)






---

## Modular System Design for LLM-Assisted Literature Search Optimisation (Opus 4.5)

**Prompt:**
Outline system design where I can optimise the search query for a specific database and add database alternatives as I gain access and also choose the LLM used and easily update it. Tell me which software packages and tools to use to achieve an easy to maintain and stable agentic AI solution for this.

**Answer:**

### Architecture Overview

The system should follow a **plugin-based architecture** with clear separation of concerns, allowing you to swap databases, LLMs, and processing components independently.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                               │
│                    (Workflow Engine / Agent Controller)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│   LLM ADAPTER   │        │  DATABASE       │        │   HUMAN-IN-     │
│     LAYER       │        │  ADAPTER LAYER  │        │   THE-LOOP      │
│                 │        │                 │        │   INTERFACE     │
│ • Claude        │        │ • OpenAlex      │        │                 │
│ • GPT-4         │        │ • PubMed        │        │ • Review UI     │
│ • Local (Llama) │        │ • Scopus        │        │ • Feedback      │
│ • Gemini        │        │ • Semantic Sch. │        │ • Override      │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│              (Query History, Classifications, Configurations)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Core Design Principles

#### 1. Adapter Pattern for Extensibility

Each external service (database, LLM) is accessed through a common interface, making it trivial to add new providers.

```python
# Abstract interfaces that all adapters must implement

class DatabaseAdapter(ABC):
    @abstractmethod
    def search(self, query: str, params: dict) -> list[Record]: ...
    
    @abstractmethod
    def get_query_syntax(self) -> QuerySyntaxSpec: ...
    
    @abstractmethod
    def translate_query(self, generic_query: GenericQuery) -> str: ...

class LLMAdapter(ABC):
    @abstractmethod
    def classify(self, record: Record, criteria: Criteria) -> Classification: ...
    
    @abstractmethod
    def suggest_query_refinement(self, results: SearchResults) -> QuerySuggestion: ...
```

#### 2. Configuration-Driven Behaviour

All provider configurations, prompts, and parameters live in external files (YAML/TOML), not code.

#### 3. Stateful Iteration Tracking

Every search iteration, classification decision, and query modification is logged for reproducibility and analysis.

---

### Recommended Technology Stack

#### Core Framework

| Component | Recommendation | Rationale |
|-----------|----------------|-----------|
| **Language** | Python 3.11+ | Best ecosystem for scientific computing, all database wrappers available |
| **Package Manager** | `uv` or `poetry` | Dependency locking, reproducible environments |
| **Configuration** | `pydantic-settings` + YAML | Type-safe config with validation |
| **CLI Interface** | `typer` | Clean CLI with automatic help generation |
| **Task Orchestration** | `Prefect` or `Temporal` | Workflow management with retries, observability |

#### LLM Integration

| Component | Recommendation | Rationale |
|-----------|----------------|-----------|
| **LLM Abstraction** | `litellm` | Single interface to 100+ LLM providers; drop-in switching |
| **Prompt Management** | `promptic` or `jinja2` templates | Version-controlled prompts separate from code |
| **Structured Output** | `instructor` | Reliable JSON/Pydantic extraction from LLMs |
| **Local Models** | `ollama` + `litellm` | Run Llama/Mistral locally with same interface |

#### Database Clients

| Database | Package | Notes |
|----------|---------|-------|
| OpenAlex | `pyalex` | Official, well-maintained |
| PubMed | `metapub` or `biopython` | `metapub` is simpler; `biopython` more comprehensive |
| Scopus | `pybliometrics` | Mature, handles auth well |
| Semantic Scholar | `semanticscholar` | Unofficial but solid |
| Crossref | `habanero` | rOpenSci quality |
| Dimensions | `dimcli` | Official, requires auth |

#### Data & State Management

| Component | Recommendation | Rationale |
|-----------|----------------|-----------|
| **Local Database** | `SQLite` + `SQLModel` | Zero config, portable, SQL when needed |
| **Data Validation** | `pydantic` | Type safety, serialisation |
| **Data Processing** | `polars` | Faster than pandas, better memory |
| **Export Formats** | `RIS`, `BibTeX`, `CSV` via `rispy`, `bibtexparser` | Standard formats for reference managers |

#### Human-in-the-Loop Interface

| Component | Recommendation | Rationale |
|-----------|----------------|-----------|
| **Review Interface** | `Streamlit` or `Gradio` | Rapid UI development, good for data apps |
| **Notifications** | `apprise` | Multi-channel (email, Slack, etc.) |

#### Observability & Debugging

| Component | Recommendation | Rationale |
|-----------|----------------|-----------|
| **Logging** | `structlog` | Structured JSON logs, easy to parse |
| **LLM Tracing** | `langfuse` (self-hosted) or `Langsmith` | Track prompts, costs, latency |
| **Experiment Tracking** | `MLflow` or `DVC` | Version queries, track iterations |

---

### Detailed Module Design

#### Module 1: Configuration Management

```
config/
├── settings.yaml           # Global settings
├── databases/
│   ├── openalex.yaml
│   ├── pubmed.yaml
│   ├── scopus.yaml
│   └── _template.yaml
├── llms/
│   ├── claude.yaml
│   ├── gpt4.yaml
│   ├── local_llama.yaml
│   └── _template.yaml
├── prompts/
│   ├── classifier_system.jinja2
│   ├── classifier_user.jinja2
│   ├── query_optimiser_system.jinja2
│   └── query_optimiser_user.jinja2
└── projects/
    └── reproducibility_review/
        ├── criteria.yaml       # Inclusion/exclusion criteria
        ├── seed_queries.yaml   # Starting queries per database
        └── known_relevant.yaml # Gold standard papers
```

**Example `databases/openalex.yaml`:**

```yaml
name: openalex
display_name: OpenAlex
adapter_class: adapters.databases.OpenAlexAdapter
enabled: true

connection:
  base_url: https://api.openalex.org
  email: your.email@ncku.edu.tw  # For polite pool
  rate_limit: 10  # requests per second

capabilities:
  has_abstracts: true
  has_full_text: false
  has_citations: true
  has_references: true
  max_results_per_query: 10000
  supports_boolean: true
  supports_proximity: false

field_mapping:
  title: display_name
  abstract: abstract
  authors: authorships
  year: publication_year
  doi: doi
  
query_syntax:
  and_operator: ","
  or_operator: "|"
  not_operator: "!"
  phrase_delimiter: '"'
  wildcard: "*"
```

#### Module 2: Database Adapter Layer

```python
# src/adapters/databases/base.py

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Iterator

class Record(BaseModel):
    """Standardised record format across all databases"""
    id: str
    source_db: str
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    doi: str | None
    url: str | None
    raw_data: dict  # Original response preserved

class SearchResults(BaseModel):
    query: str
    database: str
    total_results: int
    records: list[Record]
    execution_time_ms: int
    cursor: str | None  # For pagination

class DatabaseAdapter(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.name = config["name"]
    
    @abstractmethod
    def search(self, query: str, limit: int = 100) -> SearchResults:
        """Execute search and return standardised results"""
        ...
    
    @abstractmethod
    def search_iter(self, query: str) -> Iterator[Record]:
        """Iterate through all results with automatic pagination"""
        ...
    
    @abstractmethod
    def translate_query(self, generic_query: "GenericQuery") -> str:
        """Convert generic query AST to database-specific syntax"""
        ...
    
    @abstractmethod
    def validate_query(self, query: str) -> tuple[bool, str | None]:
        """Check if query is syntactically valid for this database"""
        ...
    
    def get_capabilities(self) -> dict:
        return self.config.get("capabilities", {})
```

```python
# src/adapters/databases/openalex.py

import pyalex
from .base import DatabaseAdapter, Record, SearchResults

class OpenAlexAdapter(DatabaseAdapter):
    def __init__(self, config: dict):
        super().__init__(config)
        pyalex.config.email = config["connection"]["email"]
    
    def search(self, query: str, limit: int = 100) -> SearchResults:
        import time
        start = time.perf_counter()
        
        works = pyalex.Works().search(query).get(per_page=limit)
        
        records = [self._to_record(w) for w in works]
        
        return SearchResults(
            query=query,
            database=self.name,
            total_results=len(records),  # Would need separate count query
            records=records,
            execution_time_ms=int((time.perf_counter() - start) * 1000),
            cursor=None
        )
    
    def _to_record(self, work: dict) -> Record:
        return Record(
            id=work.get("id", ""),
            source_db=self.name,
            title=work.get("display_name", ""),
            abstract=work.get("abstract", None),
            authors=[a["author"]["display_name"] 
                     for a in work.get("authorships", [])],
            year=work.get("publication_year"),
            doi=work.get("doi"),
            url=work.get("id"),
            raw_data=work
        )
    
    def translate_query(self, generic_query: "GenericQuery") -> str:
        # Implementation of query translation
        ...
```

#### Module 3: LLM Adapter Layer

```python
# src/adapters/llms/base.py

from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum

class RelevanceClassification(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    UNCERTAIN = "uncertain"

class ClassificationResult(BaseModel):
    record_id: str
    classification: RelevanceClassification
    confidence: float  # 0-1
    reasoning: str
    processing_time_ms: int
    model_used: str
    prompt_tokens: int
    completion_tokens: int

class QuerySuggestion(BaseModel):
    original_query: str
    suggested_query: str
    rationale: str
    expected_impact: str  # "increase_precision" | "increase_recall" | "both"
    changes: list[dict]  # Detailed changes made

class LLMAdapter(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]
    
    @abstractmethod
    def classify_record(
        self, 
        record: Record, 
        criteria: dict,
        examples: list[tuple[Record, RelevanceClassification]] = None
    ) -> ClassificationResult:
        """Classify a single record against inclusion criteria"""
        ...
    
    @abstractmethod
    def classify_batch(
        self,
        records: list[Record],
        criteria: dict,
        examples: list[tuple[Record, RelevanceClassification]] = None
    ) -> list[ClassificationResult]:
        """Classify multiple records (may use batching optimisations)"""
        ...
    
    @abstractmethod
    def suggest_query_refinement(
        self,
        current_query: str,
        database_name: str,
        false_positives: list[Record],
        false_negatives: list[Record],
        uncertain: list[Record]
    ) -> QuerySuggestion:
        """Analyse results and suggest query improvements"""
        ...
```

```python
# src/adapters/llms/litellm_adapter.py

import litellm
import instructor
from jinja2 import Environment, FileSystemLoader
from .base import LLMAdapter, ClassificationResult, RelevanceClassification

class LiteLLMAdapter(LLMAdapter):
    """Universal adapter using litellm for any supported provider"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Configure litellm
        if api_key := config.get("api_key"):
            litellm.api_key = api_key
        if api_base := config.get("api_base"):
            litellm.api_base = api_base
        
        # Patch with instructor for structured outputs
        self.client = instructor.from_litellm(litellm.completion)
        
        # Load prompt templates
        self.jinja_env = Environment(
            loader=FileSystemLoader("config/prompts")
        )
    
    def classify_record(
        self,
        record: Record,
        criteria: dict,
        examples: list = None
    ) -> ClassificationResult:
        import time
        start = time.perf_counter()
        
        # Render prompts from templates
        system_prompt = self.jinja_env.get_template(
            "classifier_system.jinja2"
        ).render(criteria=criteria)
        
        user_prompt = self.jinja_env.get_template(
            "classifier_user.jinja2"
        ).render(record=record, examples=examples)
        
        # Call LLM with structured output
        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=ClassificationOutput,  # Pydantic model
            temperature=self.config.get("temperature", 0.1)
        )
        
        return ClassificationResult(
            record_id=record.id,
            classification=response.classification,
            confidence=response.confidence,
            reasoning=response.reasoning,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            model_used=self.model_name,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens
        )
```

#### Module 4: Orchestration Layer

```python
# src/orchestration/search_loop.py

from prefect import flow, task
from prefect.artifacts import create_table_artifact

@task(retries=3, retry_delay_seconds=60)
def execute_search(db_adapter: DatabaseAdapter, query: str) -> SearchResults:
    """Execute search with automatic retries"""
    return db_adapter.search(query)

@task
def classify_records(
    llm_adapter: LLMAdapter,
    records: list[Record],
    criteria: dict
) -> list[ClassificationResult]:
    """Classify all records, routing uncertain ones for human review"""
    return llm_adapter.classify_batch(records, criteria)

@task
def human_review_uncertain(
    uncertain_records: list[Record],
    classifications: list[ClassificationResult]
) -> list[ClassificationResult]:
    """Queue uncertain records for human review and wait"""
    # This would integrate with the Streamlit UI
    ...

@task
def generate_query_refinement(
    llm_adapter: LLMAdapter,
    current_query: str,
    database_name: str,
    results: ClassificationSummary
) -> QuerySuggestion:
    """Use LLM to suggest query improvements"""
    return llm_adapter.suggest_query_refinement(
        current_query=current_query,
        database_name=database_name,
        false_positives=results.false_positives,
        false_negatives=results.false_negatives,
        uncertain=results.uncertain
    )

@flow(name="search-optimisation-iteration")
def run_iteration(
    project_id: str,
    database_name: str,
    query: str,
    iteration: int
) -> IterationResult:
    """Single iteration of the search-classify-optimise loop"""
    
    # Load configuration
    config = load_project_config(project_id)
    db_adapter = get_database_adapter(database_name)
    classifier_llm = get_llm_adapter(config.classifier_llm)
    optimiser_llm = get_llm_adapter(config.optimiser_llm)
    
    # Execute search
    results = execute_search(db_adapter, query)
    
    # Classify results
    classifications = classify_records(
        classifier_llm, 
        results.records,
        config.criteria
    )
    
    # Route uncertain to human review
    uncertain = [c for c in classifications 
                 if c.classification == RelevanceClassification.UNCERTAIN]
    if uncertain:
        human_classifications = human_review_uncertain(
            [r for r in results.records if r.id in {c.record_id for c in uncertain}],
            uncertain
        )
        # Merge human decisions back
        classifications = merge_classifications(classifications, human_classifications)
    
    # Summarise results
    summary = summarise_classifications(classifications, results.records)
    
    # Generate query refinement suggestion
    suggestion = generate_query_refinement(
        optimiser_llm,
        query,
        database_name,
        summary
    )
    
    # Log iteration
    log_iteration(project_id, iteration, query, results, classifications, suggestion)
    
    # Create artifact for Prefect UI
    create_table_artifact(
        key=f"iteration-{iteration}-summary",
        table=summary.to_table(),
        description=f"Results for iteration {iteration}"
    )
    
    return IterationResult(
        iteration=iteration,
        query=query,
        total_results=results.total_results,
        relevant=summary.relevant_count,
        irrelevant=summary.irrelevant_count,
        uncertain=summary.uncertain_count,
        suggested_query=suggestion.suggested_query,
        suggestion_rationale=suggestion.rationale
    )

@flow(name="search-optimisation-loop")
def run_optimisation_loop(
    project_id: str,
    database_name: str,
    max_iterations: int = 5,
    stopping_threshold: float = 0.9  # Stop when precision > 90%
):
    """Run the full optimisation loop until convergence or max iterations"""
    
    config = load_project_config(project_id)
    query = config.seed_queries[database_name]
    
    for iteration in range(1, max_iterations + 1):
        result = run_iteration(project_id, database_name, query, iteration)
        
        # Check stopping criteria
        precision = result.relevant / (result.relevant + result.irrelevant)
        if precision >= stopping_threshold:
            logger.info(f"Reached precision {precision:.2%}, stopping")
            break
        
        # Human approval for query change
        if await_human_approval(result.suggested_query, result.suggestion_rationale):
            query = result.suggested_query
        else:
            logger.info("Human rejected suggestion, stopping loop")
            break
    
    return FinalReport(project_id, database_name, iterations=iteration)
```

#### Module 5: Human-in-the-Loop Interface

```python
# src/ui/review_app.py

import streamlit as st
from src.data.repository import get_pending_reviews, submit_review

st.set_page_config(page_title="Literature Review Assistant", layout="wide")

def main():
    st.title("🔬 Literature Review - Human Classification")
    
    # Sidebar: Project and criteria
    with st.sidebar:
        project = st.selectbox("Project", get_projects())
        st.markdown("### Inclusion Criteria")
        criteria = load_criteria(project)
        for c in criteria.inclusion:
            st.markdown(f"✅ {c}")
        st.markdown("### Exclusion Criteria")
        for c in criteria.exclusion:
            st.markdown(f"❌ {c}")
    
    # Main area: Records to review
    pending = get_pending_reviews(project)
    
    if not pending:
        st.success("No records pending review!")
        return
    
    st.info(f"{len(pending)} records awaiting your decision")
    
    for record, llm_assessment in pending:
        with st.expander(f"📄 {record.title[:80]}...", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Year:** {record.year} | **DOI:** {record.doi}")
                st.markdown(f"**Authors:** {', '.join(record.authors[:3])}...")
                st.markdown("**Abstract:**")
                st.write(record.abstract or "*No abstract available*")
            
            with col2:
                st.markdown("**LLM Assessment:**")
                st.write(f"Classification: `{llm_assessment.classification}`")
                st.write(f"Confidence: {llm_assessment.confidence:.0%}")
                st.write(f"Reasoning: {llm_assessment.reasoning}")
            
            # Human decision buttons
            decision_col1, decision_col2, decision_col3, decision_col4 = st.columns(4)
            
            with decision_col1:
                if st.button("✅ Relevant", key=f"rel_{record.id}"):
                    submit_review(record.id, "relevant", project)
                    st.rerun()
            
            with decision_col2:
                if st.button("❌ Irrelevant", key=f"irr_{record.id}"):
                    submit_review(record.id, "irrelevant", project)
                    st.rerun()
            
            with decision_col3:
                if st.button("🔍 Need Full Text", key=f"ft_{record.id}"):
                    submit_review(record.id, "need_fulltext", project)
                    st.rerun()
            
            with decision_col4:
                if st.button("⏭️ Skip", key=f"skip_{record.id}"):
                    st.rerun()

if __name__ == "__main__":
    main()
```

---

### Project Structure

```
literature-search-optimiser/
├── pyproject.toml
├── README.md
├── config/
│   ├── settings.yaml
│   ├── databases/
│   ├── llms/
│   ├── prompts/
│   └── projects/
├── src/
│   ├── __init__.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── databases/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── openalex.py
│   │   │   ├── pubmed.py
│   │   │   ├── scopus.py
│   │   │   └── semantic_scholar.py
│   │   └── llms/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       └── litellm_adapter.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── query.py          # Generic query representation
│   │   ├── criteria.py       # Inclusion/exclusion criteria
│   │   └── models.py         # Shared Pydantic models
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── search_loop.py    # Prefect flows
│   │   └── workflows.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── repository.py     # Data access layer
│   │   ├── migrations/       # Database migrations
│   │   └── export.py         # RIS/BibTeX export
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── review_app.py     # Streamlit human review
│   │   └── dashboard.py      # Progress dashboard
│   └── cli/
│       ├── __init__.py
│       └── main.py           # Typer CLI
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
│   └── setup_databases.py
└── docker-compose.yml        # For Prefect server, optional DB
```

---

### Getting Started Implementation

#### Step 1: Environment Setup

```toml
# pyproject.toml
[project]
name = "literature-search-optimiser"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # Core
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "typer>=0.9",
    "structlog>=23.0",
    "pyyaml>=6.0",
    "jinja2>=3.0",
    
    # LLM
    "litellm>=1.0",
    "instructor>=1.0",
    
    # Databases
    "pyalex>=0.13",
    "pybliometrics>=3.5",
    "metapub>=0.5",
    "semanticscholar>=0.6",
    "habanero>=1.2",
    
    # Data
    "sqlmodel>=0.0.14",
    "polars>=0.20",
    "rispy>=0.7",
    "bibtexparser>=1.4",
    
    # Orchestration
    "prefect>=2.14",
    
    # UI
    "streamlit>=1.30",
    
    # Observability
    "langfuse>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.1",
    "mypy>=1.8",
]
```

#### Step 2: Minimal Working Example

```python
# scripts/minimal_example.py
"""
Minimal example: Search OpenAlex, classify with Claude, suggest refinement
"""

import os
from dotenv import load_dotenv
load_dotenv()

# 1. Setup adapters
from src.adapters.databases.openalex import OpenAlexAdapter
from src.adapters.llms.litellm_adapter import LiteLLMAdapter

db = OpenAlexAdapter({
    "name": "openalex",
    "connection": {"email": os.getenv("EMAIL")}
})

llm = LiteLLMAdapter({
    "model": "claude-sonnet-4-20250514",
    "api_key": os.getenv("ANTHROPIC_API_KEY")
})

# 2. Define criteria
criteria = {
    "inclusion": [
        "Reports an attempt to reproduce/replicate a previous study",
        "Describes methods for assessing reproducibility",
        "Is a review or meta-analysis of reproducibility research"
    ],
    "exclusion": [
        "Merely mentions reproducibility without it being a focus",
        "Is about computational/software reproducibility only",
        "Is a protocol without results"
    ]
}

# 3. Initial search
query = "reproducibility AND (replication study OR meta-research)"
results = db.search(query, limit=20)

print(f"Found {results.total_results} results")

# 4. Classify
classifications = []
for record in results.records[:5]:  # Just first 5 for demo
    result = llm.classify_record(record, criteria)
    classifications.append(result)
    print(f"\n{record.title[:60]}...")
    print(f"  -> {result.classification} ({result.confidence:.0%})")
    print(f"  -> {result.reasoning[:100]}...")

# 5. Get query suggestion
from collections import defaultdict
by_class = defaultdict(list)
for c, r in zip(classifications, results.records[:5]):
    by_class[c.classification].append(r)

suggestion = llm.suggest_query_refinement(
    current_query=query,
    database_name="openalex",
    false_positives=by_class["irrelevant"],
    false_negatives=[],  # Would need known relevant not found
    uncertain=by_class["uncertain"]
)

print(f"\n\nSuggested refined query:\n{suggestion.suggested_query}")
print(f"\nRationale: {suggestion.rationale}")
```

---

### Key Design Decisions Explained

#### Why `litellm` for LLM abstraction?

- Single interface to Claude, GPT-4, Gemini, local models via Ollama
- Swap models by changing one config value
- Built-in rate limiting, retries, fallbacks
- Cost tracking across providers
- Works with `instructor` for structured outputs

#### Why `Prefect` for orchestration?

- Visual workflow monitoring
- Automatic retries with backoff
- Pause workflows for human input
- Caching of expensive operations
- Easy to run locally or deploy

#### Why `SQLite` + `SQLModel`?

- Zero infrastructure to start
- Portable (single file, can share)
- SQLModel gives you Pydantic + SQLAlchemy in one
- Easy to migrate to PostgreSQL later if needed

#### Why separate classifier and optimiser LLMs?

- Can use cheaper/faster model for classification (high volume)
- Use more capable model for query optimisation (requires reasoning)
- Can test different models independently
- Example: Claude Haiku for classification, Claude Sonnet for optimisation

---

### Maintenance and Stability Considerations

#### Adding a New Database

1. Create `config/databases/newdb.yaml` (copy from template)
2. Create `src/adapters/databases/newdb.py` implementing `DatabaseAdapter`
3. Register in adapter factory
4. Add to project's `seed_queries.yaml`

#### Adding a New LLM Provider

1. Create `config/llms/newllm.yaml`
2. If supported by `litellm`, just configure—no code changes
3. If not, extend `LiteLLMAdapter` or create new adapter

#### Updating Prompts

1. Edit files in `config/prompts/`
2. No code changes needed
3. Version control tracks changes
4. Can A/B test prompts via config

#### Ensuring Reproducibility

- All iterations logged with: query, results, classifications, model used, timestamps
- Prompt templates versioned in git
- Random seeds configurable
- Export full audit trail





---

## Implementation--Version 0.1.0 (Gemini 3.0 Pro)

My credits in Claude ended and it is good practice to have multiple LLMs involved in design, so I switched to Gemini 3.0 Pro

**Prompt:**
I am a researcher conducting a literature review and need help to optimise academic reference database queries based on automatic analysis of the results. I want to do this in a closed loop using an LLM to analyse the hits and classify each as relevant or irrelevant, asking for human input when not able to determine if a hit is relevant or not, and then another LLM to optimse the query to reduce the number of false positive and false negative hits. My first literature review will focus on ways to evaluate reproducibility of research. More precisely, I want to perform the following search:
1. reports of attempts to reproduce original studies,
2. ways to evaluate reproducibility of research from written descriptions
3. assessments of ways to evaluate reproducibility of research from written descriptions, and 
4. review and meta-research articles that synthesise ways to evaluate reproducibility or attempts to reproduce studies.

I have made the following system design that I now need help implementing. In the first iteration, I want to use Gemini 2.5 through API as my LLM and optimise the query for OpenAlex. Tell me how to setup and structure a GitHub repository for containing all code and run it on MacOS or Linux.

A copy of Modular System Design for LLM-Assisted Literature Search Optimisation from Opus 4.5

**Answer:**
This is a sophisticated system design. To implement this for your first iteration (OpenAlex + Gemini + Human-in-the-loop), we need to translate your architectural blueprint into a concrete, runnable file structure.

Since "Gemini 2.5" is not a standard release version yet (current SOTA is 1.5 Pro or 2.0 Flash Experimental), I will configure the system to use `gemini-1.5-pro`, which is excellent for complex reasoning, but keep it swappable via configuration.

Here is the complete setup guide to initialize your GitHub repository on MacOS/Linux.

### 1. Repository Initialization & Structure

First, open your terminal. We will create the project structure and set up the virtual environment using `uv` (as recommended in your stack) or standard `venv`.

```bash
# Create project directory
mkdir reproducibility-review
cd reproducibility-review

# Initialize git
git init

# Create directory structure
mkdir -p config/databases config/llms config/prompts config/projects
mkdir -p src/adapters/databases src/adapters/llms src/core src/orchestration src/ui src/data
mkdir -p logs
touch .env .gitignore README.md

```

### 2. Dependency Management (`pyproject.toml`)

Create a `pyproject.toml` file in the root. This defines your environment.

```toml
[project]
name = "reproducibility-review"
version = "0.1.0"
description = "LLM-assisted literature search optimisation"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "typer>=0.9",
    "pyyaml>=6.0",
    "jinja2>=3.0",
    "litellm>=1.0",      # LLM Abstraction
    "instructor>=1.0",   # Structured Output
    "pyalex>=0.13",      # OpenAlex Wrapper
    "rich>=13.0",        # Pretty terminal output
    "streamlit>=1.30",   # UI
    "pandas>=2.0",
    "python-dotenv>=1.0"
]

[tool.setuptools]
packages = ["src"]

```

**Install dependencies:**

```bash
# Using standard pip (easiest for setup)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

```

### 3. Configuration Setup

We need to define your specific search criteria and connection settings.

**Create `.env` file** (Add to `.gitignore`):

```bash
# Get this from Google AI Studio
GEMINI_API_KEY=your_gemini_api_key_here 
# Optional: Your email specifically for the OpenAlex "Polite Pool" (faster access)
OPENALEX_EMAIL=your_email@university.edu 

```

**Create `config/settings.yaml**`:

```yaml
project:
  name: "Reproducibility Evaluation Review"
  data_dir: "./data"

llm:
  model: "gemini/gemini-1.5-pro-latest" # LiteLLM format
  temperature: 0.1

search:
  initial_query: 'reproducibility AND ("replication study" OR "meta-research")'
  max_results_per_iter: 20

criteria:
  inclusion:
    - "Reports of actual attempts to reproduce/replicate original studies."
    - "Methodologies or frameworks specifically for evaluating reproducibility from written descriptions."
    - "Assessments/Audits of how well reproducibility can be evaluated."
    - "Systematic reviews or meta-research synthesizing reproducibility evaluations."
  exclusion:
    - "Studies that merely mention reproducibility as a keyword without focus."
    - "Purely computational/software reproducibility (unless discussing general evaluation methods)."
    - "Statistical protocols without empirical results or synthesis."

```

### 4. Core Implementation

Here is the Python code to make the loop work.

#### A. Data Models (`src/core/models.py`)

This ensures type safety across the application.

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Record(BaseModel):
    id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    year: Optional[int] = None
    doi: Optional[str] = None
    
    def to_text(self):
        return f"Title: {self.title}\nAbstract: {self.abstract or 'No Abstract'}\nYear: {self.year}"

class Classification(BaseModel):
    relevance: Literal["relevant", "irrelevant", "uncertain"]
    confidence: float
    reasoning: str

class QuerySuggestion(BaseModel):
    critique: str = Field(..., description="Analysis of why false positives occurred")
    new_query: str = Field(..., description="The optimised query string for OpenAlex")
    expected_improvement: str

```

#### B. The OpenAlex Adapter (`src/adapters/databases/openalex_adapter.py`)

```python
import pyalex
from src.core.models import Record
import os

class OpenAlexAdapter:
    def __init__(self):
        email = os.getenv("OPENALEX_EMAIL")
        if email:
            pyalex.config.email = email

    def search(self, query: str, limit: int = 20) -> List[Record]:
        print(f"🔎 Searching OpenAlex for: {query}")
        try:
            # OpenAlex boolean search works best with search= parameter for keywords
            results = pyalex.Works().search(query).get(per_page=limit)
            
            records = []
            for w in results:
                # Handle missing abstracts safely
                abstract = w.get("abstract_inverted_index")
                abstract_text = None
                if abstract:
                    # Reconstruct abstract from inverted index
                    try:
                        words = sorted([(pos, word) for word, positions in abstract.items() for pos in positions])
                        abstract_text = " ".join(word for _, word in words)
                    except:
                        abstract_text = "Error parsing abstract"

                records.append(Record(
                    id=w.get("id"),
                    title=w.get("display_name", "No Title"),
                    abstract=abstract_text,
                    authors=[a['author']['display_name'] for a in w.get('authorships', [])],
                    year=w.get("publication_year"),
                    doi=w.get("doi")
                ))
            return records
        except Exception as e:
            print(f"Error querying OpenAlex: {e}")
            return []

```

#### C. The LLM Adapter (`src/adapters/llms/gemini_adapter.py`)

This uses `instructor` to force Gemini to return valid JSON fitting our Pydantic models.

```python
import instructor
import litellm
import os
from src.core.models import Record, Classification, QuerySuggestion

class GeminiAdapter:
    def __init__(self, model_name: str = "gemini/gemini-1.5-pro-latest"):
        self.model_name = model_name
        # Instructor patches the litellm client for structured output
        self.client = instructor.from_litellm(litellm.completion)
        self.api_key = os.getenv("GEMINI_API_KEY")

    def classify(self, record: Record, criteria: dict) -> Classification:
        prompt = f"""
        Analyze the following academic paper against the research criteria.
        
        PAPER:
        {record.to_text()}
        
        INCLUSION CRITERIA:
        {criteria['inclusion']}
        
        EXCLUSION CRITERIA:
        {criteria['exclusion']}
        
        Task: Classify as 'relevant', 'irrelevant', or 'uncertain'.
        Provide a confidence score (0.0 to 1.0) and brief reasoning.
        """
        
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_model=Classification,
            api_key=self.api_key
        )

    def optimize_query(self, current_query: str, false_positives: list[Record], false_negatives: list[str]) -> QuerySuggestion:
        # Note: False negatives are hard to know in a blind search, 
        # so we focus heavily on eliminating false positives (irrelevant items) 
        # while maintaining keywords from the original intent.
        
        fp_text = "\n".join([f"- {r.title}" for r in false_positives[:5]])
        
        prompt = f"""
        I am conducting a systematic review on reproducibility evaluation.
        
        CURRENT QUERY: {current_query}
        
        PROBLEM: The query returned these IRRELEVANT papers (False Positives):
        {fp_text}
        
        TASK: 
        1. Analyze why these papers were caught (e.g., polysemy of 'reproducibility', or 'evaluation' in wrong context).
        2. Construct a new, boolean OpenAlex-compatible query string to exclude these types of papers while keeping relevant ones.
        3. Explain your logic.
        """
        
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_model=QuerySuggestion,
            api_key=self.api_key
        )

```

#### D. The Orchestrator (`main.py`)

This ties it all together in a loop.

```python
import yaml
import asyncio
from rich.console import Console
from rich.prompt import Prompt
from src.adapters.databases.openalex_adapter import OpenAlexAdapter
from src.adapters.llms.gemini_adapter import GeminiAdapter
from src.core.models import Record

console = Console()

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def human_review(record: Record, llm_reason: str):
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
    
    for iteration in range(1, 4): # Run 3 iterations for demo
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
                    
                    # 3. Human in the Loop
                    if result.relevance == "uncertain":
                        final_decision = human_review(record, result.reasoning)
                    
                    if final_decision == "relevant":
                        relevant_count += 1
                        console.print(f"[blue]Relevant:[/blue] {record.title[:50]}...")
                    elif final_decision == "irrelevant":
                        irrelevant_records.append(record)
                        # console.print(f"[dim]Irrelevant: {record.title[:50]}...[/dim]")
                        
                except Exception as e:
                    console.print(f"Error classifying {record.id}: {e}")

        # Summary
        precision = relevant_count / len(records) if records else 0
        console.print(f"\nIteration Summary: Precision: {precision:.1%}")
        
        if precision > 0.8:
            console.print("[bold green]Precision goal met![/bold green]")
            break
            
        # 4. Optimise
        if irrelevant_records:
            console.print("\n[bold purple]Optimising Query...[/bold purple]")
            suggestion = llm.optimize_query(current_query, irrelevant_records, [])
            console.print(f"Critique: {suggestion.critique}")
            console.print(f"New Query: {suggestion.new_query}")
            current_query = suggestion.new_query
        else:
            console.print("No false positives to optimise against.")
            break

if __name__ == "__main__":
    main()

```

### 5. Running the System

1. **Environment Variables:** Ensure `GEMINI_API_KEY` is in your `.env`.
2. **Execute:**
```bash
python main.py

```


### How this meets your requirements:

1. **Closed Loop:** The `main.py` script feeds the "Irrelevant" papers back into Gemini to ask *why* they were retrieved and generates a better query.
2. **Modular:** The database and LLM logic are in separate classes (`OpenAlexAdapter`, `GeminiAdapter`). You can easily swap Gemini for GPT-4 by creating a `GPTAdapter` and changing one line in `main.py`.
3. **Human-in-the-Loop:** The script pauses when the LLM is "uncertain" and asks you via the terminal to decide. (Ideally, this moves to the Streamlit UI for larger datasets).
4. **Reproducibility:** All configuration is in `settings.yaml`, not hardcoded.

The actual first implementation is based on this output, but correcting repository names and asking Gemini to output it in a README structure, which surprisingly resulted in shorter code.






---

## Implementation--Version 0.2.0 (Gemini 3.0 Pro)


**Prompt:**
Add code to use Scopus with institutional API key and WOS with Starter API key.

**Answer:**

This update adds support for **Scopus** (via `pybliometrics` with Institutional Token support) and **Web of Science** (via the Starter API).

### 1. Update Dependencies

Add `pybliometrics` and `requests` to your `pyproject.toml`.

```toml
dependencies = [
    # ... existing dependencies ...
    "pybliometrics>=3.5", # Scopus Wrapper
    "requests>=2.31",     # For Web of Science API
]

```

### 2. Update Configuration (`.env`)

Add your new API keys to your environment file.

```ini
# ... existing keys ...
SCOPUS_API_KEY=your_scopus_key
SCOPUS_INST_TOKEN=your_institutional_token_if_needed
WOS_STARTER_API_KEY=your_wos_starter_key

```

### 3. Create Scopus Adapter

Create `src/adapters/databases/scopus_adapter.py`. This adapter handles the automatic configuration of `pybliometrics` so you don't have to interact with the terminal prompts on the first run.

```python
import os
import pybliometrics
from pybliometrics.scopus import ScopusSearch
from src.core.models import Record
from typing import List

class ScopusAdapter:
    def __init__(self):
        self.api_key = os.getenv("SCOPUS_API_KEY")
        self.inst_token = os.getenv("SCOPUS_INST_TOKEN")
        self._ensure_config()

    def _ensure_config(self):
        """
        Check if pybliometrics is configured. If not, create config programmatically
        to avoid interactive prompts blocking execution.
        """
        if not self.api_key:
            print("⚠️ SCOPUS_API_KEY not found in .env. Scopus search may fail.")
            return

        # Check if config exists, if not, create it
        try:
            # We trigger a lightweight check or create config explicitly
            if not pybliometrics.scopus.config['Authentication']['APIKey']:
                raise KeyError("Key missing")
        except (KeyError, ImportError, AttributeError):
            print("⚙️ Configuring Scopus for first-time use...")
            pybliometrics.scopus.utils.create_config(
                keys=[self.api_key],
                inst_token=self.inst_token
            )

    def search(self, query: str, limit: int = 20) -> List[Record]:
        print(f"🔎 Searching Scopus for: {query}")
        try:
            # Scopus API separates boolean operators with AND/OR, similar to standard syntax
            s = ScopusSearch(query, count=limit, download=True)
            
            records = []
            if s.results:
                for doc in s.results:
                    # Note: Scopus Abstract retrieval often requires the Abstract API, 
                    # but ScopusSearch result objects often contain the description/abstract
                    # if the subscriber level allows it.
                    records.append(Record(
                        id=doc.eid,
                        title=doc.title,
                        abstract=doc.description if doc.description else "Abstract not available via Search API",
                        authors=[doc.author_names] if doc.author_names else [],
                        year=int(doc.coverDate[:4]) if doc.coverDate else None,
                        doi=doc.doi
                    ))
            return records
        except Exception as e:
            print(f"Error querying Scopus: {e}")
            return []

```

### 4. Create Web of Science Adapter

Create `src/adapters/databases/wos_adapter.py`. This uses the "Starter API" (formerly Lite) which is commonly available.

**Note:** The Starter API often limits access to abstracts. The code below handles this gracefully.

```python
import os
import requests
from src.core.models import Record
from typing import List

class WosAdapter:
    def __init__(self):
        self.api_key = os.getenv("WOS_STARTER_API_KEY")
        self.base_url = "https://api.clarivate.com/apis/wos-starter/v1/documents"

    def search(self, query: str, limit: int = 20) -> List[Record]:
        print(f"🔎 Searching Web of Science (Starter) for: {query}")
        
        if not self.api_key:
            print("⚠️ WOS_STARTER_API_KEY not found. Skipping.")
            return []

        headers = {
            "X-ApiKey": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Starter API uses specific field tags (TS=Topic)
        # We assume the incoming query is a standard boolean string. 
        # Ideally, we wrap it in parentheses and prepend TS=, or pass it as 'q' param directly.
        params = {
            "q": query, 
            "limit": limit,
            "page": 1
        }

        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"WoS API Error {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            records = []
            
            for doc in data.get('hits', []):
                # WoS Starter API often does NOT return the full abstract text in the 'hits'
                # It returns metadata. We map what we can.
                
                # Extract year
                source = doc.get('source', {})
                pub_year = source.get('publishYear')
                
                # Authors
                authors = [
                    f"{a.get('displayName', a.get('name', 'Unknown'))}" 
                    for a in doc.get('names', {}).get('authors', [])
                ]

                records.append(Record(
                    id=doc.get('uid', ''),
                    title=doc.get('title', {}).get('title', ['No Title'])[0],
                    # Starter API limitation: Abstract is often missing
                    abstract=None, 
                    authors=authors,
                    year=int(pub_year) if pub_year else None,
                    doi=doc.get('identifiers', {}).get('doi', '')
                ))
            
            return records
            
        except Exception as e:
            print(f"Error querying Web of Science: {e}")
            return []

```

### 5. Update Main Loop (`main.py`)

Update `main.py` to allow switching between databases.

```python
# ... imports ...
from src.adapters.databases.scopus_adapter import ScopusAdapter
from src.adapters.databases.wos_adapter import WosAdapter

# ... inside main() ...

    # Allow user or config to select database
    db_choice = config.get('search', {}).get('database', 'openalex').lower()
    
    if db_choice == 'scopus':
        db = ScopusAdapter()
    elif db_choice == 'wos':
        db = WosAdapter()
    else:
        db = OpenAlexAdapter()

    # ... continue with search loop ...

```


