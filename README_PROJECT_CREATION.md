# AgenticAI Literature Review & Query Optimisation

**Repository:** [nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation](https://github.com/nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation)  

**Description:** AgenticAI for systematic literature search using an LLM to optimise the search query and analyse the search result.

This project implements a **closed-loop literature search agent**. It performs three key functions iteratively:
1.  **Search:** Queries academic databases (OpenAlex) for papers.
2.  **Classify:** Uses an LLM (Gemini via LiteLLM) to read titles/abstracts and classify papers as *Relevant*, *Irrelevant*, or *Uncertain*.
3.  **Optimise:** Analyses "Irrelevant" results (false positives) to refine the Boolean search string, reducing noise in the next iteration.

**Author:** Torbjörn E.M. Nordling
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




