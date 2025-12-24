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

## Quick Start Guide (by Gemini 3.0 Pro)

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
SCOPUS_API_KEY=your_scopus_key
SCOPUS_INST_TOKEN=your_institutional_token_if_needed
WOS_STARTER_API_KEY=your_wos_starter_key

```


### Usage

Run a specific search project by name (corresponding to the file in `config/projects/`):

```bash
# Run the "reproduction attempts" search
python main.py run 01_attempts

# Run the "evaluation methods" search
python main.py run 02_methods

```

---

## Project Structure (by Gemini 3.0 Pro)

The structure separates global settings from specific research questions.

```text
.
├── config/
│   ├── settings.yaml              # Global defaults (LLM models, API settings)
│   ├── projects/                  # Project definitions
│   │   ├── 01_attempts.yaml
│   │   ├── 02_methods.yaml
│   │   ├── 03_assessments.yaml
│   │   └── 04_reviews.yaml
│   ...
│
├── src/
│   ├── adapters/                  # Database & LLM connections
│   ├── core/                      # Data models & Config loader
│   ├── orchestration/             # Main logic
│   ...
│
├── main.py                        # CLI Entry point
├── pyproject.toml
└── README.md

```

## Configuration Files  (by Gemini 3.0 Pro)

### Project Definitions (`config/projects/*.yaml`)

One file for each search project in `config/projects/`.

Current projects:

1. `config/projects/01_attempts.yaml`
2. `config/projects/02_methods.yaml`
3. `config/projects/03_assessments.yaml`
4. `config/projects/04_reviews.yaml`

**Example: `config/projects/01_attempts.yaml**`

```yaml
name: "Reports of Reproduction Attempts"
description: "Search for empirical studies that attempt to reproduce previous research."

search:
  initial_query: 'reproducibility AND ("replication study" OR "reproduction attempt" OR "failed to reproduce")'

criteria:
  inclusion:
    - "Primary studies reporting an actual attempt to reproduce or replicate an original study."
    - "Reports of successful or failed replications."
  exclusion:
    - "Theoretical discussions about reproducibility without new empirical data."
    - "Studies using 'replication' in a biological context (e.g., DNA replication)."

```


### Global Settings (`config/settings.yaml`)

Keeps shared infrastructure settings.

```yaml
system:
  # Default LLM (Can be overridden per project if needed)
  llm_model: "gemini/gemini-1.5-pro-latest"
  llm_temperature: 0.1

defaults:
  database: "openalex" # openalex, scopus, wos
  max_results_per_iter: 20
  max_iterations: 3
  precision_threshold: 0.95

```

---

## Project Plan and Creation

Documentation of the creation of this project with the help of AI is in [https://github.com/nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation/blob/main/README_PROJECT_CREATION.md](https://github.com/nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation/blob/main/README_PROJECT_CREATION.md).


