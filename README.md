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

```


### Usage

Run the main orchestration loop:

```bash
python main.py

```

---

## Project Plan and Creation

Documentation of the creation of this project with the help of AI is in [https://github.com/nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation/blob/main/README_PROJECT_CREATION.md](https://github.com/nordlinglab/AgenticAI-LiteratureReview-AutoQueryOptimisation/blob/main/README_PROJECT_CREATION.md).


