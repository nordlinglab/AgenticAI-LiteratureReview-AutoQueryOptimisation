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
