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

