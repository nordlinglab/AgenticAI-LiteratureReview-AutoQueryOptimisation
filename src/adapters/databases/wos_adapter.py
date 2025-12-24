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
