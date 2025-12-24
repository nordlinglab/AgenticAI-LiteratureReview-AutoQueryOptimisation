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

