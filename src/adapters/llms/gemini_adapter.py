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
