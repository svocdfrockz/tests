"""
Azure GPT-4 Hallucination Detection and Mitigation Framework
"""

import openai
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

@dataclass
class HallucinationCheck:
    """Result of hallucination detection check"""
    is_hallucination: bool
    confidence: float
    reason: str
    category: str  # 'factual', 'source', 'contextual'

class HallucinationDetector:
    """Detects potential hallucinations in GPT-4 responses"""
    
    def __init__(self, azure_endpoint: str, api_key: str):
        self.client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01"
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize text analytics for sentiment and key phrase extraction
        self.text_analytics = TextAnalyticsClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(api_key)
        )
        
        # Common hallucination patterns
        self.hallucination_patterns = [
            r"according to a study by.*(?:university|institute|research)",
            r"research shows that.*(?:\d{1,3}%|\d+\.\d+%)",
            r"experts say that",
            r"it is well known that",
            r"studies have shown",
            r"according to.*(?:published|journal|paper)",
        ]
    
    def check_factual_consistency(self, response: str, context: str = "") -> HallucinationCheck:
        """Check for factual inconsistencies in the response"""
        
        # Look for specific claims that can be fact-checked
        suspicious_patterns = [
            r"\d{4}.*(?:discovered|invented|founded)",  # Date claims
            r"\d+(?:\.\d+)?%.*(?:of|percent)",  # Percentage claims
            r"(?:costs?|prices?|worth).*\$\d+",  # Price claims
            r"located in.*(?:city|country|state)",  # Location claims
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return HallucinationCheck(
                    is_hallucination=True,
                    confidence=0.7,
                    reason=f"Contains specific factual claim matching pattern: {pattern}",
                    category="factual"
                )
        
        return HallucinationCheck(
            is_hallucination=False,
            confidence=0.8,
            reason="No suspicious factual patterns detected",
            category="factual"
        )
    
    def check_source_citations(self, response: str) -> HallucinationCheck:
        """Check for fabricated sources or citations"""
        
        citation_patterns = [
            r"(?:according to|citing|references?).*(?:study|paper|research|journal)",
            r"(?:dr\.|professor|researcher).*(?:said|stated|found|concluded)",
            r"(?:published in|journal of|proceedings of)",
            r"(?:university of|institute of|center for)",
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return HallucinationCheck(
                    is_hallucination=True,
                    confidence=0.8,
                    reason=f"Contains potential fabricated citation: {pattern}",
                    category="source"
                )
        
        return HallucinationCheck(
            is_hallucination=False,
            confidence=0.9,
            reason="No suspicious citation patterns detected",
            category="source"
        )
    
    def check_contextual_relevance(self, response: str, query: str) -> HallucinationCheck:
        """Check if response is contextually relevant to the query"""
        
        # Extract key phrases from both query and response
        try:
            query_phrases = self.text_analytics.extract_key_phrases([query])[0].key_phrases
            response_phrases = self.text_analytics.extract_key_phrases([response])[0].key_phrases
            
            # Calculate overlap
            overlap = len(set(query_phrases) & set(response_phrases))
            total_query_phrases = len(query_phrases)
            
            if total_query_phrases == 0:
                relevance_score = 0.5
            else:
                relevance_score = overlap / total_query_phrases
            
            if relevance_score < 0.3:
                return HallucinationCheck(
                    is_hallucination=True,
                    confidence=0.7,
                    reason=f"Low contextual relevance: {relevance_score:.2f}",
                    category="contextual"
                )
                
        except Exception as e:
            self.logger.warning(f"Error in contextual analysis: {e}")
        
        return HallucinationCheck(
            is_hallucination=False,
            confidence=0.8,
            reason="Response appears contextually relevant",
            category="contextual"
        )
    
    def comprehensive_check(self, response: str, query: str = "", context: str = "") -> List[HallucinationCheck]:
        """Run all hallucination checks"""
        
        checks = [
            self.check_factual_consistency(response, context),
            self.check_source_citations(response),
            self.check_contextual_relevance(response, query)
        ]
        
        return checks

class SafeGPTClient:
    """Wrapper around Azure GPT-4 with hallucination mitigation"""
    
    def __init__(self, azure_endpoint: str, api_key: str, deployment_name: str):
        self.client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01"
        )
        self.deployment_name = deployment_name
        self.detector = HallucinationDetector(azure_endpoint, api_key)
        self.logger = logging.getLogger(__name__)
    
    def safe_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        check_hallucinations: bool = True
    ) -> Dict:
        """Generate completion with hallucination checking"""
        
        # Add system message for hallucination prevention
        safe_messages = messages.copy()
        if not any(msg.get("role") == "system" for msg in safe_messages):
            safe_messages.insert(0, {
                "role": "system",
                "content": """You are a helpful assistant. Follow these guidelines:
1. Only provide information you are confident about
2. Clearly state when you are uncertain about something
3. Do not fabricate sources, citations, or specific statistics
4. If you don't know something, say so explicitly
5. Ground your responses in the provided context when available"""
            })
        
        try:
            # Generate response with conservative settings
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=safe_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.8  # Reduce randomness
            )
            
            content = response.choices[0].message.content
            
            # Check for hallucinations if enabled
            hallucination_checks = []
            if check_hallucinations:
                user_query = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_query = msg.get("content", "")
                        break
                
                hallucination_checks = self.detector.comprehensive_check(
                    content, user_query
                )
            
            # Log potential issues
            for check in hallucination_checks:
                if check.is_hallucination:
                    self.logger.warning(
                        f"Potential {check.category} hallucination detected: {check.reason}"
                    )
            
            return {
                "content": content,
                "usage": response.usage.model_dump(),
                "hallucination_checks": [
                    {
                        "category": check.category,
                        "is_hallucination": check.is_hallucination,
                        "confidence": check.confidence,
                        "reason": check.reason
                    }
                    for check in hallucination_checks
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in safe completion: {e}")
            raise

def create_rag_prompt(query: str, context_documents: List[str]) -> str:
    """Create a RAG prompt to ground the response in provided context"""
    
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_documents)])
    
    return f"""Based on the following context documents, please answer the question. Only use information that is explicitly stated in the provided documents. If the answer cannot be found in the documents, please state that clearly.

Context:
{context}

Question: {query}

Answer based only on the provided context:"""

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    AZURE_ENDPOINT = "your-azure-endpoint"
    API_KEY = "your-api-key"
    DEPLOYMENT_NAME = "gpt-4"
    
    # Initialize safe client
    safe_client = SafeGPTClient(AZURE_ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    
    # Example 1: Basic safe completion
    messages = [
        {"role": "user", "content": "What are the benefits of regular exercise for blood pressure?"}
    ]
    
    try:
        result = safe_client.safe_completion(messages)
        print("Response:", result["content"])
        print("Hallucination checks:", result["hallucination_checks"])
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: RAG-based query
    context_docs = [
        "Regular physical activity helps lower blood pressure by strengthening the heart muscle, which allows it to pump blood more efficiently with less effort."
    ]
    
    rag_prompt = create_rag_prompt(
        "How does exercise affect blood pressure?",
        context_docs
    )
    
    rag_messages = [{"role": "user", "content": rag_prompt}]
    
    try:
        rag_result = safe_client.safe_completion(rag_messages)
        print("RAG Response:", rag_result["content"])
    except Exception as e:
        print(f"RAG Error: {e}")