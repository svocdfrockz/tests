"""
Azure GPT-4 Hallucination Mitigation Handler

This module provides strategies and implementations to reduce and handle hallucinations
in Azure GPT-4 model responses.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from openai import AzureOpenAI
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationStrategy(Enum):
    """Strategies for handling hallucinations"""
    PROMPT_ENGINEERING = "prompt_engineering"
    TEMPERATURE_CONTROL = "temperature_control"
    FACT_CHECKING = "fact_checking"
    CONFIDENCE_SCORING = "confidence_scoring"
    RESPONSE_VALIDATION = "response_validation"
    MULTI_SHOT_VERIFICATION = "multi_shot_verification"


@dataclass
class ResponseMetadata:
    """Metadata for model responses"""
    confidence_score: float
    hallucination_risk: float
    fact_check_results: Dict[str, bool]
    validation_passed: bool
    strategy_used: List[HallucinationStrategy]
    timestamp: datetime


class AzureGPT4HallucinationHandler:
    """
    Handler for mitigating hallucinations in Azure GPT-4 responses
    """
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "gpt-4",
        default_temperature: float = 0.3,
        max_retries: int = 3
    ):
        """
        Initialize the hallucination handler
        
        Args:
            azure_endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            api_version: API version
            deployment_name: Deployment name for GPT-4
            default_temperature: Default temperature for responses
            max_retries: Maximum number of retries for verification
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.fact_check_sources = {}
        
    def add_fact_source(self, source_name: str, facts: Dict[str, Any]):
        """Add a fact source for validation"""
        self.fact_check_sources[source_name] = facts
        
    def build_system_prompt(self, context: Optional[str] = None) -> str:
        """
        Build a system prompt that reduces hallucination likelihood
        """
        base_prompt = """You are a helpful AI assistant powered by GPT-4. Follow these guidelines strictly:

1. ACCURACY: Only provide information you are certain about. If you're unsure, say so explicitly.
2. SOURCES: When possible, mention the source of your information or indicate if it's based on your training data.
3. LIMITATIONS: Acknowledge when a topic is outside your knowledge cutoff or expertise.
4. VERIFICATION: Double-check facts and calculations before presenting them.
5. CLARITY: Be clear about what is fact vs. opinion or speculation.
6. UNCERTAINTY: Use phrases like "I believe", "likely", "possibly" when not 100% certain.

Remember: It's better to say "I don't know" than to provide incorrect information."""

        if context:
            base_prompt += f"\n\nAdditional Context:\n{context}"
            
        return base_prompt
    
    def apply_prompt_engineering(
        self,
        user_prompt: str,
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Apply prompt engineering techniques to reduce hallucinations
        """
        messages = [
            {"role": "system", "content": self.build_system_prompt(context)}
        ]
        
        # Add few-shot examples if provided
        if examples:
            for example in examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})
        
        # Add chain-of-thought prompting
        enhanced_prompt = f"""{user_prompt}

Please think through this step-by-step:
1. First, identify what information is being requested
2. Consider what you know with certainty
3. Identify any areas of uncertainty
4. Provide your response with appropriate caveats"""
        
        messages.append({"role": "user", "content": enhanced_prompt})
        return messages
    
    def validate_response(self, response: str, original_prompt: str) -> Tuple[bool, List[str]]:
        """
        Validate response for common hallucination patterns
        """
        issues = []
        
        # Check for overly specific dates/numbers without source
        specific_patterns = [
            (r'\b\d{4}\b(?![\s-]\d)', "Specific year without context"),
            (r'\b\d+\.\d{3,}\b', "Overly precise decimal"),
            (r'\b(?:exactly|precisely)\s+\d+', "Overly precise claim"),
        ]
        
        for pattern, issue in specific_patterns:
            if re.search(pattern, response):
                issues.append(issue)
        
        # Check for confidence indicators
        low_confidence_phrases = [
            "i think", "probably", "maybe", "possibly", "likely",
            "i believe", "it seems", "appears to be"
        ]
        
        confidence_count = sum(1 for phrase in low_confidence_phrases if phrase in response.lower())
        
        # Check for admission of uncertainty
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot confirm",
            "outside my knowledge", "i don't have information"
        ]
        
        has_uncertainty = any(phrase in response.lower() for phrase in uncertainty_phrases)
        
        # Validation passes if no major issues and appropriate uncertainty is expressed
        validation_passed = len(issues) == 0 or (has_uncertainty and confidence_count > 0)
        
        return validation_passed, issues
    
    def calculate_confidence_score(self, response: str, metadata: Dict[str, Any]) -> float:
        """
        Calculate a confidence score for the response
        """
        score = 1.0
        
        # Reduce score for uncertainty phrases
        uncertainty_phrases = [
            ("i don't know", 0.3),
            ("i'm not sure", 0.2),
            ("possibly", 0.1),
            ("likely", 0.1),
            ("might be", 0.15),
            ("could be", 0.15)
        ]
        
        for phrase, penalty in uncertainty_phrases:
            if phrase in response.lower():
                score -= penalty
        
        # Reduce score for lack of specific details
        if len(response) < 100:
            score -= 0.1
        
        # Adjust based on temperature used
        if metadata.get("temperature", 0) > 0.7:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def fact_check_response(self, response: str) -> Dict[str, bool]:
        """
        Check response against known facts
        """
        fact_results = {}
        
        for source_name, facts in self.fact_check_sources.items():
            for fact_key, fact_value in facts.items():
                if str(fact_key) in response:
                    # Simple string matching - can be enhanced with NLP
                    fact_results[fact_key] = str(fact_value) in response
        
        return fact_results
    
    def multi_shot_verification(
        self,
        prompt: str,
        initial_response: str,
        num_shots: int = 3
    ) -> Tuple[str, float]:
        """
        Verify response by asking the model multiple times and checking consistency
        """
        responses = [initial_response]
        
        verification_prompt = f"""Original question: {prompt}

You previously answered: {initial_response}

Please answer the original question again, independently. If you're not certain about any part, please say so."""
        
        for _ in range(num_shots - 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": self.build_system_prompt()},
                        {"role": "user", "content": verification_prompt}
                    ],
                    temperature=0.2,  # Lower temperature for verification
                    max_tokens=1000
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"Verification shot failed: {e}")
        
        # Calculate consistency score
        if len(responses) > 1:
            # Simple consistency check - can be enhanced with semantic similarity
            common_words = set(responses[0].lower().split())
            consistency_scores = []
            
            for resp in responses[1:]:
                resp_words = set(resp.lower().split())
                overlap = len(common_words.intersection(resp_words))
                consistency_scores.append(overlap / max(len(common_words), len(resp_words)))
            
            consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        else:
            consistency = 0.5  # Default if verification failed
        
        # Return the most consistent response or the initial one
        return initial_response, consistency
    
    def query_with_hallucination_mitigation(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        strategies: Optional[List[HallucinationStrategy]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, ResponseMetadata]:
        """
        Query Azure GPT-4 with hallucination mitigation strategies
        
        Args:
            prompt: User prompt
            context: Additional context
            temperature: Temperature override
            strategies: List of strategies to apply
            examples: Few-shot examples
            
        Returns:
            Tuple of (response, metadata)
        """
        if strategies is None:
            strategies = [
                HallucinationStrategy.PROMPT_ENGINEERING,
                HallucinationStrategy.TEMPERATURE_CONTROL,
                HallucinationStrategy.RESPONSE_VALIDATION
            ]
        
        temperature = temperature or self.default_temperature
        used_strategies = []
        
        # Apply prompt engineering
        if HallucinationStrategy.PROMPT_ENGINEERING in strategies:
            messages = self.apply_prompt_engineering(prompt, context, examples)
            used_strategies.append(HallucinationStrategy.PROMPT_ENGINEERING)
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Apply temperature control
        if HallucinationStrategy.TEMPERATURE_CONTROL in strategies:
            temperature = min(temperature, 0.3)  # Cap temperature for accuracy
            used_strategies.append(HallucinationStrategy.TEMPERATURE_CONTROL)
        
        try:
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            response_text = response.choices[0].message.content
            
            # Apply response validation
            validation_passed = True
            validation_issues = []
            if HallucinationStrategy.RESPONSE_VALIDATION in strategies:
                validation_passed, validation_issues = self.validate_response(response_text, prompt)
                used_strategies.append(HallucinationStrategy.RESPONSE_VALIDATION)
            
            # Apply fact checking
            fact_check_results = {}
            if HallucinationStrategy.FACT_CHECKING in strategies:
                fact_check_results = self.fact_check_response(response_text)
                used_strategies.append(HallucinationStrategy.FACT_CHECKING)
            
            # Calculate confidence score
            confidence_score = 0.5  # Default
            if HallucinationStrategy.CONFIDENCE_SCORING in strategies:
                confidence_score = self.calculate_confidence_score(
                    response_text,
                    {"temperature": temperature}
                )
                used_strategies.append(HallucinationStrategy.CONFIDENCE_SCORING)
            
            # Apply multi-shot verification if confidence is low
            if (HallucinationStrategy.MULTI_SHOT_VERIFICATION in strategies and 
                confidence_score < 0.7):
                response_text, consistency = self.multi_shot_verification(
                    prompt, response_text
                )
                confidence_score = (confidence_score + consistency) / 2
                used_strategies.append(HallucinationStrategy.MULTI_SHOT_VERIFICATION)
            
            # Calculate hallucination risk
            hallucination_risk = 1.0 - confidence_score
            if not validation_passed:
                hallucination_risk += 0.2
            if len(fact_check_results) > 0:
                failed_facts = sum(1 for v in fact_check_results.values() if not v)
                hallucination_risk += (failed_facts / len(fact_check_results)) * 0.3
            
            hallucination_risk = min(1.0, hallucination_risk)
            
            metadata = ResponseMetadata(
                confidence_score=confidence_score,
                hallucination_risk=hallucination_risk,
                fact_check_results=fact_check_results,
                validation_passed=validation_passed,
                strategy_used=used_strategies,
                timestamp=datetime.now()
            )
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"Error querying Azure GPT-4: {e}")
            raise
    
    def format_response_with_metadata(
        self,
        response: str,
        metadata: ResponseMetadata
    ) -> str:
        """
        Format response with hallucination risk indicators
        """
        risk_level = "LOW" if metadata.hallucination_risk < 0.3 else \
                     "MEDIUM" if metadata.hallucination_risk < 0.6 else "HIGH"
        
        formatted = f"""**Response** (Hallucination Risk: {risk_level})
        
{response}

---
*Confidence Score: {metadata.confidence_score:.2f}*
*Validation: {'Passed' if metadata.validation_passed else 'Failed'}*
*Strategies Applied: {', '.join([s.value for s in metadata.strategy_used])}*"""

        if metadata.fact_check_results:
            formatted += f"\n*Fact Checks: {sum(1 for v in metadata.fact_check_results.values() if v)}/{len(metadata.fact_check_results)} passed*"
        
        return formatted


# Example usage functions
def create_safe_azure_client(
    endpoint: str,
    api_key: str,
    deployment_name: str = "gpt-4"
) -> AzureGPT4HallucinationHandler:
    """
    Create a pre-configured Azure GPT-4 client with hallucination mitigation
    """
    handler = AzureGPT4HallucinationHandler(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        default_temperature=0.3,
        max_retries=3
    )
    
    # Add some example fact sources
    handler.add_fact_source("constants", {
        "speed_of_light": "299,792,458 m/s",
        "earth_radius": "6,371 km",
        "python_creator": "Guido van Rossum"
    })
    
    return handler