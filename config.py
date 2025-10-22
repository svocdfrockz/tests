"""
Configuration management for Azure GPT-4 hallucination mitigation
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AzureConfig:
    """Azure OpenAI configuration"""
    endpoint: str
    api_key: str
    deployment_name: str
    api_version: str = "2024-02-01"
    
    @classmethod
    def from_env(cls) -> "AzureConfig":
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )

@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection"""
    
    # Detection thresholds
    factual_confidence_threshold: float = 0.7
    source_confidence_threshold: float = 0.8
    contextual_relevance_threshold: float = 0.3
    
    # Model parameters for safety
    safe_temperature: float = 0.3
    safe_top_p: float = 0.8
    max_tokens: int = 1000
    
    # Monitoring settings
    log_all_checks: bool = True
    alert_on_high_risk: bool = True
    
    # Domain-specific settings
    medical_mode: bool = False
    financial_mode: bool = False
    legal_mode: bool = False
    
    def get_system_prompt(self) -> str:
        """Get appropriate system prompt based on configuration"""
        
        base_prompt = """You are a helpful assistant. Follow these guidelines:
1. Only provide information you are confident about
2. Clearly state when you are uncertain about something
3. Do not fabricate sources, citations, or specific statistics
4. If you don't know something, say so explicitly
5. Ground your responses in the provided context when available"""

        if self.medical_mode:
            base_prompt += """
6. For medical information, always recommend consulting healthcare professionals
7. Never provide specific medical diagnoses or treatment recommendations
8. Include appropriate medical disclaimers"""
        
        if self.financial_mode:
            base_prompt += """
6. For financial information, include appropriate disclaimers about market risks
7. Never provide specific investment advice
8. Verify all numerical data and calculations"""
        
        if self.legal_mode:
            base_prompt += """
6. For legal information, always recommend consulting qualified legal professionals
7. Never provide specific legal advice
8. Include appropriate legal disclaimers"""
        
        return base_prompt

# Global configuration instances
AZURE_CONFIG = AzureConfig.from_env()
HALLUCINATION_CONFIG = HallucinationConfig()

# Validation patterns for different types of hallucinations
HALLUCINATION_PATTERNS = {
    "factual": [
        r"\d{4}.*(?:discovered|invented|founded)",  # Date claims
        r"\d+(?:\.\d+)?%.*(?:of|percent)",  # Percentage claims
        r"(?:costs?|prices?|worth).*\$\d+",  # Price claims
        r"located in.*(?:city|country|state)",  # Location claims
        r"population of.*\d+",  # Population claims
        r"established in \d{4}",  # Establishment dates
    ],
    "source": [
        r"(?:according to|citing|references?).*(?:study|paper|research|journal)",
        r"(?:dr\.|professor|researcher).*(?:said|stated|found|concluded)",
        r"(?:published in|journal of|proceedings of)",
        r"(?:university of|institute of|center for)",
        r"(?:peer.?reviewed|scientific study|clinical trial)",
        r"(?:research shows|studies indicate|data suggests)",
    ],
    "contextual": [
        r"as mentioned earlier",
        r"as we discussed",
        r"referring back to",
        r"as you know",
        r"obviously",
        r"clearly",
    ]
}

# Fact-checking APIs and resources
FACT_CHECK_APIS = {
    "google_fact_check": {
        "url": "https://factchecktools.googleapis.com/v1alpha1/claims:search",
        "key_env": "GOOGLE_FACT_CHECK_API_KEY"
    },
    "snopes": {
        "url": "https://www.snopes.com/api/",
        "key_env": "SNOPES_API_KEY"
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "hallucination_detection.log",
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}