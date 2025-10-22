"""
Example usage of Azure GPT-4 Hallucination Handler
"""

import os
from azure_gpt4_hallucination_handler import (
    AzureGPT4HallucinationHandler,
    HallucinationStrategy,
    create_safe_azure_client
)


def main():
    # Initialize the handler with your Azure OpenAI credentials
    # Replace these with your actual credentials
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    # Create handler instance
    handler = AzureGPT4HallucinationHandler(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        default_temperature=0.3
    )
    
    # Add fact sources for validation
    handler.add_fact_source("company_facts", {
        "founded": "2015",
        "headquarters": "San Francisco",
        "ceo": "John Doe"
    })
    
    print("=== Example 1: Basic Query with Hallucination Mitigation ===")
    response1, metadata1 = handler.query_with_hallucination_mitigation(
        prompt="What is the exact population of Tokyo in 2024?",
        strategies=[
            HallucinationStrategy.PROMPT_ENGINEERING,
            HallucinationStrategy.TEMPERATURE_CONTROL,
            HallucinationStrategy.RESPONSE_VALIDATION
        ]
    )
    
    print(handler.format_response_with_metadata(response1, metadata1))
    print("\n" + "="*50 + "\n")
    
    # Example 2: Query with context and few-shot examples
    print("=== Example 2: Query with Context and Examples ===")
    
    examples = [
        {
            "user": "What is the capital of France?",
            "assistant": "The capital of France is Paris. This is a well-established fact that has been true for centuries."
        },
        {
            "user": "What is the exact number of tourists visiting Paris each day?",
            "assistant": "I don't have access to real-time tourist data for Paris. The number of daily tourists varies significantly by season and events. According to pre-pandemic data, Paris received about 38 million tourists annually, but I cannot provide an exact daily figure as it fluctuates."
        }
    ]
    
    response2, metadata2 = handler.query_with_hallucination_mitigation(
        prompt="How many people visit the Eiffel Tower every hour?",
        context="You are helping a travel planning application provide accurate information.",
        examples=examples,
        strategies=[
            HallucinationStrategy.PROMPT_ENGINEERING,
            HallucinationStrategy.TEMPERATURE_CONTROL,
            HallucinationStrategy.RESPONSE_VALIDATION,
            HallucinationStrategy.CONFIDENCE_SCORING
        ]
    )
    
    print(handler.format_response_with_metadata(response2, metadata2))
    print("\n" + "="*50 + "\n")
    
    # Example 3: Technical query with multi-shot verification
    print("=== Example 3: Technical Query with Multi-shot Verification ===")
    
    response3, metadata3 = handler.query_with_hallucination_mitigation(
        prompt="Explain the time complexity of quicksort algorithm and its worst-case scenario.",
        strategies=[
            HallucinationStrategy.PROMPT_ENGINEERING,
            HallucinationStrategy.TEMPERATURE_CONTROL,
            HallucinationStrategy.MULTI_SHOT_VERIFICATION,
            HallucinationStrategy.CONFIDENCE_SCORING
        ]
    )
    
    print(handler.format_response_with_metadata(response3, metadata3))
    print("\n" + "="*50 + "\n")
    
    # Example 4: Query that might trigger hallucination
    print("=== Example 4: Query Likely to Trigger Hallucination ===")
    
    response4, metadata4 = handler.query_with_hallucination_mitigation(
        prompt="What was the exact revenue of Microsoft in Q3 2024 and how much did their stock price change on October 15, 2024?",
        strategies=[
            HallucinationStrategy.PROMPT_ENGINEERING,
            HallucinationStrategy.TEMPERATURE_CONTROL,
            HallucinationStrategy.RESPONSE_VALIDATION,
            HallucinationStrategy.FACT_CHECKING,
            HallucinationStrategy.CONFIDENCE_SCORING
        ]
    )
    
    print(handler.format_response_with_metadata(response4, metadata4))
    print("\n" + "="*50 + "\n")
    
    # Example 5: Using the pre-configured safe client
    print("=== Example 5: Using Pre-configured Safe Client ===")
    
    safe_handler = create_safe_azure_client(
        endpoint=azure_endpoint,
        api_key=api_key,
        deployment_name=deployment_name
    )
    
    response5, metadata5 = safe_handler.query_with_hallucination_mitigation(
        prompt="Who created Python and what is the speed of light?",
        strategies=[
            HallucinationStrategy.PROMPT_ENGINEERING,
            HallucinationStrategy.FACT_CHECKING,
            HallucinationStrategy.CONFIDENCE_SCORING
        ]
    )
    
    print(safe_handler.format_response_with_metadata(response5, metadata5))
    

def demonstrate_strategies():
    """Demonstrate different hallucination mitigation strategies"""
    
    print("\n=== Hallucination Mitigation Strategies ===\n")
    
    print("1. PROMPT ENGINEERING")
    print("   - Uses system prompts that encourage accuracy")
    print("   - Implements chain-of-thought reasoning")
    print("   - Includes few-shot examples for consistency")
    print("   - Explicitly asks model to acknowledge uncertainty\n")
    
    print("2. TEMPERATURE CONTROL")
    print("   - Lowers temperature (default 0.3) for more deterministic outputs")
    print("   - Reduces randomness that can lead to hallucinations")
    print("   - Balances creativity with accuracy\n")
    
    print("3. RESPONSE VALIDATION")
    print("   - Checks for overly specific claims without sources")
    print("   - Identifies suspicious patterns (exact numbers, dates)")
    print("   - Validates presence of uncertainty acknowledgments\n")
    
    print("4. FACT CHECKING")
    print("   - Compares responses against known fact databases")
    print("   - Flags discrepancies for review")
    print("   - Can be customized with domain-specific facts\n")
    
    print("5. CONFIDENCE SCORING")
    print("   - Analyzes response for confidence indicators")
    print("   - Scores based on certainty language")
    print("   - Adjusts score based on response characteristics\n")
    
    print("6. MULTI-SHOT VERIFICATION")
    print("   - Asks the same question multiple times")
    print("   - Compares responses for consistency")
    print("   - Higher consistency indicates lower hallucination risk\n")


if __name__ == "__main__":
    # Run the examples
    print("Azure GPT-4 Hallucination Mitigation Examples\n")
    
    # Note: Set your environment variables before running
    # export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
    # export AZURE_OPENAI_API_KEY="your-api-key"
    # export AZURE_OPENAI_DEPLOYMENT="gpt-4"
    
    try:
        main()
    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nMake sure to set your Azure OpenAI credentials as environment variables:")
        print("  AZURE_OPENAI_ENDPOINT")
        print("  AZURE_OPENAI_API_KEY")
        print("  AZURE_OPENAI_DEPLOYMENT")
    
    # Show strategy explanations
    demonstrate_strategies()