"""
Example usage of the Azure GPT-4 hallucination mitigation framework
"""

import asyncio
import logging
from typing import List, Dict
from hallucination_detection import SafeGPTClient, create_rag_prompt
from config import AZURE_CONFIG, HallucinationConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_example():
    """Basic example of safe GPT-4 usage"""
    
    print("=== Basic Safe Completion Example ===")
    
    # Initialize the safe client
    client = SafeGPTClient(
        azure_endpoint=AZURE_CONFIG.endpoint,
        api_key=AZURE_CONFIG.api_key,
        deployment_name=AZURE_CONFIG.deployment_name
    )
    
    # Example query that might trigger hallucinations
    messages = [
        {
            "role": "user", 
            "content": "What are the latest statistics on blood pressure medication effectiveness?"
        }
    ]
    
    try:
        result = client.safe_completion(messages, temperature=0.2)
        
        print(f"Response: {result['content']}")
        print(f"\nHallucination Checks:")
        for check in result['hallucination_checks']:
            status = "⚠️  POTENTIAL ISSUE" if check['is_hallucination'] else "✅ PASSED"
            print(f"  {check['category'].upper()}: {status}")
            print(f"    Confidence: {check['confidence']:.2f}")
            print(f"    Reason: {check['reason']}")
        
    except Exception as e:
        logger.error(f"Error in basic example: {e}")

async def rag_example():
    """Example using Retrieval-Augmented Generation"""
    
    print("\n=== RAG-based Safe Completion Example ===")
    
    client = SafeGPTClient(
        azure_endpoint=AZURE_CONFIG.endpoint,
        api_key=AZURE_CONFIG.api_key,
        deployment_name=AZURE_CONFIG.deployment_name
    )
    
    # Simulated context documents (in practice, these would come from your knowledge base)
    context_documents = [
        """
        A study published in the Journal of Hypertension (2023) found that regular aerobic exercise 
        can reduce systolic blood pressure by an average of 4-9 mmHg in adults with hypertension. 
        The study followed 500 participants over 6 months.
        """,
        """
        The American Heart Association recommends at least 150 minutes of moderate-intensity 
        aerobic activity per week for adults to help manage blood pressure. This can include 
        activities like brisk walking, swimming, or cycling.
        """,
        """
        Dietary approaches such as the DASH diet (Dietary Approaches to Stop Hypertension) 
        have been shown to lower blood pressure. The diet emphasizes fruits, vegetables, 
        whole grains, and lean proteins while limiting sodium intake.
        """
    ]
    
    # Create RAG prompt
    query = "How can lifestyle changes help manage blood pressure?"
    rag_prompt = create_rag_prompt(query, context_documents)
    
    messages = [{"role": "user", "content": rag_prompt}]
    
    try:
        result = client.safe_completion(messages, temperature=0.1)
        
        print(f"RAG Response: {result['content']}")
        print(f"\nHallucination Analysis:")
        
        hallucination_detected = any(check['is_hallucination'] for check in result['hallucination_checks'])
        if not hallucination_detected:
            print("✅ No hallucinations detected - response is grounded in provided context")
        else:
            print("⚠️  Potential hallucinations detected:")
            for check in result['hallucination_checks']:
                if check['is_hallucination']:
                    print(f"  - {check['category']}: {check['reason']}")
        
    except Exception as e:
        logger.error(f"Error in RAG example: {e}")

async def medical_mode_example():
    """Example with medical mode configuration"""
    
    print("\n=== Medical Mode Example ===")
    
    # Configure for medical domain
    medical_config = HallucinationConfig(
        medical_mode=True,
        safe_temperature=0.1,  # Very conservative for medical
        factual_confidence_threshold=0.9  # Higher threshold for medical facts
    )
    
    client = SafeGPTClient(
        azure_endpoint=AZURE_CONFIG.endpoint,
        api_key=AZURE_CONFIG.api_key,
        deployment_name=AZURE_CONFIG.deployment_name
    )
    
    # Medical query
    messages = [
        {
            "role": "system",
            "content": medical_config.get_system_prompt()
        },
        {
            "role": "user",
            "content": "What are the side effects of ACE inhibitors for blood pressure?"
        }
    ]
    
    try:
        result = client.safe_completion(
            messages, 
            temperature=medical_config.safe_temperature,
            check_hallucinations=True
        )
        
        print(f"Medical Response: {result['content']}")
        
        # Check for medical disclaimers
        if "consult" in result['content'].lower() and "healthcare" in result['content'].lower():
            print("✅ Appropriate medical disclaimer included")
        else:
            print("⚠️  Consider adding medical disclaimer")
        
        print(f"\nSafety Analysis:")
        for check in result['hallucination_checks']:
            print(f"  {check['category']}: {'SAFE' if not check['is_hallucination'] else 'REVIEW NEEDED'}")
        
    except Exception as e:
        logger.error(f"Error in medical mode example: {e}")

async def batch_processing_example():
    """Example of processing multiple queries safely"""
    
    print("\n=== Batch Processing Example ===")
    
    client = SafeGPTClient(
        azure_endpoint=AZURE_CONFIG.endpoint,
        api_key=AZURE_CONFIG.api_key,
        deployment_name=AZURE_CONFIG.deployment_name
    )
    
    queries = [
        "What is normal blood pressure range?",
        "How does salt intake affect blood pressure?",
        "What are the symptoms of high blood pressure?",
        "When should someone see a doctor about blood pressure?"
    ]
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nProcessing query {i}/{len(queries)}: {query}")
        
        messages = [{"role": "user", "content": query}]
        
        try:
            result = client.safe_completion(messages)
            
            # Analyze safety
            hallucination_count = sum(1 for check in result['hallucination_checks'] 
                                    if check['is_hallucination'])
            
            safety_status = "✅ SAFE" if hallucination_count == 0 else f"⚠️  {hallucination_count} ISSUES"
            print(f"  Status: {safety_status}")
            print(f"  Response length: {len(result['content'])} characters")
            
            results.append({
                "query": query,
                "response": result['content'],
                "safety_score": len(result['hallucination_checks']) - hallucination_count,
                "total_checks": len(result['hallucination_checks'])
            })
            
        except Exception as e:
            logger.error(f"Error processing query {i}: {e}")
            results.append({
                "query": query,
                "error": str(e)
            })
    
    # Summary
    successful_queries = [r for r in results if 'error' not in r]
    print(f"\n=== Batch Summary ===")
    print(f"Processed: {len(successful_queries)}/{len(queries)} queries successfully")
    
    if successful_queries:
        avg_safety = sum(r['safety_score'] for r in successful_queries) / len(successful_queries)
        print(f"Average safety score: {avg_safety:.1f}/3.0")

async def monitoring_example():
    """Example of monitoring and alerting"""
    
    print("\n=== Monitoring Example ===")
    
    client = SafeGPTClient(
        azure_endpoint=AZURE_CONFIG.endpoint,
        api_key=AZURE_CONFIG.api_key,
        deployment_name=AZURE_CONFIG.deployment_name
    )
    
    # Simulate queries with varying risk levels
    test_queries = [
        ("What is hypertension?", "low_risk"),
        ("According to a recent Harvard study, 95% of doctors recommend this new treatment.", "high_risk"),
        ("The FDA approved this medication in 2019 for treating high blood pressure.", "medium_risk"),
    ]
    
    risk_summary = {"low_risk": 0, "medium_risk": 0, "high_risk": 0}
    
    for query, expected_risk in test_queries:
        messages = [{"role": "user", "content": query}]
        
        try:
            result = client.safe_completion(messages)
            
            # Calculate risk score
            hallucination_count = sum(1 for check in result['hallucination_checks'] 
                                    if check['is_hallucination'])
            
            if hallucination_count == 0:
                actual_risk = "low_risk"
            elif hallucination_count <= 1:
                actual_risk = "medium_risk"
            else:
                actual_risk = "high_risk"
            
            risk_summary[actual_risk] += 1
            
            print(f"Query: {query[:50]}...")
            print(f"  Expected risk: {expected_risk}")
            print(f"  Detected risk: {actual_risk}")
            print(f"  Hallucination count: {hallucination_count}")
            
            # Simulate alerting for high-risk responses
            if actual_risk == "high_risk":
                print("  🚨 HIGH RISK ALERT: Manual review recommended")
            
        except Exception as e:
            logger.error(f"Error in monitoring example: {e}")
    
    print(f"\n=== Risk Summary ===")
    for risk_level, count in risk_summary.items():
        print(f"{risk_level.replace('_', ' ').title()}: {count} queries")

async def main():
    """Run all examples"""
    
    print("Azure GPT-4 Hallucination Mitigation Examples")
    print("=" * 50)
    
    # Check if configuration is available
    if not AZURE_CONFIG.endpoint or not AZURE_CONFIG.api_key:
        print("⚠️  Azure configuration not found. Please set up your .env file.")
        print("   Copy .env.example to .env and fill in your Azure OpenAI details.")
        return
    
    try:
        await basic_example()
        await rag_example()
        await medical_mode_example()
        await batch_processing_example()
        await monitoring_example()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated responses and hallucination checks")
        print("2. Adjust configuration parameters based on your needs")
        print("3. Implement additional domain-specific validation")
        print("4. Set up monitoring and alerting in production")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Azure OpenAI configuration in .env")
        print("2. Verify your API key and endpoint are correct")
        print("3. Ensure your deployment is active and accessible")

if __name__ == "__main__":
    asyncio.run(main())