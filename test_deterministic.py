"""
Test script to verify GPT-4 responses are identical across multiple runs.
"""

import os
from deterministic_gpt4 import DeterministicGPT4
import json


def test_determinism():
    """Test that running the same prompt 5 times produces identical results."""
    
    print("=" * 80)
    print("DETERMINISM TEST SUITE")
    print("=" * 80)
    print()
    
    gpt4 = DeterministicGPT4()
    
    # Test cases
    test_cases = [
        {
            "name": "Simple factual question",
            "prompt": "What is the capital of France?",
            "system_message": "You are a helpful assistant. Provide accurate, concise answers."
        },
        {
            "name": "List generation",
            "prompt": "List 3 ways to reduce stress.",
            "system_message": "You are a health advisor. Provide clear, numbered lists."
        },
        {
            "name": "Technical explanation",
            "prompt": "Explain what the 'seed' parameter does in GPT-4.",
            "system_message": "You are a technical expert. Be precise and concise."
        }
    ]
    
    all_tests_passed = True
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"Test {idx}: {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        print()
        
        # Run 5 times
        results = gpt4.run_multiple_times(
            prompt=test_case['prompt'],
            system_message=test_case['system_message'],
            num_runs=5,
            temperature=0.0,
            seed=42,
            max_tokens=300
        )
        
        # Verify consistency
        verification = gpt4.verify_consistency(results)
        
        print(f"Result: {'✓ PASS' if verification['consistent'] else '✗ FAIL'}")
        print(f"Unique responses: {verification['unique_responses']}/5")
        
        if verification['consistent']:
            print(f"Response hash: {results[0]['hash']}")
        else:
            all_tests_passed = False
            print("ERROR: Responses were not identical!")
            for i, result in enumerate(results, 1):
                print(f"  Run {i} hash: {result.get('hash', 'ERROR')}")
        
        print()
        print("-" * 80)
        print()
    
    # Final summary
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED - Responses are deterministic!")
    else:
        print("✗ SOME TESTS FAILED - Check your configuration")
    print()
    
    return all_tests_passed


def test_temperature_comparison():
    """Compare deterministic (temp=0) vs non-deterministic (temp=0.7) responses."""
    
    print("=" * 80)
    print("TEMPERATURE COMPARISON TEST")
    print("=" * 80)
    print()
    
    gpt4 = DeterministicGPT4()
    prompt = "Suggest a creative name for a coffee shop."
    
    print("Test: Running same prompt with different temperatures")
    print(f"Prompt: {prompt}")
    print()
    
    # Test with temperature=0 (deterministic)
    print("A) Temperature = 0.0 (Deterministic)")
    results_deterministic = gpt4.run_multiple_times(
        prompt=prompt,
        num_runs=5,
        temperature=0.0,
        seed=42
    )
    verification_det = gpt4.verify_consistency(results_deterministic)
    print(f"   Unique responses: {verification_det['unique_responses']}/5")
    print(f"   Consistent: {'✓ YES' if verification_det['consistent'] else '✗ NO'}")
    print()
    
    # Test with temperature=0.7 (creative)
    print("B) Temperature = 0.7 (Creative)")
    results_creative = gpt4.run_multiple_times(
        prompt=prompt,
        num_runs=5,
        temperature=0.7,
        seed=42  # Even with seed, high temp may vary
    )
    verification_cre = gpt4.verify_consistency(results_creative)
    print(f"   Unique responses: {verification_cre['unique_responses']}/5")
    print(f"   Consistent: {'✓ YES' if verification_cre['consistent'] else '✗ NO'}")
    print()
    
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("With temperature=0.0:")
    print("  - Responses should be identical (deterministic)")
    print("  - Best for factual questions, data extraction, structured outputs")
    print()
    print("With temperature=0.7:")
    print("  - Responses may vary (creative)")
    print("  - Better for creative tasks, brainstorming, varied responses")
    print()


if __name__ == "__main__":
    # Run determinism tests
    test_determinism()
    
    print()
    
    # Run temperature comparison
    test_temperature_comparison()
