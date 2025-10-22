"""
Deterministic Azure GPT-4 Responses
This script demonstrates how to get identical responses from GPT-4 on every run.
"""

import os
from openai import AzureOpenAI
from typing import List, Dict
import hashlib


class DeterministicGPT4:
    """
    A wrapper for Azure OpenAI that ensures deterministic responses.
    """
    
    def __init__(
        self,
        api_key: str = None,
        azure_endpoint: str = None,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = None
    ):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_API_KEY env var)
            azure_endpoint: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            api_version: API version to use
            deployment_name: Deployment name (defaults to AZURE_OPENAI_DEPLOYMENT env var)
        """
        self.client = AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version
        )
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    def get_deterministic_response(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant that provides accurate and consistent responses.",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 500,
        top_p: float = 1.0
    ) -> Dict[str, any]:
        """
        Get a deterministic response from GPT-4.
        
        Key parameters for determinism:
        - temperature=0.0: Makes the model deterministic (no randomness)
        - seed: Fixed seed ensures reproducibility
        - All other parameters kept constant
        
        Args:
            prompt: The user's question/prompt
            system_message: System message to set context
            temperature: Controls randomness (0.0 = deterministic)
            seed: Random seed for reproducibility
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,  # CRITICAL: 0.0 for determinism
                seed=seed,                # CRITICAL: Fixed seed for reproducibility
                max_tokens=max_tokens,
                top_p=top_p,
                n=1  # Generate only 1 response
            )
            
            response_text = response.choices[0].message.content
            
            return {
                "response": response_text,
                "prompt": prompt,
                "system_fingerprint": response.system_fingerprint,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "hash": self._hash_response(response_text)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "response": None
            }
    
    def _hash_response(self, text: str) -> str:
        """Generate a hash of the response for easy comparison."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def run_multiple_times(
        self,
        prompt: str,
        num_runs: int = 5,
        **kwargs
    ) -> List[Dict[str, any]]:
        """
        Run the same prompt multiple times and return all results.
        
        Args:
            prompt: The prompt to test
            num_runs: Number of times to run
            **kwargs: Additional arguments for get_deterministic_response
            
        Returns:
            List of response dictionaries
        """
        results = []
        for i in range(num_runs):
            print(f"Run {i + 1}/{num_runs}...")
            result = self.get_deterministic_response(prompt, **kwargs)
            result["run_number"] = i + 1
            results.append(result)
        return results
    
    def verify_consistency(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Verify that all results are identical.
        
        Args:
            results: List of response dictionaries from run_multiple_times
            
        Returns:
            Dictionary with verification results
        """
        if not results:
            return {"consistent": False, "error": "No results to verify"}
        
        # Check if all hashes are the same
        hashes = [r.get("hash") for r in results if r.get("hash")]
        responses = [r.get("response") for r in results if r.get("response")]
        
        if not hashes or not responses:
            return {"consistent": False, "error": "Some requests failed"}
        
        all_same_hash = len(set(hashes)) == 1
        all_same_text = len(set(responses)) == 1
        
        return {
            "consistent": all_same_hash and all_same_text,
            "unique_responses": len(set(responses)),
            "total_runs": len(results),
            "hashes": hashes,
            "first_response": responses[0] if responses else None,
            "all_responses_identical": all_same_text
        }


def main():
    """Example usage demonstrating deterministic responses."""
    
    print("=" * 80)
    print("DETERMINISTIC GPT-4 RESPONSE DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize the client
    gpt4 = DeterministicGPT4()
    
    # Test prompt
    test_prompt = "What are the three main benefits of regular exercise for cardiovascular health?"
    
    print(f"Test Prompt: {test_prompt}")
    print()
    print("Running the same prompt 5 times with deterministic settings...")
    print("(temperature=0, seed=42)")
    print()
    
    # Run 5 times
    results = gpt4.run_multiple_times(
        prompt=test_prompt,
        num_runs=5,
        temperature=0.0,  # Deterministic
        seed=42           # Fixed seed
    )
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # Display each response with its hash
    for result in results:
        if result.get("error"):
            print(f"Run {result['run_number']}: ERROR - {result['error']}")
        else:
            print(f"Run {result['run_number']}:")
            print(f"  Hash: {result['hash']}")
            print(f"  Tokens: {result['usage']['total_tokens']}")
            print()
    
    # Verify consistency
    verification = gpt4.verify_consistency(results)
    
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"All responses identical: {verification['all_responses_identical']}")
    print(f"Unique responses: {verification['unique_responses']} out of {verification['total_runs']}")
    print(f"Consistent: {'✓ YES' if verification['consistent'] else '✗ NO'}")
    print()
    
    if verification['consistent']:
        print("SUCCESS! All 5 runs produced identical responses.")
        print()
        print("Response received:")
        print("-" * 80)
        print(verification['first_response'])
        print("-" * 80)
    else:
        print("WARNING: Responses were not identical!")
        print("This might happen if:")
        print("  1. API version doesn't support 'seed' parameter")
        print("  2. Temperature is not set to 0")
        print("  3. Network/API issues occurred")


if __name__ == "__main__":
    main()
