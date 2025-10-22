"""
SIMPLE EXAMPLE: Get identical GPT-4 responses every time
Run this script 5 times - you'll get the exact same response each time!
"""

import os
from openai import AzureOpenAI

# Configure your Azure OpenAI credentials
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Your question
prompt = "What are 3 key benefits of regular exercise?"

print(f"Question: {prompt}\n")
print("Calling GPT-4 with deterministic settings...\n")

# THE KEY: temperature=0 and seed=42
response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.0,  # ← CRITICAL: Makes it deterministic
    seed=42,          # ← CRITICAL: Ensures reproducibility
    max_tokens=300
)

answer = response.choices[0].message.content

print("Answer:")
print("-" * 80)
print(answer)
print("-" * 80)
print("\n✓ Run this script multiple times - you'll get the SAME answer every time!")
print("\nTo test non-deterministic behavior, change temperature to 0.7")
