# Deterministic Azure GPT-4 Responses

This project demonstrates how to get **identical responses** from Azure GPT-4 every time you run the same query, solving the hallucination and consistency problem.

## 🎯 Problem Solved

When you run GPT-4 multiple times with the same prompt, you usually get different responses. This project shows you how to get **exactly the same response every time**.

## 🔑 Key Techniques

### 1. **Temperature = 0**
The most critical parameter for determinism.
- `temperature=0.0` → Deterministic (always same response)
- `temperature=0.7` → Creative (different responses each time)
- `temperature=1.0` → Very random

### 2. **Fixed Seed**
The `seed` parameter ensures reproducibility:
```python
seed=42  # Any fixed integer
```

### 3. **Consistent Parameters**
Keep all other parameters the same:
- Same `max_tokens`
- Same `top_p`
- Same `system_message`
- Same `prompt`

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI
Copy the example environment file and add your credentials:
```bash
cp .env.example .env
```

Edit `.env` with your Azure OpenAI details:
```
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-gpt4-deployment-name
```

### 3. Run the Demo
```bash
python deterministic_gpt4.py
```

This will run the same prompt **5 times** and verify all responses are identical.

### 4. Run Tests
```bash
python test_deterministic.py
```

## 📝 Usage Examples

### Basic Usage

```python
from deterministic_gpt4 import DeterministicGPT4

# Initialize
gpt4 = DeterministicGPT4()

# Get deterministic response
result = gpt4.get_deterministic_response(
    prompt="What are 3 benefits of exercise?",
    temperature=0.0,  # CRITICAL for determinism
    seed=42           # CRITICAL for reproducibility
)

print(result['response'])
```

### Run Multiple Times and Verify

```python
# Run same prompt 5 times
results = gpt4.run_multiple_times(
    prompt="What is the capital of France?",
    num_runs=5,
    temperature=0.0,
    seed=42
)

# Verify all responses are identical
verification = gpt4.verify_consistency(results)

if verification['consistent']:
    print("✓ All responses are identical!")
else:
    print("✗ Responses differ")
```

## 🔬 How It Works

### The Magic Formula

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Your question here"}
    ],
    temperature=0.0,     # ← Deterministic sampling
    seed=42,             # ← Reproducibility
    max_tokens=500,      # ← Keep constant
    top_p=1.0,          # ← Keep constant
    n=1                  # ← Single response
)
```

### Why This Works

1. **Temperature=0**: Makes the model select the most likely token at each step (no randomness)
2. **Seed**: Ensures any remaining randomness is reproducible
3. **Consistent params**: Same input → same output

## 📊 Test Results

When you run `test_deterministic.py`, you should see:

```
Test 1: Simple factual question
Result: ✓ PASS
Unique responses: 1/5

Test 2: List generation
Result: ✓ PASS
Unique responses: 1/5

Test 3: Technical explanation
Result: ✓ PASS
Unique responses: 1/5

✓ ALL TESTS PASSED - Responses are deterministic!
```

## 🎯 Use Cases

### When to Use Deterministic Responses (temperature=0)

✅ **Factual Q&A**: Always get the same fact
✅ **Data Extraction**: Consistent JSON/structured outputs
✅ **Classification**: Same input → same category
✅ **Testing**: Reproducible results for unit tests
✅ **Production Systems**: Predictable behavior
✅ **Compliance**: Auditable, consistent responses

### When NOT to Use (use higher temperature)

❌ **Creative Writing**: Want variety
❌ **Brainstorming**: Need different ideas
❌ **Content Generation**: Avoid repetition
❌ **Conversation**: More natural variation

## 🛡️ Anti-Hallucination Benefits

Deterministic responses help with hallucination in several ways:

1. **Testable**: You can verify the response is correct once, then it stays correct
2. **Consistent**: No "sometimes it works, sometimes it doesn't"
3. **Debuggable**: Same input always produces same output
4. **Cacheable**: Can cache responses safely
5. **Reliable**: Production systems behave predictably

## 📚 Additional Tips

### Combine with Other Anti-Hallucination Techniques

```python
system_message = """
You are a helpful assistant.
IMPORTANT RULES:
1. Only provide information you are certain about
2. If unsure, say "I don't know"
3. Cite sources when possible
4. Distinguish between facts and opinions
"""

result = gpt4.get_deterministic_response(
    prompt="What is the population of Tokyo in 2024?",
    system_message=system_message,
    temperature=0.0,
    seed=42
)
```

### Use with RAG (Retrieval-Augmented Generation)

```python
# Include relevant context in prompt
context = "According to WHO data from 2023, regular exercise reduces cardiovascular disease risk by 30-40%."

prompt = f"""
Context: {context}

Question: What are the cardiovascular benefits of exercise?

Instructions: Base your answer ONLY on the provided context.
"""

result = gpt4.get_deterministic_response(prompt, temperature=0.0, seed=42)
```

## 🔍 Troubleshooting

### Responses Still Vary?

1. **Check API version**: Older versions may not support `seed` parameter
2. **Verify temperature=0**: Must be exactly 0.0
3. **Check all parameters**: All must be identical between runs
4. **API changes**: Azure may update model behavior occasionally

### System Fingerprint

The `system_fingerprint` in the response indicates the exact model version:
```python
print(result['system_fingerprint'])
```

If this changes, responses might vary even with same parameters.

## 📖 References

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [GPT-4 Best Practices](https://platform.openai.com/docs/guides/gpt-best-practices)

## 📄 License

MIT License - Feel free to use in your projects!

## 🤝 Contributing

Found a better way to ensure determinism? Open a PR!

---

**Made with ❤️ to solve GPT-4 hallucination and consistency issues**
