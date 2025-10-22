# Azure GPT-4 Hallucination Mitigation

A comprehensive Python library for detecting and mitigating hallucinations in Azure GPT-4 model responses.

## Overview

This library provides multiple strategies to reduce hallucinations (false or fabricated information) in GPT-4 responses, including:

- **Prompt Engineering**: Optimized prompts that encourage accuracy
- **Temperature Control**: Parameter tuning for more reliable outputs
- **Response Validation**: Pattern detection for common hallucination indicators
- **Fact Checking**: Validation against known facts
- **Confidence Scoring**: Quantitative assessment of response reliability
- **Multi-shot Verification**: Consistency checking across multiple queries

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from azure_gpt4_hallucination_handler import AzureGPT4HallucinationHandler, HallucinationStrategy

# Initialize handler
handler = AzureGPT4HallucinationHandler(
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-api-key",
    deployment_name="gpt-4"
)

# Query with hallucination mitigation
response, metadata = handler.query_with_hallucination_mitigation(
    prompt="What is the population of Tokyo?",
    strategies=[
        HallucinationStrategy.PROMPT_ENGINEERING,
        HallucinationStrategy.TEMPERATURE_CONTROL,
        HallucinationStrategy.RESPONSE_VALIDATION
    ]
)

# Check confidence
if metadata.hallucination_risk < 0.3:
    print(f"High confidence response: {response}")
else:
    print(f"Lower confidence response (verify details): {response}")
```

## Key Features

### 1. Intelligent Prompt Engineering
- Automatically enhances prompts for accuracy
- Implements chain-of-thought reasoning
- Supports few-shot examples

### 2. Response Validation
- Detects overly specific claims
- Identifies missing uncertainty acknowledgments
- Checks for logical consistency

### 3. Confidence Scoring
- Quantifies response reliability (0-1 scale)
- Factors in multiple indicators
- Provides actionable thresholds

### 4. Fact Checking
- Validates against custom fact databases
- Supports domain-specific knowledge
- Real-time verification

### 5. Multi-shot Verification
- Queries multiple times for consistency
- Reduces random hallucinations
- Improves reliability for critical queries

## Usage Examples

### Basic Usage

```python
# Simple query with default strategies
response, metadata = handler.query_with_hallucination_mitigation(
    prompt="Explain quantum computing"
)
```

### Advanced Usage with Context

```python
# Query with context and examples
examples = [
    {
        "user": "What is the speed of light?",
        "assistant": "The speed of light in vacuum is approximately 299,792,458 meters per second."
    }
]

response, metadata = handler.query_with_hallucination_mitigation(
    prompt="What is the speed of sound?",
    context="Provide accurate physics information",
    examples=examples,
    temperature=0.2
)
```

### Fact Checking

```python
# Add fact sources
handler.add_fact_source("company_facts", {
    "founded": "2010",
    "headquarters": "Seattle",
    "employees": "50000"
})

# Query with fact checking
response, metadata = handler.query_with_hallucination_mitigation(
    prompt="Tell me about our company",
    strategies=[HallucinationStrategy.FACT_CHECKING]
)
```

## Configuration

### Environment Variables

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
```

### Handler Parameters

- `azure_endpoint`: Your Azure OpenAI endpoint
- `api_key`: Your API key
- `deployment_name`: GPT-4 deployment name (default: "gpt-4")
- `default_temperature`: Default temperature (default: 0.3)
- `max_retries`: Maximum verification retries (default: 3)

## Best Practices

1. **Use Multiple Strategies**: Combine strategies for best results
2. **Set Appropriate Temperature**: Lower = more accurate, higher = more creative
3. **Provide Context**: Give the model relevant background information
4. **Check Metadata**: Always review confidence scores and validation results
5. **Add Domain Facts**: Customize fact checking for your use case

## Testing

Run the test suite:

```bash
python -m pytest test_hallucination_handler.py -v
```

## Documentation

- `hallucination_mitigation_guide.md`: Comprehensive guide on hallucination mitigation
- `example_usage.py`: Full examples with different scenarios
- `test_hallucination_handler.py`: Unit tests and test patterns

## Common Issues and Solutions

### Issue: High hallucination risk scores
**Solution**: Lower temperature, add more context, use multi-shot verification

### Issue: Slow response times
**Solution**: Reduce number of strategies, cache fact checking results

### Issue: Overly conservative responses
**Solution**: Adjust confidence thresholds, balance strategies

## Contributing

When contributing, please:
1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Test with various prompts

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions:
1. Check the comprehensive guide in `hallucination_mitigation_guide.md`
2. Review examples in `example_usage.py`
3. Run tests to verify setup

## Acknowledgments

Built for reliable AI applications using Azure OpenAI GPT-4.