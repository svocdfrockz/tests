"""
Unit tests for Azure GPT-4 Hallucination Handler
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from azure_gpt4_hallucination_handler import (
    AzureGPT4HallucinationHandler,
    HallucinationStrategy,
    ResponseMetadata
)


class TestAzureGPT4HallucinationHandler(unittest.TestCase):
    """Test cases for hallucination handler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = AzureGPT4HallucinationHandler(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="gpt-4",
            default_temperature=0.3
        )
        
    def test_system_prompt_building(self):
        """Test system prompt construction"""
        # Test without context
        prompt = self.handler.build_system_prompt()
        self.assertIn("ACCURACY", prompt)
        self.assertIn("Only provide information you are certain about", prompt)
        
        # Test with context
        context = "You are helping with medical information"
        prompt_with_context = self.handler.build_system_prompt(context)
        self.assertIn(context, prompt_with_context)
        
    def test_prompt_engineering(self):
        """Test prompt engineering techniques"""
        user_prompt = "What is the capital of France?"
        
        # Test without examples
        messages = self.handler.apply_prompt_engineering(user_prompt)
        self.assertEqual(len(messages), 2)  # System + user
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("think through this step-by-step", messages[1]["content"])
        
        # Test with examples
        examples = [
            {"user": "What is 2+2?", "assistant": "2+2 equals 4."}
        ]
        messages_with_examples = self.handler.apply_prompt_engineering(
            user_prompt, examples=examples
        )
        self.assertEqual(len(messages_with_examples), 4)  # System + example + user
        
    def test_response_validation(self):
        """Test response validation logic"""
        # Test response with specific year
        response1 = "This happened in 1995"
        valid1, issues1 = self.handler.validate_response(response1, "When did X happen?")
        self.assertIn("Specific year without context", issues1)
        
        # Test response with uncertainty
        response2 = "I'm not sure, but I think it might be around 1995"
        valid2, issues2 = self.handler.validate_response(response2, "When did X happen?")
        self.assertTrue(valid2)
        
        # Test overly precise numbers
        response3 = "The exact value is 3.14159265359"
        valid3, issues3 = self.handler.validate_response(response3, "What is pi?")
        self.assertIn("Overly precise decimal", issues3)
        
    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        # High confidence response
        response1 = "Paris is the capital of France."
        score1 = self.handler.calculate_confidence_score(response1, {"temperature": 0.2})
        self.assertGreater(score1, 0.7)
        
        # Low confidence response
        response2 = "I'm not sure, but I think it might be Paris, possibly."
        score2 = self.handler.calculate_confidence_score(response2, {"temperature": 0.2})
        self.assertLess(score2, 0.5)
        
        # Short response penalty
        response3 = "Maybe Paris"
        score3 = self.handler.calculate_confidence_score(response3, {"temperature": 0.2})
        self.assertLess(score3, score1)
        
    def test_fact_checking(self):
        """Test fact checking functionality"""
        # Add fact sources
        self.handler.add_fact_source("test_facts", {
            "capital_france": "Paris",
            "year_founded": "1900"
        })
        
        # Test matching facts
        response1 = "The capital_france is Paris"
        results1 = self.handler.fact_check_response(response1)
        self.assertTrue(results1.get("capital_france", False))
        
        # Test non-matching facts
        response2 = "The capital_france is London"
        results2 = self.handler.fact_check_response(response2)
        self.assertFalse(results2.get("capital_france", True))
        
    @patch('azure_gpt4_hallucination_handler.AzureOpenAI')
    def test_multi_shot_verification(self, mock_client):
        """Test multi-shot verification"""
        # Mock responses
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Paris is the capital"))]
        
        self.handler.client.chat.completions.create = MagicMock(return_value=mock_response)
        
        initial_response = "Paris is the capital of France"
        final_response, consistency = self.handler.multi_shot_verification(
            "What is the capital of France?",
            initial_response,
            num_shots=3
        )
        
        self.assertEqual(final_response, initial_response)
        self.assertIsInstance(consistency, float)
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
        
    @patch('azure_gpt4_hallucination_handler.AzureOpenAI')
    def test_query_with_mitigation(self, mock_client):
        """Test full query with hallucination mitigation"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content="I believe Paris is the capital of France.")
        )]
        
        self.handler.client.chat.completions.create = MagicMock(return_value=mock_response)
        
        response, metadata = self.handler.query_with_hallucination_mitigation(
            prompt="What is the capital of France?",
            strategies=[
                HallucinationStrategy.PROMPT_ENGINEERING,
                HallucinationStrategy.TEMPERATURE_CONTROL,
                HallucinationStrategy.RESPONSE_VALIDATION,
                HallucinationStrategy.CONFIDENCE_SCORING
            ]
        )
        
        # Verify response
        self.assertIn("Paris", response)
        
        # Verify metadata
        self.assertIsInstance(metadata, ResponseMetadata)
        self.assertIsInstance(metadata.confidence_score, float)
        self.assertIsInstance(metadata.hallucination_risk, float)
        self.assertIsInstance(metadata.validation_passed, bool)
        self.assertIsInstance(metadata.strategy_used, list)
        self.assertIn(HallucinationStrategy.PROMPT_ENGINEERING, metadata.strategy_used)
        
    def test_format_response_with_metadata(self):
        """Test response formatting with metadata"""
        response = "This is a test response"
        metadata = ResponseMetadata(
            confidence_score=0.8,
            hallucination_risk=0.2,
            fact_check_results={"fact1": True, "fact2": False},
            validation_passed=True,
            strategy_used=[HallucinationStrategy.PROMPT_ENGINEERING],
            timestamp=datetime.now()
        )
        
        formatted = self.handler.format_response_with_metadata(response, metadata)
        
        self.assertIn("LOW", formatted)  # Low risk
        self.assertIn("0.80", formatted)  # Confidence score
        self.assertIn("Passed", formatted)  # Validation
        self.assertIn("1/2 passed", formatted)  # Fact checks
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty response validation
        empty_response = ""
        valid, issues = self.handler.validate_response(empty_response, "Test prompt")
        self.assertIsInstance(valid, bool)
        self.assertIsInstance(issues, list)
        
        # Very long response
        long_response = "This is a test. " * 100
        score = self.handler.calculate_confidence_score(long_response, {})
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # No fact sources
        handler_no_facts = AzureGPT4HallucinationHandler(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key"
        )
        results = handler_no_facts.fact_check_response("Some response")
        self.assertEqual(results, {})


class TestHallucinationPatterns(unittest.TestCase):
    """Test specific hallucination patterns"""
    
    def setUp(self):
        self.handler = AzureGPT4HallucinationHandler(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key"
        )
        
    def test_date_hallucination_detection(self):
        """Test detection of date-related hallucinations"""
        responses = [
            ("On March 23, 2024, exactly at 3:45 PM", False),
            ("Sometime in March 2024", True),
            ("In the year 1995", False),
            ("Around the mid-1990s", True)
        ]
        
        for response, should_pass in responses:
            valid, issues = self.handler.validate_response(response, "When did it happen?")
            if should_pass:
                self.assertTrue(len(issues) == 0 or "possibly" in response.lower(),
                              f"Response '{response}' should pass validation")
            else:
                self.assertTrue(len(issues) > 0,
                              f"Response '{response}' should have validation issues")
                
    def test_number_hallucination_detection(self):
        """Test detection of number-related hallucinations"""
        responses = [
            ("The exact value is 3.14159265359", False),
            ("Approximately 3.14", True),
            ("Exactly 12,345 people", False),
            ("Roughly 12,000 people", True)
        ]
        
        for response, should_pass in responses:
            valid, issues = self.handler.validate_response(response, "How many?")
            if should_pass:
                self.assertTrue(len(issues) == 0 or "roughly" in response.lower())
            else:
                self.assertGreater(len(issues), 0)


if __name__ == "__main__":
    unittest.main()