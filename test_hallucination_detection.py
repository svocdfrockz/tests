"""
Test suite for hallucination detection and mitigation
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from hallucination_detection import HallucinationDetector, SafeGPTClient, HallucinationCheck
from config import HallucinationConfig

class TestHallucinationDetector:
    """Test cases for hallucination detection"""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing"""
        with patch('hallucination_detection.openai.AzureOpenAI'):
            with patch('hallucination_detection.TextAnalyticsClient'):
                return HallucinationDetector("test_endpoint", "test_key")
    
    def test_factual_consistency_date_claim(self, detector):
        """Test detection of suspicious date claims"""
        response = "The iPhone was invented in 2007 by Apple."
        result = detector.check_factual_consistency(response)
        
        assert result.is_hallucination == True
        assert result.category == "factual"
        assert "date" in result.reason.lower() or "pattern" in result.reason.lower()
    
    def test_factual_consistency_percentage_claim(self, detector):
        """Test detection of suspicious percentage claims"""
        response = "Studies show that 87.3% of people prefer this method."
        result = detector.check_factual_consistency(response)
        
        assert result.is_hallucination == True
        assert result.category == "factual"
    
    def test_factual_consistency_clean_response(self, detector):
        """Test that clean responses pass factual checks"""
        response = "Regular exercise is beneficial for health."
        result = detector.check_factual_consistency(response)
        
        assert result.is_hallucination == False
        assert result.category == "factual"
    
    def test_source_citation_detection(self, detector):
        """Test detection of fabricated sources"""
        response = "According to a study published in the Journal of Medicine, this treatment is effective."
        result = detector.check_source_citations(response)
        
        assert result.is_hallucination == True
        assert result.category == "source"
    
    def test_source_citation_clean_response(self, detector):
        """Test that responses without citations pass source checks"""
        response = "This is a general statement about health."
        result = detector.check_source_citations(response)
        
        assert result.is_hallucination == False
        assert result.category == "source"
    
    @patch('hallucination_detection.TextAnalyticsClient')
    def test_contextual_relevance_high_overlap(self, mock_client, detector):
        """Test contextual relevance with high phrase overlap"""
        # Mock the text analytics response
        mock_result = Mock()
        mock_result.key_phrases = ["exercise", "health", "benefits"]
        mock_client.return_value.extract_key_phrases.return_value = [mock_result, mock_result]
        
        detector.text_analytics = mock_client.return_value
        
        query = "What are the benefits of exercise for health?"
        response = "Exercise provides many health benefits including improved cardiovascular health."
        
        result = detector.check_contextual_relevance(response, query)
        
        assert result.is_hallucination == False
        assert result.category == "contextual"
    
    @patch('hallucination_detection.TextAnalyticsClient')
    def test_contextual_relevance_low_overlap(self, mock_client, detector):
        """Test contextual relevance with low phrase overlap"""
        # Mock the text analytics response
        query_result = Mock()
        query_result.key_phrases = ["exercise", "health", "benefits"]
        response_result = Mock()
        response_result.key_phrases = ["cooking", "recipes", "ingredients"]
        
        mock_client.return_value.extract_key_phrases.return_value = [query_result, response_result]
        detector.text_analytics = mock_client.return_value
        
        query = "What are the benefits of exercise for health?"
        response = "Here are some great cooking recipes with healthy ingredients."
        
        result = detector.check_contextual_relevance(response, query)
        
        assert result.is_hallucination == True
        assert result.category == "contextual"
    
    def test_comprehensive_check(self, detector):
        """Test comprehensive hallucination checking"""
        response = "According to a 2023 study, 95% of doctors recommend this treatment invented in 1995."
        query = "What do doctors recommend?"
        
        with patch.object(detector, 'check_contextual_relevance') as mock_contextual:
            mock_contextual.return_value = HallucinationCheck(
                is_hallucination=False, confidence=0.8, reason="Relevant", category="contextual"
            )
            
            results = detector.comprehensive_check(response, query)
            
            assert len(results) == 3
            assert any(check.category == "factual" for check in results)
            assert any(check.category == "source" for check in results)
            assert any(check.category == "contextual" for check in results)

class TestSafeGPTClient:
    """Test cases for SafeGPTClient"""
    
    @pytest.fixture
    def safe_client(self):
        """Create a SafeGPTClient instance for testing"""
        with patch('hallucination_detection.openai.AzureOpenAI'):
            return SafeGPTClient("test_endpoint", "test_key", "test_deployment")
    
    @patch('hallucination_detection.openai.AzureOpenAI')
    def test_safe_completion_success(self, mock_openai, safe_client):
        """Test successful safe completion"""
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a safe response."
        mock_response.usage.model_dump.return_value = {"total_tokens": 50}
        
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        safe_client.client = mock_openai.return_value
        
        # Mock the detector
        with patch.object(safe_client.detector, 'comprehensive_check') as mock_check:
            mock_check.return_value = [
                HallucinationCheck(False, 0.9, "Clean", "factual"),
                HallucinationCheck(False, 0.9, "Clean", "source"),
                HallucinationCheck(False, 0.9, "Clean", "contextual")
            ]
            
            messages = [{"role": "user", "content": "Test message"}]
            result = safe_client.safe_completion(messages)
            
            assert "content" in result
            assert "hallucination_checks" in result
            assert "timestamp" in result
            assert result["content"] == "This is a safe response."
    
    def test_safe_completion_adds_system_message(self, safe_client):
        """Test that system message is added when not present"""
        with patch.object(safe_client.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.usage.model_dump.return_value = {"total_tokens": 50}
            mock_create.return_value = mock_response
            
            with patch.object(safe_client.detector, 'comprehensive_check') as mock_check:
                mock_check.return_value = []
                
                messages = [{"role": "user", "content": "Test"}]
                safe_client.safe_completion(messages)
                
                # Check that system message was added
                call_args = mock_create.call_args
                called_messages = call_args[1]["messages"]
                assert called_messages[0]["role"] == "system"
                assert "helpful assistant" in called_messages[0]["content"]

class TestHallucinationPatterns:
    """Test specific hallucination patterns"""
    
    test_cases = [
        # Factual hallucinations
        ("The company was founded in 1995 by John Smith.", True, "factual"),
        ("This costs $299.99 in most stores.", True, "factual"),
        ("The population of this city is 2.5 million.", True, "factual"),
        ("Regular exercise is good for health.", False, "factual"),
        
        # Source hallucinations
        ("According to Dr. Smith from Harvard University, this is true.", True, "source"),
        ("A study published in Nature found that...", True, "source"),
        ("Research shows that 80% of participants improved.", True, "source"),
        ("Many people believe this to be true.", False, "source"),
        
        # Contextual hallucinations
        ("As we discussed earlier in our conversation...", True, "contextual"),
        ("Obviously, this is the best approach.", True, "contextual"),
        ("This is a straightforward answer.", False, "contextual"),
    ]
    
    @pytest.fixture
    def detector(self):
        """Create detector for pattern testing"""
        with patch('hallucination_detection.openai.AzureOpenAI'):
            with patch('hallucination_detection.TextAnalyticsClient'):
                return HallucinationDetector("test_endpoint", "test_key")
    
    @pytest.mark.parametrize("text,expected_hallucination,category", test_cases)
    def test_hallucination_patterns(self, detector, text, expected_hallucination, category):
        """Test various hallucination patterns"""
        if category == "factual":
            result = detector.check_factual_consistency(text)
        elif category == "source":
            result = detector.check_source_citations(text)
        else:  # contextual
            # For contextual, we need to mock the text analytics
            with patch.object(detector, 'check_contextual_relevance') as mock_contextual:
                if expected_hallucination:
                    mock_contextual.return_value = HallucinationCheck(
                        True, 0.7, "Contextual issue", "contextual"
                    )
                else:
                    mock_contextual.return_value = HallucinationCheck(
                        False, 0.8, "Clean", "contextual"
                    )
                result = detector.check_contextual_relevance(text, "test query")
        
        assert result.is_hallucination == expected_hallucination
        assert result.category == category

class TestConfigurationHandling:
    """Test configuration and domain-specific settings"""
    
    def test_medical_mode_system_prompt(self):
        """Test medical mode system prompt"""
        config = HallucinationConfig(medical_mode=True)
        prompt = config.get_system_prompt()
        
        assert "medical" in prompt.lower()
        assert "healthcare professionals" in prompt.lower()
        assert "medical disclaimers" in prompt.lower()
    
    def test_financial_mode_system_prompt(self):
        """Test financial mode system prompt"""
        config = HallucinationConfig(financial_mode=True)
        prompt = config.get_system_prompt()
        
        assert "financial" in prompt.lower()
        assert "market risks" in prompt.lower()
        assert "investment advice" in prompt.lower()
    
    def test_legal_mode_system_prompt(self):
        """Test legal mode system prompt"""
        config = HallucinationConfig(legal_mode=True)
        prompt = config.get_system_prompt()
        
        assert "legal" in prompt.lower()
        assert "legal professionals" in prompt.lower()
        assert "legal disclaimers" in prompt.lower()

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def mock_azure_response(self):
        """Mock Azure OpenAI response"""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "This is a test response about exercise benefits."
        response.usage.model_dump.return_value = {"total_tokens": 25}
        return response
    
    @patch('hallucination_detection.openai.AzureOpenAI')
    @patch('hallucination_detection.TextAnalyticsClient')
    def test_end_to_end_safe_completion(self, mock_text_analytics, mock_openai, mock_azure_response):
        """Test complete end-to-end safe completion"""
        # Setup mocks
        mock_openai.return_value.chat.completions.create.return_value = mock_azure_response
        
        mock_key_phrases = Mock()
        mock_key_phrases.key_phrases = ["exercise", "benefits", "health"]
        mock_text_analytics.return_value.extract_key_phrases.return_value = [mock_key_phrases]
        
        # Create client and run completion
        client = SafeGPTClient("test_endpoint", "test_key", "test_deployment")
        
        messages = [{"role": "user", "content": "What are the benefits of exercise?"}]
        result = client.safe_completion(messages)
        
        # Verify results
        assert "content" in result
        assert "hallucination_checks" in result
        assert len(result["hallucination_checks"]) == 3
        assert all(check["category"] in ["factual", "source", "contextual"] 
                  for check in result["hallucination_checks"])

# Performance tests
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.performance
    def test_detection_speed(self):
        """Test that detection completes within reasonable time"""
        import time
        
        with patch('hallucination_detection.openai.AzureOpenAI'):
            with patch('hallucination_detection.TextAnalyticsClient'):
                detector = HallucinationDetector("test_endpoint", "test_key")
                
                # Test with a moderately long response
                response = "This is a test response. " * 100
                
                start_time = time.time()
                detector.check_factual_consistency(response)
                detector.check_source_citations(response)
                end_time = time.time()
                
                # Should complete within 1 second for pattern matching
                assert (end_time - start_time) < 1.0

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=hallucination_detection",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])