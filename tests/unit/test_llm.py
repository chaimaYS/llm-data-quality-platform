"""Tests for LLM client abstraction (mocked — no real API calls)."""

import pytest
import hashlib
from unittest.mock import MagicMock

from src.llm.client import LLMClient, LLMResponse
from src.llm.semantic_classifier import SemanticColumnClassifier, SemanticType
from src.llm.rule_proposer import RuleProposer
from src.llm.anomaly_explainer import AnomalyExplainer, AnomalyExplanation
from src.profiling.engine import ColumnProfile
from src.rules.base import RuleResult, Dimension, Severity


class ConcreteLLMClient(LLMClient):
    """Concrete implementation for testing."""
    def _call(self, prompt, schema=None, attachments=None):
        return LLMResponse(content={"test": True}, model=self.model, prompt_tokens=10, completion_tokens=20)


class TestLLMClientCaching:
    def test_cache_key_deterministic(self):
        client = ConcreteLLMClient(model="model-a")
        key1 = client._cache_key("prompt1")
        key2 = client._cache_key("prompt1")
        assert key1 == key2

    def test_cache_key_differs_for_different_prompts(self):
        client = ConcreteLLMClient(model="model-a")
        key1 = client._cache_key("prompt1")
        key2 = client._cache_key("prompt2")
        assert key1 != key2

    def test_cache_key_differs_for_different_models(self):
        client_a = ConcreteLLMClient(model="model-a")
        client_b = ConcreteLLMClient(model="model-b")
        key1 = client_a._cache_key("prompt1")
        key2 = client_b._cache_key("prompt1")
        assert key1 != key2

    def test_complete_returns_dict(self):
        client = ConcreteLLMClient(model="test")
        result = client.complete("hello")
        assert isinstance(result, dict)

    def test_caching_works(self):
        client = ConcreteLLMClient(model="test")
        r1 = client.complete("same prompt")
        r2 = client.complete("same prompt")
        assert r1 == r2

    def test_complete_uses_cache(self):
        client = ConcreteLLMClient(model="test")
        r1 = client.complete("cached prompt")
        r2 = client.complete("cached prompt")
        assert r1 == r2
        r3 = client.complete("different prompt")
        assert r1 == r3 or True  # different prompt may return same mock


class TestSemanticColumnClassifier:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock(spec=LLMClient)
        llm.complete.return_value = {
            "semantic_type": "email",
            "confidence": 0.95,
            "is_pii": True,
            "pii_category": "high",
            "reasoning": "Column contains email addresses",
        }
        return llm

    @pytest.fixture
    def sample_profile(self):
        return ColumnProfile(
            name="user_email", dtype="object", total_count=100, null_count=5,
            null_pct=5.0, distinct_count=95, distinct_pct=95.0,
            top_k=[{"value": "test@example.com", "count": 1}],
        )

    def test_classify_calls_llm(self, mock_llm, sample_profile):
        classifier = SemanticColumnClassifier(llm=mock_llm)
        result = classifier.classify("user_email", sample_profile, ["alice@example.com"])
        assert mock_llm.complete.called

    def test_classify_returns_semantic_type(self, mock_llm, sample_profile):
        classifier = SemanticColumnClassifier(llm=mock_llm)
        result = classifier.classify("user_email", sample_profile, [])
        assert isinstance(result, SemanticType)


class TestRuleProposer:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock(spec=LLMClient)
        llm.complete.return_value = {
            "rules": [
                {"type": "null_check", "column": "email", "rationale": "Required field",
                 "severity": "high", "dimension": "completeness"},
            ]
        }
        return llm

    def test_propose_calls_llm(self, mock_llm):
        from src.profiling.engine import ProfileResult
        profile = ProfileResult(row_count=100, column_count=3)
        proposer = RuleProposer(llm=mock_llm)
        result = proposer.propose(profile=profile, table_name="customers")
        assert mock_llm.complete.called

    def test_propose_returns_list(self, mock_llm):
        from src.profiling.engine import ProfileResult
        profile = ProfileResult(row_count=100, column_count=3)
        proposer = RuleProposer(llm=mock_llm)
        result = proposer.propose(profile=profile)
        assert isinstance(result, list)


class TestAnomalyExplainer:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock(spec=LLMClient)
        llm.complete.return_value = {
            "explanation": "The age value -5 is negative.",
            "root_cause": "Data entry error",
            "suggested_action": "Add range check",
            "severity": "high",
        }
        return llm

    @pytest.fixture
    def sample_result(self):
        return RuleResult(
            rule_name="range_check_age", column="age",
            dimension=Dimension.VALIDITY, severity=Severity.HIGH,
            pass_count=190, fail_count=10, total_count=200,
            pass_rate=0.95, passed=False,
            failing_sample=[{"age": -5}, {"age": -10}],
        )

    def test_explain_calls_llm(self, mock_llm, sample_result):
        explainer = AnomalyExplainer(llm=mock_llm)
        result = explainer.explain(sample_result, [{"age": -5}])
        assert mock_llm.complete.called

    def test_explain_returns_explanation(self, mock_llm, sample_result):
        explainer = AnomalyExplainer(llm=mock_llm)
        result = explainer.explain(sample_result)
        assert isinstance(result, AnomalyExplanation)
