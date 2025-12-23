"""Unit tests for the Risk Identifier Agent.

Tests cover:
- Prompt formatting with various clause types
- JSON output parsing and validation
- Risk type classification
- Severity scoring
- Overall risk rating calculation
- Edge cases (clean clauses, multiple risks, undefined terms)
- Retry logic
- Cost tracking
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from core.agents.risk_identifier_agent import (
    Risk,
    RiskIdentifierAgent,
    RiskIdentifierInput,
    RiskIdentifierOutput,
    ClauseMetadata,
    RetryConfig,
)
from core.agents.prompts.risk_identifier_prompt import (
    RISK_IDENTIFIER_SYSTEM_PROMPT,
    RISK_IDENTIFIER_USER_PROMPT_TEMPLATE,
    EXAMPLE_CLAUSE_CLEAN,
    EXAMPLE_CLAUSE_HIGH_RISK,
    EXAMPLE_CLAUSE_AMBIGUOUS,
    EXAMPLE_CLAUSE_MULTI_RISK,
    format_risk_identifier_prompt,
)
from core.agents.utils.risk_taxonomy import (
    RiskType,
    Severity,
    OverallRiskRating,
    calculate_overall_rating,
    aggregate_risks,
    normalize_risk_type,
    normalize_severity,
    get_risk_description,
)
from core.cost_tracker import CostTracker, ExecutionStatus


class TestPromptFormatting:
    """Tests for prompt formatting with various clause types."""

    def test_format_prompt_with_full_metadata(self):
        """Test prompt formatting with all metadata fields."""
        prompt = format_risk_identifier_prompt(
            clause_id="clause_001",
            clause_text="The Seller shall indemnify without limitation...",
            clause_type="Indemnity",
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="Delaware",
        )
        
        assert "clause_001" in prompt
        assert "without limitation" in prompt
        assert "Indemnity" in prompt
        assert "SPA" in prompt
        assert "Seller, Buyer" in prompt
        assert "Delaware" in prompt

    def test_format_prompt_with_empty_metadata(self):
        """Test prompt formatting with missing metadata."""
        prompt = format_risk_identifier_prompt(
            clause_id="clause_002",
            clause_text="Some clause text",
        )
        
        assert "clause_002" in prompt
        assert "Not specified" in prompt

    def test_format_prompt_high_risk_clause(self):
        """Test prompt with known high-risk clause."""
        prompt = format_risk_identifier_prompt(
            clause_id="clause_003",
            clause_text=EXAMPLE_CLAUSE_HIGH_RISK,
            clause_type="Indemnity",
        )
        
        assert "without limitation" in prompt.lower()
        assert "indemnify" in prompt.lower()

    def test_format_prompt_ambiguous_clause(self):
        """Test prompt with ambiguous clause."""
        prompt = format_risk_identifier_prompt(
            clause_id="clause_004",
            clause_text=EXAMPLE_CLAUSE_AMBIGUOUS,
        )
        
        assert "Material Adverse Effect" in prompt


class TestRiskModel:
    """Tests for the Risk Pydantic model."""

    def test_valid_risk(self):
        """Test creating a valid risk."""
        risk = Risk(
            risk_type="unlimited_exposure",
            risk_description="Unlimited indemnification without caps",
            severity="high",
            remediation_suggestion="Add liability cap",
            confidence=0.9,
            supporting_quote="without limitation as to amount",
        )
        
        assert risk.risk_type == "unlimited_exposure"
        assert risk.severity == "high"
        assert risk.confidence == 0.9

    def test_confidence_clamping_high(self):
        """Test confidence above 1.0 is clamped."""
        risk = Risk(
            risk_type="ambiguous_definition",
            risk_description="Unclear definition",
            severity="medium",
            confidence=1.5,
        )
        
        assert risk.confidence == 1.0

    def test_confidence_clamping_low(self):
        """Test confidence below 0.0 is clamped."""
        risk = Risk(
            risk_type="missing_safeguard",
            risk_description="Missing cure period",
            severity="high",
            confidence=-0.3,
        )
        
        assert risk.confidence == 0.0

    def test_risk_type_normalization(self):
        """Test risk type is normalized."""
        risk = Risk(
            risk_type="Unlimited Exposure",  # Should normalize
            risk_description="No caps",
            severity="high",
            confidence=0.8,
        )
        
        assert risk.risk_type == "unlimited_exposure"

    def test_severity_normalization(self):
        """Test severity is normalized."""
        risk = Risk(
            risk_type="unusual_language",
            risk_description="Archaic terms",
            severity="HIGH",  # Should normalize
            confidence=0.7,
        )
        
        assert risk.severity == "high"


class TestRiskIdentifierOutput:
    """Tests for the RiskIdentifierOutput model."""

    def test_output_with_risks(self):
        """Test output with multiple risks."""
        output = RiskIdentifierOutput(
            clause_id="clause_001",
            risks_identified=[
                Risk(
                    risk_type="unlimited_exposure",
                    risk_description="No cap on liability",
                    severity="critical",
                    confidence=0.95,
                ),
                Risk(
                    risk_type="missing_safeguard",
                    risk_description="No cure period",
                    severity="high",
                    confidence=0.85,
                ),
            ],
            overall_risk_rating="critical_issues",
        )
        
        assert len(output.risks_identified) == 2
        assert output.overall_risk_rating == "critical_issues"

    def test_output_clean(self):
        """Test output with no risks."""
        output = RiskIdentifierOutput(
            clause_id="clause_002",
            risks_identified=[],
            overall_risk_rating="clean",
        )
        
        assert len(output.risks_identified) == 0
        assert output.overall_risk_rating == "clean"


class TestRiskTaxonomy:
    """Tests for risk taxonomy utilities."""

    def test_all_risk_types_defined(self):
        """Test all risk types have descriptions."""
        for risk_type in RiskType:
            description = get_risk_description(risk_type)
            assert description != "Unknown risk type."

    def test_normalize_risk_type_valid(self):
        """Test normalizing valid risk types."""
        assert normalize_risk_type("unusual_language") == "unusual_language"
        assert normalize_risk_type("UNLIMITED_EXPOSURE") == "unlimited_exposure"

    def test_normalize_risk_type_variations(self):
        """Test normalizing risk type variations."""
        assert normalize_risk_type("archaic language") == "unusual_language"
        assert normalize_risk_type("vague terms") == "ambiguous_definition"
        assert normalize_risk_type("uncapped liability") == "unlimited_exposure"

    def test_normalize_severity_valid(self):
        """Test normalizing valid severities."""
        assert normalize_severity("critical") == "critical"
        assert normalize_severity("HIGH") == "high"
        assert normalize_severity("Medium") == "medium"

    def test_normalize_severity_variations(self):
        """Test normalizing severity variations."""
        assert normalize_severity("severe") == "critical"
        assert normalize_severity("major") == "high"
        assert normalize_severity("moderate") == "medium"


class TestOverallRatingCalculation:
    """Tests for overall risk rating calculation."""

    def test_clean_rating_no_risks(self):
        """Test clean rating with no risks."""
        rating = calculate_overall_rating([])
        assert rating == OverallRiskRating.CLEAN

    def test_minor_issues_low_only(self):
        """Test minor issues with only low severity."""
        rating = calculate_overall_rating([
            {"severity": "low"},
            {"severity": "low"},
        ])
        assert rating == OverallRiskRating.MINOR_ISSUES

    def test_significant_issues_medium(self):
        """Test significant issues with medium severity."""
        rating = calculate_overall_rating([
            {"severity": "medium"},
            {"severity": "low"},
        ])
        assert rating == OverallRiskRating.SIGNIFICANT_ISSUES

    def test_significant_issues_high(self):
        """Test significant issues with high severity."""
        rating = calculate_overall_rating([
            {"severity": "high"},
        ])
        assert rating == OverallRiskRating.SIGNIFICANT_ISSUES

    def test_critical_issues(self):
        """Test critical issues with critical severity."""
        rating = calculate_overall_rating([
            {"severity": "critical"},
            {"severity": "low"},
        ])
        assert rating == OverallRiskRating.CRITICAL_ISSUES


class TestRiskAggregation:
    """Tests for risk aggregation utilities."""

    def test_aggregate_empty(self):
        """Test aggregating empty risks."""
        result = aggregate_risks([])
        
        assert result["total_risks"] == 0
        assert result["risk_score"] == 0

    def test_aggregate_single_risk(self):
        """Test aggregating single risk."""
        result = aggregate_risks([
            {"severity": "high", "risk_type": "unlimited_exposure"}
        ])
        
        assert result["total_risks"] == 1
        assert result["by_severity"]["high"] == 1
        assert result["by_type"]["unlimited_exposure"] == 1

    def test_aggregate_multiple_risks(self):
        """Test aggregating multiple risks."""
        result = aggregate_risks([
            {"severity": "critical", "risk_type": "unlimited_exposure"},
            {"severity": "high", "risk_type": "missing_safeguard"},
            {"severity": "medium", "risk_type": "ambiguous_definition"},
            {"severity": "low", "risk_type": "unusual_language"},
        ])
        
        assert result["total_risks"] == 4
        assert result["by_severity"]["critical"] == 1
        assert result["by_severity"]["high"] == 1
        assert result["highest_severity"] == "critical"
        assert result["risk_score"] > 0


class TestJSONParsing:
    """Tests for JSON output parsing."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return RiskIdentifierAgent()

    def test_parse_valid_json(self, agent: RiskIdentifierAgent):
        """Test parsing valid JSON."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "risks_identified": [
                {
                    "risk_type": "unlimited_exposure",
                    "risk_description": "No cap",
                    "severity": "high",
                    "remediation_suggestion": "Add cap",
                    "confidence": 0.9,
                    "supporting_quote": "without limitation"
                }
            ],
            "overall_risk_rating": "significant_issues"
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert len(result.risks_identified) == 1
        assert result.risks_identified[0].risk_type == "unlimited_exposure"

    def test_parse_json_with_markdown(self, agent: RiskIdentifierAgent):
        """Test parsing JSON wrapped in markdown."""
        json_str = """```json
{
  "clause_id": "clause_001",
  "risks_identified": [],
  "overall_risk_rating": "clean"
}
```"""
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.overall_risk_rating == "clean"

    def test_parse_json_with_trailing_comma(self, agent: RiskIdentifierAgent):
        """Test parsing JSON with trailing commas."""
        json_str = """{
  "clause_id": "clause_001",
  "risks_identified": [
    {
      "risk_type": "unusual_language",
      "risk_description": "Archaic terms",
      "severity": "low",
      "confidence": 0.7,
    },
  ],
  "overall_risk_rating": "minor_issues",
}"""
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert len(result.risks_identified) == 1

    def test_parse_clean_clause(self, agent: RiskIdentifierAgent):
        """Test parsing response with no risks."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "risks_identified": [],
            "overall_risk_rating": "clean"
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert len(result.risks_identified) == 0
        assert result.overall_risk_rating == "clean"


class TestRetryLogic:
    """Tests for retry logic."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0
        assert "timeout" in config.retryable_errors

    def test_is_retryable_timeout(self):
        """Test timeout is retryable."""
        agent = RiskIdentifierAgent()
        
        error = Exception("Connection timeout")
        assert agent._is_retryable_error(error) is True

    def test_is_retryable_rate_limit(self):
        """Test rate limit is retryable."""
        agent = RiskIdentifierAgent()
        
        error = Exception("Rate limit exceeded")
        assert agent._is_retryable_error(error) is True

    def test_is_not_retryable_validation(self):
        """Test validation errors are not retryable."""
        agent = RiskIdentifierAgent()
        
        error = Exception("Invalid request format")
        assert agent._is_retryable_error(error) is False


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return RiskIdentifierAgent()

    def test_empty_clause_text(self):
        """Test handling empty clause text."""
        input_data = RiskIdentifierInput(
            clause_id="clause_empty",
            clause_text="",
        )
        
        assert input_data.clause_id == "clause_empty"

    def test_malformed_json_fallback(self, agent: RiskIdentifierAgent):
        """Test fallback for malformed JSON."""
        malformed = "This is not JSON"
        
        result = agent._parse_json_response(malformed, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert result.overall_risk_rating == "clean"

    def test_multiple_risks_single_clause(self, agent: RiskIdentifierAgent):
        """Test multiple risks in single clause."""
        json_str = json.dumps({
            "clause_id": "clause_multi",
            "risks_identified": [
                {"risk_type": "unlimited_exposure", "risk_description": "No cap", "severity": "critical", "confidence": 0.9},
                {"risk_type": "missing_safeguard", "risk_description": "No cure", "severity": "high", "confidence": 0.85},
                {"risk_type": "ambiguous_definition", "risk_description": "Vague", "severity": "medium", "confidence": 0.7},
            ],
            "overall_risk_rating": "critical_issues"
        })
        
        result = agent._parse_json_response(json_str, "clause_multi")
        
        assert len(result.risks_identified) == 3
        assert result.overall_risk_rating == "critical_issues"

    def test_special_characters_in_clause(self):
        """Test handling special characters."""
        input_data = RiskIdentifierInput(
            clause_id="clause_special",
            clause_text='The "Buyer" shall pay < $5M & > $1M',
        )
        
        assert "&" in input_data.clause_text
        assert '"' in input_data.clause_text


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_cost_calculation_basic(self):
        """Test basic cost calculation."""
        agent = RiskIdentifierAgent()
        
        cost = agent._calculate_cost(input_tokens=1000, output_tokens=500)
        
        expected = (1000 * 0.05 / 1_000_000) + (500 * 0.4 / 1_000_000)
        assert cost == pytest.approx(expected, rel=0.01)

    def test_cost_calculation_zero(self):
        """Test cost calculation with zero tokens."""
        agent = RiskIdentifierAgent()
        
        cost = agent._calculate_cost(input_tokens=0, output_tokens=0)
        
        assert cost == 0.0


class TestCostTracker:
    """Tests for cost tracking integration."""

    def test_create_log(self):
        """Test creating execution log."""
        tracker = CostTracker()
        
        log = tracker.create_log(
            agent_type="RISK_IDENTIFICATION",
            clause_id="clause_001",
            model="gpt-5-nano",
            input_tokens=400,
            output_tokens=300,
            execution_time_ms=1500,
            status=ExecutionStatus.SUCCESS,
            extra_data={"risks_found": 3, "overall_rating": "significant_issues"},
        )
        
        assert log.agent_type == "RISK_IDENTIFICATION"
        assert log.extra_data["risks_found"] == 3


class TestAgentConfiguration:
    """Tests for agent configuration."""

    def test_model_name(self):
        """Test model name."""
        agent = RiskIdentifierAgent()
        
        assert agent.MODEL_NAME == "gpt-5-nano"

    def test_seed_for_determinism(self):
        """Test seed is set."""
        agent = RiskIdentifierAgent()
        
        assert agent.SEED == 42

    def test_max_completion_tokens(self):
        """Test max tokens for detailed responses."""
        agent = RiskIdentifierAgent()
        
        assert agent.MAX_COMPLETION_TOKENS == 1000


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input(self):
        """Test creating valid input."""
        input_data = RiskIdentifierInput(
            clause_id="clause_001",
            clause_text="The Seller shall...",
            clause_type="Indemnity",
            metadata=ClauseMetadata(
                contract_type="SPA",
                parties=["Seller", "Buyer"],
                jurisdiction="Delaware",
            ),
        )
        
        assert input_data.clause_id == "clause_001"
        assert input_data.metadata.contract_type == "SPA"

    def test_input_requires_clause_id(self):
        """Test clause_id is required."""
        with pytest.raises(ValueError):
            RiskIdentifierInput(clause_text="Some text")

    def test_input_requires_clause_text(self):
        """Test clause_text is required."""
        with pytest.raises(ValueError):
            RiskIdentifierInput(clause_id="clause_001")
