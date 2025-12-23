"""Unit tests for the Comparison Agent.

Tests cover:
- Prompt formatting with various clause types
- JSON output parsing and validation
- Deviation type classification
- Alignment score calculation
- Impact assessment
- Edge cases
- Retry logic
- Cost tracking
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from core.agents.comparison_agent import (
    Deviation,
    ComparisonAgent,
    ComparisonInput,
    ComparisonOutput,
    ClauseMetadata,
    ReferenceStandardInput,
    RetryConfig,
)
from core.agents.prompts.comparison_agent_prompt import (
    COMPARISON_AGENT_SYSTEM_PROMPT,
    COMPARISON_AGENT_USER_PROMPT_TEMPLATE,
    EXAMPLE_CLAUSE_DEVIATION,
    EXAMPLE_REFERENCE_STANDARD,
    format_comparison_agent_prompt,
)
from core.agents.utils.deviation_taxonomy import (
    DeviationType,
    Severity,
    OverallAssessment,
    calculate_alignment_score,
    determine_overall_assessment,
    aggregate_deviations,
    normalize_deviation_type,
    normalize_severity,
)
from core.agents.utils.reference_standards import (
    get_reference_standard,
    format_reference_standards_for_prompt,
)
from core.cost_tracker import CostTracker, ExecutionStatus


class TestPromptFormatting:
    """Tests for prompt formatting with various clause types."""

    def test_format_prompt_with_full_metadata(self):
        """Test prompt formatting with all metadata fields."""
        prompt = format_comparison_agent_prompt(
            clause_id="clause_001",
            clause_text="The Seller shall indemnify the Buyer...",
            clause_type="Indemnity",
            reference_standards=EXAMPLE_REFERENCE_STANDARD,
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="Delaware",
        )
        
        assert "clause_001" in prompt
        assert "indemnify" in prompt.lower()
        assert "Indemnity" in prompt
        assert "SPA" in prompt
        assert "Delaware" in prompt

    def test_format_prompt_without_standards(self):
        """Test prompt formatting without reference standards."""
        prompt = format_comparison_agent_prompt(
            clause_id="clause_002",
            clause_text="Some clause text",
        )
        
        assert "clause_002" in prompt
        assert "No reference standards" in prompt

    def test_format_prompt_with_reference_standards(self):
        """Test prompt with reference standards included."""
        prompt = format_comparison_agent_prompt(
            clause_id="clause_003",
            clause_text=EXAMPLE_CLAUSE_DEVIATION,
            reference_standards=EXAMPLE_REFERENCE_STANDARD,
        )
        
        assert "market_standard_2023" in prompt
        assert "85% of M&A transactions" in prompt


class TestDeviationModel:
    """Tests for the Deviation Pydantic model."""

    def test_valid_deviation(self):
        """Test creating a valid deviation."""
        deviation = Deviation(
            deviation_type="more_restrictive",
            standard_language="limited to 12 months",
            actual_language="perpetual period",
            impact="Favors Buyer significantly",
            market_prevalence="Only 5% of SPAs",
            severity="critical",
            remediation_suggestion="Add sunset provision",
        )
        
        assert deviation.deviation_type == "more_restrictive"
        assert deviation.severity == "critical"

    def test_deviation_type_normalization(self):
        """Test deviation type is normalized."""
        deviation = Deviation(
            deviation_type="MORE_RESTRICTIVE",
            severity="high",
        )
        
        assert deviation.deviation_type == "more_restrictive"

    def test_severity_normalization(self):
        """Test severity is normalized."""
        deviation = Deviation(
            deviation_type="non_standard",
            severity="HIGH",
        )
        
        assert deviation.severity == "high"


class TestComparisonOutput:
    """Tests for the ComparisonOutput model."""

    def test_output_with_deviations(self):
        """Test output with deviations."""
        output = ComparisonOutput(
            clause_id="clause_001",
            deviations_from_standard=[
                Deviation(
                    deviation_type="less_protective",
                    severity="high",
                    impact="Buyer exposed",
                ),
            ],
            alignment_score=0.75,
            overall_assessment="minor_deviations",
            confidence_score=0.85,
        )
        
        assert len(output.deviations_from_standard) == 1
        assert output.overall_assessment == "minor_deviations"

    def test_output_market_standard(self):
        """Test output when clause is market standard."""
        output = ComparisonOutput(
            clause_id="clause_002",
            deviations_from_standard=[],
            alignment_score=1.0,
            overall_assessment="market_standard",
            confidence_score=0.95,
        )
        
        assert len(output.deviations_from_standard) == 0
        assert output.alignment_score == 1.0


class TestDeviationTaxonomy:
    """Tests for deviation taxonomy utilities."""

    def test_all_deviation_types_defined(self):
        """Test all deviation types are defined."""
        assert len(DeviationType) == 5

    def test_normalize_deviation_type_valid(self):
        """Test normalizing valid deviation types."""
        assert normalize_deviation_type("more_restrictive") == "more_restrictive"
        assert normalize_deviation_type("LESS_PROTECTIVE") == "less_protective"

    def test_normalize_deviation_type_variations(self):
        """Test normalizing deviation type variations."""
        assert normalize_deviation_type("strict") == "more_restrictive"
        assert normalize_deviation_type("missing protection") == "missing_safeguard"
        assert normalize_deviation_type("extra obligation") == "additional_obligation"

    def test_normalize_severity_valid(self):
        """Test normalizing valid severities."""
        assert normalize_severity("critical") == "critical"
        assert normalize_severity("HIGH") == "high"


class TestAlignmentScoring:
    """Tests for alignment score calculation."""

    def test_perfect_alignment_no_deviations(self):
        """Test perfect score with no deviations."""
        score = calculate_alignment_score([])
        assert score == 1.0

    def test_score_with_low_deviation(self):
        """Test score with low severity deviation."""
        score = calculate_alignment_score([{"severity": "low"}])
        assert score == pytest.approx(0.95, rel=0.01)

    def test_score_with_critical_deviation(self):
        """Test score with critical deviation."""
        score = calculate_alignment_score([{"severity": "critical"}])
        assert score == pytest.approx(0.60, rel=0.01)

    def test_score_with_multiple_deviations(self):
        """Test score with multiple deviations."""
        score = calculate_alignment_score([
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "low"},
        ])
        assert score < 1.0
        assert score > 0.5


class TestOverallAssessment:
    """Tests for overall assessment determination."""

    def test_market_standard_assessment(self):
        """Test market standard assessment."""
        assessment = determine_overall_assessment(0.95)
        assert assessment == OverallAssessment.MARKET_STANDARD

    def test_minor_deviations_assessment(self):
        """Test minor deviations assessment."""
        assessment = determine_overall_assessment(0.80)
        assert assessment == OverallAssessment.MINOR_DEVIATIONS

    def test_significant_deviations_assessment(self):
        """Test significant deviations assessment."""
        assessment = determine_overall_assessment(0.60)
        assert assessment == OverallAssessment.SIGNIFICANT_DEVIATIONS

    def test_highly_unusual_assessment(self):
        """Test highly unusual assessment."""
        assessment = determine_overall_assessment(0.30)
        assert assessment == OverallAssessment.HIGHLY_UNUSUAL


class TestDeviationAggregation:
    """Tests for deviation aggregation utilities."""

    def test_aggregate_empty(self):
        """Test aggregating empty deviations."""
        result = aggregate_deviations([])
        
        assert result["total_deviations"] == 0
        assert result["alignment_score"] == 1.0

    def test_aggregate_single_deviation(self):
        """Test aggregating single deviation."""
        result = aggregate_deviations([
            {"severity": "high", "deviation_type": "less_protective"}
        ])
        
        assert result["total_deviations"] == 1
        assert result["by_severity"]["high"] == 1

    def test_aggregate_multiple_deviations(self):
        """Test aggregating multiple deviations."""
        result = aggregate_deviations([
            {"severity": "critical", "deviation_type": "less_protective"},
            {"severity": "high", "deviation_type": "more_restrictive"},
            {"severity": "medium", "deviation_type": "non_standard"},
        ])
        
        assert result["total_deviations"] == 3
        assert result["highest_severity"] == "critical"


class TestReferenceStandards:
    """Tests for reference standards utilities."""

    def test_get_reference_standard_indemnity(self):
        """Test getting indemnity reference standard."""
        standards = get_reference_standard("Indemnity")
        
        assert len(standards) > 0
        assert standards[0].clause_type == "Indemnity"

    def test_get_reference_standard_unknown(self):
        """Test getting unknown clause type."""
        standards = get_reference_standard("UnknownType")
        
        assert len(standards) == 0

    def test_format_reference_standards(self):
        """Test formatting reference standards for prompt."""
        standards = get_reference_standard("Indemnity")
        formatted = format_reference_standards_for_prompt(standards)
        
        assert "STANDARD 1" in formatted
        assert "Prevalence:" in formatted


class TestJSONParsing:
    """Tests for JSON output parsing."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return ComparisonAgent()

    def test_parse_valid_json(self, agent: ComparisonAgent):
        """Test parsing valid JSON."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "deviations_from_standard": [
                {
                    "deviation_type": "less_protective",
                    "standard_language": "limited liability",
                    "actual_language": "unlimited liability",
                    "impact": "Seller exposed",
                    "market_prevalence": "10% of SPAs",
                    "severity": "critical",
                    "remediation_suggestion": "Add cap"
                }
            ],
            "alignment_score": 0.6,
            "overall_assessment": "significant_deviations",
            "confidence_score": 0.85
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert len(result.deviations_from_standard) == 1

    def test_parse_json_with_markdown(self, agent: ComparisonAgent):
        """Test parsing JSON wrapped in markdown."""
        json_str = """```json
{
  "clause_id": "clause_001",
  "deviations_from_standard": [],
  "alignment_score": 1.0,
  "overall_assessment": "market_standard",
  "confidence_score": 0.95
}
```"""
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.overall_assessment == "market_standard"

    def test_parse_market_standard(self, agent: ComparisonAgent):
        """Test parsing response with no deviations."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "deviations_from_standard": [],
            "alignment_score": 1.0,
            "overall_assessment": "market_standard",
            "confidence_score": 0.95
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert len(result.deviations_from_standard) == 0
        assert result.alignment_score == 1.0


class TestRetryLogic:
    """Tests for retry logic."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0

    def test_is_retryable_timeout(self):
        """Test timeout is retryable."""
        agent = ComparisonAgent()
        
        error = Exception("Connection timeout")
        assert agent._is_retryable_error(error) is True

    def test_is_retryable_rate_limit(self):
        """Test rate limit is retryable."""
        agent = ComparisonAgent()
        
        error = Exception("Rate limit exceeded")
        assert agent._is_retryable_error(error) is True

    def test_is_not_retryable_validation(self):
        """Test validation errors are not retryable."""
        agent = ComparisonAgent()
        
        error = Exception("Invalid request format")
        assert agent._is_retryable_error(error) is False


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return ComparisonAgent()

    def test_empty_clause_text(self):
        """Test handling empty clause text."""
        input_data = ComparisonInput(
            clause_id="clause_empty",
            clause_text="",
        )
        
        assert input_data.clause_id == "clause_empty"

    def test_malformed_json_fallback(self, agent: ComparisonAgent):
        """Test fallback for malformed JSON."""
        malformed = "This is not JSON"
        
        result = agent._parse_json_response(malformed, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert result.alignment_score == 1.0

    def test_multiple_deviations(self, agent: ComparisonAgent):
        """Test multiple deviations in single clause."""
        json_str = json.dumps({
            "clause_id": "clause_multi",
            "deviations_from_standard": [
                {"deviation_type": "less_protective", "severity": "critical"},
                {"deviation_type": "more_restrictive", "severity": "high"},
                {"deviation_type": "non_standard", "severity": "medium"},
            ],
            "alignment_score": 0.25,
            "overall_assessment": "highly_unusual",
            "confidence_score": 0.8
        })
        
        result = agent._parse_json_response(json_str, "clause_multi")
        
        assert len(result.deviations_from_standard) == 3


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_cost_calculation_basic(self):
        """Test basic cost calculation."""
        agent = ComparisonAgent()
        
        cost = agent._calculate_cost(input_tokens=1000, output_tokens=500)
        
        expected = (1000 * 0.05 / 1_000_000) + (500 * 0.4 / 1_000_000)
        assert cost == pytest.approx(expected, rel=0.01)

    def test_cost_calculation_zero(self):
        """Test cost calculation with zero tokens."""
        agent = ComparisonAgent()
        
        cost = agent._calculate_cost(input_tokens=0, output_tokens=0)
        
        assert cost == 0.0


class TestCostTracker:
    """Tests for cost tracking integration."""

    def test_create_log(self):
        """Test creating execution log."""
        tracker = CostTracker()
        
        log = tracker.create_log(
            agent_type="COMPARISON",
            clause_id="clause_001",
            model="gpt-5-nano",
            input_tokens=500,
            output_tokens=400,
            execution_time_ms=1500,
            status=ExecutionStatus.SUCCESS,
            extra_data={
                "deviation_count": 2,
                "alignment_score": 0.75,
            },
        )
        
        assert log.agent_type == "COMPARISON"
        assert log.extra_data["deviation_count"] == 2


class TestAgentConfiguration:
    """Tests for agent configuration."""

    def test_model_name(self):
        """Test model name."""
        agent = ComparisonAgent()
        
        assert agent.MODEL_NAME == "gpt-5-nano"

    def test_seed_for_determinism(self):
        """Test seed is set."""
        agent = ComparisonAgent()
        
        assert agent.SEED == 42

    def test_max_completion_tokens(self):
        """Test max tokens configuration."""
        agent = ComparisonAgent()
        
        assert agent.MAX_COMPLETION_TOKENS == 1000


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input(self):
        """Test creating valid input."""
        input_data = ComparisonInput(
            clause_id="clause_001",
            clause_text="The Seller shall indemnify...",
            clause_type="Indemnity",
            reference_standards=[
                ReferenceStandardInput(
                    standard_type="market_standard_2023",
                    standard_text="Standard indemnity...",
                    prevalence="85% of SPAs",
                )
            ],
        )
        
        assert input_data.clause_id == "clause_001"
        assert len(input_data.reference_standards) == 1

    def test_input_requires_clause_id(self):
        """Test clause_id is required."""
        with pytest.raises(ValueError):
            ComparisonInput(clause_text="Some text")

    def test_input_requires_clause_text(self):
        """Test clause_text is required."""
        with pytest.raises(ValueError):
            ComparisonInput(clause_id="clause_001")
