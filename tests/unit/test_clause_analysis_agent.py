"""Unit tests for the Clause Analysis Agent.

Tests cover:
- Prompt formatting with various clause types
- JSON output parsing and validation
- Finding type classification
- Importance scoring
- Risk level assessment
- Clarification flagging
- Edge cases
- Retry logic
- Cost tracking
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from core.agents.clause_analysis_agent import (
    Finding,
    ClauseAnalysisAgent,
    ClauseAnalysisInput,
    ClauseAnalysisOutput,
    ClauseMetadata,
    RetryConfig,
)
from core.agents.prompts.clause_analysis_prompt import (
    CLAUSE_ANALYSIS_SYSTEM_PROMPT,
    CLAUSE_ANALYSIS_USER_PROMPT_TEMPLATE,
    EXAMPLE_CLAUSE_INDEMNITY,
    EXAMPLE_CLAUSE_TERMINATION,
    EXAMPLE_CLAUSE_PAYMENT,
    format_clause_analysis_prompt,
)
from core.agents.utils.finding_taxonomy import (
    FindingType,
    Importance,
    RiskLevel,
    calculate_risk_level,
    aggregate_findings,
    normalize_finding_type,
    normalize_importance,
    normalize_risk_level,
    get_finding_description,
)
from core.cost_tracker import CostTracker, ExecutionStatus


class TestPromptFormatting:
    """Tests for prompt formatting with various clause types."""

    def test_format_prompt_with_full_metadata(self):
        """Test prompt formatting with all metadata fields."""
        prompt = format_clause_analysis_prompt(
            clause_id="clause_001",
            clause_text="The Seller shall indemnify the Buyer...",
            clause_type="Indemnity",
            task_instructions="Extract indemnification obligations and caps",
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="Delaware",
        )
        
        assert "clause_001" in prompt
        assert "indemnify" in prompt.lower()
        assert "Indemnity" in prompt
        assert "SPA" in prompt
        assert "Seller, Buyer" in prompt
        assert "Delaware" in prompt
        assert "indemnification obligations" in prompt.lower()

    def test_format_prompt_with_empty_metadata(self):
        """Test prompt formatting with missing metadata."""
        prompt = format_clause_analysis_prompt(
            clause_id="clause_002",
            clause_text="Some clause text",
        )
        
        assert "clause_002" in prompt
        assert "Not specified" in prompt

    def test_format_prompt_indemnity_clause(self):
        """Test prompt with indemnity clause."""
        prompt = format_clause_analysis_prompt(
            clause_id="clause_003",
            clause_text=EXAMPLE_CLAUSE_INDEMNITY,
            clause_type="Indemnity",
        )
        
        assert "indemnify" in prompt.lower()
        assert "hold harmless" in prompt.lower()

    def test_format_prompt_termination_clause(self):
        """Test prompt with termination clause."""
        prompt = format_clause_analysis_prompt(
            clause_id="clause_004",
            clause_text=EXAMPLE_CLAUSE_TERMINATION,
            clause_type="Termination",
        )
        
        assert "terminate" in prompt.lower()
        assert "ninety (90) days" in prompt

    def test_format_prompt_payment_clause(self):
        """Test prompt with payment clause."""
        prompt = format_clause_analysis_prompt(
            clause_id="clause_005",
            clause_text=EXAMPLE_CLAUSE_PAYMENT,
            clause_type="Payment",
        )
        
        assert "thirty (30)" in prompt
        assert "interest" in prompt.lower()


class TestFindingModel:
    """Tests for the Finding Pydantic model."""

    def test_valid_finding(self):
        """Test creating a valid finding."""
        finding = Finding(
            finding_type="obligation",
            finding_text="Seller must indemnify Buyer",
            importance="high",
            confidence=0.9,
            supporting_quote="The Seller shall indemnify",
        )
        
        assert finding.finding_type == "obligation"
        assert finding.importance == "high"
        assert finding.confidence == 0.9

    def test_confidence_clamping_high(self):
        """Test confidence above 1.0 is clamped."""
        finding = Finding(
            finding_type="cap",
            finding_text="$5M liability cap",
            importance="medium",
            confidence=1.5,
        )
        
        assert finding.confidence == 1.0

    def test_confidence_clamping_low(self):
        """Test confidence below 0.0 is clamped."""
        finding = Finding(
            finding_type="exception",
            finding_text="Excludes gross negligence",
            importance="high",
            confidence=-0.3,
        )
        
        assert finding.confidence == 0.0

    def test_finding_type_normalization(self):
        """Test finding type is normalized."""
        finding = Finding(
            finding_type="OBLIGATION",
            finding_text="Must do something",
            importance="medium",
            confidence=0.8,
        )
        
        assert finding.finding_type == "obligation"

    def test_importance_normalization(self):
        """Test importance is normalized."""
        finding = Finding(
            finding_type="trigger",
            finding_text="Upon breach",
            importance="HIGH",
            confidence=0.7,
        )
        
        assert finding.importance == "high"


class TestClauseAnalysisOutput:
    """Tests for the ClauseAnalysisOutput model."""

    def test_output_with_findings(self):
        """Test output with multiple findings."""
        output = ClauseAnalysisOutput(
            clause_id="clause_001",
            findings=[
                Finding(
                    finding_type="obligation",
                    finding_text="Indemnification duty",
                    importance="high",
                    confidence=0.9,
                ),
                Finding(
                    finding_type="cap",
                    finding_text="$10M cap",
                    importance="critical",
                    confidence=0.95,
                ),
            ],
            summary="Seller indemnifies Buyer up to $10M.",
            risk_level="critical",
        )
        
        assert len(output.findings) == 2
        assert output.risk_level == "critical"
        assert "$10M" in output.summary

    def test_output_informational(self):
        """Test output with no concerning findings."""
        output = ClauseAnalysisOutput(
            clause_id="clause_002",
            findings=[],
            summary="Standard notice provision.",
            risk_level="informational",
        )
        
        assert len(output.findings) == 0
        assert output.risk_level == "informational"

    def test_output_with_clarification(self):
        """Test output requiring clarification."""
        output = ClauseAnalysisOutput(
            clause_id="clause_003",
            findings=[
                Finding(
                    finding_type="definition",
                    finding_text="Material Adverse Effect",
                    importance="high",
                    confidence=0.6,
                ),
            ],
            summary="MAE definition is vague.",
            risk_level="high",
            clarification_required=True,
            clarification_note="Definition of 'material' is ambiguous",
        )
        
        assert output.clarification_required is True
        assert "ambiguous" in output.clarification_note


class TestFindingTaxonomy:
    """Tests for finding taxonomy utilities."""

    def test_all_finding_types_defined(self):
        """Test all finding types have descriptions."""
        for finding_type in FindingType:
            description = get_finding_description(finding_type)
            assert description != "Unknown finding type."

    def test_normalize_finding_type_valid(self):
        """Test normalizing valid finding types."""
        assert normalize_finding_type("obligation") == "obligation"
        assert normalize_finding_type("CAP") == "cap"
        assert normalize_finding_type("Exception") == "exception"

    def test_normalize_finding_type_variations(self):
        """Test normalizing finding type variations."""
        assert normalize_finding_type("duty") == "obligation"
        assert normalize_finding_type("maximum limit") == "cap"
        assert normalize_finding_type("carve-out") == "exception"

    def test_normalize_importance_valid(self):
        """Test normalizing valid importance levels."""
        assert normalize_importance("critical") == "critical"
        assert normalize_importance("HIGH") == "high"
        assert normalize_importance("Medium") == "medium"

    def test_normalize_risk_level_valid(self):
        """Test normalizing valid risk levels."""
        assert normalize_risk_level("critical") == "critical"
        assert normalize_risk_level("INFORMATIONAL") == "informational"


class TestRiskLevelCalculation:
    """Tests for risk level calculation."""

    def test_informational_no_findings(self):
        """Test informational rating with no findings."""
        rating = calculate_risk_level([])
        assert rating == RiskLevel.INFORMATIONAL

    def test_low_with_low_only(self):
        """Test low rating with only low importance."""
        rating = calculate_risk_level([
            {"importance": "low"},
            {"importance": "low"},
        ])
        assert rating == RiskLevel.LOW

    def test_medium_with_medium_importance(self):
        """Test medium rating with medium importance."""
        rating = calculate_risk_level([
            {"importance": "medium"},
            {"importance": "low"},
        ])
        assert rating == RiskLevel.MEDIUM

    def test_high_with_high_importance(self):
        """Test high rating with high importance."""
        rating = calculate_risk_level([
            {"importance": "high"},
        ])
        assert rating == RiskLevel.HIGH

    def test_critical_with_critical_importance(self):
        """Test critical rating with critical importance."""
        rating = calculate_risk_level([
            {"importance": "critical"},
            {"importance": "low"},
        ])
        assert rating == RiskLevel.CRITICAL


class TestFindingAggregation:
    """Tests for finding aggregation utilities."""

    def test_aggregate_empty(self):
        """Test aggregating empty findings."""
        result = aggregate_findings([])
        
        assert result["total_findings"] == 0
        assert result["importance_score"] == 0

    def test_aggregate_single_finding(self):
        """Test aggregating single finding."""
        result = aggregate_findings([
            {"importance": "high", "finding_type": "obligation"}
        ])
        
        assert result["total_findings"] == 1
        assert result["by_importance"]["high"] == 1
        assert result["by_type"]["obligation"] == 1

    def test_aggregate_multiple_findings(self):
        """Test aggregating multiple findings."""
        result = aggregate_findings([
            {"importance": "critical", "finding_type": "cap"},
            {"importance": "high", "finding_type": "obligation"},
            {"importance": "medium", "finding_type": "exception"},
            {"importance": "low", "finding_type": "definition"},
        ])
        
        assert result["total_findings"] == 4
        assert result["by_importance"]["critical"] == 1
        assert result["by_importance"]["high"] == 1
        assert result["highest_importance"] == "critical"
        assert result["importance_score"] > 0


class TestJSONParsing:
    """Tests for JSON output parsing."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return ClauseAnalysisAgent()

    def test_parse_valid_json(self, agent: ClauseAnalysisAgent):
        """Test parsing valid JSON."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "findings": [
                {
                    "finding_type": "obligation",
                    "finding_text": "Must indemnify",
                    "importance": "high",
                    "confidence": 0.9,
                    "supporting_quote": "shall indemnify"
                }
            ],
            "summary": "Indemnification clause.",
            "risk_level": "high",
            "clarification_required": False,
            "clarification_note": None
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert len(result.findings) == 1
        assert result.summary == "Indemnification clause."

    def test_parse_json_with_markdown(self, agent: ClauseAnalysisAgent):
        """Test parsing JSON wrapped in markdown."""
        json_str = """```json
{
  "clause_id": "clause_001",
  "findings": [],
  "summary": "Standard clause.",
  "risk_level": "informational",
  "clarification_required": false,
  "clarification_note": null
}
```"""
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.risk_level == "informational"

    def test_parse_json_with_trailing_comma(self, agent: ClauseAnalysisAgent):
        """Test parsing JSON with trailing commas."""
        json_str = """{
  "clause_id": "clause_001",
  "findings": [
    {
      "finding_type": "cap",
      "finding_text": "$5M cap",
      "importance": "medium",
      "confidence": 0.8,
    },
  ],
  "summary": "Liability cap.",
  "risk_level": "medium",
}"""
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert len(result.findings) == 1

    def test_parse_empty_findings(self, agent: ClauseAnalysisAgent):
        """Test parsing response with no findings."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "findings": [],
            "summary": "No concerns.",
            "risk_level": "informational"
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert len(result.findings) == 0
        assert result.risk_level == "informational"


class TestRetryLogic:
    """Tests for retry logic."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0

    def test_is_retryable_timeout(self):
        """Test timeout is retryable."""
        agent = ClauseAnalysisAgent()
        
        error = Exception("Connection timeout")
        assert agent._is_retryable_error(error) is True

    def test_is_retryable_rate_limit(self):
        """Test rate limit is retryable."""
        agent = ClauseAnalysisAgent()
        
        error = Exception("Rate limit exceeded")
        assert agent._is_retryable_error(error) is True

    def test_is_not_retryable_validation(self):
        """Test validation errors are not retryable."""
        agent = ClauseAnalysisAgent()
        
        error = Exception("Invalid request format")
        assert agent._is_retryable_error(error) is False


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return ClauseAnalysisAgent()

    def test_empty_clause_text(self):
        """Test handling empty clause text."""
        input_data = ClauseAnalysisInput(
            clause_id="clause_empty",
            clause_text="",
        )
        
        assert input_data.clause_id == "clause_empty"

    def test_malformed_json_fallback(self, agent: ClauseAnalysisAgent):
        """Test fallback for malformed JSON."""
        malformed = "This is not JSON"
        
        result = agent._parse_json_response(malformed, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert result.risk_level == "informational"

    def test_multiple_findings_single_clause(self, agent: ClauseAnalysisAgent):
        """Test multiple findings in single clause."""
        json_str = json.dumps({
            "clause_id": "clause_multi",
            "findings": [
                {"finding_type": "obligation", "finding_text": "Must do X", "importance": "high", "confidence": 0.9},
                {"finding_type": "cap", "finding_text": "$5M cap", "importance": "critical", "confidence": 0.95},
                {"finding_type": "duration", "finding_text": "5 years", "importance": "medium", "confidence": 0.8},
            ],
            "summary": "Multi-finding clause.",
            "risk_level": "critical"
        })
        
        result = agent._parse_json_response(json_str, "clause_multi")
        
        assert len(result.findings) == 3
        assert result.risk_level == "critical"

    def test_clarification_required_flag(self, agent: ClauseAnalysisAgent):
        """Test clarification required flag."""
        json_str = json.dumps({
            "clause_id": "clause_unclear",
            "findings": [
                {"finding_type": "definition", "finding_text": "MAE", "importance": "high", "confidence": 0.5}
            ],
            "summary": "Unclear definition.",
            "risk_level": "high",
            "clarification_required": True,
            "clarification_note": "MAE definition needs review"
        })
        
        result = agent._parse_json_response(json_str, "clause_unclear")
        
        assert result.clarification_required is True
        assert "MAE" in result.clarification_note


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_cost_calculation_basic(self):
        """Test basic cost calculation."""
        agent = ClauseAnalysisAgent()
        
        cost = agent._calculate_cost(input_tokens=1000, output_tokens=500)
        
        expected = (1000 * 0.05 / 1_000_000) + (500 * 0.4 / 1_000_000)
        assert cost == pytest.approx(expected, rel=0.01)

    def test_cost_calculation_zero(self):
        """Test cost calculation with zero tokens."""
        agent = ClauseAnalysisAgent()
        
        cost = agent._calculate_cost(input_tokens=0, output_tokens=0)
        
        assert cost == 0.0


class TestCostTracker:
    """Tests for cost tracking integration."""

    def test_create_log(self):
        """Test creating execution log."""
        tracker = CostTracker()
        
        log = tracker.create_log(
            agent_type="CLAUSE_ANALYSIS",
            clause_id="clause_001",
            model="gpt-5-nano",
            input_tokens=500,
            output_tokens=400,
            execution_time_ms=1800,
            status=ExecutionStatus.SUCCESS,
            extra_data={
                "findings_count": 3,
                "risk_level": "high",
                "clarification_required": True,
            },
        )
        
        assert log.agent_type == "CLAUSE_ANALYSIS"
        assert log.extra_data["findings_count"] == 3
        assert log.extra_data["clarification_required"] is True


class TestAgentConfiguration:
    """Tests for agent configuration."""

    def test_model_name(self):
        """Test model name."""
        agent = ClauseAnalysisAgent()
        
        assert agent.MODEL_NAME == "gpt-5-nano"

    def test_seed_for_determinism(self):
        """Test seed is set."""
        agent = ClauseAnalysisAgent()
        
        assert agent.SEED == 42

    def test_max_completion_tokens(self):
        """Test max tokens for comprehensive findings."""
        agent = ClauseAnalysisAgent()
        
        assert agent.MAX_COMPLETION_TOKENS == 1200


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input(self):
        """Test creating valid input."""
        input_data = ClauseAnalysisInput(
            clause_id="clause_001",
            clause_type="Indemnity",
            clause_text="The Seller shall...",
            task_instructions="Extract obligations",
            metadata=ClauseMetadata(
                contract_type="SPA",
                parties=["Seller", "Buyer"],
                jurisdiction="Delaware",
            ),
        )
        
        assert input_data.clause_id == "clause_001"
        assert input_data.clause_type == "Indemnity"
        assert input_data.task_instructions == "Extract obligations"

    def test_input_requires_clause_id(self):
        """Test clause_id is required."""
        with pytest.raises(ValueError):
            ClauseAnalysisInput(clause_text="Some text")

    def test_input_requires_clause_text(self):
        """Test clause_text is required."""
        with pytest.raises(ValueError):
            ClauseAnalysisInput(clause_id="clause_001")
