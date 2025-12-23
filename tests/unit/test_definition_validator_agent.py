"""Unit tests for the Definition Validator Agent.

Tests cover:
- Prompt formatting with various clause types
- JSON output parsing and validation
- Term extraction and definition checking
- Severity scoring
- Cross-reference detection
- Edge cases
- Retry logic
- Cost tracking
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from core.agents.definition_validator_agent import (
    UndefinedTerm,
    DefinitionValidatorAgent,
    DefinitionValidatorInput,
    DefinitionValidatorOutput,
    ClauseMetadata,
    RetryConfig,
)
from core.agents.prompts.definition_validator_prompt import (
    DEFINITION_VALIDATOR_SYSTEM_PROMPT,
    DEFINITION_VALIDATOR_USER_PROMPT_TEMPLATE,
    EXAMPLE_CLAUSE_WITH_UNDEFINED,
    EXAMPLE_DEFINITIONS_SECTION,
    format_definition_validator_prompt,
)
from core.agents.utils.definition_utilities import (
    DefinitionSeverity,
    extract_defined_terms,
    find_capitalized_terms,
    is_term_defined,
    suggest_definition_location,
    classify_term_severity,
    estimate_definition_elsewhere,
    aggregate_undefined_terms,
    normalize_severity,
)
from core.cost_tracker import CostTracker, ExecutionStatus


class TestPromptFormatting:
    """Tests for prompt formatting with various clause types."""

    def test_format_prompt_with_full_metadata(self):
        """Test prompt formatting with all metadata fields."""
        prompt = format_definition_validator_prompt(
            clause_id="clause_001",
            clause_text="The Seller represents that no Material Adverse Effect...",
            clause_type="Representations",
            definitions_section=EXAMPLE_DEFINITIONS_SECTION,
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="Delaware",
        )
        
        assert "clause_001" in prompt
        assert "Material Adverse Effect" in prompt
        assert "Representations" in prompt
        assert "SPA" in prompt
        assert "Delaware" in prompt

    def test_format_prompt_without_definitions(self):
        """Test prompt formatting without definitions section."""
        prompt = format_definition_validator_prompt(
            clause_id="clause_002",
            clause_text="Some clause text with Undefined Term",
        )
        
        assert "clause_002" in prompt
        assert "Not provided" in prompt

    def test_format_prompt_with_definitions(self):
        """Test prompt with definitions section included."""
        prompt = format_definition_validator_prompt(
            clause_id="clause_003",
            clause_text=EXAMPLE_CLAUSE_WITH_UNDEFINED,
            definitions_section=EXAMPLE_DEFINITIONS_SECTION,
        )
        
        assert "Material Adverse Effect" in prompt
        assert "Balance Sheet Date" in prompt


class TestUndefinedTermModel:
    """Tests for the UndefinedTerm Pydantic model."""

    def test_valid_undefined_term(self):
        """Test creating a valid undefined term."""
        term = UndefinedTerm(
            term="Material Adverse Effect",
            context_usage="no Material Adverse Effect on the Business",
            severity="critical",
            likelihood_defined_elsewhere=True,
            search_suggestions=["Section 1 - Definitions"],
        )
        
        assert term.term == "Material Adverse Effect"
        assert term.severity == "critical"
        assert term.likelihood_defined_elsewhere is True

    def test_severity_normalization(self):
        """Test severity is normalized."""
        term = UndefinedTerm(
            term="Some Term",
            context_usage="usage",
            severity="HIGH",
        )
        
        assert term.severity == "high"

    def test_default_values(self):
        """Test default values are applied."""
        term = UndefinedTerm(
            term="Some Term",
            context_usage="",
            severity="medium",
        )
        
        assert term.likelihood_defined_elsewhere is False
        assert term.search_suggestions == []


class TestDefinitionValidatorOutput:
    """Tests for the DefinitionValidatorOutput model."""

    def test_output_with_undefined_terms(self):
        """Test output with undefined terms."""
        output = DefinitionValidatorOutput(
            clause_id="clause_001",
            undefined_terms=[
                UndefinedTerm(
                    term="Permitted Change",
                    context_usage="any Permitted Change shall not constitute",
                    severity="high",
                ),
            ],
            all_terms_defined=False,
            definition_gaps_summary="Missing definition for 'Permitted Change'",
            confidence_score=0.85,
        )
        
        assert len(output.undefined_terms) == 1
        assert output.all_terms_defined is False
        assert "Permitted Change" in output.definition_gaps_summary

    def test_output_all_defined(self):
        """Test output when all terms are defined."""
        output = DefinitionValidatorOutput(
            clause_id="clause_002",
            undefined_terms=[],
            all_terms_defined=True,
            confidence_score=0.9,
        )
        
        assert len(output.undefined_terms) == 0
        assert output.all_terms_defined is True


class TestDefinitionUtilities:
    """Tests for definition utility functions."""

    def test_extract_defined_terms_quotes(self):
        """Test extracting terms with quoted definitions."""
        definitions = '''
        "Material Adverse Effect" means any change that is materially adverse.
        "Business Day" means any day other than Saturday or Sunday.
        '''
        
        terms = extract_defined_terms(definitions)
        
        assert "Material Adverse Effect" in terms
        assert "Business Day" in terms

    def test_extract_defined_terms_shall_mean(self):
        """Test extracting terms with 'shall mean'."""
        definitions = '"Closing" shall mean the closing of the transactions.'
        
        terms = extract_defined_terms(definitions)
        
        assert "Closing" in terms

    def test_find_capitalized_terms(self):
        """Test finding capitalized terms in clause."""
        clause = "The Seller shall deliver the Target Company shares on the Closing Date."
        
        terms = find_capitalized_terms(clause)
        
        # Should find capitalized terms (not at sentence start)
        assert any("Target" in t for t in terms) or any("Closing" in t for t in terms)

    def test_is_term_defined_exact_match(self):
        """Test exact match for defined terms."""
        defined = {"Material Adverse Effect", "Business Day"}
        
        assert is_term_defined("Material Adverse Effect", defined) is True
        assert is_term_defined("Unknown Term", defined) is False

    def test_is_term_defined_case_insensitive(self):
        """Test case-insensitive matching."""
        defined = {"Material Adverse Effect"}
        
        assert is_term_defined("material adverse effect", defined) is True


class TestSeverityClassification:
    """Tests for severity classification."""

    def test_classify_critical_term(self):
        """Test critical severity for key deal terms."""
        severity = classify_term_severity("Purchase Price")
        assert severity == DefinitionSeverity.CRITICAL

    def test_classify_high_term(self):
        """Test high severity for operational terms."""
        severity = classify_term_severity("Confidential Information")
        assert severity == DefinitionSeverity.HIGH

    def test_classify_medium_term(self):
        """Test medium severity for secondary terms."""
        severity = classify_term_severity("Some Minor Term")
        assert severity == DefinitionSeverity.MEDIUM

    def test_classify_quoted_term(self):
        """Test quoted terms are critical."""
        severity = classify_term_severity("Something", is_quoted=True)
        assert severity == DefinitionSeverity.CRITICAL

    def test_classify_frequent_term(self):
        """Test frequently used terms are critical."""
        severity = classify_term_severity("Some Term", usage_count=5)
        assert severity == DefinitionSeverity.CRITICAL


class TestCrossReferenceDetection:
    """Tests for cross-reference detection."""

    def test_suggest_location_general(self):
        """Test general location suggestions."""
        suggestions = suggest_definition_location("Unknown Term")
        
        assert "Section 1 - Definitions" in suggestions

    def test_suggest_location_mac(self):
        """Test location suggestions for MAC/MAE terms."""
        suggestions = suggest_definition_location("Material Adverse Change")
        
        assert any("MAC" in s or "MAE" in s for s in suggestions)

    def test_suggest_location_ip(self):
        """Test location suggestions for IP terms."""
        suggestions = suggest_definition_location("Intellectual Property")
        
        assert any("IP" in s or "Schedule" in s for s in suggestions)

    def test_estimate_defined_elsewhere_common(self):
        """Test estimating if common terms are defined elsewhere."""
        assert estimate_definition_elsewhere("Material Adverse Effect") is True
        assert estimate_definition_elsewhere("Purchase Price") is True

    def test_estimate_defined_elsewhere_uncommon(self):
        """Test estimating uncommon terms."""
        assert estimate_definition_elsewhere("xyz") is False


class TestUndefinedTermsAggregation:
    """Tests for undefined terms aggregation."""

    def test_aggregate_empty(self):
        """Test aggregating empty terms list."""
        result = aggregate_undefined_terms([])
        
        assert result["total_undefined"] == 0

    def test_aggregate_single_term(self):
        """Test aggregating single term."""
        result = aggregate_undefined_terms([
            {"severity": "critical", "likelihood_defined_elsewhere": True}
        ])
        
        assert result["total_undefined"] == 1
        assert result["by_severity"]["critical"] == 1
        assert result["likely_elsewhere_count"] == 1

    def test_aggregate_multiple_terms(self):
        """Test aggregating multiple terms."""
        result = aggregate_undefined_terms([
            {"severity": "critical", "likelihood_defined_elsewhere": True},
            {"severity": "high", "likelihood_defined_elsewhere": False},
            {"severity": "medium", "likelihood_defined_elsewhere": True},
        ])
        
        assert result["total_undefined"] == 3
        assert result["by_severity"]["critical"] == 1
        assert result["by_severity"]["high"] == 1
        assert result["likely_elsewhere_count"] == 2


class TestJSONParsing:
    """Tests for JSON output parsing."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return DefinitionValidatorAgent()

    def test_parse_valid_json(self, agent: DefinitionValidatorAgent):
        """Test parsing valid JSON."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "undefined_terms": [
                {
                    "term": "Permitted Change",
                    "context_usage": "any Permitted Change",
                    "severity": "high",
                    "likelihood_defined_elsewhere": True,
                    "search_suggestions": ["Section 1"]
                }
            ],
            "all_terms_defined": False,
            "definition_gaps_summary": "Missing Permitted Change",
            "confidence_score": 0.85
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert len(result.undefined_terms) == 1
        assert result.all_terms_defined is False

    def test_parse_json_with_markdown(self, agent: DefinitionValidatorAgent):
        """Test parsing JSON wrapped in markdown."""
        json_str = """```json
{
  "clause_id": "clause_001",
  "undefined_terms": [],
  "all_terms_defined": true,
  "definition_gaps_summary": null,
  "confidence_score": 0.9
}
```"""
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.all_terms_defined is True

    def test_parse_all_defined(self, agent: DefinitionValidatorAgent):
        """Test parsing response with all terms defined."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "undefined_terms": [],
            "all_terms_defined": True,
            "confidence_score": 0.95
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert len(result.undefined_terms) == 0
        assert result.all_terms_defined is True


class TestRetryLogic:
    """Tests for retry logic."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0

    def test_is_retryable_timeout(self):
        """Test timeout is retryable."""
        agent = DefinitionValidatorAgent()
        
        error = Exception("Connection timeout")
        assert agent._is_retryable_error(error) is True

    def test_is_retryable_rate_limit(self):
        """Test rate limit is retryable."""
        agent = DefinitionValidatorAgent()
        
        error = Exception("Rate limit exceeded")
        assert agent._is_retryable_error(error) is True

    def test_is_not_retryable_validation(self):
        """Test validation errors are not retryable."""
        agent = DefinitionValidatorAgent()
        
        error = Exception("Invalid request format")
        assert agent._is_retryable_error(error) is False


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return DefinitionValidatorAgent()

    def test_empty_clause_text(self):
        """Test handling empty clause text."""
        input_data = DefinitionValidatorInput(
            clause_id="clause_empty",
            clause_text="",
        )
        
        assert input_data.clause_id == "clause_empty"

    def test_malformed_json_fallback(self, agent: DefinitionValidatorAgent):
        """Test fallback for malformed JSON."""
        malformed = "This is not JSON"
        
        result = agent._parse_json_response(malformed, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert result.all_terms_defined is True

    def test_multiple_undefined_terms(self, agent: DefinitionValidatorAgent):
        """Test multiple undefined terms in single clause."""
        json_str = json.dumps({
            "clause_id": "clause_multi",
            "undefined_terms": [
                {"term": "Term A", "context_usage": "using Term A", "severity": "critical"},
                {"term": "Term B", "context_usage": "using Term B", "severity": "high"},
                {"term": "Term C", "context_usage": "using Term C", "severity": "medium"},
            ],
            "all_terms_defined": False,
            "confidence_score": 0.8
        })
        
        result = agent._parse_json_response(json_str, "clause_multi")
        
        assert len(result.undefined_terms) == 3
        assert result.all_terms_defined is False


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_cost_calculation_basic(self):
        """Test basic cost calculation."""
        agent = DefinitionValidatorAgent()
        
        cost = agent._calculate_cost(input_tokens=1000, output_tokens=500)
        
        expected = (1000 * 0.05 / 1_000_000) + (500 * 0.4 / 1_000_000)
        assert cost == pytest.approx(expected, rel=0.01)

    def test_cost_calculation_zero(self):
        """Test cost calculation with zero tokens."""
        agent = DefinitionValidatorAgent()
        
        cost = agent._calculate_cost(input_tokens=0, output_tokens=0)
        
        assert cost == 0.0


class TestCostTracker:
    """Tests for cost tracking integration."""

    def test_create_log(self):
        """Test creating execution log."""
        tracker = CostTracker()
        
        log = tracker.create_log(
            agent_type="DEFINITION_VALIDATION",
            clause_id="clause_001",
            model="gpt-5-nano",
            input_tokens=400,
            output_tokens=300,
            execution_time_ms=1200,
            status=ExecutionStatus.SUCCESS,
            extra_data={
                "undefined_terms_count": 2,
                "all_terms_defined": False,
            },
        )
        
        assert log.agent_type == "DEFINITION_VALIDATION"
        assert log.extra_data["undefined_terms_count"] == 2


class TestAgentConfiguration:
    """Tests for agent configuration."""

    def test_model_name(self):
        """Test model name."""
        agent = DefinitionValidatorAgent()
        
        assert agent.MODEL_NAME == "gpt-5-nano"

    def test_seed_for_determinism(self):
        """Test seed is set."""
        agent = DefinitionValidatorAgent()
        
        assert agent.SEED == 42

    def test_max_completion_tokens(self):
        """Test max tokens configuration."""
        agent = DefinitionValidatorAgent()
        
        assert agent.MAX_COMPLETION_TOKENS == 800


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input(self):
        """Test creating valid input."""
        input_data = DefinitionValidatorInput(
            clause_id="clause_001",
            clause_text="The Seller represents...",
            definitions_section=EXAMPLE_DEFINITIONS_SECTION,
            metadata=ClauseMetadata(
                contract_type="SPA",
                parties=["Seller", "Buyer"],
            ),
        )
        
        assert input_data.clause_id == "clause_001"
        assert input_data.definitions_section is not None

    def test_input_requires_clause_id(self):
        """Test clause_id is required."""
        with pytest.raises(ValueError):
            DefinitionValidatorInput(clause_text="Some text")

    def test_input_requires_clause_text(self):
        """Test clause_text is required."""
        with pytest.raises(ValueError):
            DefinitionValidatorInput(clause_id="clause_001")
