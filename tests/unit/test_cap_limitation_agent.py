"""Unit tests for the Cap & Limitation Agent.

Tests cover:
- Prompt formatting with various liability clause types
- JSON output parsing and validation
- Cap amount parsing (string → numeric conversion)
- Exception extraction and validation
- Retry logic (3 retries with exponential backoff, factor 2)
- Confidence score validation (0-1 range)
- Edge cases (no caps, multiple caps, relative caps)
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from core.agents.cap_limitation_agent import (
    Cap,
    CapLimitationAgent,
    CapLimitationInput,
    CapLimitationOutput,
    CapType,
    ClauseMetadata,
    RetryConfig,
)
from core.agents.prompts.cap_limitation_prompt import (
    CAP_LIMITATION_SYSTEM_PROMPT,
    CAP_LIMITATION_USER_PROMPT_TEMPLATE,
    EXAMPLE_CLAUSE_DEDUCTIBLE,
    EXAMPLE_CLAUSE_INDEMNITY_CAP,
    EXAMPLE_CLAUSE_LIABILITY_CAP,
    EXAMPLE_CLAUSE_NO_CAP,
    format_cap_limitation_prompt,
)
from core.agents.utils.cap_parser import (
    parse_cap_amount,
    detect_relative_cap,
    currency_to_usd,
    extract_all_caps_from_text,
    normalize_cap_display,
)
from core.cost_tracker import CostTracker, ExecutionStatus


class TestPromptFormatting:
    """Tests for prompt formatting with various liability clause types."""

    def test_format_prompt_with_full_metadata(self):
        """Test prompt formatting with all metadata fields populated."""
        prompt = format_cap_limitation_prompt(
            clause_id="clause_001",
            clause_text="The Seller's liability shall not exceed $5,000,000.",
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="Delaware",
        )
        
        assert "clause_001" in prompt
        assert "$5,000,000" in prompt
        assert "SPA" in prompt
        assert "Seller, Buyer" in prompt
        assert "Delaware" in prompt

    def test_format_prompt_with_empty_metadata(self):
        """Test prompt formatting with missing metadata fields."""
        prompt = format_cap_limitation_prompt(
            clause_id="clause_002",
            clause_text="Liability is limited.",
        )
        
        assert "clause_002" in prompt
        assert "Not specified" in prompt

    def test_format_prompt_liability_cap_clause(self):
        """Test prompt formatting for liability cap clause."""
        prompt = format_cap_limitation_prompt(
            clause_id="clause_003",
            clause_text=EXAMPLE_CLAUSE_LIABILITY_CAP,
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="New York",
        )
        
        assert "Five Million Dollars" in prompt
        assert "$5,000,000" in prompt
        assert "fraud" in prompt.lower()

    def test_format_prompt_indemnity_cap_clause(self):
        """Test prompt formatting for indemnity cap clause."""
        prompt = format_cap_limitation_prompt(
            clause_id="clause_004",
            clause_text=EXAMPLE_CLAUSE_INDEMNITY_CAP,
        )
        
        assert "fifteen percent" in prompt.lower()
        assert "15%" in prompt
        assert "Purchase Price" in prompt


class TestCapModel:
    """Tests for the Cap Pydantic model."""

    def test_valid_cap(self):
        """Test creating a valid cap."""
        cap = Cap(
            cap_type="liability_cap",
            cap_amount="$5,000,000",
            cap_amount_numeric=5000000.0,
            applies_to="All claims under this Agreement",
            exceptions=["fraud", "willful misconduct"],
            confidence=0.95,
        )
        
        assert cap.cap_type == "liability_cap"
        assert cap.cap_amount == "$5,000,000"
        assert cap.cap_amount_numeric == 5000000.0
        assert len(cap.exceptions) == 2
        assert cap.confidence == 0.95

    def test_confidence_clamping_high(self):
        """Test that confidence values above 1.0 are clamped."""
        cap = Cap(
            cap_type="indemnity_cap",
            cap_amount="$1M",
            applies_to="Indemnification claims",
            confidence=1.5,
        )
        
        assert cap.confidence == 1.0

    def test_confidence_clamping_low(self):
        """Test that confidence values below 0.0 are clamped."""
        cap = Cap(
            cap_type="deductible",
            cap_amount="$250,000",
            applies_to="Losses",
            confidence=-0.5,
        )
        
        assert cap.confidence == 0.0

    def test_cap_type_normalization(self):
        """Test cap type is normalized to valid enum."""
        cap = Cap(
            cap_type="Liability Cap",  # Should normalize
            cap_amount="$1M",
            applies_to="Claims",
            confidence=0.8,
        )
        
        assert cap.cap_type == "liability_cap"

    def test_cap_with_null_numeric(self):
        """Test cap with relative/non-numeric amount."""
        cap = Cap(
            cap_type="indemnity_cap",
            cap_amount="15% of Purchase Price",
            cap_amount_numeric=None,
            applies_to="Indemnification obligations",
            confidence=0.85,
        )
        
        assert cap.cap_amount_numeric is None
        assert "15%" in cap.cap_amount


class TestCapLimitationOutput:
    """Tests for the CapLimitationOutput model."""

    def test_output_with_caps(self):
        """Test output with multiple caps."""
        output = CapLimitationOutput(
            clause_id="clause_001",
            caps=[
                Cap(
                    cap_type="liability_cap",
                    cap_amount="$5M",
                    cap_amount_numeric=5000000.0,
                    applies_to="General liability",
                    confidence=0.9,
                ),
                Cap(
                    cap_type="deductible",
                    cap_amount="$250K",
                    cap_amount_numeric=250000.0,
                    applies_to="First party claims",
                    confidence=0.85,
                ),
            ],
            no_cap_found=False,
        )
        
        assert len(output.caps) == 2
        assert output.no_cap_found is False

    def test_output_no_caps(self):
        """Test output when no caps are found."""
        output = CapLimitationOutput(
            clause_id="clause_002",
            caps=[],
            no_cap_found=True,
        )
        
        assert len(output.caps) == 0
        assert output.no_cap_found is True


class TestCapAmountParsing:
    """Tests for cap amount parsing utilities."""

    def test_parse_usd_with_commas(self):
        """Test parsing USD amount with commas."""
        original, numeric = parse_cap_amount("$5,000,000")
        
        assert original == "$5,000,000"
        assert numeric == 5000000.0

    def test_parse_usd_millions(self):
        """Test parsing USD amount with M suffix."""
        original, numeric = parse_cap_amount("$5M")
        
        assert numeric == 5000000.0

    def test_parse_usd_with_word(self):
        """Test parsing USD amount with 'million' word."""
        original, numeric = parse_cap_amount("$2.5 million")
        
        assert numeric == 2500000.0

    def test_parse_eur(self):
        """Test parsing EUR amount (converts to USD)."""
        original, numeric = parse_cap_amount("€5M")
        
        # EUR rate is 1.08
        assert numeric == pytest.approx(5400000.0, rel=0.01)

    def test_parse_gbp(self):
        """Test parsing GBP amount (converts to USD)."""
        original, numeric = parse_cap_amount("£1M")
        
        # GBP rate is 1.27
        assert numeric == pytest.approx(1270000.0, rel=0.01)

    def test_parse_thousands(self):
        """Test parsing amount with K suffix."""
        original, numeric = parse_cap_amount("$250K")
        
        assert numeric == 250000.0

    def test_parse_billions(self):
        """Test parsing billion amounts."""
        original, numeric = parse_cap_amount("$1B")
        
        assert numeric == 1000000000.0

    def test_detect_relative_cap_percentage(self):
        """Test detecting percentage-based cap."""
        result = detect_relative_cap("15% of the Purchase Price")
        
        assert result == "15% of Purchase Price"

    def test_detect_relative_cap_multiplier(self):
        """Test detecting multiplier-based cap."""
        result = detect_relative_cap("2x annual fees")
        
        assert result == "2x annual fees"

    def test_detect_no_relative_cap(self):
        """Test that absolute amounts return None."""
        result = detect_relative_cap("$5,000,000")
        
        assert result is None


class TestCurrencyConversion:
    """Tests for currency conversion."""

    def test_usd_to_usd(self):
        """Test USD to USD (no conversion)."""
        result = currency_to_usd(1000000, "USD")
        
        assert result == 1000000.0

    def test_eur_to_usd(self):
        """Test EUR to USD conversion."""
        result = currency_to_usd(1000000, "EUR")
        
        assert result == pytest.approx(1080000.0, rel=0.01)

    def test_gbp_to_usd(self):
        """Test GBP to USD conversion."""
        result = currency_to_usd(1000000, "GBP")
        
        assert result == pytest.approx(1270000.0, rel=0.01)


class TestNormalizeCapDisplay:
    """Tests for cap display normalization."""

    def test_display_billions(self):
        """Test display of billions."""
        result = normalize_cap_display(2500000000)
        
        assert result == "$2.5B"

    def test_display_millions(self):
        """Test display of millions."""
        result = normalize_cap_display(5000000)
        
        assert result == "$5.0M"

    def test_display_thousands(self):
        """Test display of thousands."""
        result = normalize_cap_display(250000)
        
        assert result == "$250.0K"

    def test_display_relative(self):
        """Test display of relative cap."""
        result = normalize_cap_display(None, is_relative=True, relative_str="15% of Purchase Price")
        
        assert result == "15% of Purchase Price"


class TestJSONParsing:
    """Tests for JSON output parsing and validation."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        return CapLimitationAgent()

    def test_parse_valid_json(self, agent: CapLimitationAgent):
        """Test parsing valid JSON response."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "caps": [
                {
                    "cap_type": "liability_cap",
                    "cap_amount": "$5M",
                    "cap_amount_numeric": 5000000,
                    "applies_to": "All claims",
                    "exceptions": ["fraud"],
                    "confidence": 0.9
                }
            ],
            "no_cap_found": False
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.clause_id == "clause_001"
        assert len(result.caps) == 1
        assert result.caps[0].cap_amount == "$5M"

    def test_parse_json_with_markdown(self, agent: CapLimitationAgent):
        """Test parsing JSON wrapped in markdown code blocks."""
        json_str = """```json
{
  "clause_id": "clause_001",
  "caps": [],
  "no_cap_found": true
}
```"""
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert result.no_cap_found is True

    def test_parse_json_with_trailing_comma(self, agent: CapLimitationAgent):
        """Test parsing JSON with trailing commas (common LLM error)."""
        json_str = """{
  "clause_id": "clause_001",
  "caps": [
    {
      "cap_type": "liability_cap",
      "cap_amount": "$5M",
      "applies_to": "Claims",
      "exceptions": [],
      "confidence": 0.8,
    },
  ],
  "no_cap_found": false,
}"""
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert len(result.caps) == 1

    def test_parse_empty_caps(self, agent: CapLimitationAgent):
        """Test parsing response with no caps."""
        json_str = json.dumps({
            "clause_id": "clause_001",
            "caps": [],
            "no_cap_found": True
        })
        
        result = agent._parse_json_response(json_str, "clause_001")
        
        assert len(result.caps) == 0
        assert result.no_cap_found is True


class TestRetryLogic:
    """Tests for retry logic configuration."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0
        assert config.initial_delay_seconds == 1.0
        assert "timeout" in config.retryable_errors

    def test_retry_config_custom(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            backoff_factor=1.5,
            initial_delay_seconds=0.5,
        )
        
        assert config.max_retries == 5
        assert config.backoff_factor == 1.5

    def test_is_retryable_timeout(self):
        """Test that timeout errors are retryable."""
        agent = CapLimitationAgent()
        
        error = Exception("Connection timeout")
        assert agent._is_retryable_error(error) is True

    def test_is_retryable_rate_limit(self):
        """Test that rate limit errors are retryable."""
        agent = CapLimitationAgent()
        
        error = Exception("Rate limit exceeded")
        assert agent._is_retryable_error(error) is True

    def test_is_retryable_server_error(self):
        """Test that 500 errors are retryable."""
        agent = CapLimitationAgent()
        
        error = Exception("Server returned 503")
        assert agent._is_retryable_error(error) is True

    def test_is_not_retryable_validation(self):
        """Test that validation errors are not retryable."""
        agent = CapLimitationAgent()
        
        error = Exception("Invalid request format")
        assert agent._is_retryable_error(error) is False


class TestConfidenceScoreValidation:
    """Tests for confidence score validation."""

    def test_confidence_within_range(self):
        """Test confidence values within 0-1 range."""
        cap = Cap(
            cap_type="liability_cap",
            cap_amount="$5M",
            applies_to="Claims",
            confidence=0.85,
        )
        
        assert 0.0 <= cap.confidence <= 1.0

    def test_confidence_boundary_values(self):
        """Test confidence at boundary values."""
        cap_zero = Cap(
            cap_type="liability_cap",
            cap_amount="$5M",
            applies_to="Claims",
            confidence=0.0,
        )
        cap_one = Cap(
            cap_type="liability_cap",
            cap_amount="$5M",
            applies_to="Claims",
            confidence=1.0,
        )
        
        assert cap_zero.confidence == 0.0
        assert cap_one.confidence == 1.0


class TestEdgeCases:
    """Tests for edge cases in cap extraction."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        return CapLimitationAgent()

    def test_empty_clause_text(self):
        """Test handling of empty clause text."""
        input_data = CapLimitationInput(
            clause_id="clause_empty",
            clause_text="",
        )
        
        # Should not raise, just create a valid input
        assert input_data.clause_id == "clause_empty"

    def test_malformed_json_fallback(self, agent: CapLimitationAgent):
        """Test fallback parsing for malformed JSON."""
        malformed = "This is not JSON at all"
        
        result = agent._parse_json_response(malformed, "clause_001")
        
        # Should return empty caps with fallback
        assert result.clause_id == "clause_001"
        assert result.no_cap_found is True

    def test_multiple_caps_single_clause(self, agent: CapLimitationAgent):
        """Test extraction of multiple caps from single clause."""
        json_str = json.dumps({
            "clause_id": "clause_multi",
            "caps": [
                {"cap_type": "liability_cap", "cap_amount": "$5M", "applies_to": "General", "confidence": 0.9},
                {"cap_type": "deductible", "cap_amount": "$250K", "applies_to": "First dollar", "confidence": 0.85},
                {"cap_type": "indemnity_cap", "cap_amount": "15% of Purchase Price", "applies_to": "Indemnity", "confidence": 0.8},
            ],
            "no_cap_found": False
        })
        
        result = agent._parse_json_response(json_str, "clause_multi")
        
        assert len(result.caps) == 3

    def test_special_characters_in_clause(self):
        """Test handling clauses with special characters."""
        input_data = CapLimitationInput(
            clause_id="clause_special",
            clause_text='Liability < $5M & > $1M; "quoted" amounts',
        )
        
        assert "&" in input_data.clause_text
        assert '"' in input_data.clause_text

    def test_extract_all_caps_from_text(self):
        """Test utility to extract all caps from text."""
        text = """
        The liability cap is $5,000,000 and the deductible is $250K.
        The indemnity cap is 15% of the Purchase Price.
        """
        
        caps = extract_all_caps_from_text(text)
        
        # Should find numeric and relative caps
        assert len(caps) >= 2
        numeric_caps = [c for c in caps if not c["is_relative"]]
        relative_caps = [c for c in caps if c["is_relative"]]
        
        assert len(numeric_caps) >= 2
        assert len(relative_caps) >= 1


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_cost_calculation_basic(self):
        """Test basic cost calculation."""
        agent = CapLimitationAgent()
        
        cost = agent._calculate_cost(input_tokens=1000, output_tokens=500)
        
        # GPT-5 Nano: $0.05/1M input, $0.4/1M output
        expected = (1000 * 0.05 / 1_000_000) + (500 * 0.4 / 1_000_000)
        assert cost == pytest.approx(expected, rel=0.01)

    def test_cost_calculation_typical(self):
        """Test cost calculation with typical token counts."""
        agent = CapLimitationAgent()
        
        cost = agent._calculate_cost(input_tokens=300, output_tokens=200)
        
        # Should be roughly $0.0001 per clause
        assert cost < 0.001

    def test_cost_calculation_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        agent = CapLimitationAgent()
        
        cost = agent._calculate_cost(input_tokens=0, output_tokens=0)
        
        assert cost == 0.0


class TestCostTracker:
    """Tests for cost tracking integration."""

    def test_create_log(self):
        """Test creating an execution log."""
        tracker = CostTracker()
        
        log = tracker.create_log(
            agent_type="CAP_LIMITATION_EXTRACTION",
            clause_id="clause_001",
            model="gpt-5-nano",
            input_tokens=300,
            output_tokens=200,
            execution_time_ms=1500,
            status=ExecutionStatus.SUCCESS,
            extra_data={"caps_found": 2, "total_cap_value_usd": 5000000},
        )
        
        assert log.agent_type == "CAP_LIMITATION_EXTRACTION"
        assert log.clause_id == "clause_001"
        assert log.extra_data["caps_found"] == 2

    def test_batch_summary(self):
        """Test batch summary generation."""
        tracker = CostTracker()
        
        for i in range(5):
            log = tracker.create_log(
                agent_type="CAP_LIMITATION_EXTRACTION",
                clause_id=f"clause_{i}",
                model="gpt-5-nano",
                input_tokens=300,
                output_tokens=200,
                execution_time_ms=1500,
                status=ExecutionStatus.SUCCESS,
            )
            tracker.log_execution(log)
        
        summary = tracker.get_batch_summary("test_batch")
        
        assert summary.total_clauses == 5
        assert summary.successful_clauses == 5
        assert summary.success_rate_percent == 100.0


class TestAgentConfiguration:
    """Tests for agent configuration."""

    def test_model_name(self):
        """Test that model name is correct."""
        agent = CapLimitationAgent()
        
        assert agent.MODEL_NAME == "gpt-5-nano"

    def test_max_completion_tokens(self):
        """Test max completion tokens setting."""
        agent = CapLimitationAgent()
        
        assert agent.MAX_COMPLETION_TOKENS == 800

    def test_seed_for_determinism(self):
        """Test that seed is set for deterministic outputs."""
        agent = CapLimitationAgent()
        
        assert agent.SEED == 42

    def test_pricing_rates(self):
        """Test pricing rates are set correctly."""
        agent = CapLimitationAgent()
        
        assert agent.INPUT_TOKEN_RATE == 0.05
        assert agent.OUTPUT_TOKEN_RATE == 0.4


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input(self):
        """Test creating a valid input."""
        input_data = CapLimitationInput(
            clause_id="clause_001",
            clause_text="Liability shall not exceed $5M.",
            clause_type="Liability",
            metadata=ClauseMetadata(
                contract_type="SPA",
                parties=["Seller", "Buyer"],
                jurisdiction="Delaware",
            ),
        )
        
        assert input_data.clause_id == "clause_001"
        assert input_data.metadata.contract_type == "SPA"

    def test_input_requires_clause_id(self):
        """Test that clause_id is required."""
        with pytest.raises(ValueError):
            CapLimitationInput(
                clause_text="Some text",
            )

    def test_input_requires_clause_text(self):
        """Test that clause_text is required."""
        with pytest.raises(ValueError):
            CapLimitationInput(
                clause_id="clause_001",
            )
