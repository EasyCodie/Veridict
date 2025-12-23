"""Unit tests for the Obligation Extractor Agent.

Tests cover:
- Prompt formatting with various clause types
- JSON output parsing and validation
- Retry logic (3 retries with exponential backoff, factor 2)
- Confidence score validation (0-1 range)
- Edge cases: empty obligations, malformed text, missing metadata
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.agents.obligation_extractor import (
    AgentExecutionResult,
    ClauseMetadata,
    Obligation,
    ObligationExtractorAgent,
    ObligationExtractorInput,
    ObligationExtractorOutput,
    RetryConfig,
)
from core.agents.prompts.obligation_extractor_prompt import (
    EXAMPLE_CLAUSE_INDEMNIFICATION,
    EXAMPLE_CLAUSE_NO_OBLIGATIONS,
    EXAMPLE_CLAUSE_PAYMENT,
    format_obligation_extractor_prompt,
)
from core.cost_tracker import CostTracker, ExecutionStatus


class TestPromptFormatting:
    """Tests for prompt formatting with various clause types."""

    def test_format_prompt_with_full_metadata(self) -> None:
        """Test prompt formatting with all metadata fields populated."""
        prompt = format_obligation_extractor_prompt(
            clause_id="clause_001",
            clause_text="The Seller shall deliver the goods within 30 days.",
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="Delaware",
        )

        assert "clause_001" in prompt
        assert "The Seller shall deliver the goods" in prompt
        assert "SPA" in prompt
        assert "Seller, Buyer" in prompt
        assert "Delaware" in prompt

    def test_format_prompt_with_empty_metadata(self) -> None:
        """Test prompt formatting with missing metadata fields."""
        prompt = format_obligation_extractor_prompt(
            clause_id="clause_002",
            clause_text="Test clause text.",
            contract_type="",
            parties=[],
            jurisdiction="",
        )

        assert "clause_002" in prompt
        assert "Not specified" in prompt

    def test_format_prompt_indemnification_clause(self) -> None:
        """Test prompt formatting for indemnification clause."""
        prompt = format_obligation_extractor_prompt(
            clause_id="indem_001",
            clause_text=EXAMPLE_CLAUSE_INDEMNIFICATION,
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="New York",
        )

        assert "indemnify" in prompt.lower()
        assert "Seller" in prompt
        assert "Buyer" in prompt

    def test_format_prompt_payment_clause(self) -> None:
        """Test prompt formatting for payment clause."""
        prompt = format_obligation_extractor_prompt(
            clause_id="payment_001",
            clause_text=EXAMPLE_CLAUSE_PAYMENT,
            contract_type="SPA",
            parties=["Seller", "Buyer"],
            jurisdiction="Delaware",
        )

        assert "Payment" in prompt or "payment" in prompt.lower()
        assert "Purchase Price" in prompt or "purchase price" in prompt.lower()

    def test_format_prompt_definition_clause(self) -> None:
        """Test prompt formatting for definition clause (no obligations expected)."""
        prompt = format_obligation_extractor_prompt(
            clause_id="def_001",
            clause_text=EXAMPLE_CLAUSE_NO_OBLIGATIONS,
            contract_type="NDA",
            parties=["Party A", "Party B"],
            jurisdiction="California",
        )

        assert "def_001" in prompt
        assert "Definitions" in prompt or "definitions" in prompt.lower()


class TestObligationModel:
    """Tests for the Obligation Pydantic model."""

    def test_valid_obligation(self) -> None:
        """Test creating a valid obligation."""
        obligation = Obligation(
            obligor="Seller",
            obligation_description="Deliver goods",
            trigger_condition="Upon contract execution",
            deadline="30 days",
            consequences_of_breach="Penalty of $10,000",
            confidence=0.85,
        )

        assert obligation.obligor == "Seller"
        assert obligation.confidence == 0.85

    def test_confidence_clamping_high(self) -> None:
        """Test that confidence values above 1.0 are clamped."""
        obligation = Obligation(
            obligor="Seller",
            obligation_description="Test",
            trigger_condition="Test",
            confidence=1.5,
        )
        assert obligation.confidence == 1.0

    def test_confidence_clamping_low(self) -> None:
        """Test that confidence values below 0.0 are clamped."""
        obligation = Obligation(
            obligor="Seller",
            obligation_description="Test",
            trigger_condition="Test",
            confidence=-0.5,
        )
        assert obligation.confidence == 0.0

    def test_obligation_with_null_fields(self) -> None:
        """Test obligation with optional fields as None."""
        obligation = Obligation(
            obligor="Buyer",
            obligation_description="Pay the purchase price",
            trigger_condition="At closing",
            deadline=None,
            consequences_of_breach=None,
            confidence=0.9,
        )

        assert obligation.deadline is None
        assert obligation.consequences_of_breach is None


class TestObligationExtractorOutput:
    """Tests for the ObligationExtractorOutput model."""

    def test_output_with_obligations(self) -> None:
        """Test output with multiple obligations."""
        output = ObligationExtractorOutput(
            clause_id="test_001",
            obligations=[
                Obligation(
                    obligor="Seller",
                    obligation_description="Deliver goods",
                    trigger_condition="Upon request",
                    confidence=0.9,
                ),
                Obligation(
                    obligor="Buyer",
                    obligation_description="Pay for goods",
                    trigger_condition="Upon delivery",
                    confidence=0.85,
                ),
            ],
            no_obligations_found=False,
        )

        assert len(output.obligations) == 2
        assert not output.no_obligations_found

    def test_output_no_obligations(self) -> None:
        """Test output when no obligations are found."""
        output = ObligationExtractorOutput(
            clause_id="test_002",
            obligations=[],
            no_obligations_found=True,
        )

        assert len(output.obligations) == 0
        assert output.no_obligations_found


class TestJSONParsing:
    """Tests for JSON output parsing and validation."""

    @pytest.fixture
    def agent(self) -> ObligationExtractorAgent:
        """Create agent instance for testing."""
        return ObligationExtractorAgent()

    def test_parse_valid_json(self, agent: ObligationExtractorAgent) -> None:
        """Test parsing valid JSON response."""
        json_str = json.dumps({
            "clause_id": "test_001",
            "obligations": [
                {
                    "obligor": "Seller",
                    "obligation_description": "Deliver goods",
                    "trigger_condition": "Upon request",
                    "deadline": "30 days",
                    "consequences_of_breach": "Penalty",
                    "confidence": 0.9,
                }
            ],
            "no_obligations_found": False,
        })

        output = agent._parse_json_response(json_str, "test_001")
        assert output.clause_id == "test_001"
        assert len(output.obligations) == 1
        assert output.obligations[0].obligor == "Seller"

    def test_parse_json_with_markdown(self, agent: ObligationExtractorAgent) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        json_str = """```json
{
    "clause_id": "test_002",
    "obligations": [],
    "no_obligations_found": true
}
```"""

        output = agent._parse_json_response(json_str, "test_002")
        assert output.clause_id == "test_002"
        assert output.no_obligations_found

    def test_parse_json_with_trailing_comma(self, agent: ObligationExtractorAgent) -> None:
        """Test parsing JSON with trailing commas (common LLM error)."""
        json_str = """{
    "clause_id": "test_003",
    "obligations": [
        {
            "obligor": "Seller",
            "obligation_description": "Test",
            "trigger_condition": "Test",
            "confidence": 0.8,
        },
    ],
    "no_obligations_found": false,
}"""

        output = agent._parse_json_response(json_str, "test_003")
        assert output.clause_id == "test_003"

    def test_parse_empty_obligations(self, agent: ObligationExtractorAgent) -> None:
        """Test parsing response with no obligations."""
        json_str = json.dumps({
            "clause_id": "def_001",
            "obligations": [],
            "no_obligations_found": True,
        })

        output = agent._parse_json_response(json_str, "def_001")
        assert len(output.obligations) == 0
        assert output.no_obligations_found

    def test_parse_with_missing_fields(self, agent: ObligationExtractorAgent) -> None:
        """Test parsing JSON with missing optional fields."""
        json_str = json.dumps({
            "clause_id": "test_004",
            "obligations": [
                {
                    "obligor": "Seller",
                    "obligation_description": "Test",
                    "trigger_condition": "Test",
                    "confidence": 0.7,
                    # Missing deadline and consequences_of_breach
                }
            ],
            "no_obligations_found": False,
        })

        output = agent._parse_json_response(json_str, "test_004")
        assert output.obligations[0].deadline is None
        assert output.obligations[0].consequences_of_breach is None


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    def test_retry_config_defaults(self) -> None:
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0
        assert config.initial_delay_seconds == 1.0

    def test_retry_config_custom(self) -> None:
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            backoff_factor=3.0,
            initial_delay_seconds=0.5,
        )
        assert config.max_retries == 5
        assert config.backoff_factor == 3.0
        assert config.initial_delay_seconds == 0.5

    @pytest.fixture
    def agent_with_config(self) -> ObligationExtractorAgent:
        """Create agent with custom retry config for testing."""
        config = RetryConfig(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay_seconds=0.01,  # Short delay for tests
        )
        return ObligationExtractorAgent(retry_config=config)

    def test_is_retryable_timeout(self, agent_with_config: ObligationExtractorAgent) -> None:
        """Test that timeout errors are retryable."""
        error = Exception("Request timeout")
        assert agent_with_config._is_retryable_error(error)

    def test_is_retryable_rate_limit(self, agent_with_config: ObligationExtractorAgent) -> None:
        """Test that rate limit errors are retryable."""
        error = Exception("Rate limit exceeded")
        assert agent_with_config._is_retryable_error(error)

    def test_is_retryable_server_error(self, agent_with_config: ObligationExtractorAgent) -> None:
        """Test that server errors (5xx) are retryable."""
        for code in ["500", "502", "503", "504"]:
            error = Exception(f"Server error {code}")
            assert agent_with_config._is_retryable_error(error)

    def test_is_retryable_connection_error(self, agent_with_config: ObligationExtractorAgent) -> None:
        """Test that connection errors are retryable."""
        error = Exception("Connection refused")
        assert agent_with_config._is_retryable_error(error)

    def test_is_not_retryable_validation(self, agent_with_config: ObligationExtractorAgent) -> None:
        """Test that validation errors are NOT retryable."""
        error = Exception("Invalid request format")
        assert not agent_with_config._is_retryable_error(error)


class TestConfidenceScoreValidation:
    """Tests for confidence score validation (0-1 range)."""

    def test_confidence_within_range(self) -> None:
        """Test confidence scores within valid range."""
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            obligation = Obligation(
                obligor="Test",
                obligation_description="Test",
                trigger_condition="Test",
                confidence=conf,
            )
            assert 0.0 <= obligation.confidence <= 1.0
            assert obligation.confidence == conf

    def test_confidence_boundary_values(self) -> None:
        """Test confidence at boundary values."""
        # Exactly 0
        ob0 = Obligation(
            obligor="Test",
            obligation_description="Test",
            trigger_condition="Test",
            confidence=0.0,
        )
        assert ob0.confidence == 0.0

        # Exactly 1
        ob1 = Obligation(
            obligor="Test",
            obligation_description="Test",
            trigger_condition="Test",
            confidence=1.0,
        )
        assert ob1.confidence == 1.0


class TestEdgeCases:
    """Tests for edge cases: empty obligations, malformed text, missing metadata."""

    @pytest.fixture
    def agent(self) -> ObligationExtractorAgent:
        """Create agent instance for testing."""
        return ObligationExtractorAgent()

    def test_empty_clause_text(self) -> None:
        """Test handling of empty clause text."""
        input_data = ObligationExtractorInput(
            clause_id="empty_001",
            clause_text="",
            metadata=ClauseMetadata(),
        )
        assert input_data.clause_text == ""

    def test_malformed_json_fallback(self, agent: ObligationExtractorAgent) -> None:
        """Test fallback parsing for malformed JSON."""
        malformed = "This is not JSON at all"
        output = agent._parse_json_response(malformed, "fallback_001")

        assert output.clause_id == "fallback_001"
        assert output.no_obligations_found

    def test_missing_metadata_fields(self) -> None:
        """Test input with missing metadata fields uses defaults."""
        input_data = ObligationExtractorInput(
            clause_id="meta_001",
            clause_text="Test clause",
        )

        assert input_data.metadata.contract_type == ""
        assert input_data.metadata.parties == []
        assert input_data.metadata.jurisdiction == ""
        assert input_data.metadata.execution_date is None

    def test_very_long_clause_text(self) -> None:
        """Test handling of very long clause text."""
        long_text = "Test obligation. " * 1000
        input_data = ObligationExtractorInput(
            clause_id="long_001",
            clause_text=long_text,
            metadata=ClauseMetadata(contract_type="SPA"),
        )

        assert len(input_data.clause_text) > 10000

    def test_special_characters_in_clause(self) -> None:
        """Test handling of special characters in clause text."""
        special_text = """
        The Seller shall pay $1,000,000 (one million dollars) to the Buyer.
        This represents 50% of the "Purchase Price" under Section 2.1(a)(i).
        Terms & Conditions apply. <See Exhibit A>
        """
        input_data = ObligationExtractorInput(
            clause_id="special_001",
            clause_text=special_text,
            metadata=ClauseMetadata(parties=["Seller", "Buyer"]),
        )

        assert "$1,000,000" in input_data.clause_text
        assert '"Purchase Price"' in input_data.clause_text

    def test_unicode_in_clause(self) -> None:
        """Test handling of unicode characters."""
        unicode_text = "The Käufer shall pay €500,000 to the Verkäufer. § 433 BGB applies."
        input_data = ObligationExtractorInput(
            clause_id="unicode_001",
            clause_text=unicode_text,
        )

        assert "€500,000" in input_data.clause_text
        assert "§ 433" in input_data.clause_text


class TestCostCalculation:
    """Tests for cost calculation accuracy."""

    @pytest.fixture
    def agent(self) -> ObligationExtractorAgent:
        """Create agent instance for testing."""
        return ObligationExtractorAgent()

    def test_cost_calculation_basic(self, agent: ObligationExtractorAgent) -> None:
        """Test basic cost calculation."""
        # GPT-5 Nano: $0.05/M input, $0.4/M output
        cost = agent._calculate_cost(input_tokens=1000, output_tokens=500)

        expected_input_cost = (1000 * 0.05) / 1_000_000  # $0.00005
        expected_output_cost = (500 * 0.4) / 1_000_000   # $0.0002
        expected_total = expected_input_cost + expected_output_cost

        assert abs(cost - expected_total) < 0.0000001

    def test_cost_calculation_typical(self, agent: ObligationExtractorAgent) -> None:
        """Test cost calculation for typical usage (~$0.002-0.003)."""
        # Typical: ~300 input tokens, ~200 output tokens
        cost = agent._calculate_cost(input_tokens=300, output_tokens=200)

        # Should be around $0.0000015 + $0.00008 = $0.0000815
        assert cost < 0.001  # Well under $0.001 per clause

    def test_cost_calculation_zero_tokens(self, agent: ObligationExtractorAgent) -> None:
        """Test cost calculation with zero tokens."""
        cost = agent._calculate_cost(input_tokens=0, output_tokens=0)
        assert cost == 0.0


class TestCostTracker:
    """Tests for the CostTracker utility."""

    def test_create_log(self) -> None:
        """Test creating an execution log."""
        tracker = CostTracker()
        log = tracker.create_log(
            agent_type="OBLIGATION_EXTRACTION",
            clause_id="test_001",
            model="gpt-5-nano",
            input_tokens=300,
            output_tokens=200,
            execution_time_ms=1500,
            status="SUCCESS",
            extra_data={"confidence": 0.85, "obligations_found": 2},
        )

        assert log.agent_type == "OBLIGATION_EXTRACTION"
        assert log.clause_id == "test_001"
        assert log.status == ExecutionStatus.SUCCESS
        assert log.cost_usd > 0

    def test_batch_summary(self) -> None:
        """Test batch summary generation."""
        tracker = CostTracker()

        # Add some logs
        for i in range(5):
            log = tracker.create_log(
                agent_type="OBLIGATION_EXTRACTION",
                clause_id=f"clause_{i}",
                model="gpt-5-nano",
                input_tokens=300,
                output_tokens=200,
                execution_time_ms=1500,
                status="SUCCESS",
            )
            tracker.log_execution(log)

        # Add a failure
        fail_log = tracker.create_log(
            agent_type="OBLIGATION_EXTRACTION",
            clause_id="clause_fail",
            model="gpt-5-nano",
            input_tokens=0,
            output_tokens=0,
            execution_time_ms=500,
            status="FAILURE",
            error_message="Test error",
        )
        tracker.log_execution(fail_log)

        summary = tracker.get_batch_summary("batch_001")

        assert summary.total_clauses == 6
        assert summary.successful_clauses == 5
        assert summary.failed_clauses == 1
        assert summary.success_rate_percent == pytest.approx(83.33, rel=0.1)

    def test_projected_cost(self) -> None:
        """Test projected cost calculation."""
        tracker = CostTracker()
        projection = tracker.calculate_projected_cost(
            model="gpt-5-nano",
            estimated_clauses=100,
            avg_input_tokens=300,
            avg_output_tokens=200,
        )

        assert projection["estimated_clauses"] == 100
        assert projection["total_projected_cost_usd"] > 0
        assert projection["cost_per_clause_usd"] > 0


class TestAgentConfiguration:
    """Tests for agent configuration and constants."""

    def test_model_name(self) -> None:
        """Test correct model name is configured."""
        agent = ObligationExtractorAgent()
        assert agent.MODEL_NAME == "gpt-5-nano"

    def test_max_completion_tokens(self) -> None:
        """Test max completion tokens is configured correctly."""
        agent = ObligationExtractorAgent()
        assert agent.MAX_COMPLETION_TOKENS == 800

    def test_seed_for_determinism(self) -> None:
        """Test seed is set for deterministic outputs."""
        agent = ObligationExtractorAgent()
        assert agent.SEED == 42

    def test_pricing_rates(self) -> None:
        """Test pricing rates are correct for GPT-5 Nano."""
        agent = ObligationExtractorAgent()
        assert agent.INPUT_TOKEN_RATE == 0.05  # $0.05 per 1M tokens
        assert agent.OUTPUT_TOKEN_RATE == 0.4  # $0.4 per 1M tokens


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input(self) -> None:
        """Test creating valid input."""
        input_data = ObligationExtractorInput(
            clause_id="valid_001",
            clause_text="The Seller shall deliver the goods.",
            clause_type="delivery",
            metadata=ClauseMetadata(
                contract_type="SPA",
                parties=["Seller", "Buyer"],
                jurisdiction="Delaware",
                execution_date="2024-01-15T00:00:00Z",
            ),
        )

        assert input_data.clause_id == "valid_001"
        assert input_data.metadata.parties == ["Seller", "Buyer"]

    def test_input_requires_clause_id(self) -> None:
        """Test that clause_id is required."""
        with pytest.raises(ValueError):
            ObligationExtractorInput(
                clause_text="Test",
            )  # type: ignore

    def test_input_requires_clause_text(self) -> None:
        """Test that clause_text is required."""
        with pytest.raises(ValueError):
            ObligationExtractorInput(
                clause_id="test",
            )  # type: ignore
