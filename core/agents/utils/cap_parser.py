"""Cap Amount Parsing Utilities for Cap & Limitation Agent.

This module provides utilities for parsing and normalizing cap amounts from
legal contract text. Supports numeric amounts ($5M, €2.5M) and relative
amounts (X% of Purchase Price).

Usage:
    from core.agents.utils.cap_parser import parse_cap_amount, detect_relative_cap
    
    # Numeric cap
    amount_str, numeric = parse_cap_amount("$5,000,000")
    # -> ("$5,000,000", 5000000.0)
    
    # Relative cap
    relative = detect_relative_cap("15% of the Purchase Price")
    # -> "15% of the Purchase Price"
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Tuple


# Currency conversion rates (static for MVP - could be made configurable)
CURRENCY_RATES_TO_USD = {
    "USD": 1.0,
    "$": 1.0,
    "EUR": 1.08,  # Approximate rate
    "€": 1.08,
    "GBP": 1.27,  # Approximate rate
    "£": 1.27,
}

# Multiplier keywords
MULTIPLIERS = {
    "k": 1_000,
    "K": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "M": 1_000_000,
    "mm": 1_000_000,
    "MM": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "B": 1_000_000_000,
    "billion": 1_000_000_000,
}

# Regex patterns
CURRENCY_SYMBOL_PATTERN = r"[\$€£]"
NUMERIC_AMOUNT_PATTERN = re.compile(
    r"(" + CURRENCY_SYMBOL_PATTERN + r")?\s*"
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(K|M|MM|B|thousand|million|billion)?",
    re.IGNORECASE
)

RELATIVE_CAP_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:the\s+)?([^.,;]+)",
    re.IGNORECASE
)

MULTIPLIER_CAP_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:x|times)\s+([^.,;]+)",
    re.IGNORECASE
)


def parse_numeric_amount(amount_str: str) -> float | None:
    """Parse a numeric string to float, handling commas.
    
    Args:
        amount_str: String like "5,000,000" or "5.5"
        
    Returns:
        Float value or None if parsing fails.
    """
    try:
        # Remove commas and parse
        cleaned = amount_str.replace(",", "")
        return float(Decimal(cleaned))
    except (InvalidOperation, ValueError):
        return None


def parse_cap_amount(text: str) -> Tuple[str, float | None]:
    """Parse a cap amount string and return original + numeric value.
    
    Handles formats like:
    - "$5,000,000" -> ("$5,000,000", 5000000.0)
    - "$5M" -> ("$5M", 5000000.0)
    - "€2.5 million" -> ("€2.5 million", 2700000.0)  # Converted to USD
    - "5 million USD" -> ("5 million USD", 5000000.0)
    
    Args:
        text: The text containing the cap amount.
        
    Returns:
        Tuple of (original_string, numeric_usd or None).
    """
    text = text.strip()
    
    # Try to find a numeric amount pattern
    match = NUMERIC_AMOUNT_PATTERN.search(text)
    if not match:
        return (text, None)
    
    currency_symbol = match.group(1) or "$"  # Default to USD
    amount_str = match.group(2)
    multiplier_str = match.group(3)
    
    # Parse the base amount
    amount = parse_numeric_amount(amount_str)
    if amount is None:
        return (text, None)
    
    # Apply multiplier if present
    if multiplier_str:
        multiplier = MULTIPLIERS.get(multiplier_str, MULTIPLIERS.get(multiplier_str.lower(), 1))
        amount *= multiplier
    
    # Convert to USD if needed
    rate = CURRENCY_RATES_TO_USD.get(currency_symbol, 1.0)
    amount_usd = amount * rate
    
    return (text, amount_usd)


def detect_relative_cap(text: str) -> str | None:
    """Detect a relative cap amount (percentage-based).
    
    Handles formats like:
    - "15% of the Purchase Price"
    - "50% of fees paid"
    - "2x annual fees"
    
    Args:
        text: The text to search for relative caps.
        
    Returns:
        Full relative cap string or None if not found.
    """
    text = text.strip()
    
    # Check for percentage-based caps
    match = RELATIVE_CAP_PATTERN.search(text)
    if match:
        percentage = match.group(1)
        base = match.group(2).strip()
        return f"{percentage}% of {base}"
    
    # Check for multiplier-based caps (e.g., "2x annual fees")
    match = MULTIPLIER_CAP_PATTERN.search(text)
    if match:
        multiplier = match.group(1)
        base = match.group(2).strip()
        return f"{multiplier}x {base}"
    
    return None


def currency_to_usd(amount: float, currency: str) -> float:
    """Convert an amount from a currency to USD.
    
    Args:
        amount: The amount in the source currency.
        currency: Currency code or symbol (USD, EUR, €, £, etc.)
        
    Returns:
        Amount in USD.
    """
    rate = CURRENCY_RATES_TO_USD.get(currency, CURRENCY_RATES_TO_USD.get(currency.upper(), 1.0))
    return amount * rate


def extract_all_caps_from_text(text: str) -> list[dict]:
    """Extract all potential cap amounts from a text block.
    
    This is a utility function to find all monetary amounts in text,
    useful for validation and testing.
    
    Args:
        text: Full text to search.
        
    Returns:
        List of dicts with 'original', 'numeric_usd', 'is_relative' keys.
    """
    results = []
    
    # Find all numeric amounts
    for match in NUMERIC_AMOUNT_PATTERN.finditer(text):
        original = match.group(0).strip()
        _, numeric = parse_cap_amount(original)
        if numeric is not None:
            results.append({
                "original": original,
                "numeric_usd": numeric,
                "is_relative": False,
            })
    
    # Find all relative amounts
    for match in RELATIVE_CAP_PATTERN.finditer(text):
        original = match.group(0).strip()
        results.append({
            "original": original,
            "numeric_usd": None,
            "is_relative": True,
        })
    
    return results


def normalize_cap_display(amount: float | None, is_relative: bool = False, relative_str: str = "") -> str:
    """Format a cap amount for display.
    
    Args:
        amount: Numeric amount in USD.
        is_relative: Whether this is a relative cap.
        relative_str: Original relative string if applicable.
        
    Returns:
        Formatted display string.
    """
    if is_relative:
        return relative_str
    
    if amount is None:
        return "Unknown"
    
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    else:
        return f"${amount:,.2f}"
