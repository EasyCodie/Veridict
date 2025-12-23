"""Reference Standards Management for Comparison Agent.

This module provides utilities for managing market-standard reference data
used by the Comparison Agent.

Usage:
    from core.agents.utils.reference_standards import get_reference_standard
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ReferenceStandard:
    """A market-standard reference for a clause type."""
    standard_type: str
    standard_text: str
    prevalence: str
    clause_type: str
    jurisdiction: str | None = None
    contract_type: str | None = None


# Sample market standards for common clause types
MARKET_STANDARDS: dict[str, list[ReferenceStandard]] = {
    "Indemnity": [
        ReferenceStandard(
            standard_type="market_standard_2023",
            standard_text=(
                "The Indemnifying Party shall indemnify, defend, and hold harmless "
                "the Indemnified Party from and against any and all claims, damages, "
                "losses, costs, and expenses (including reasonable attorneys' fees) "
                "arising out of or relating to any breach of this Agreement by the "
                "Indemnifying Party."
            ),
            prevalence="85% of M&A transactions",
            clause_type="Indemnity",
        ),
    ],
    "Termination": [
        ReferenceStandard(
            standard_type="market_standard_2023",
            standard_text=(
                "Either Party may terminate this Agreement for convenience upon "
                "thirty (30) days' prior written notice to the other Party. Either "
                "Party may terminate immediately for cause upon material breach that "
                "remains uncured for thirty (30) days after written notice."
            ),
            prevalence="75% of commercial contracts",
            clause_type="Termination",
        ),
    ],
    "Liability": [
        ReferenceStandard(
            standard_type="market_standard_2023",
            standard_text=(
                "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, SPECIAL, "
                "INCIDENTAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES. THE TOTAL LIABILITY "
                "OF EACH PARTY SHALL NOT EXCEED THE AMOUNTS PAID OR PAYABLE UNDER "
                "THIS AGREEMENT DURING THE TWELVE (12) MONTHS PRECEDING THE CLAIM."
            ),
            prevalence="80% of technology contracts",
            clause_type="Liability",
        ),
    ],
    "Confidentiality": [
        ReferenceStandard(
            standard_type="market_standard_2023",
            standard_text=(
                "Each Party agrees to maintain the confidentiality of all Confidential "
                "Information received from the other Party and to use such information "
                "solely for the purposes contemplated by this Agreement. Confidential "
                "Information excludes information that: (a) is publicly available; "
                "(b) was known prior to disclosure; (c) is independently developed; "
                "or (d) is rightfully obtained from third parties."
            ),
            prevalence="90% of commercial contracts",
            clause_type="Confidentiality",
        ),
    ],
    "Warranty": [
        ReferenceStandard(
            standard_type="market_standard_2023",
            standard_text=(
                "The Seller represents and warrants that: (a) it has full power and "
                "authority to enter into this Agreement; (b) the Products will conform "
                "to specifications for a period of twelve (12) months from delivery; "
                "and (c) the Products do not infringe any third-party intellectual "
                "property rights."
            ),
            prevalence="70% of purchase agreements",
            clause_type="Warranty",
        ),
    ],
}


def get_reference_standard(
    clause_type: str,
    jurisdiction: str | None = None,
    contract_type: str | None = None,
) -> list[ReferenceStandard]:
    """Get reference standards for a clause type.
    
    Args:
        clause_type: Type of clause (e.g., 'Indemnity').
        jurisdiction: Optional jurisdiction filter.
        contract_type: Optional contract type filter.
        
    Returns:
        List of matching reference standards.
    """
    standards = MARKET_STANDARDS.get(clause_type, [])
    
    if jurisdiction:
        standards = [s for s in standards if s.jurisdiction is None or s.jurisdiction == jurisdiction]
    
    if contract_type:
        standards = [s for s in standards if s.contract_type is None or s.contract_type == contract_type]
    
    return standards


def format_reference_standards_for_prompt(standards: list[ReferenceStandard]) -> str:
    """Format reference standards for inclusion in prompt.
    
    Args:
        standards: List of reference standards.
        
    Returns:
        Formatted string for prompt.
    """
    if not standards:
        return "No reference standards available for this clause type."
    
    parts = []
    for i, standard in enumerate(standards, 1):
        parts.append(
            f"STANDARD {i} ({standard.standard_type}):\n"
            f"Prevalence: {standard.prevalence}\n"
            f"Language: {standard.standard_text}"
        )
    
    return "\n\n".join(parts)


async def fetch_reference_standards_from_db(
    pool: Any,
    clause_type: str,
    jurisdiction: str | None = None,
) -> list[ReferenceStandard]:
    """Fetch reference standards from database.
    
    Args:
        pool: asyncpg connection pool.
        clause_type: Type of clause.
        jurisdiction: Optional jurisdiction filter.
        
    Returns:
        List of reference standards.
    """
    async with pool.acquire() as conn:
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'reference_standards'
            )
        """)
        
        if not table_exists:
            # Fall back to in-memory standards
            return get_reference_standard(clause_type, jurisdiction)
        
        query = """
            SELECT standard_type, standard_text, prevalence, clause_type,
                   jurisdiction, contract_type
            FROM reference_standards
            WHERE clause_type = $1
        """
        params = [clause_type]
        
        if jurisdiction:
            query += " AND (jurisdiction IS NULL OR jurisdiction = $2)"
            params.append(jurisdiction)
        
        rows = await conn.fetch(query, *params)
        
        if rows:
            return [
                ReferenceStandard(
                    standard_type=row["standard_type"],
                    standard_text=row["standard_text"],
                    prevalence=row["prevalence"],
                    clause_type=row["clause_type"],
                    jurisdiction=row["jurisdiction"],
                    contract_type=row["contract_type"],
                )
                for row in rows
            ]
        
        # Fall back to in-memory standards
        return get_reference_standard(clause_type, jurisdiction)
