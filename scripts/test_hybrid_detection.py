"""Test the hybrid AI-powered clause detection."""

import sys
import os
import json
sys.path.append(os.getcwd())

from core.decomposition.ingestion import pdf_service
from core.decomposition.hybrid_detector import hybrid_detector


def test_hybrid_detection(pdf_path: str = "tests/SampleContract-Shuttle.pdf", use_ai: bool = True):
    """Test hybrid clause detection on a PDF file."""
    print(f"📄 Loading PDF: {pdf_path}")
    
    # Extract text
    content = pdf_service.extract_from_file(pdf_path)
    print(f"✅ Extracted {content.page_count} pages, {content.total_word_count} words")
    
    # Detect clauses with AI
    mode = "AI-Powered" if use_ai else "Structure Only"
    print(f"\n🤖 Running {mode} clause detection...")
    
    result = hybrid_detector.extract_to_json(content.raw_text, use_ai=use_ai)
    
    print(f"✅ Found {result['total_clauses']} clauses")
    print(f"📊 Clause types: {result['clause_types']}")
    print(f"⚠️  Risk summary: {result['risk_summary']}")
    print(f"🚩 Total red flags: {result['total_red_flags']}")
    
    if result['red_flags']:
        print("\n🚩 Red Flags Detected:")
        for flag in result['red_flags']:
            print(f"   • {flag}")
    
    print("\n📋 Clause Details:")
    print("-" * 70)
    for i, clause in enumerate(result['clauses'], 1):
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(clause['risk_level'], "⚪")
        print(f"{i}. {risk_emoji} [{clause['clause_type'].upper()}] {clause['title']}")
        print(f"   Confidence: {clause['confidence']:.0%} | Risk: {clause['risk_level']}")
        if clause['reasoning']:
            print(f"   Reasoning: {clause['reasoning']}")
        if clause['key_obligations']:
            print(f"   Obligations: {', '.join(clause['key_obligations'][:3])}")
        if clause['red_flags']:
            print(f"   ⚠️  Flags: {', '.join(clause['red_flags'])}")
        print()
    
    # Display usage summary
    if 'usage' in result:
        usage = result['usage']
        print("\n💰 API Usage Summary:")
        print(f"   Input tokens:  {usage['total_input_tokens']:,}")
        print(f"   Output tokens: {usage['total_output_tokens']:,}")
        print(f"   Total tokens:  {usage['total_tokens']:,}")
        print(f"   Total cost:    ${usage['total_cost_usd']:.4f}")
    
    # Save output
    output_path = "tests/hybrid_clause_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full output saved to: {output_path}")
    print("\n✨ Hybrid detection complete!")


if __name__ == "__main__":
    use_ai = "--no-ai" not in sys.argv
    pdf_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "tests/SampleContract-Shuttle.pdf"
    test_hybrid_detection(pdf_path, use_ai=use_ai)
