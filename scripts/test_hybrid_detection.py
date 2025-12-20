"""Test the hybrid AI-powered clause detection."""

import sys
import os
import json
import time
import logging

# Configure logging to see reliability filter activity
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

sys.path.append(os.getcwd())

from core.decomposition.ingestion import pdf_service
from core.decomposition.hybrid_detector import hybrid_detector


def select_pdf_from_tests() -> str:
    """Display a menu of PDF files from the tests directory for user selection."""
    tests_dir = "tests"
    
    # Find all PDF files in the tests directory
    pdf_files = [f for f in os.listdir(tests_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in the tests/ directory.")
        sys.exit(1)
    
    pdf_files.sort()
    
    print("\n📁 Available PDF files in tests/:")
    print("-" * 40)
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf}")
    print("-" * 40)
    
    while True:
        try:
            choice = input(f"\nSelect a PDF (1-{len(pdf_files)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(pdf_files):
                return os.path.join(tests_dir, pdf_files[index])
            print(f"⚠️  Please enter a number between 1 and {len(pdf_files)}")
        except ValueError:
            print("⚠️  Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Cancelled.")
            sys.exit(0)


def test_hybrid_detection(pdf_path: str, use_ai: bool = True) -> None:
    """Test hybrid clause detection on a PDF file."""
    print(f"\n📄 Loading PDF: {pdf_path}")
    
    content = pdf_service.extract_from_file(pdf_path)
    print(f"✅ Extracted {content.page_count} pages, {content.total_word_count} words")
    
    mode = "AI-Powered" if use_ai else "Structure Only"
    print(f"\n🤖 Running {mode} clause detection...")
    
    start_time = time.time()
    result = hybrid_detector.extract_to_json(content.raw_text, use_ai=use_ai)
    elapsed_time = time.time() - start_time
    
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
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
    
    for i, clause in enumerate(result['clauses'], 1):
        emoji = risk_emoji.get(clause['risk_level'], "⚪")
        print(f"{i}. {emoji} [{clause['clause_type'].upper()}] {clause['title']}")
        label = clause.get('confidence_label', 'Medium')
        print(f"   Confidence: {label} | Risk: {clause['risk_level']}")
        if clause['reasoning']:
            print(f"   Reasoning: {clause['reasoning']}")
        if clause['key_obligations']:
            print(f"   Obligations: {', '.join(clause['key_obligations'][:3])}")
        if clause['red_flags']:
            print(f"   ⚠️  Flags: {', '.join(clause['red_flags'])}")
        print()
    
    if 'usage' in result:
        usage = result['usage']
        print("\n💰 API Usage Summary:")
        print(f"   Input tokens:  {usage['total_input_tokens']:,}")
        print(f"   Output tokens: {usage['total_output_tokens']:,}")
        print(f"   Total tokens:  {usage['total_tokens']:,}")
        print(f"   Total cost:    ${usage['total_cost_usd']:.4f}")
    
    # Display timing
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n⏱️  Time taken: {int(minutes)}m {seconds:.1f}s")
    
    output_path = "tests/hybrid_clause_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full output saved to: {output_path}")
    print("\n✨ Hybrid detection complete!")


if __name__ == "__main__":
    use_ai = "--no-ai" not in sys.argv
    
    # Check for command-line PDF path, otherwise show selection menu
    pdf_path = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            pdf_path = arg
            break
    
    if not pdf_path:
        pdf_path = select_pdf_from_tests()
    
    test_hybrid_detection(pdf_path, use_ai=use_ai)
