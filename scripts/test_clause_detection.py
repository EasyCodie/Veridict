"""Test the clause detection on the sample contract."""

import sys
import os
import json
sys.path.append(os.getcwd())

from core.decomposition.ingestion import pdf_service
from core.decomposition.clause_detector import clause_detector


def test_clause_detection(pdf_path: str = "tests/SampleContract-Shuttle.pdf"):
    """Test clause detection on a PDF file."""
    print(f"📄 Loading PDF: {pdf_path}")
    
    # Extract text
    content = pdf_service.extract_from_file(pdf_path)
    print(f"✅ Extracted {content.page_count} pages, {content.total_word_count} words")
    
    # Detect clauses
    print("\n🔍 Detecting clauses...")
    result = clause_detector.extract_to_json(content.raw_text)
    
    print(f"✅ Found {result['total_clauses']} clauses")
    print(f"📊 Clause types detected: {result['clause_types']}")
    
    print("\n📋 Clause Breakdown:")
    print("-" * 60)
    for i, clause in enumerate(result['clauses'], 1):
        # Truncate text for display
        preview = clause['text'][:100].replace('\n', ' ')
        print(f"{i}. [{clause['clause_type'].upper()}] {clause['section_number']}. {clause['title']}")
        print(f"   Preview: {preview}...")
        print()
    
    # Save full JSON output
    output_path = "tests/clause_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full output saved to: {output_path}")
    
    print("\n✨ Clause detection complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_clause_detection(sys.argv[1])
    else:
        test_clause_detection()
