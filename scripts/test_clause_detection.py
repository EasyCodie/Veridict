"""Test the clause detection on PDF contracts."""

import sys
import os
import json

sys.path.append(os.getcwd())

from core.decomposition.ingestion import pdf_service
from core.decomposition.clause_detector import clause_detector


def select_pdf_from_tests() -> str:
    """Display a menu of PDF files from the tests directory for user selection."""
    tests_dir = "tests"
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


def test_clause_detection(pdf_path: str) -> None:
    """Test clause detection on a PDF file."""
    print(f"\n📄 Loading PDF: {pdf_path}")
    
    content = pdf_service.extract_from_file(pdf_path)
    print(f"✅ Extracted {content.page_count} pages, {content.total_word_count} words")
    
    print("\n🔍 Detecting clauses...")
    result = clause_detector.extract_to_json(content.raw_text)
    
    print(f"✅ Found {result['total_clauses']} clauses")
    print(f"📊 Clause types detected: {result['clause_types']}")
    
    print("\n📋 Clause Breakdown:")
    print("-" * 60)
    for i, clause in enumerate(result['clauses'], 1):
        preview = clause['text'][:100].replace('\n', ' ')
        print(f"{i}. [{clause['clause_type'].upper()}] {clause['section_number']}. {clause['title']}")
        print(f"   Preview: {preview}...")
        print()
    
    output_path = "tests/clause_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"💾 Full output saved to: {output_path}")
    print("\n✨ Clause detection complete!")


if __name__ == "__main__":
    # Check for command-line PDF path, otherwise show selection menu
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else select_pdf_from_tests()
    test_clause_detection(pdf_path)
