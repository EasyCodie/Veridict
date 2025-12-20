"""Test the PDF ingestion service with the sample contract."""

import sys
import os
sys.path.append(os.getcwd())

from core.decomposition.ingestion import pdf_service


def test_sample_contract(pdf_path: str = "tests/SampleContract-Shuttle.pdf"):
    """Test extracting text from the sample contract PDF."""
    
    print(f"📄 Testing PDF: {pdf_path}")
    
    # Extract content
    content = pdf_service.extract_from_file(pdf_path)
    
    print(f"✅ Filename: {content.filename}")
    print(f"✅ Pages: {content.page_count}")
    print(f"✅ Total words: {content.total_word_count}")
    print(f"✅ Is empty: {content.is_empty}")
    
    print("\n📝 Extracted Text Preview (first 1000 chars):")
    print("-" * 50)
    print(content.raw_text[:1000])
    print("-" * 50)
    
    # Check for expected clauses
    expected_clauses = [
        "INDEMNIFICATION",
        "TERMINATION",
        "CONFIDENTIALITY",
        "LIMITATION OF LIABILITY",
        "GOVERNING LAW",
        "FORCE MAJEURE",
    ]
    
    print("\n🔍 Checking for expected clause headers:")
    for clause in expected_clauses:
        if clause in content.raw_text:
            print(f"  ✅ Found: {clause}")
        else:
            print(f"  ❌ Missing: {clause}")
    
    print("\n✨ Ingestion test complete!")


if __name__ == "__main__":
    test_sample_contract()
