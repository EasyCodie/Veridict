"""Test the PDF ingestion service."""

import sys
import os

sys.path.append(os.getcwd())

from core.decomposition.ingestion import pdf_service


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
            print("\n\n� Cancelled.")
            sys.exit(0)


def test_ingestion(pdf_path: str) -> None:
    """Test extracting text from a PDF file."""
    print(f"\n📄 Testing PDF: {pdf_path}")
    
    content = pdf_service.extract_from_file(pdf_path)
    
    print(f"✅ Filename: {content.filename}")
    print(f"✅ Pages: {content.page_count}")
    print(f"✅ Total words: {content.total_word_count}")
    print(f"✅ Is empty: {content.is_empty}")
    
    print("\n📝 Extracted Text Preview (first 1000 chars):")
    print("-" * 50)
    print(content.raw_text[:1000])
    print("-" * 50)
    
    # Check for common clause headers
    common_headers = [
        "INDEMNIFICATION", "TERMINATION", "CONFIDENTIALITY",
        "LIABILITY", "GOVERNING LAW", "FORCE MAJEURE",
    ]
    
    print("\n🔍 Checking for common clause headers:")
    for header in common_headers:
        status = "✅" if header in content.raw_text.upper() else "❌"
        print(f"  {status} {header}")
    
    print("\n✨ Ingestion test complete!")


if __name__ == "__main__":
    # Check for command-line PDF path, otherwise show selection menu
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else select_pdf_from_tests()
    test_ingestion(pdf_path)
