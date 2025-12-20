"""Script to upload a PDF to the Veridict API for testing."""

import sys
import requests
from pathlib import Path

def upload_pdf(file_path: str):
    url = "http://localhost:8000/api/v1/documents/upload"
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return

    print(f"📤 Uploading {path.name} to {url}...")
    
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "application/pdf")}
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            
            result = response.json()
            doc_id = result["id"]
            print(f"✅ Upload Successful!")
            print(f"🆔 Document ID: {doc_id}")
            print(f"📄 Pages: {result['page_count']}")
            print(f"📊 Status: {result['status']}")
            
            # Now fetch the extracted text
            print(f"\n🔍 Fetching extracted text...")
            text_url = f"http://localhost:8000/api/v1/documents/{doc_id}/text"
            text_response = requests.get(text_url)
            text_data = text_response.json()
            
            print("\n📝 Text Preview (first 500 chars):")
            print("-" * 50)
            print(text_data["text"][:500] + "...")
            print("-" * 50)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Upload failed: {e}")
            print("💡 Make sure the server is running (uvicorn app.main:app --reload)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/upload_pdf.py <path_to_your_pdf>")
    else:
        upload_pdf(sys.argv[1])
