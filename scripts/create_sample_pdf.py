"""Script to create a sample contract PDF for testing."""

import fitz  # PyMuPDF


def create_sample_contract():
    """Create a sample legal contract PDF."""
    doc = fitz.open()
    
    # Page 1 - Title and Parties
    page1 = doc.new_page()
    text1 = """
SAMPLE LEGAL CONTRACT

AGREEMENT made as of January 1, 2024

BETWEEN:

ABC Corporation ("Company")
AND
XYZ Enterprises ("Client")

1. INDEMNIFICATION

The Client agrees to indemnify, defend, and hold harmless the Company, its officers, 
directors, employees, and agents from and against any and all claims, damages, losses, 
costs, and expenses (including reasonable attorneys' fees) arising out of or relating 
to any breach of this Agreement by the Client.

2. TERMINATION

Either party may terminate this Agreement upon thirty (30) days written notice to the 
other party. Upon termination, all rights granted hereunder shall immediately cease.
"""
    page1.insert_text((50, 50), text1, fontsize=11)
    
    # Page 2 - More Clauses
    page2 = doc.new_page()
    text2 = """
3. CONFIDENTIALITY

Each party agrees to maintain the confidentiality of any proprietary or confidential 
information disclosed by the other party during the term of this Agreement. This 
obligation shall survive termination of this Agreement for a period of three (3) years.

4. LIMITATION OF LIABILITY

IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, 
SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING OUT OF OR RELATED TO THIS AGREEMENT, 
REGARDLESS OF WHETHER SUCH DAMAGES ARE BASED ON CONTRACT, TORT, STRICT LIABILITY, OR 
ANY OTHER THEORY.

5. GOVERNING LAW

This Agreement shall be governed by and construed in accordance with the laws of the 
State of Delaware, without giving effect to any choice of law or conflict of law provisions.

6. FORCE MAJEURE

Neither party shall be liable for any failure or delay in performing its obligations 
under this Agreement if such failure or delay results from circumstances beyond the 
reasonable control of that party, including but not limited to acts of God, natural 
disasters, war, terrorism, riots, embargoes, acts of civil or military authorities, 
fire, floods, accidents, strikes, or shortages of transportation, facilities, fuel, 
energy, labor, or materials.
"""
    page2.insert_text((50, 50), text2, fontsize=11)
    
    # Save to file
    output_path = "tests/sample_contract.pdf"
    doc.save(output_path)
    doc.close()
    print(f"✅ Sample contract created: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_contract()
