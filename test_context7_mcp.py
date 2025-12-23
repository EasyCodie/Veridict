#!/usr/bin/env python3
"""
Context7 MCP Documentation Retrieval Test Script

This script tests the Context7 MCP tool for the Veridict clause retrieval system.
It validates documentation retrieval for key technologies:
- FastAPI (web framework)
- PyMuPDF (PDF parsing)
- pgvector/PostgreSQL (vector database)
"""

import json
import subprocess
import time
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Test configuration
MCP_ENDPOINT = "http://localhost:3001/mcp"
HEADERS = [
    "-H", "Content-Type: application/json",
    "-H", "Accept: application/json, text/event-stream"
]


@dataclass
class LibraryMatch:
    """A single library match from resolve-library-id."""
    title: str
    library_id: str
    description: str
    code_snippets: int
    trust_score: float


@dataclass
class TestResult:
    """Stores the result of a Context7 MCP query test."""
    query: str
    library_name: str
    library_id: Optional[str]
    all_matches: List[LibraryMatch]
    docs_retrieved: bool
    docs_length: int
    docs_preview: str
    response_time_ms: float
    success: bool
    error: Optional[str]


def call_mcp(method: str, params: dict) -> dict:
    """Make a JSON-RPC call to the Context7 MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    
    cmd = [
        "curl", "-s", "-X", "POST", MCP_ENDPOINT,
        *HEADERS,
        "-d", json.dumps(payload)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Parse SSE response
    output = result.stdout
    if output.startswith("event:"):
        # Extract data from SSE format
        for line in output.split("\n"):
            if line.startswith("data:"):
                output = line[5:].strip()
                break
    
    try:
        response = json.loads(output)
        return {"response": response, "elapsed_ms": elapsed_ms}
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": output, "elapsed_ms": elapsed_ms}


def resolve_library_id(library_name: str) -> dict:
    """Resolve a library name to a Context7-compatible library ID."""
    return call_mcp("tools/call", {
        "name": "resolve-library-id",
        "arguments": {"libraryName": library_name}
    })


def get_library_docs(library_id: str, topic: str = "", tokens: int = 5000) -> dict:
    """Fetch documentation for a library."""
    args = {"context7CompatibleLibraryID": library_id, "tokens": tokens}
    if topic:
        args["topic"] = topic
    return call_mcp("tools/call", {
        "name": "get-library-docs",
        "arguments": args
    })


def parse_library_matches(text: str) -> List[LibraryMatch]:
    """Parse the resolve-library-id response into structured matches."""
    matches = []
    
    # Split by separator
    sections = text.split("----------")
    
    for section in sections:
        section = section.strip()
        if not section or "Available Libraries" in section:
            continue
        
        # Extract fields
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', section)
        id_match = re.search(r'Context7-compatible library ID:\s*(/[^\s]+)', section)
        desc_match = re.search(r'Description:\s*(.+?)(?:\n-|$)', section, re.DOTALL)
        snippets_match = re.search(r'Code Snippets:\s*(\d+)', section)
        trust_match = re.search(r'Trust Score:\s*([\d.]+)', section)
        
        if id_match:
            matches.append(LibraryMatch(
                title=title_match.group(1).strip() if title_match else "",
                library_id=id_match.group(1).strip(),
                description=desc_match.group(1).strip() if desc_match else "",
                code_snippets=int(snippets_match.group(1)) if snippets_match else 0,
                trust_score=float(trust_match.group(1)) if trust_match else 0.0
            ))
    
    return matches


def select_best_match(matches: List[LibraryMatch], query: str) -> Optional[LibraryMatch]:
    """Select the best library match based on trust score and name similarity."""
    if not matches:
        return None
    
    # Prioritize exact name matches, then high trust scores
    query_lower = query.lower().replace(" ", "")
    
    for match in matches:
        # Check for exact or near-exact matches
        lib_name = match.library_id.split("/")[-1].lower().replace("-", "").replace("_", "")
        if query_lower == lib_name or query_lower in lib_name:
            return match
    
    # Fall back to highest trust score
    return max(matches, key=lambda m: m.trust_score)


def test_library(name: str, topic: str = "", expected_id: str = None) -> TestResult:
    """Test documentation retrieval for a library."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}" + (f" (topic: {topic})" if topic else ""))
    print('='*70)
    
    # Step 1: Resolve library ID
    print(f"\n1. Resolving library ID for '{name}'...")
    resolve_result = resolve_library_id(name)
    
    if "error" in resolve_result:
        return TestResult(
            query=name,
            library_name=name,
            library_id=None,
            all_matches=[],
            docs_retrieved=False,
            docs_length=0,
            docs_preview="",
            response_time_ms=resolve_result["elapsed_ms"],
            success=False,
            error=f"Resolve error: {resolve_result['error']}"
        )
    
    # Parse response
    try:
        content = resolve_result["response"]["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        content = ""
    
    print(f"   Response time: {resolve_result['elapsed_ms']:.0f}ms")
    
    # Parse all library matches
    all_matches = parse_library_matches(content)
    print(f"   Found {len(all_matches)} library matches")
    
    # Select best match or use expected_id
    if expected_id:
        lib_id = expected_id
        print(f"   Using specified library ID: {lib_id}")
    else:
        best_match = select_best_match(all_matches, name)
        if best_match:
            lib_id = best_match.library_id
            print(f"   Best match: {best_match.title} ({lib_id})")
            print(f"   Trust Score: {best_match.trust_score}, Snippets: {best_match.code_snippets}")
        else:
            print(f"   No matches found!")
            # Show first 3 matches for debugging
            if all_matches:
                print("\n   Top matches found:")
                for m in all_matches[:3]:
                    print(f"   - {m.title}: {m.library_id} (Trust: {m.trust_score})")
            
            return TestResult(
                query=name,
                library_name=name,
                library_id=None,
                all_matches=all_matches,
                docs_retrieved=False,
                docs_length=0,
                docs_preview="",
                response_time_ms=resolve_result["elapsed_ms"],
                success=False,
                error="No library matches found"
            )
    
    # Step 2: Get documentation
    print(f"\n2. Fetching documentation for {lib_id}...")
    docs_result = get_library_docs(lib_id, topic=topic, tokens=5000)
    
    if "error" in docs_result:
        return TestResult(
            query=name,
            library_name=name,
            library_id=lib_id,
            all_matches=all_matches,
            docs_retrieved=False,
            docs_length=0,
            docs_preview="",
            response_time_ms=resolve_result["elapsed_ms"] + docs_result["elapsed_ms"],
            success=False,
            error=f"Docs error: {docs_result['error']}"
        )
    
    print(f"   Response time: {docs_result['elapsed_ms']:.0f}ms")
    
    # Extract docs content
    try:
        docs_content = docs_result["response"]["result"]["content"][0]["text"]
        docs_length = len(docs_content)
        # Check if it's a real doc or an error message
        if "does not exist" in docs_content.lower() or docs_length < 200:
            docs_retrieved = False
            docs_preview = docs_content[:500]
        else:
            docs_retrieved = True
            docs_preview = docs_content[:800]
    except (KeyError, IndexError, TypeError):
        docs_content = ""
        docs_length = 0
        docs_preview = ""
        docs_retrieved = False
    
    total_time = resolve_result["elapsed_ms"] + docs_result["elapsed_ms"]
    
    print(f"\n3. Documentation Retrieved: {docs_retrieved} ({docs_length} chars)")
    print("-" * 50)
    if docs_preview:
        # Clean up the preview for display
        preview_lines = docs_preview.split('\n')[:15]
        for line in preview_lines:
            print(f"   {line[:80]}")
        if len(preview_lines) >= 15:
            print("   ...")
    else:
        print("   (No documentation retrieved)")
    print("-" * 50)
    
    return TestResult(
        query=name,
        library_name=name,
        library_id=lib_id,
        all_matches=all_matches,
        docs_retrieved=docs_retrieved,
        docs_length=docs_length,
        docs_preview=docs_preview,
        response_time_ms=total_time,
        success=docs_retrieved,
        error=None if docs_retrieved else "Documentation not found or too short"
    )


def run_all_tests():
    """Run all Context7 MCP tests for Veridict technologies."""
    print("\n" + "="*80)
    print("CONTEXT7 MCP DOCUMENTATION RETRIEVAL TESTS")
    print("Testing key technologies for Veridict Clause Retrieval System")
    print("="*80)
    
    # Tests with (query, topic, expected_library_id)
    tests: List[Tuple[str, str, Optional[str]]] = [
        # FastAPI tests - specify the exact library ID
        ("fastapi", "", "/fastapi/fastapi"),
        ("fastapi", "async endpoints", "/fastapi/fastapi"),
        ("fastapi", "file upload", "/fastapi/fastapi"),
        ("fastapi", "dependency injection", "/fastapi/fastapi"),
        
        # PyMuPDF tests
        ("pymupdf", "", None),
        ("pymupdf", "text extraction", None),
        
        # pgvector tests
        ("pgvector", "", None),
        ("pgvector", "similarity search", None),
        
        # asyncpg - PostgreSQL async driver
        ("asyncpg", "", None),
        
        # Pydantic - data validation
        ("pydantic", "", None),
        ("pydantic", "validation", None),
        
        # Additional relevant libraries
        ("openai python", "", None),
        ("anthropic python", "", None),
    ]
    
    results = []
    
    for query, topic, expected_id in tests:
        result = test_library(query, topic, expected_id)
        results.append(result)
        time.sleep(0.3)  # Rate limiting
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\n{'Query':<25} {'Library ID':<35} {'Docs':<8} {'Length':<10} {'Time':<10} {'Status'}")
    print("-" * 100)
    
    successes = 0
    failures = 0
    
    for r in results:
        query = r.query[:23] + ".." if len(r.query) > 25 else r.query
        lib_id = r.library_id[:33] + ".." if r.library_id and len(r.library_id) > 35 else (r.library_id or "N/A")
        docs = "✓" if r.docs_retrieved else "✗"
        length = f"{r.docs_length:,}" if r.docs_length else "-"
        time_str = f"{r.response_time_ms:.0f}ms"
        status = "✓ PASS" if r.success else "✗ FAIL"
        print(f"{query:<25} {lib_id:<35} {docs:<8} {length:<10} {time_str:<10} {status}")
        
        if r.success:
            successes += 1
        else:
            failures += 1
    
    print("-" * 100)
    print(f"\nTotal: {len(results)} tests | Passed: {successes} | Failed: {failures}")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # Group by technology
    tech_groups = {
        "FastAPI": [r for r in results if "fastapi" in r.query.lower()],
        "PyMuPDF": [r for r in results if "pymupdf" in r.query.lower()],
        "pgvector": [r for r in results if "pgvector" in r.query.lower()],
        "asyncpg": [r for r in results if "asyncpg" in r.query.lower()],
        "Pydantic": [r for r in results if "pydantic" in r.query.lower()],
        "OpenAI/Anthropic": [r for r in results if "openai" in r.query.lower() or "anthropic" in r.query.lower()],
    }
    
    for tech, tech_results in tech_groups.items():
        if not tech_results:
            continue
        
        success_count = sum(1 for r in tech_results if r.success)
        print(f"\n{tech}:")
        print(f"  Tests: {len(tech_results)} | Success: {success_count}/{len(tech_results)}")
        
        for r in tech_results:
            status = "✓" if r.success else "✗"
            if r.success:
                print(f"  {status} {r.query}: {r.docs_length:,} chars of docs retrieved")
            else:
                print(f"  {status} {r.query}: {r.error}")
            
            # Show available matches if failed
            if not r.success and r.all_matches:
                print(f"      Available matches: {[m.library_id for m in r.all_matches[:3]]}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    failed_techs = [tech for tech, rs in tech_groups.items() if rs and not any(r.success for r in rs)]
    
    if failed_techs:
        print(f"\nMissing coverage for: {', '.join(failed_techs)}")
        print("Consider adding these libraries to Context7 or using alternative sources:")
        for tech in failed_techs:
            if tech == "PyMuPDF":
                print(f"  - PyMuPDF: Use official docs at https://pymupdf.readthedocs.io/")
            elif tech == "pgvector":
                print(f"  - pgvector: Use GitHub README at https://github.com/pgvector/pgvector")
            elif tech == "asyncpg":
                print(f"  - asyncpg: Use official docs at https://magicstack.github.io/asyncpg/")
    
    return results


if __name__ == "__main__":
    run_all_tests()
