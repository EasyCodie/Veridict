# Context7 MCP Tool - Summary for Veridict

## Quick Reference

### What is Context7 MCP?

Context7 MCP is a Model Context Protocol server that provides up-to-date, version-specific documentation for software libraries directly into your development workflow. It's available at:
- **Package:** `@upstash/context7-mcp` (v1.0.25)
- **Remote Endpoint:** `https://mcp.context7.com/mcp`
- **Local Server:** Can run via `npx @upstash/context7-mcp`

### Available Tools

| Tool | Purpose |
|------|---------|
| `resolve-library-id` | Find the Context7 library ID for a package name |
| `get-library-docs` | Fetch documentation with optional topic filtering |

---

## Veridict Technology Coverage

### ✅ Fully Supported

| Technology | Context7 Library ID | Notes |
|------------|---------------------|-------|
| **FastAPI** | `/fastapi/fastapi` | 1000+ code snippets, Trust Score 10 |
| **PyMuPDF** | `/pymupdf/pymupdf` | Excellent text extraction docs |
| **Pydantic** | `/pydantic/pydantic` | 1600+ snippets, v2 support |
| **asyncpg** | `/websites/magicstack_github_io_asyncpg` | Async PostgreSQL driver |
| **OpenAI SDK** | `/openai/openai-python` | Embeddings, completions |
| **pgvector** | `/pgvector/pgvector` | Vector similarity search |

### ⚠️ Partial/Alternative Coverage

| Technology | Recommendation |
|------------|----------------|
| **Unstructured.io** | Not in Context7 - use official docs |
| **Anthropic Python** | Use `/anthropics/anthropic-sdk-python` explicitly |

---

## Sample Usage

### 1. Resolve Library ID

```bash
# Find the library ID for FastAPI
curl -X POST https://mcp.context7.com/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"resolve-library-id",
                 "arguments":{"libraryName":"fastapi"}}}'
```

### 2. Get Documentation

```bash
# Fetch FastAPI file upload documentation
curl -X POST https://mcp.context7.com/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get-library-docs",
                 "arguments":{
                   "context7CompatibleLibraryID":"/fastapi/fastapi",
                   "topic":"file upload",
                   "tokens":5000}}}'
```

---

## Best Practices for Veridict Development

1. **Pre-define Library IDs** - Skip the resolution step by using known IDs
2. **Use Topic Filtering** - Focus documentation on specific features
3. **Combine Sources** - For pgvector, supplement with GitHub README
4. **Check Trust Scores** - Prefer libraries with scores > 7.0
5. **Consider API Key** - For heavy usage, get a key from context7.com/dashboard

---

## Performance Expectations

| Metric | Typical Value |
|--------|---------------|
| Response Time | 750-850ms |
| Doc Size | 20,000-30,000 chars |
| Success Rate | 100% (with correct IDs) |

---

## Integration with Veridict Workflow

### PDF Upload Endpoint
- Query: `/fastapi/fastapi` + topic "file upload"
- Returns: UploadFile patterns, async handling

### Text Extraction
- Query: `/pymupdf/pymupdf` + topic "text extraction"
- Returns: get_text() patterns, extraction flags

### Clause Embedding & Search
- Query: `/openai/openai-python` + topic "embeddings"
- Query: `/pgvector/pgvector` + topic "similarity search"
- Returns: Embedding generation, vector queries

---

## Files Created

- `test_context7_mcp.py` - Automated test script
- `docs/CONTEXT7_MCP_TEST_REPORT.md` - Detailed test report
- `docs/CONTEXT7_MCP_SUMMARY.md` - This quick reference
