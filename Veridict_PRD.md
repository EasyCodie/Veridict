# VERIDICT PRD
## Autonomous Legal Due Diligence Engine
### Version 1.0 | December 2025

---

## EXECUTIVE SUMMARY

**Product Name:** Veridict  
**Product Vision:** An autonomous, zero-error legal due diligence engine designed to automate the review of high-volume legal contracts in data room environments.

**Core Value Proposition:**
- Replaces manual junior associate contract review
- Achieves 99.99%+ accuracy through MAKER Framework decomposition
- Processes M&A data rooms with statistical guarantee of correctness
- Dramatically reduces legal review time and costs

**Technical Foundation:** Veridict implements the **MAKER Framework** (Massively Decomposed Agentic Processes) to decompose complex contracts into atomic micro-tasks, execute them with high reliability through First-to-Ahead-by-k voting, and aggregate results with mathematical precision.

**Scope:** This PRD covers **3 core components** of the Veridict system:
1. The Decomposition Engine (The Manager)
2. The Micro-Agent Factory (The Workers)
3. The Voting Mechanism (The Verifier)

The Red-Flagging component (The Filter) is documented separately as a quality assurance subsystem.

---

## SECTION 1: PRODUCT OVERVIEW & MARKET CONTEXT

### 1.1 Problem Statement

**Current State of Legal Due Diligence:**
- M&A data rooms contain thousands of contracts requiring manual review
- Junior associates spend weeks reviewing contracts at ~$200-300/hour
- Average review accuracy: 85-92% (significant risk of missed issues)
- Critical clauses are frequently overlooked due to document fatigue
- Redlining and revision cycles add weeks to transaction timelines

**Why Existing Solutions Fail:**
- Document AI tools struggle with legal language nuance
- Standard LLM contracts review shows 3-8% per-document error rate
- No error correction mechanism for long-form legal analysis
- Hallucination rates increase with document complexity
- Cannot guarantee accuracy on financial/liability clauses (where errors are catastrophic)

**The Specific Challenge We Solve:**
Legal contracts require extreme precision on high-stakes decisions:
- Indemnity caps and scope
- Termination conditions and penalties
- Warranties and representations
- Liability limitations
- Data protection obligations

A 1% error rate over 500 contracts = **5 contracts with missed critical issues**. In M&A, this can mean millions in exposure.

### 1.2 Product Mission

**Veridict's mission:** To transform legal due diligence from a bottleneck requiring human junior associates into a systematic, autonomous process with **zero tolerance for error**—achieved through extreme task decomposition and statistical voting rather than building progressively larger language models.

### 1.3 Market Opportunity

**Target Market:** M&A Legal Teams & Law Firms
- **TAM:** $4.2B global legal services market for contract review (est. 15% of $28B legal tech market)
- **SAM:** $800M addressable market (major law firms, corporate legal teams, deal management platforms)
- **SOM:** $15-25M Year 1 (top-tier firms + venture/PE groups)

**Key Customer Segments:**
1. Large law firms (100+ partners) conducting 20+ M&A deals/year
2. In-house legal teams at PE/VC firms reviewing portfolio company transactions
3. Specialized M&A deal management platforms (Intralinks, Firmex)
4. Corporate development teams at Fortune 500 companies

---

## SECTION 2: TECHNICAL ARCHITECTURE OVERVIEW

### 2.1 The MAKER Framework Applied to Legal Documents

Veridict decomposes legal contract analysis using the MAKER framework:

```
Raw Contract PDF
         ↓
[Decomposition Engine] → Atomic Clause Extraction
         ↓
[Micro-Agent Factory] → Focused Analysis Workers
         ↓
[Voting Mechanism] → First-to-Ahead-by-k Voting
         ↓
Zero-Error Consensus Output
```

**Why This Works for Legal Documents:**
- **Extreme Decomposition:** Each clause is analyzed independently (atomic unit = single clause analysis)
- **Focused Context:** Workers receive only the clause text + relevant surrounding context (no full contract cognitive burden)
- **Statistical Reliability:** Multiple independent agents vote on each clause finding
- **Error Correlation Reduction:** Red-flagging filters outputs with structural errors (indicating reasoning failure)

### 2.2 System Architecture (High Level)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          VERIDICT SYSTEM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Decomposition   │  │  Micro-Agent     │  │  Voting          │  │
│  │  Engine          │→ │  Factory         │→ │  Mechanism       │  │
│  │  (Manager)       │  │  (Workers)       │  │  (Verifier)      │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│           ↓                    ↓                      ↓              │
│       Processes            Executes               Aggregates        │
│       Clauses              Micro-Tasks            Results           │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │               Red-Flagging (Quality Filter)                  │  │
│  │        Discards malformed/suspicious outputs                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SECTION 3: COMPONENT 1 – DECOMPOSITION ENGINE (THE MANAGER)

### 3.1 Purpose & Role

**Decomposition Engine Mission:**
Transform raw, unstructured legal PDFs into a structured stream of atomic, independently-analyzable units (clauses) with preserved context and metadata.

**Key Principle:** "Atomic Decomposition for Context Isolation"
- Each clause becomes a standalone analysis unit
- Minimal but sufficient context is preserved (e.g., parent agreement type, date, parties)
- Boundaries are deterministic and reproducible
- Output format is JSON-streamed for downstream processing

### 3.2 Input & Output Specifications

#### 3.2.1 Input Specifications

**Input Format:**
- PDF files (scanned or digital text)
- File size range: 50 KB - 50 MB (typical contract = 200 KB - 10 MB)
- Supported contract types: NDA, SPA, APA, Lease, License, Employment, IP Assignment, Data Processing Addendum, Escrow, etc.

**Input Metadata:**
```json
{
  "document_id": "string (unique SHA256 hash)",
  "file_name": "string",
  "contract_type": "enum (SPA | APA | NDA | Lease | License | Employment | IP | DPA | Other)",
  "parties": ["string"],
  "execution_date": "ISO8601 | null",
  "effective_date": "ISO8601 | null",
  "jurisdiction": "string (e.g., 'Delaware' or 'England & Wales')",
  "language": "string (default: 'en')",
  "upload_timestamp": "ISO8601"
}
```

**Quality Assumptions:**
- PDF is machine-readable (OCR NOT required for digital PDFs; optional for scanned documents)
- Text layer exists or can be generated via OCR
- Contract is in English (other languages supported as extension)

#### 3.2.2 Output Specifications

**Output Format:** JSON-streamed chunks (one JSON object per line, newline-delimited)

```json
{
  "decomposition_batch_id": "string (UUID)",
  "clause_id": "string (format: '{document_id}_{sequence_number}')",
  "sequence_number": "integer (1-based index)",
  "clause_type": "enum (Indemnity | Termination | Warranty | Liability | Confidentiality | IP | Payment | Definitions | Other)",
  "clause_title": "string (extracted heading or inferred title)",
  "clause_text": "string (full text of clause)",
  "start_page": "integer",
  "start_offset": "integer (character offset in document)",
  "end_page": "integer",
  "end_offset": "integer",
  "context_before": "string (50 words prior for reference)",
  "context_after": "string (50 words after for reference)",
  "parent_section": "string (e.g., 'Section 5. Indemnification')",
  "metadata": {
    "contract_id": "string",
    "contract_type": "string",
    "parties": ["string"],
    "execution_date": "ISO8601 | null",
    "jurisdiction": "string"
  },
  "extraction_confidence": "number (0-1, confidence that boundaries are correct)",
  "is_numbered": "boolean (whether clause has a section number)",
  "section_number": "string | null (e.g., '5.1.2')",
  "character_count": "integer",
  "word_count": "integer",
  "has_table": "boolean",
  "has_schedule_reference": "boolean"
}
```

### 3.3 Decomposition Algorithm

#### 3.3.1 Phase 1: PDF Processing & Text Extraction

**Input:** Raw PDF file  
**Output:** Machine-readable text with layout metadata

**Process:**
1. **PDF Parsing:**
   - Use [PDFPlumber](https://github.com/jsvine/pdfplumber) or [PyPDF4](https://github.com/claydh/PyPDF4) for text extraction
   - Preserve page boundaries and text coordinates
   - Extract tables and special formatting markers

2. **OCR Integration (Optional):**
   - If PDF is scanned (no text layer), apply Tesseract OCR or similar
   - Confidence threshold: ≥0.85 for character recognition
   - Flag low-confidence regions for manual review

3. **Layout Analysis:**
   - Identify heading hierarchy (H1 → H2 → H3)
   - Detect section numbering patterns (e.g., "5.1.2")
   - Map visual structure to logical document tree

**Implementation Note:** Use layout-aware PDF processing to preserve section hierarchies and clause boundaries without relying on heuristic splitting.

#### 3.3.2 Phase 2: Clause Boundary Detection

**Input:** Structured text with layout metadata  
**Output:** Identified clause boundaries (start/end positions)

**Detection Methods (Hierarchical):**

1. **Explicit Numbering** (Highest Priority)
   ```
   Pattern: "5.1.2 Indemnification by Seller"
   → Boundary = from "5.1.2" to start of "5.1.3" or "5.2"
   Confidence: 0.95-1.0
   ```

2. **Heading-Based Segmentation** (High Priority)
   ```
   Pattern: "INDEMNIFICATION" (bold, larger font)
   → Boundary = from heading to next same-level heading
   Confidence: 0.85-0.95
   ```

3. **Semantic Markers** (Medium Priority)
   ```
   Patterns: "This section", "shall", "provided that", "notwithstanding"
   → Use NLP entity recognition to identify clause boundaries
   Confidence: 0.70-0.85
   ```

4. **Visual Clustering** (Fallback)
   ```
   Uses paragraph breaks, indentation, whitespace analysis
   Confidence: 0.60-0.75
   ```

**Algorithm Pseudocode:**
```python
def detect_clause_boundaries(document_tree):
    clauses = []
    
    for section in document_tree.sections:
        # Method 1: Numbered sections
        numbered_clauses = extract_numbered_subsections(section)
        clauses.extend(numbered_clauses)
        
        # Method 2: If no numbering, detect by headings
        if not numbered_clauses:
            heading_clauses = extract_by_headings(section)
            clauses.extend(heading_clauses)
        
        # Method 3: If insufficient, use semantic markers
        if len(clauses) < min_expected_count:
            semantic_clauses = extract_by_semantic_markers(section)
            clauses.extend(semantic_clauses)
    
    return sort_by_position(clauses)
```

#### 3.3.3 Phase 3: Clause Classification

**Input:** Identified clause text  
**Output:** Clause type label + confidence score

**Classification Taxonomy:**

| Clause Type | Key Indicators | Examples |
|---|---|---|
| **Indemnity** | "indemnify", "hold harmless", "losses", "claims" | Indemnification by Seller; Indemnification Procedures |
| **Termination** | "terminate", "termination", "grounds", "condition precedent" | Termination Rights; Termination Fees; Termination Assistance |
| **Warranty** | "represents", "warrants", "warranty", "condition" | Seller's Representations; No Infringement Warranty |
| **Liability** | "liability", "limit", "cap", "exclude" | Limitation of Liability; Liability Caps |
| **Confidentiality** | "confidential", "confidentiality", "trade secret", "non-disclosure" | Confidentiality; NDA; Confidential Information |
| **IP/Intellectual Property** | "intellectual property", "IP", "patent", "trademark", "copyright" | IP Ownership; IP Indemnification; IP Licensing |
| **Payment/Financial** | "payment", "price", "consideration", "amount", "invoice" | Purchase Price; Payment Terms; Adjustment Mechanism |
| **Definitions** | "defined as", "shall mean", "includes but not limited to" | Definitions; Interpretation |
| **Other** | Miscellaneous clauses not fitting above | Governing Law; Entire Agreement; Severability |

**Classification Algorithm:**
```python
def classify_clause(clause_text, clause_title, context):
    # Step 1: Score against keyword patterns
    scores = {}
    for clause_type, keywords in KEYWORD_PATTERNS.items():
        match_count = sum(1 for kw in keywords if kw in clause_text.lower())
        scores[clause_type] = match_count / len(keywords)
    
    # Step 2: Apply LLM classifier for edge cases
    if max(scores.values()) < 0.6:
        llm_classification = classifier_llm.classify(clause_text, clause_title)
        scores.update(llm_classification)
    
    # Step 3: Return top classification with confidence
    best_type = max(scores, key=scores.get)
    confidence = scores[best_type]
    
    return {
        "clause_type": best_type,
        "confidence": min(confidence, 1.0),
        "alternative_types": sorted(scores.items(), key=lambda x: -x[1])[:2]
    }
```

### 3.4 Decomposition Quality Assurance

**Validation Checks (Post-Decomposition):**

```python
def validate_decomposition(clauses, original_document):
    errors = []
    
    # Check 1: Coverage (all text accounted for)
    total_extracted = sum(len(c['clause_text']) for c in clauses)
    if total_extracted / len(original_document) < 0.95:
        errors.append({
            "type": "INCOMPLETE_COVERAGE",
            "message": f"Only {total_extracted}/{len(original_document)} chars extracted",
            "severity": "HIGH"
        })
    
    # Check 2: Overlap detection
    for i, clause_i in enumerate(clauses):
        for j, clause_j in enumerate(clauses):
            if i != j:
                if overlaps(clause_i, clause_j):
                    errors.append({
                        "type": "OVERLAPPING_CLAUSES",
                        "clause_ids": [clause_i['clause_id'], clause_j['clause_id']],
                        "severity": "MEDIUM"
                    })
    
    # Check 3: Orphaned text (gaps between clauses)
    gaps = find_gaps(clauses)
    for gap in gaps:
        if gap['length'] > 100:  # Non-trivial gap
            errors.append({
                "type": "ORPHANED_TEXT",
                "gap_context": gap['context'],
                "severity": "MEDIUM"
            })
    
    # Check 4: Boundary accuracy
    for clause in clauses:
        if is_boundary_cut_midsentence(clause):
            errors.append({
                "type": "MALFORMED_BOUNDARY",
                "clause_id": clause['clause_id'],
                "severity": "MEDIUM"
            })
    
    return {
        "is_valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "coverage_ratio": total_extracted / len(original_document)
    }
```

### 3.5 Implementation Stack

**Technologies:**
- **PDF Processing:** PDFPlumber + PyMuPDF (fitz)
- **OCR (Optional):** Tesseract OCR + Pytesseract wrapper
- **Layout Analysis:** Detectron2 (for document structure detection) OR simple heuristic-based approach (for MVP)
- **NLP/Classification:** spaCy (lightweight) + optional LLM classifier (Claude/GPT-4-mini)
- **Streaming Output:** NDJSON format (one JSON per line)
- **Language:** Python 3.10+ with async I/O (asyncio)

**Performance Targets:**
| Metric | Target |
|---|---|
| PDF parsing time | < 2 seconds per page |
| Clause extraction time | < 5 seconds per document (avg 50 clauses) |
| Classification accuracy | ≥ 95% top-1 accuracy on labeled test set |
| Decomposition latency (E2E) | < 30 seconds for 50-page contract |
| Coverage ratio (text extracted) | ≥ 99% |

### 3.6 Error Handling & Fallbacks

**Degradation Strategies:**

| Failure Mode | Detection | Fallback |
|---|---|---|
| **PDF unreadable** | PDF parsing exception | Flag for manual review; reject document |
| **OCR confidence < 80%** | OCR confidence < 0.80 | Escalate to human; optionally re-upload |
| **Clause boundaries ambiguous** | Multiple overlapping boundaries | Conservative approach: use smaller clauses + longer context |
| **Classification confidence < 60%** | Classifier confidence score | Label as "Other"; flag for verification in voting phase |
| **Document malformed** | Validation errors > threshold | Reject with specific error message; request clean PDF |

---

## SECTION 4: COMPONENT 2 – MICRO-AGENT FACTORY (THE WORKERS)

### 4.1 Purpose & Role

**Micro-Agent Factory Mission:**
Execute stateless, highly-focused micro-tasks on individual clauses using small-to-medium language models, where each agent analyzes exactly one atomic unit (one clause analysis) with minimal context to prevent hallucination.

**Key Principle:** "Stateless Hyper-Specialization"
- Each agent = one task
- Each task = one clause analysis
- Minimal context (clause + metadata only)
- No multi-step reasoning required
- Output is deterministic, structured JSON

### 4.2 Agent Taxonomy & Specialization

**Agent Types (by task):**

#### 4.2.1 Clause Analysis Agent

**Task:** Extract and structure key legal findings from a single clause

**Input:**
```json
{
  "clause_id": "string",
  "clause_type": "enum (Indemnity | Termination | Warranty | etc.)",
  "clause_text": "string",
  "metadata": {
    "contract_type": "string (e.g., 'SPA')",
    "parties": ["string"],
    "jurisdiction": "string",
    "execution_date": "ISO8601"
  },
  "task_instructions": "string (e.g., 'Extract indemnification obligations and caps')"
}
```

**Output:**
```json
{
  "clause_id": "string",
  "findings": [
    {
      "finding_type": "enum (obligation | cap | exception | definition | trigger | duration | process)",
      "finding_text": "string (direct quote or paraphrase)",
      "importance": "enum (critical | high | medium | low)",
      "confidence": "number (0-1)",
      "supporting_quote": "string (exact text from clause)"
    }
  ],
  "summary": "string (1-2 sentence summary of clause effect)",
  "risk_level": "enum (critical | high | medium | low | informational)",
  "clarification_required": "boolean",
  "clarification_note": "string | null",
  "execution_timestamp": "ISO8601",
  "model_used": "string (e.g., 'gpt-4-mini')"
}
```

#### 4.2.2 Obligation Extractor Agent

**Task:** Identify who must do what and under what conditions

**Input:** Clause text + party names + agreement type  
**Output:**
```json
{
  "clause_id": "string",
  "obligations": [
    {
      "obligor": "string (e.g., 'Seller')",
      "obligation_description": "string",
      "trigger_condition": "string (e.g., 'Within 30 days of closing')",
      "deadline": "string | null (e.g., '30 days')",
      "consequences_of_breach": "string | null",
      "confidence": "number (0-1)"
    }
  ]
}
```

#### 4.2.3 Cap & Limitation Agent

**Task:** Extract financial limits, liability caps, exceptions

**Input:** Clause text (typically Liability clause)  
**Output:**
```json
{
  "clause_id": "string",
  "caps": [
    {
      "cap_type": "enum (liability_cap | indemnity_cap | damages_limitation | deductible)",
      "cap_amount": "string (e.g., '$5M' or 'X% of purchase price')",
      "cap_amount_numeric": "number | null (in USD if convertible)",
      "applies_to": "string (what losses does this cap apply to)",
      "exceptions": ["string"] (carve-outs from cap)",
      "confidence": "number (0-1)"
    }
  ],
  "no_cap_found": "boolean"
}
```

#### 4.2.4 Risk Identifier Agent

**Task:** Flag high-risk legal language or unusual provisions

**Input:** Clause text + clause type + context  
**Output:**
```json
{
  "clause_id": "string",
  "risks_identified": [
    {
      "risk_type": "enum (unusual_language | ambiguous_definition | contradicts_standard | missing_safeguard | unlimited_exposure | undefined_term)",
      "risk_description": "string",
      "severity": "enum (critical | high | medium | low)",
      "remediation_suggestion": "string",
      "confidence": "number (0-1)",
      "supporting_quote": "string"
    }
  ],
  "overall_risk_rating": "enum (clean | minor_issues | significant_issues | critical_issues)"
}
```

#### 4.2.5 Definition Validator Agent

**Task:** Check if all key terms in a clause are properly defined; flag undefined references

**Input:** Clause text + definitions section (if available)  
**Output:**
```json
{
  "clause_id": "string",
  "undefined_terms": [
    {
      "term": "string",
      "context_usage": "string (how it's used in clause)",
      "severity": "enum (critical | high | medium)",
      "likelihood_defined_elsewhere": "boolean",
      "search_suggestions": ["string"]
    }
  ],
  "all_terms_defined": "boolean"
}
```

#### 4.2.6 Comparison Agent

**Task:** Compare a clause against market-standard language (for contracts with similar types)

**Input:** Clause text + clause type + jurisdiction + reference standard language  
**Output:**
```json
{
  "clause_id": "string",
  "deviations_from_standard": [
    {
      "deviation_type": "enum (more_restrictive | less_protective | non_standard | missing_safeguard | additional_obligation)",
      "standard_language": "string",
      "actual_language": "string",
      "impact": "string (who does this favor)",
      "market_prevalence": "string (e.g., '15% of SPAs include this')",
      "severity": "enum (critical | high | medium | low)"
    }
  ]
}
```

### 4.3 Agent Prompt Engineering

**Design Principles:**

1. **Hyper-specific tasks:** No multi-step reasoning; one clause = one agent task
2. **Minimal context:** Only clause text + essential metadata
3. **Structured output:** JSON schema validation (not prose)
4. **No hallucination slack:** Explicit instruction to output "NOT FOUND" if uncertain
5. **Quote-based evidence:** All findings backed by direct text quotes

**Example Prompt Template (Obligation Extractor):**

```
You are a legal obligation extractor analyzing a single contract clause.

CLAUSE TEXT:
{clause_text}

CONTRACT METADATA:
- Type: {contract_type}
- Parties: {parties}
- Jurisdiction: {jurisdiction}

TASK:
Extract all obligations (who must do what, under what conditions) from this clause ONLY.

RULES:
1. Extract only what is explicitly stated in this clause.
2. Do NOT infer obligations from other sections.
3. For each obligation, identify:
   - WHO must act (the obligor)
   - WHAT they must do (obligation description)
   - WHEN/IF it applies (trigger condition and deadline)
   - WHAT happens if they don't (consequences)
4. If an obligation is unclear or partially stated, mark confidence < 0.7.
5. If no obligations exist, return empty obligations array with "no_obligations_found": true.

OUTPUT FORMAT (JSON):
{
  "clause_id": "{clause_id}",
  "obligations": [
    {
      "obligor": "string",
      "obligation_description": "string",
      "trigger_condition": "string",
      "deadline": "string or null",
      "consequences_of_breach": "string or null",
      "confidence": number between 0 and 1
    }
  ],
  "no_obligations_found": boolean
}

RESPONSE (JSON ONLY, NO EXPLANATIONS):
```

### 4.4 LLM Selection & Configuration

**Model Selection Criteria:**

| Metric | Rationale |
|---|---|
| **Model Size** | Smaller is better (3B-7B parameters preferred); MAKER scales reliability via voting, not model size |
| **Instruction Following** | Must follow structured output requirements perfectly |
| **Cost per Token** | Cost directly impacts voting overhead; prefer <$0.50/1M tokens |
| **Latency** | < 2 seconds per task (allows parallelization) |
| **Consistency** | Low variance between runs (high semantic consistency) |

**Recommended Models (Ranked by Performance/Cost):**

| Model | Provider | Cost | Latency | Strengths | Notes |
|---|---|---|---|---|---|
| **GPT-4-mini** | OpenAI | $0.00075 in / $0.003 out | ~1.5s | Reliable instruction following; low error rate | Preferred for voting |
| **Claude 3.5 Haiku** | Anthropic | $0.008 in / $0.0004 out | ~2s | Strong reasoning; good with complex tasks | Good alternative |
| **Mistral-7B** | Mistral | $0.00007 in / $0.00021 out | ~3s | Cost-effective; good instruction following | For cost-sensitive deployments |
| **Llama 3.1 8B** | Meta | $0.0002 in / $0.001 out | ~2.5s | Open-weight; low cost | Requires self-hosting |

**Configuration (per task):**

```json
{
  "model": "gpt-4-mini",
  "temperature": 0.0,
  "max_tokens": 800,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "response_format": "json",
  "timeout": 10,
  "retry_policy": {
    "max_retries": 3,
    "backoff_factor": 2,
    "retry_on": ["timeout", "rate_limit", "server_error"]
  }
}
```

**Rationale:**
- **Temperature 0:** Deterministic outputs (critical for voting consistency)
- **JSON response format:** Enforces structured output
- **Timeout 10s:** Prevents hung requests
- **Retries:** Handles transient API failures

### 4.5 Agent Execution Flow

**Sequential Processing (MVP):**

```python
async def execute_agent_task(clause, task_config, agent_instructions):
    """
    Execute a single agent task on a clause.
    
    Args:
        clause: Decomposed clause dict with text, metadata, ID
        task_config: Agent configuration (model, temperature, etc.)
        agent_instructions: Task-specific prompt template
    
    Returns:
        task_result: Structured JSON output from agent
        metadata: Execution metadata (model, tokens, latency, etc.)
    """
    
    # Step 1: Prepare prompt
    prompt = agent_instructions.format(
        clause_id=clause['clause_id'],
        clause_text=clause['clause_text'],
        contract_type=clause['metadata']['contract_type'],
        parties=", ".join(clause['metadata']['parties']),
        jurisdiction=clause['metadata']['jurisdiction']
    )
    
    # Step 2: Call LLM
    start_time = time.time()
    try:
        response = await llm_client.create_completion(
            model=task_config['model'],
            messages=[
                {"role": "system", "content": "You are a legal contract analyzer."},
                {"role": "user", "content": prompt}
            ],
            **task_config
        )
    except APIError as e:
        return retry_with_backoff(prompt, task_config, max_retries=3)
    
    # Step 3: Parse output
    response_text = response.choices[0].message.content
    try:
        task_result = json.loads(response_text)
    except json.JSONDecodeError:
        # JSON parsing failed; apply repair heuristics
        task_result = repair_json(response_text)
    
    # Step 4: Return result + metadata
    return {
        "task_result": task_result,
        "metadata": {
            "execution_time": time.time() - start_time,
            "model": task_config['model'],
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_cost": calculate_cost(response.usage),
            "success": True
        }
    }
```

**Parallel Execution (Scaling):**

```python
async def execute_agents_parallel(clauses, task_configs):
    """
    Execute multiple agent tasks in parallel.
    
    Uses asyncio.gather with concurrency limit to prevent rate limit issues.
    """
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # e.g., 10
    
    async def bounded_execute(clause, task_config):
        async with semaphore:
            return await execute_agent_task(clause, task_config, PROMPT_TEMPLATES[task_config['type']])
    
    tasks = [
        bounded_execute(clause, task_configs[i % len(task_configs)])
        for i, clause in enumerate(clauses)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        "results": [r for r in results if not isinstance(r, Exception)],
        "errors": [str(r) for r in results if isinstance(r, Exception)],
        "total_executed": len([r for r in results if not isinstance(r, Exception)]),
        "total_failed": len([r for r in results if isinstance(r, Exception)])
    }
```

### 4.6 Stateless Architecture

**Why Stateless?**

1. **Prevents hallucination chain:** Each agent doesn't "remember" previous clause analyses; no cascading errors
2. **Enables parallelization:** Tasks can run in any order or simultaneously
3. **Simplifies recovery:** Failed tasks can be retried without state reconstruction
4. **Reduces cost:** No context carryover means no token waste

**Stateless Design Pattern:**

```python
# BAD: Stateful agent (carries previous context)
class StatefulAgent:
    def __init__(self):
        self.conversation_history = []
    
    def analyze_clause(self, clause):
        # Agent remembers previous clauses!
        # Risk: confuses context from clause N with clause N+1
        self.conversation_history.append(clause)
        return self.model.chat(self.conversation_history)

# GOOD: Stateless agent (fresh analysis each time)
async def stateless_analyze_clause(clause, task_instructions):
    # Fresh prompt with ONLY this clause's data
    # No history; no context bleeding
    prompt = task_instructions.format(
        clause_text=clause['text'],
        clause_id=clause['id']
    )
    return await llm_client.create_completion(messages=[{"role": "user", "content": prompt}])
```

### 4.7 Quality Metrics & Monitoring

**Per-Task Metrics:**

```json
{
  "task_metrics": {
    "execution_time_ms": 1500,
    "model": "gpt-4-mini",
    "tokens_used": 450,
    "cost_usd": 0.00135,
    "output_format_valid": true,
    "json_parseable": true,
    "fields_present": true,
    "all_fields_non_null": true,
    "confidence_scores_valid": true,
    "supporting_quotes_present": true,
    "confidence_average": 0.87
  },
  "health_status": "healthy"
}
```

**Aggregated Metrics (Dashboard):**

| Metric | Target | Alert Threshold |
|---|---|---|
| Task success rate | ≥ 99.5% | < 98% |
| JSON parse success rate | ≥ 99% | < 95% |
| Average confidence score | ≥ 0.75 | < 0.60 |
| Average execution time | < 2s | > 5s |
| API error rate | < 1% | > 2% |
| Cost per task | $0.002-0.003 | > $0.005 |

---

## SECTION 5: COMPONENT 3 – VOTING MECHANISM (THE VERIFIER)

### 5.1 Purpose & Role

**Voting Mechanism Mission:**
Aggregate multiple independent agent outputs using First-to-Ahead-by-k voting to achieve statistical guarantee of correctness (mathematical proof of zero-error at target confidence level).

**Key Principle:** "Statistical Reliability Through Redundancy"
- Spawn multiple independent agents for the same task
- Spawn continues until one answer leads by k votes
- Result: Mathematically-proven correctness at target confidence (e.g., 99.9%)

### 5.2 Mathematical Foundation (MAKER Framework)

#### 5.2.1 Core Voting Algorithm: First-to-Ahead-by-k

**Definition:**
Given a task with per-step success rate `p > 0.5`, spawn independent agents until one answer receives `k` more votes than any other answer. This process guarantees the correct answer wins with probability:

$$P(\text{correct}) = \frac{1}{1 + \left(\frac{1-p}{p}\right)^k}$$

**Key Insight:** For `p = 0.95` (95% single-agent accuracy):
- `k=1`: 95.24% correctness (insufficient)
- `k=3`: 99.99% correctness (target for legal docs)
- `k=5`: 99.9999% correctness (overkill for most cases)

#### 5.2.2 Scaling Laws for Veridict

**Theorem (MAKER):** For a contract with `s` clauses and per-clause accuracy `p`, using MAD (Maximal Agentic Decomposition) with voting parameter `k`, the probability of analyzing the entire contract with zero errors is:

$$P(\text{full contract success}) = \left(1 + \left(\frac{1-p}{p}\right)^k\right)^{-s/m}$$

Where `m=1` (one clause per agent).

**For Veridict:**
- `s` = number of clauses (typically 30-200)
- `p` = per-clause single-agent accuracy (typically 0.92-0.98)
- `k` = voting threshold (typically 3-5)

**Example Calculation:**
- Contract with 100 clauses
- Single-agent accuracy per clause: 95% (p=0.95)
- Voting parameter: k=3
- Expected success rate: 99.99%+ (effectively zero-error)

$$P = \left(1 + \left(\frac{0.05}{0.95}\right)^3\right)^{-100/1} ≈ 99.999\%$$

#### 5.2.3 Cost Scaling (Critical for Economics)

**Cost Formula:**
With `k` voting threshold and `p` per-task accuracy, expected cost per task is:

$$E[\text{Cost per Task}] = \frac{c}{p} \cdot 2k_{\min}$$

And for full contract with `s` clauses:

$$E[\text{Total Cost}] = \Theta(s \cdot \ln(s)) = \text{log-linear}$$

**Key Finding:** Cost grows **log-linearly** with contract size, not exponentially. This is what makes MAKER economically feasible.

**Example Cost Scaling:**
- 50-clause contract: ~$2.50 in votes (k=3, p=0.95)
- 100-clause contract: ~$4.20 (not $5.00)
- 500-clause contract: ~$12.80 (not $12.50)

### 5.3 Voting Architecture

#### 5.3.1 Voting Process (Detailed)

**Algorithm 1: Vote Spawning**

```python
async def first_to_ahead_by_k_voting(
    clause,
    task_config,
    k=3,
    max_votes=20,
    timeout=300
):
    """
    Spawn agents until one answer leads by k votes.
    
    Args:
        clause: Clause to analyze
        task_config: Agent configuration
        k: Voting threshold (lead needed to win)
        max_votes: Safety limit on total votes spawned
        timeout: Max time to spend voting (seconds)
    
    Returns:
        winning_answer: The agreed-upon result
        vote_distribution: How votes were distributed
        votes_needed: Number of votes to reach consensus
    """
    
    vote_counts = {}  # answer -> count
    votes_cast = []   # log of all votes
    start_time = time.time()
    round_num = 0
    
    while time.time() - start_time < timeout:
        round_num += 1
        
        # Spawn independent agent
        agent_result = await execute_agent_task(clause, task_config)
        
        # Normalize answer (canonicalize JSON for comparison)
        answer = normalize_answer(agent_result['task_result'])
        answer_key = json.dumps(answer, sort_keys=True)  # Use JSON string as key
        
        if answer_key not in vote_counts:
            vote_counts[answer_key] = 0
        
        vote_counts[answer_key] += 1
        votes_cast.append({
            "round": round_num,
            "answer_key": answer_key,
            "raw_answer": agent_result['task_result'],
            "execution_time": agent_result['metadata']['execution_time'],
            "model": agent_result['metadata']['model']
        })
        
        # Check convergence: does any answer lead by k?
        sorted_votes = sorted(vote_counts.values(), reverse=True)
        if len(sorted_votes) >= 1:
            leader_votes = sorted_votes[0]
            runner_up_votes = sorted_votes[1] if len(sorted_votes) > 1 else 0
            
            if leader_votes - runner_up_votes >= k:
                # Convergence reached!
                winning_key = max(vote_counts, key=vote_counts.get)
                winning_answer = next(
                    v['raw_answer'] for v in votes_cast 
                    if json.dumps(normalize_answer(v['raw_answer']), sort_keys=True) == winning_key
                )
                
                return {
                    "status": "converged",
                    "winning_answer": winning_answer,
                    "vote_counts": vote_counts,
                    "votes_cast": votes_cast,
                    "rounds_needed": round_num,
                    "total_votes": sum(vote_counts.values()),
                    "success_probability": calculate_success_probability(vote_counts, p),
                    "execution_time": time.time() - start_time
                }
        
        # Safety checks
        if round_num >= max_votes:
            # Fallback: return leader even if not converged
            winning_key = max(vote_counts, key=vote_counts.get)
            winning_answer = next(
                v['raw_answer'] for v in votes_cast 
                if json.dumps(normalize_answer(v['raw_answer']), sort_keys=True) == winning_key
            )
            
            return {
                "status": "max_votes_reached",
                "winning_answer": winning_answer,
                "vote_counts": vote_counts,
                "rounds_needed": round_num,
                "total_votes": sum(vote_counts.values()),
                "warning": f"Did not converge after {max_votes} votes"
            }
    
    # Timeout reached
    return {
        "status": "timeout",
        "vote_counts": vote_counts,
        "rounds_needed": round_num,
        "winning_answer": max(vote_counts, key=vote_counts.get),
        "error": f"Voting did not converge within {timeout}s"
    }
```

#### 5.3.2 Answer Normalization (Critical for Voting)

**Challenge:** Two agents may give semantically identical answers in different JSON formats:

```json
// Agent 1 response
{
  "obligations": [
    {"obligor": "Seller", "obligation": "Indemnify"}
  ]
}

// Agent 2 response  
{
  "obligations": [
    {"obligor": "seller", "obligation": "indemnify"}
  ]
}
```

These should be treated as **identical votes**, not different answers.

**Normalization Strategy:**

```python
def normalize_answer(answer_json):
    """
    Canonicalize agent responses for voting comparison.
    
    Handles:
    - Case normalization (lowercase strings)
    - Whitespace normalization (trim, collapse)
    - Number precision (round to 2 decimals)
    - Confidence score binning (0.85-0.95 → 0.9)
    - Boolean standardization
    - Array ordering (sort by key for consistency)
    """
    
    def normalize_value(v):
        if isinstance(v, str):
            return v.lower().strip()
        elif isinstance(v, bool):
            return v
        elif isinstance(v, (int, float)):
            if isinstance(v, float) and 0 <= v <= 1:
                # Confidence score: bin to nearest 0.05
                return round(v * 20) / 20
            else:
                return round(v, 2)
        elif isinstance(v, dict):
            return {k: normalize_value(v[k]) for k in sorted(v.keys())}
        elif isinstance(v, list):
            # Sort list of dicts by their first key for consistency
            if v and isinstance(v[0], dict):
                return sorted([normalize_value(item) for item in v], key=str)
            else:
                return [normalize_value(item) for item in v]
        else:
            return v
    
    return normalize_value(answer_json)
```

#### 5.3.3 Voting at Scale (Parallel Voting)

**Optimization:** Rather than spawning agents sequentially, spawn a batch of `k` agents in parallel:

```python
async def parallel_voting_batch(clause, task_config, k=3):
    """
    Spawn k agents in parallel for faster voting convergence.
    
    For typical contracts:
    - Sequential voting: k rounds × (2-3s per round) = 6-9s per clause
    - Parallel voting: 1 round × (2-3s) + verification = 3-5s per clause
    
    This is a 2-3x speedup with the same statistical guarantees.
    """
    
    # Spawn k agents concurrently
    parallel_tasks = [
        execute_agent_task(clause, task_config)
        for _ in range(k)
    ]
    
    results = await asyncio.gather(*parallel_tasks)
    
    # Count votes
    vote_counts = {}
    for result in results:
        answer_key = json.dumps(normalize_answer(result['task_result']), sort_keys=True)
        vote_counts[answer_key] = vote_counts.get(answer_key, 0) + 1
    
    # Check if any answer has k votes (impossible to lose at this point)
    if max(vote_counts.values()) >= k:
        # Convergence guaranteed
        winning_key = max(vote_counts, key=vote_counts.get)
        return {
            "status": "converged_in_first_batch",
            "winning_answer": next(
                r['task_result'] for r in results
                if json.dumps(normalize_answer(r['task_result']), sort_keys=True) == winning_key
            ),
            "vote_counts": vote_counts,
            "votes_needed": k
        }
    else:
        # Continue sequential voting with these k initial votes
        return await first_to_ahead_by_k_voting(clause, task_config, k=k, initial_votes=vote_counts)
```

### 5.4 Voting System Configuration

**Configuration Parameters:**

```json
{
  "voting": {
    "strategy": "first_to_ahead_by_k",
    "k": 3,
    "max_votes_per_clause": 15,
    "timeout_per_clause_ms": 60000,
    "parallel_batch_size": 3,
    "answer_normalization": {
      "case_sensitive": false,
      "confidence_binning": 0.05,
      "number_precision": 2,
      "array_ordering": "sorted"
    },
    "fallback_strategy": "plurality",
    "enable_red_flagging": true,
    "red_flag_retry_limit": 2
  }
}
```

**Rationale:**
- **k=3:** Achieves 99.99%+ correctness for p≥0.92 (typical for legal clause analysis)
- **max_votes=15:** Safety limit to prevent runaway costs
- **timeout=60s:** Per-clause voting must complete quickly (async parallel execution)
- **parallel_batch=3:** Spawn 3 agents at once; balance parallelism with cost
- **fallback=plurality:** If convergence fails, take most popular answer

### 5.5 Voting Result Output

**Per-Clause Voting Result:**

```json
{
  "clause_id": "SPA_001",
  "clause_type": "Indemnity",
  "voting_result": {
    "status": "converged",
    "winning_answer": {
      "obligations": [
        {
          "obligor": "Seller",
          "obligation_description": "Indemnify Buyer against all claims arising from breach of reps",
          "trigger_condition": "Within 30 days of Closing",
          "deadline": "30 days",
          "consequences_of_breach": "Indemnity obligation survives",
          "confidence": 0.95
        }
      ],
      "summary": "Seller indemnifies Buyer for breaches of representations for 30 days post-closing",
      "risk_level": "medium"
    },
    "vote_distribution": {
      "winning_answer_key": "{...}",
      "winner_votes": 3,
      "total_votes": 3,
      "alternatives": [
        {
          "answer_key": "{...alt...}",
          "votes": 0
        }
      ]
    },
    "voting_metadata": {
      "rounds_needed": 1,
      "parallel_batches": 1,
      "total_agent_calls": 3,
      "convergence_confidence": 0.9999,
      "total_voting_time_ms": 4500,
      "cost_usd": 0.008
    }
  },
  "red_flag_status": "passed",
  "final_verdict": "READY_FOR_REVIEW"
}
```

### 5.6 Confidence Score Calculation

**Post-Voting Confidence:**

```python
def calculate_post_voting_confidence(vote_counts, single_agent_accuracy_p, k):
    """
    Calculate the probability that the winning answer is correct,
    given:
    - vote_counts: Distribution of votes (e.g., {answer1: 4, answer2: 1})
    - p: Single-agent accuracy on this task type
    - k: Voting threshold used
    
    Returns: Confidence as probability (0-1)
    """
    
    leader_votes = max(vote_counts.values())
    runner_up_votes = sorted(vote_counts.values(), reverse=True)[1] if len(vote_counts) > 1 else 0
    
    # If winning answer has clear majority
    margin = leader_votes - runner_up_votes
    
    # Based on MAKER theoretical analysis
    if margin >= k:
        # Use formula from MAKER paper
        # P(correct | votes) ≈ 1 / (1 + ((1-p)/p)^margin)
        ratio = (1 - p) / p if p > 0 else 1.0
        confidence = 1.0 / (1.0 + ratio ** margin)
    else:
        # Fallback: use plurality
        confidence = leader_votes / sum(vote_counts.values())
    
    return min(confidence, 0.9999)  # Cap at 99.99%
```

### 5.7 Voting Quality Assurance

**Monitoring & Alerts:**

| Metric | Target | Alert |
|---|---|---|
| Convergence rate (clauses that converge in k rounds) | ≥ 95% | < 85% |
| Average votes per clause | 3-4 (k+1) | > 8 |
| Post-voting confidence average | ≥ 0.995 | < 0.99 |
| Divergence (answers that disagree) | < 2% of clauses | > 5% |
| Red flag correlation (high-flag clauses diverge more) | ≥ 0.8 | < 0.5 |

---

## SECTION 6: SYSTEM INTEGRATION & DATA FLOW

### 6.1 End-to-End Processing Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: Raw Contract PDF                                        │
└─────────────────────────────┬──────────────────────────────────┘
                              ↓
                    [DECOMPOSITION ENGINE]
                    - PDF parsing
                    - Layout analysis
                    - Clause extraction
                    - Classification
                              ↓
           ┌──────────────────────────────────────────┐
           │ Clause Stream (NDJSON)                   │
           │ [clause_1, clause_2, ..., clause_N]     │
           └──────────────────────┬───────────────────┘
                                  ↓
              [MICRO-AGENT FACTORY] + [VOTING MECHANISM]
              For each clause:
              1. Spawn agents via Micro-Agent Factory
              2. Agents analyze independently
              3. Voting Mechanism aggregates results
              4. Red-flagging filters suspicious outputs
                                  ↓
        ┌──────────────────────────────────────────┐
        │ Analyzed Clauses with Consensus Verdicts │
        │ [verdict_1, verdict_2, ..., verdict_N]   │
        └──────────────────────┬───────────────────┘
                               ↓
                     [OUTPUT AGGREGATION]
                     - Generate summary report
                     - Risk scoring
                     - Executive summary
                               ↓
                OUTPUT: Final Legal Analysis Report
```

### 6.2 Data Structure Mappings

**Stage 1: Decomposition → Clause Stream**

```
Raw PDF
  ↓
{
  "decomposition_batch_id": "batch_abc123",
  "clause_id": "SPA_001_clause_5_1_2",
  "sequence_number": 1,
  "clause_type": "Indemnity",
  "clause_title": "Indemnification by Seller",
  "clause_text": "Seller shall indemnify and hold harmless Buyer...",
  "metadata": {...}
}
```

**Stage 2: Clause → Agent Tasks**

```
{
  "clause_id": "SPA_001_clause_5_1_2",
  "task_type": "obligation_extraction",
  "task_instructions": "Extract all obligations from this clause",
  "agent_config": {
    "model": "gpt-4-mini",
    "k": 3
  }
}
```

**Stage 3: Agent Tasks → Voting**

```
Round 1: [Agent_A result, Agent_B result, Agent_C result]
Round 2: (if needed) [Agent_D result]
...
Until: max(vote_counts.values()) - second_max >= k
```

**Stage 4: Voting → Final Verdict**

```
{
  "clause_id": "SPA_001_clause_5_1_2",
  "final_verdict": {
    "obligations": [
      {
        "obligor": "Seller",
        "obligation_description": "Indemnify Buyer",
        ...
      }
    ],
    "confidence": 0.9999,
    "votes_needed": 3,
    "status": "passed_red_flagging"
  }
}
```

### 6.3 Asynchronous Processing Architecture

**Event Flow:**

```
decomposition_complete 
  → emit 1000 "clause_ready" events
    → consume in parallel (concurrency limit 10)
      → dispatch to voting_engine
        → emit "clause_analyzed" events
      → aggregate results
        → emit "batch_complete" event
          → generate report
            → emit "report_ready" event
```

**Implementation (Pseudo-Code):**

```python
import asyncio
from typing import AsyncGenerator

class VericktPipeline:
    def __init__(self, max_parallel_clauses=10):
        self.max_parallel = max_parallel_clauses
    
    async def process_contract(self, pdf_path: str) -> AsyncGenerator:
        """Main processing pipeline."""
        
        # Stage 1: Decomposition
        clauses_stream = await self.decomposition_engine.process_pdf(pdf_path)
        
        # Stage 2: Voting (parallel with concurrency limit)
        semaphore = asyncio.Semaphore(self.max_parallel)
        
        async def bounded_vote(clause):
            async with semaphore:
                return await self.voting_mechanism.vote_on_clause(clause)
        
        # Stream results as they complete
        pending = set()
        async for clause in clauses_stream:
            task = asyncio.create_task(bounded_vote(clause))
            pending.add(task)
            
            # Yield completed results as they come in
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for completed_task in done:
                result = await completed_task
                yield result
        
        # Wait for remaining tasks
        if pending:
            done, _ = await asyncio.wait(pending)
            for completed_task in done:
                result = await completed_task
                yield result
        
        # Stage 3: Aggregation & Report Generation
        yield await self.generate_final_report(clauses_analyzed)
```

### 6.4 Error Recovery & Fault Tolerance

**Retry Policies:**

| Failure Type | Detection | Retry Strategy |
|---|---|---|
| **Agent timeout** | No response in 10s | Exponential backoff (2s, 4s, 8s); max 3 retries |
| **JSON parse error** | Response not valid JSON | Apply repair heuristics; if fail, re-vote |
| **Red-flag triggered** | Output flagged for issues | Retry agent (up to 2 times) with different temp |
| **API rate limit** | 429 response | Backoff 60s; resume |
| **Decomposition failure** | PDF unreadable | Reject document; notify user |

**Degradation Paths:**

```python
async def vote_with_fallback(clause, k=3):
    try:
        # Primary: Full voting with k threshold
        return await first_to_ahead_by_k_voting(clause, k=k)
    except VotingTimeoutError:
        # Fallback 1: Reduce k
        logging.warning(f"Voting timeout; reducing k from {k} to {k-1}")
        return await first_to_ahead_by_k_voting(clause, k=k-1)
    except MaxVotesExceededError:
        # Fallback 2: Return plurality winner
        logging.warning(f"Max votes exceeded; returning plurality winner")
        return await plurality_vote(clause)
    except AllAgentsFailedError:
        # Fallback 3: Manual review flag
        logging.error(f"All agents failed on clause {clause['clause_id']}")
        return {
            "status": "agent_failure",
            "clause_id": clause['clause_id'],
            "manual_review_required": True
        }
```

---

## SECTION 7: REPORTING & OUTPUT FORMATS

### 7.1 Executive Summary Report

**Generated for each contract analyzed:**

```json
{
  "report_id": "report_abc123",
  "contract_id": "SPA_001",
  "contract_metadata": {
    "file_name": "SPA_ABC_Corp_XYZ_Corp.pdf",
    "contract_type": "SPA",
    "parties": ["ABC Corp", "XYZ Corp"],
    "execution_date": "2024-12-15",
    "jurisdiction": "Delaware"
  },
  "analysis_results": {
    "total_clauses_analyzed": 127,
    "clauses_flagged_critical": 3,
    "clauses_flagged_high": 12,
    "overall_risk_rating": "HIGH",
    "key_findings": [
      {
        "rank": 1,
        "clause_id": "SPA_001_clause_8_2",
        "clause_type": "Liability",
        "finding": "Liability cap of $5M applies to all claims except IP; lower than typical 1% of purchase price",
        "risk_level": "CRITICAL",
        "recommendation": "Negotiate cap to 1% of $500M purchase price ($5M insufficient)",
        "affected_party": "Buyer",
        "financial_impact_usd": 500000000
      }
    ],
    "voting_statistics": {
      "avg_votes_per_clause": 3.2,
      "avg_convergence_rounds": 1.1,
      "overall_confidence": 0.9998,
      "clauses_requiring_red_flag_retry": 8,
      "clauses_flagged_suspicious": 2
    }
  },
  "cost_summary": {
    "decomposition_cost_usd": 0.15,
    "voting_cost_usd": 3.24,
    "total_cost_usd": 3.39,
    "cost_per_clause": 0.027
  },
  "processing_time": {
    "decomposition_time_seconds": 18,
    "voting_time_seconds": 245,
    "total_time_seconds": 263,
    "human_equivalent_hours": 8
  },
  "generated_timestamp": "2024-12-20T14:32:00Z"
}
```

### 7.2 Detailed Clause Analysis Report

**For each analyzed clause:**

```json
{
  "clause_analysis": {
    "clause_id": "SPA_001_clause_8_2",
    "sequence_number": 89,
    "clause_type": "Liability",
    "clause_title": "Limitation of Liability",
    "clause_text": "... [full clause text] ...",
    "source_location": {
      "page_start": 12,
      "page_end": 12,
      "char_offset_start": 3421,
      "char_offset_end": 4123
    },
    "decomposition_metadata": {
      "extraction_confidence": 0.98,
      "classification_confidence": 0.96,
      "alternative_classifications": []
    },
    "voting_analysis": {
      "winning_answer": {
        "caps": [
          {
            "cap_type": "liability_cap",
            "cap_amount": "$5,000,000",
            "cap_amount_numeric": 5000000,
            "applies_to": "All claims except IP and confidentiality breaches",
            "exceptions": [
              "IP infringement claims",
              "Confidentiality breaches",
              "Willful misconduct"
            ],
            "confidence": 0.98
          }
        ],
        "no_cap_found": false
      },
      "vote_distribution": {
        "winner_votes": 3,
        "total_votes": 3,
        "convergence_achieved": true,
        "rounds_needed": 1,
        "voting_confidence": 0.9999
      }
    },
    "risk_assessment": {
      "risk_level": "CRITICAL",
      "risk_factors": [
        {
          "factor": "Below-market cap",
          "description": "$5M cap for $500M transaction = 1.0% (market standard 0.75-1.5% but typically 1%)",
          "severity": "CRITICAL",
          "recommendation": "Negotiate to at least $7.5M (1.5%) or consider representation & warranty insurance"
        },
        {
          "factor": "Broad exceptions to cap",
          "description": "IP infringement, confidentiality, and willful misconduct carved out; these can be substantial",
          "severity": "HIGH",
          "recommendation": "Define 'willful misconduct' narrowly; consider aggregate cap on carve-outs"
        }
      ]
    },
    "deviations_from_standard": [
      {
        "deviation": "Liability cap applies to consequential damages but not to direct damages",
        "prevalence_in_spas": "30% of deals",
        "market_norm": "Liability cap applies to both direct and consequential",
        "recommendation": "Negotiate to carve-out direct damages for IP and confidentiality"
      }
    ],
    "definitions_check": {
      "all_terms_defined": true,
      "undefined_terms": [],
      "critical_definitions_present": [
        "Losses",
        "Claims",
        "Consequential Damages"
      ]
    },
    "red_flag_status": "PASSED",
    "compliance_check": {
      "jurisdiction_specific_issues": [
        "Delaware: Liability caps must be materially breach-related to be enforceable; ensure 'Losses' defined clearly"
      ]
    },
    "analyst_confidence": 0.9998,
    "generated_timestamp": "2024-12-20T14:32:15Z"
  }
}
```

### 7.3 Risk Dashboard JSON

**For quick executive review:**

```json
{
  "risk_dashboard": {
    "contract_id": "SPA_001",
    "overall_risk_score": 7.2,
    "risk_categories": {
      "financial_risk": {
        "score": 8.5,
        "issues": [
          {
            "clause": "Liability Cap",
            "issue": "Below-market limitation",
            "impact_usd": 500000000,
            "priority": 1
          }
        ]
      },
      "legal_risk": {
        "score": 6.8,
        "issues": [
          {
            "clause": "IP Indemnity",
            "issue": "Seller indemnity survival period longer than warranty",
            "impact": "Extended exposure",
            "priority": 2
          }
        ]
      },
      "operational_risk": {
        "score": 4.2,
        "issues": []
      }
    },
    "required_actions": [
      {
        "priority": "CRITICAL",
        "action": "Renegotiate liability cap from $5M to $7.5M+",
        "estimated_hours": 2,
        "involves_parties": ["Buyer", "Seller"]
      },
      {
        "priority": "HIGH",
        "action": "Clarify definition of 'Willful Misconduct' in carve-out",
        "estimated_hours": 1,
        "involves_parties": ["Legal Team"]
      }
    ],
    "risk_trend": "STABLE",
    "compared_to_baseline": "Above market average for liability caps"
  }
}
```

---

## SECTION 8: TECHNICAL SPECIFICATIONS

### 8.1 System Requirements

**Infrastructure:**

| Component | Requirement |
|---|---|
| **Cloud Platform** | AWS (primary) or GCP (backup) |
| **Compute** | ECS/Fargate for serverless; scale up to 50 concurrent processes |
| **Storage** | S3 (PDFs), RDS PostgreSQL (metadata), DynamoDB (results cache) |
| **API** | REST API (FastAPI) + WebSocket for streaming results |
| **Caching** | Redis (conversation cache, voting state) |
| **Monitoring** | CloudWatch + DataDog for APM |
| **Logging** | CloudWatch Logs + Datadog; structured JSON logging |

**Software Stack:**

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI (Python 3.10+) |
| **PDF Processing** | PDFPlumber + PyMuPDF |
| **NLP/Classification** | spaCy 3.x |
| **LLM Integration** | OpenAI SDK + Anthropic SDK |
| **Async Runtime** | asyncio + aiohttp |
| **Database** | SQLAlchemy ORM + Alembic migrations |
| **Caching** | Redis client library |
| **Testing** | pytest + pytest-asyncio |
| **Deployment** | Docker + Kubernetes (EKS) |

### 8.2 API Specifications

#### 8.2.1 Contract Upload & Analysis Endpoint

**POST /api/v1/contracts/analyze**

```
Request:
{
  "file": <binary PDF>,
  "contract_type": "SPA" | "APA" | "NDA" | etc.,
  "parties": ["Company A", "Company B"],
  "jurisdiction": "Delaware",
  "execution_date": "2024-12-15",
  "priority": "normal" | "high",
  "voting_config": {
    "k": 3,
    "max_votes": 15
  },
  "webhook_url": "https://client.example.com/webhook" (optional)
}

Response (200):
{
  "contract_id": "contract_abc123",
  "analysis_id": "analysis_xyz789",
  "status": "queued",
  "estimated_completion": "2024-12-20T15:30:00Z",
  "polling_url": "/api/v1/analysis/xyz789/status"
}
```

#### 8.2.2 Analysis Status Endpoint

**GET /api/v1/analysis/{analysis_id}/status**

```
Response (200):
{
  "analysis_id": "analysis_xyz789",
  "status": "in_progress",
  "progress": {
    "clauses_decomposed": 127,
    "clauses_analyzed": 45,
    "overall_progress_percent": 35
  },
  "current_phase": "voting",
  "estimated_remaining_seconds": 240
}

Response (200) - Complete:
{
  "analysis_id": "analysis_xyz789",
  "status": "complete",
  "report_url": "/api/v1/analysis/xyz789/report",
  "summary": { ... executive summary ... },
  "timestamp_completed": "2024-12-20T15:28:00Z"
}
```

#### 8.2.3 Report Retrieval Endpoint

**GET /api/v1/analysis/{analysis_id}/report**

```
Query Parameters:
- format: "json" | "html" | "pdf" (default: json)
- detail_level: "executive" | "detailed" (default: executive)

Response (200):
{
  "report": { ... full analysis report JSON ... }
}

Response (200) - HTML:
<html>... rendered report HTML ... </html>

Response (200) - PDF:
<binary PDF file>
```

#### 8.2.4 Clause Analysis Details Endpoint

**GET /api/v1/analysis/{analysis_id}/clauses/{clause_id}**

```
Response (200):
{
  "clause_analysis": { ... full clause analysis ... }
}
```

### 8.3 Database Schema (Simplified)

```sql
-- Contracts table
CREATE TABLE contracts (
  id UUID PRIMARY KEY,
  file_name VARCHAR(255),
  contract_type VARCHAR(50),
  parties TEXT[], -- JSON array
  jurisdiction VARCHAR(100),
  execution_date DATE,
  created_at TIMESTAMP,
  file_s3_path VARCHAR(500),
  file_size_bytes INT,
  num_pages INT
);

-- Decompositions table (clause extractions)
CREATE TABLE clause_decompositions (
  id UUID PRIMARY KEY,
  contract_id UUID REFERENCES contracts(id),
  clause_id VARCHAR(255),
  sequence_number INT,
  clause_type VARCHAR(50),
  clause_title VARCHAR(255),
  clause_text TEXT,
  start_page INT,
  end_page INT,
  extraction_confidence FLOAT,
  created_at TIMESTAMP
);

-- Voting results table
CREATE TABLE voting_results (
  id UUID PRIMARY KEY,
  clause_id VARCHAR(255),
  analysis_id UUID,
  winning_answer JSONB, -- Full consensus answer
  vote_distribution JSONB,
  votes_needed INT,
  convergence_confidence FLOAT,
  red_flag_status VARCHAR(50), -- 'passed', 'failed', 'retry_passed'
  created_at TIMESTAMP
);

-- Analysis runs table
CREATE TABLE analysis_runs (
  id UUID PRIMARY KEY,
  contract_id UUID REFERENCES contracts(id),
  status VARCHAR(50), -- 'queued', 'decomposing', 'voting', 'complete', 'failed'
  total_clauses INT,
  clauses_analyzed INT,
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  total_cost_usd FLOAT,
  error_message TEXT,
  created_at TIMESTAMP
);

-- Risk assessments table
CREATE TABLE risk_assessments (
  id UUID PRIMARY KEY,
  analysis_id UUID REFERENCES analysis_runs(id),
  overall_risk_score FLOAT,
  critical_issues_count INT,
  high_issues_count INT,
  summary_json JSONB,
  created_at TIMESTAMP
);
```

### 8.4 Performance SLAs

| Metric | Target | Notes |
|---|---|---|
| **Contract processing latency** | < 10 minutes (50-page contract) | Depends on clause count & complexity |
| **Clause analysis time** | < 2 minutes per clause average | Including voting |
| **API response time (POST upload)** | < 1 second | Returns job ID; async processing |
| **Report generation** | < 30 seconds | After all voting complete |
| **System availability** | 99.5% | Excludes planned maintenance |
| **P99 latency (clause voting)** | < 5 minutes | Including red-flagging retries |

### 8.5 Scaling & Load Testing

**Projected Capacity:**

| Metric | Capacity |
|---|---|
| **Concurrent contract uploads** | 100 simultaneous users |
| **Clauses analyzed per day** | 500,000 clauses |
| **Cost per contract** | $3-8 depending on length |
| **Monthly throughput** | 50,000+ contracts at scale |

**Load Testing Plan:**
- Simulate 50 concurrent contract uploads
- Measure latency at p50, p95, p99
- Verify voting concurrency limits
- Test database connection pooling

---

## SECTION 9: SECURITY & COMPLIANCE

### 9.1 Data Security

**PDF Handling:**
- PDFs encrypted in transit (TLS 1.3)
- At-rest encryption (AES-256) in S3
- PII detection & masking optional
- Audit logs for all file access

**API Authentication:**
- OAuth 2.0 with JWT tokens
- API key authentication for service-to-service
- Rate limiting per API key

**Data Retention:**
- User can specify retention period (default 90 days)
- Automatic deletion after retention window
- Option for secure deletion (NIST guidelines)

### 9.2 Compliance

**Applicable Regulations:**
- **GDPR:** If processing EU customer PDFs; data processing agreement required
- **CCPA:** California privacy rights; cookie consent
- **SOC 2 Type II:** Target compliance for enterprise customers

**Attorney-Client Privilege:**
- Veridict does not claim privilege protection
- Customers responsible for maintaining privilege through proper controls
- Recommend: use only on non-privileged documents or with legal review

### 9.3 Audit & Logging

```python
# Logging template for compliance
{
  "timestamp": "2024-12-20T14:32:15Z",
  "event_type": "contract_analyzed",
  "contract_id": "SPA_001",
  "user_id": "user_abc123",
  "user_email": "analyst@lawfirm.com",
  "action": "analyzed_contract",
  "result": "success",
  "clauses_analyzed": 127,
  "ip_address": "203.0.113.42",
  "user_agent": "Mozilla/5.0..."
}
```

---

## SECTION 10: ROADMAP & FUTURE ENHANCEMENTS

### Phase 1 (MVP - Q1 2025)
- ✅ Decomposition Engine: PDF → Clauses
- ✅ Micro-Agent Factory: Basic clause analysis agents
- ✅ Voting Mechanism: First-to-ahead-by-k voting
- ✅ API: Contract upload + report retrieval
- Risk scoring: Executive summary only

### Phase 2 (Q2 2025)
- Expanded clause types (50+ specialized agent types)
- Comparative analysis: Market-standard language matching
- Integration with deal management platforms (Intralinks, Firmex)
- Legal team collaboration features

### Phase 3 (Q3 2025)
- Custom redline generation (automatic clauses to push back)
- Trademark/patent search integration
- Multi-language support
- Client feedback loop → model fine-tuning

### Phase 4 (Q4 2025)
- Real-time collaborative analysis UI
- Veridict Agent: Autonomous deal assistant
- Advanced risk correlation analysis
- Custom clause template library

---

## SECTION 11: GLOSSARY & DEFINITIONS

| Term | Definition |
|---|---|
| **Atomic Unit** | Smallest meaningful decomposition; for legal contracts = single clause |
| **Clause** | Discrete contractual provision with defined scope (e.g., "Indemnification") |
| **Consensus** | Agreement among k agents on a single answer (winning answer) |
| **Convergence** | Voting process reaches consensus (one answer leads by k votes) |
| **First-to-Ahead-by-k** | Voting rule: continue sampling until one answer has k more votes than any other |
| **MAD** | Maximal Agentic Decomposition: decompose task into smallest possible subtasks |
| **MAKER Framework** | Massively Decomposed Agentic Processes; framework combining MAD + voting + red-flagging |
| **Micro-Agent** | Lightweight LLM agent focused on single atomic task |
| **NDJSON** | Newline-delimited JSON; one JSON object per line |
| **Red-Flagging** | Quality filter that discards suspicious outputs (malformed, overly long, etc.) |
| **Stateless** | Agent execution has no memory of previous tasks; each task is independent |
| **Zero-Error** | Mathematical guarantee that final output is correct with ≥99.99% confidence |

---

## SECTION 12: SUCCESS METRICS & KPIs

### For Veridict (Product)

| KPI | Target (Year 1) | Rationale |
|---|---|---|
| **Contracts analyzed** | 2,000+ | Market penetration |
| **Customer satisfaction (NPS)** | ≥ 70 | Adoption signal |
| **Cost per contract** | $5-10 | Competitive vs. junior associates ($200-300/hour × 8-10 hours) |
| **Accuracy (audited)** | 99.9%+ | Core value prop |
| **Time to review (avg)** | 30 minutes | vs. 8-10 hours for manual |
| **Adoption by top 50 law firms** | 5+ firms | Enterprise traction |

### For This PRD (Development)

| Metric | Success Criteria |
|---|---|
| **Component completion** | All 3 components deployed and tested before Q2 2025 |
| **Integration testing** | E2E pipeline works on 20+ representative contracts |
| **Load testing** | System handles 50 concurrent uploads without degradation |
| **Accuracy validation** | Voting confidence ≥ 99.95% on 100-contract test set |
| **Cost per contract** | < $8 average (including voting overhead) |

---

## APPENDIX A: EXAMPLE WORKFLOW

### Scenario: M&A SPA Review

**Input:** SPA_ABC_Corp_XYZ_Corp.pdf (68 pages, 150 clauses)

**Timeline:**

```
T=0:00  - User uploads PDF via API
         - System returns job ID: "analysis_xyz789"

T=0:05  - Decomposition Engine processes PDF
         - Extracts 150 clauses via layout analysis + numbering detection
         - Classifies each: Indemnity (12), Termination (8), Warranty (15), etc.
         - Outputs clause stream to queue

T=0:10  - Micro-Agent Factory spawns agents
         - For each clause: spawn 3 independent agents
         - Agents analyze in parallel (concurrency limit: 10 clauses at a time)
         - Each agent takes ~3 seconds

T=0:15  - Voting Mechanism begins aggregating results
         - As agents complete, their answers enter voting
         - First-to-ahead-by-k voting (k=3)
         - Most clauses converge in 1 round (3 agents = 3 votes, convergence)
         - Some edge cases need additional voting (up to 5-6 votes)

T=8:00  - All 150 clauses analyzed with zero errors (99.99% confidence)
         - Red-flagging filtered 12 suspicious outputs; re-analyzed successfully
         - Risk scoring complete
         - Cost: $6.20 (vs. 10 hours × $250/hour = $2,500 for junior associate)

T=8:05  - Report generated
         - Executive summary: 3 CRITICAL issues flagged
         - Detailed analysis: 150 clauses with voting details
         - Risk dashboard: prioritized actions
         - User downloads report (PDF, HTML, or JSON)

T=8:10  - User reviews critical issues
         - Issue 1: Liability cap $5M is 1.0% of purchase price (below market 0.75-1.5%)
         - Issue 2: IP indemnity 5-year tail (vs. standard 2-3 years)
         - Issue 3: Termination fee 3% (above market 1-2%)
         - User marks issues for negotiation
         - System suggests market-standard language alternatives

T=8:15  - Negotiation begins with counterparty
         - Armed with data-driven recommendations
         - Veridict provides supporting evidence (voting confidence, clause quotes)
         - Accelerated deal close = millions in value realized faster
```

---

## APPENDIX B: TECHNICAL DEBT & LIMITATIONS

### Known Limitations (MVP)

1. **Single Language:** English contracts only (Phase 3: multilingual)
2. **Clause Extraction:** Heuristic-based (Phase 2: ML-based layout analysis)
3. **Agent Task Types:** Limited to 6 core tasks (Phase 2: 50+ specialized agents)
4. **Voting:** First-to-ahead-by-k only (Phase 3: confidence-based stopping)
5. **Red-Flagging:** Length + format only (Phase 2: semantic red-flagging)
6. **No Reasoning:** Agent tasks don't require multi-step reasoning (by design)

### Technical Debt Backlog

- [ ] Optimize voting algorithm for outlier detection
- [ ] Implement semantic similarity for answer matching (instead of exact JSON)
- [ ] Add caching layer for identical clauses across deals
- [ ] Parallel PDF processing for multi-file batches
- [ ] Custom LLM fine-tuning on legal clause corpus

---

## APPENDIX C: SAMPLE PROMPTS FOR AGENTS

### Obligation Extraction Agent Prompt

```
SYSTEM PROMPT:
You are a legal obligation extractor analyzing a single contract clause.

USER PROMPT:
Analyze this clause and extract all obligations (who must do what, when, consequences):

CLAUSE TEXT:
"The Seller shall indemnify, defend and hold harmless the Buyer, its Affiliates, and their respective officers, directors, employees and agents from and against any and all claims, damages, losses and expenses (including reasonable attorneys' fees) arising out of or resulting from (a) any breach of any representation, warranty or covenant made by Seller in this Agreement, or (b) any Seller Pre-Closing Liabilities, provided that such indemnification obligations shall not apply to any claims arising after the 18-month anniversary of the Closing Date, except for claims arising from the Fundamental Representations which shall survive until the 5-year anniversary."

CONTRACT METADATA:
- Type: Stock Purchase Agreement
- Parties: Acme Corp (Seller), Alpha Inc (Buyer)
- Jurisdiction: Delaware
- Execution Date: 2024-12-15

INSTRUCTIONS:
1. Extract ONLY obligations explicitly stated in THIS clause.
2. For each obligation identify: WHO, WHAT, WHEN, CONSEQUENCES
3. If text is ambiguous, mark confidence < 0.8
4. Return JSON only, no explanations.

OUTPUT (JSON):
```

### Risk Identifier Agent Prompt

```
SYSTEM PROMPT:
You identify high-risk language in contract clauses that deviates from market standards.

USER PROMPT:
Identify any unusual, risky, or market-divergent language in this clause:

CLAUSE TEXT:
"In no event shall the total aggregate liability of Seller for all claims arising out of or related to this Agreement, whether in contract, tort, or any other form, exceed $5,000,000, except that this limitation shall not apply to: (i) claims arising from Seller's breach of Section 2 (Representations and Warranties), (ii) claims arising from Seller's infringement of intellectual property rights, (iii) claims arising from Seller's violation of applicable law, or (iv) claims arising from Seller's willful misconduct or gross negligence."

MARKET CONTEXT:
- For a $500M stock purchase agreement
- Typical liability caps: 0.75% - 1.5% of purchase price
- This cap: $5M = 1.0% (technically within range, but see carve-outs)

RISK ASSESSMENT TASK:
1. Identify deviations from market standard
2. Assess if language favors Buyer or Seller
3. Flag ambiguities (e.g., "willful misconduct" - not defined)
4. Suggest remediation

OUTPUT (JSON):
```


