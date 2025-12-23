# Product Requirements Document (PRD): Veridict

## Agent Instructions
Use this PRD as an overview of the Veridict system. Not as a reference for development. For development, see the Veridict_PRD.md file which contains the full technical details of the system.

## 1. Executive Summary

Veridict is an autonomous, "Zero-Error" legal due diligence engine designed to automate the review of high-volume legal contracts (e.g., M&A Data Rooms). Veridict utilizes the MAKER Framework (Massively Decomposed Agentic Processes) to perform execution tasks independently with mathematical reliability.

By decomposing complex contracts into atomic micro-tasks and utilizing a First-to-Ahead-by-k Voting mechanism, Veridict replaces the need for manual junior associate review.

## 2. Problem Statement

Volume & Fatigue: In M&A, legal teams must review thousands of documents in days. Fatigue leads to missed risks ("human error").

Cost: Manual review by junior associates is prohibitively expensive.

Inconsistency: Different lawyers interpret "standard" clauses differently, leading to inconsistent risk reports.

## 3. Solution Architecture (The MAKER Framework)

### 3.1 Component 1: The Decomposition Engine (The Manager)

Role: Ingests raw PDFs and slices them into atomic units.

Function: Uses OCR and layout analysis to identify boundaries of "Clauses" (e.g., Indemnity, Termination).

Output: A JSON stream of independent text chunks.

### 3.2 Component 2: The Micro-Agent Factory (The Workers)

Role: Stateless execution of single instructions.

Constraint: Agents receive minimal context to prevent hallucination.

### 3.3 Component 3: The Voting Mechanism (The Verifier)

Logic: First-to-Ahead-by-k.

Process: Spawns multiple independent agents for the same task. Sampling continues until one answer leads the runner-up by a margin of k votes.

Result: Statistical guarantee of correctness.

### 3.4 Component 4: Red-Flagging (The Filter)

Heuristics: Discards outputs that are incorrectly formatted, excessively long, or logically inconsistent.

## 4. Technical Specifications

### 4.1 AI Models

Manager / Architect: Claude 4.5 Opus (Anthropic)

Worker / Voter: GPT-5-nano (OpenAI)

Embeddings: OpenAI text-embedding-3-small

### 4.2 Infrastructure & Stack

Language: Python 3.11+

Backend: FastAPI (Async architecture)

Database: PostgreSQL with pgvector

Document Processing: Unstructured.io / PyMuPDF

## 5. Core Features

Automated Clause Extraction: Instantly identifies 50+ standard clause types.

Consensus Meter: UI feature showing the confidence score of every finding.

Risk Heatmap: Visual dashboard highlighting high-risk documents.

Audit Trail: Logs of every decision with raw outputs of all voting agents.

Batch Ingestion: Ability to process 5,000+ PDFs asynchronously.

## 6. Development Roadmap

### Phase 1: The Foundation (Weeks 1-2)

Build "Ingestion Service" (PDF to Clean Text).

Successfully parse contracts into JSON structure.

### Phase 2: The Micro-Agent Core (Weeks 3-4)

Develop "Worker" prompt templates.

Implement "Red-Flagging" regex filters.

### Phase 3: The Voting Engine (Weeks 5-6)

Implement vote_loop logic.

Connect to OpenAI Batch API for cost-effective scaling.

### Phase 4: Interface & Scale (Weeks 7-8)

Build React/Streamlit Dashboard.

Stress test with 1,000+ documents.

### 7. Economics & Performance Estimates (Per 5,000 Contracts)

Manual Cost: $100,000+

Veridict Cost: ~$30 - $60 (via Batch API)

Manual Time: ~2 Months

Veridict Time: ~24 Hours

### 8. KPIs (Key Performance Indicators)

Zero Errors: 100% recall on "High Risk" clauses in the Golden Set.

Throughput: Ability to process 30,000 tokens/minute without API blocks.

Cost Efficiency: Average cost per document < $0.05.