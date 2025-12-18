Veridict: Zero-Error Legal Due Diligence Engine

Veridict is an autonomous, "Zero-Error" legal due diligence engine designed to automate the review of high-volume legal contracts (e.g., M&A Data Rooms). Veridict utilizes the MAKER Framework (Massively Decomposed Agentic Processes) to perform execution tasks independently with mathematical reliability.

Project Overview

LLMs have a persistent error rate that prevents scaling on long-horizon tasks.

Veridict overcomes this "execution cliff" by utilizing Massively Decomposed Agentic Processes (MDAPs).

The system decomposes complex legal documents into thousands of minimal, independent micro-tasks.

By applying a specialized voting mechanism, we drive the probability of error to near zero.

The process provides a verifiable and defensible audit trail for every legal finding.

Core Architecture: The MAKER Framework

Maximal Agentic Decomposition (MAD): Tasks are broken down into the smallest possible micro-roles to minimize the context window and reduce model confusion.

First-to-Ahead-by-k Voting: Sampling continues until one answer leads the runner-up by a margin of k votes, ensuring high statistical confidence.

Red-Flagging: A safety layer that detects and discards unreliable outputs like malformed data or excessive verbosity before they reach the voting stage.

System Components

Decomposition Engine (The Manager): Performs structural separation of raw documents by identifying boundaries of clauses.

Micro-Agent Factory (The Workers): Executes single, atomic instructions such as clause classification or risk analysis.

Voting Mechanism (The Verifier): Orchestrates parallel workers and implements the k-margin voting logic to verify findings.

Red-Flagging System (The Filter): Performs quality control by filtering outputs based on syntax and logical consistency.

Tech Stack

Language: Python 3.11+

AI Models: Claude 4.5 Opus (Manager), GPT-5.2 / GPT-4o-mini (Workers).

Orchestration: Google Antigravity / Custom Async Loops.

Data Processing: Unstructured.io / PyMuPDF.

Database: PostgreSQL with pgvector for clause retrieval.

Roadmap

Phase 1: Ingestion and OCR pipeline development.

Phase 2: Implementation of the First-to-Ahead-by-k voting engine and Red-Flagging filter suite.

Phase 3: Integration of Batch APIs for parallelization and launch of the Risk Heatmap dashboard.

Contribution Guidelines

Focus on Atomic Units: New agent roles must be maximally decomposed to ensure reliability.

Stateless Design: Workers must remain stateless to ensure independent voting results.

Benchmark-Driven: All changes must be tested against a verified set of contracts to ensure no regression in recall.

References

Solving a Million-Step LLM Task with Zero Errors (MAKER Framework).

SWE-bench: Can Language Models Resolve Real-World GitHub Issues?.

License

MIT License
