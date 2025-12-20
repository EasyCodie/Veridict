"""Test the stateless worker framework."""

import sys
import os
sys.path.append(os.getcwd())

from core.workers import worker_registry, WorkerContext, context_builder
from core.workers.classifier import ClauseClassifierWorker

print("📋 Registered Workers:")
for w in worker_registry.list_workers():
    print(f"  - {w['task_type']}: {w['description']}")

# Test raw clause with noise
raw_text = """
Page 12 of 45

SECTION 7. INDEMNIFICATION

The Consultant shall indemnify and hold harmless the Company from any claims 
arising from negligence. See Section 4.2 for additional terms. Pursuant to 
Section 3.1, Consultant agrees to defend all claims.

--- CONFIDENTIAL ---

Page 13
"""

print("\n🧹 Testing Context Builder (minimal injection)...")
print(f"   Raw text length: {len(raw_text)} chars")

# Build minimal context
ctx = context_builder.build_context(raw_text, title="INDEMNIFICATION")
print(f"   Sanitized length: {ctx['sanitized_length']} chars")
print(f"   Cross-refs replaced: {ctx['text'].count('[REF]')}")
print(f"   Page numbers stripped: {'Page' not in ctx['text']}")

print("\n🔧 Executing worker with minimal context...")
context = WorkerContext(
    text=ctx['text'],
    title=ctx['title'],
    metadata=ctx['metadata']
)

result = worker_registry.execute("classify_clause", context)

print(f"\n✅ Success: {result.success}")
print(f"📊 Clause Type: {result.data['clause_type']}")
print(f"⚠️  Risk Level: {result.data['risk_level']}")
print(f"🎯 Confidence: {result.confidence_label}")
print(f"💰 Cost: ${result.cost_usd:.6f}")
print(f"🔢 Tokens: {result.input_tokens} in / {result.output_tokens} out")

if result.data.get("red_flags"):
    print(f"\n🚩 Red Flags:")
    for flag in result.data["red_flags"]:
        print(f"  - {flag}")

print("\n✨ Worker framework test complete!")
