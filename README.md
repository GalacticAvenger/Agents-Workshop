# Writeup — Agentic AI Assignment

**Name:** Samuel Meddin
**Date:** 2026-03-30
**Agent built:** Literature review agent (Option A — Semantic Scholar + local PDF RAG)

---

## Part 2: Task Analysis

**Research question:** How do LLM agents handle tool failures and error recovery?

### Depth beyond surface-level search

The local RAG pipeline surfaced passage-level details that metadata-only search misses. Querying about tool failure recovery returned a passage from Wang et al. (2023) describing how agents "learn to perform self-debugging" — a claim in the full text but not the abstract. The ReAct paper (Yao et al., 2022) returned specific failure statistics: hallucination accounts for 14% of false positives in CoT vs. 6% in ReAct, and "non-informative search counts for 23% of error cases." These numbers wouldn't surface from abstract skimming alone.

The agent missed papers on engineering-oriented error handling (retry logic, fallback tools). The corpus and keyword framing both favor reasoning-based approaches over practical ones.

### At least one failure

**API rate limiting:** Semantic Scholar returned 429 errors on every request during testing, disabling half the agent's tools (search_papers, get_paper_details, get_citations). The server handled this gracefully with error messages, but the agent was reduced to local-only retrieval with no fallback or cache.

**Query sensitivity:** Querying "reflexion error correction self-improvement" returned Self-Refine (Madaan et al., 2023) as the top result instead of Reflexion (Shinn et al., 2023), which scored barely above threshold at 0.4975. Slight rephrasing changed which papers appeared — users might not expect this.

**Chunk boundary artifacts:** With chunk_size=512, some retrieved passages cut mid-sentence (e.g., ending with "required parameters, optional"). The agent presents incomplete text that looks complete.

For real work: usable as a discovery tool with guardrails, not as a source of truth. Local citations can be verified; external results are leads to check.

---

## Part 3: Reflection

### 3.1 Build process

The main design decision was RAG parameters. chunk_size=512 (~2-3 sentences) balances context vs. precision. I considered 1024 for more context but it would return more irrelevant text alongside relevant portions. The similarity_threshold of 0.3 was deliberately permissive — relevant content scores 0.45-0.65, so 0.3 errs on recall over precision.

Claude implemented chunk_text() and retrieve() correctly on the first try — the sliding window and ChromaDB query pattern are straightforward. Setup required iteration: the embedding model download timed out once, and .mcp.json needed the Python path updated to use the venv.

### 3.2 System prompt engineering

I compared "default" and "concise" prompts. Default starts with Semantic Scholar, uses multiple tool calls, and produces prose reviews with thematic grouping. Concise flips the order (local library first), limits to two tool calls total, and requires bullet-point output.

The concise prompt produced higher information density per token. It also had fewer hallucination opportunities since it made fewer claims. The default prompt was better for comprehensive literature review writing. I'd define "better" as information density, so concise wins — but for a grad student writing a review section, default is more useful.

### 3.4 Architecture limitations

To make this viable for real work: (1) add a caching layer for Semantic Scholar results so the system works during rate limiting, (2) replace character-level chunking with sentence-boundary-aware splitting, (3) add the ability to incrementally add papers without full re-ingestion.

The biggest surprise was how sensitive retrieval is to query phrasing. Semantic embeddings don't handle paraphrasing as well as expected — querying "reflexion error correction" vs. "agent self-improvement from failure" returned different top results for the same concept. The similarity score range is also very compressed (useful range ~0.4-0.65), making the threshold parameter surprisingly sensitive.