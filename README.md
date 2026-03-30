# Writeup — Agentic AI Assignment

**Name:** Samuel Meddin
**Date:** 2026-03-30
**Agent built:** Literature review agent (Option A — Semantic Scholar + local PDF RAG)

---

## Part 2: Task Analysis

**Research question:** How do LLM agents handle tool failures and error recovery?

### Depth beyond surface-level search

The local RAG pipeline surfaced passage-level details that metadata-only search
misses. Querying about tool failure recovery returned a passage from Wang et al.
(2023) describing how agents "learn to perform self-debugging" — a claim present
in the full text but not in the abstract. The ReAct paper (Yao et al., 2022)
returned specific failure statistics: hallucination accounts for 14% of false
positives in CoT vs. 6% in ReAct, and "non-informative search counts for 23% of
error cases." Neither number appears in the abstract, so abstract-only search
would have missed both.

The initial corpus was weighted toward reasoning-based error recovery (Reflexion,
Self-Refine, ReAct) and missed the engineering side of the problem — retry logic,
fallback tool selection, and structured error correction. To address this, four
papers were added to the corpus: CRITIC (Gou et al., 2023), ExpeL (Zhao et al.,
2024), Chen et al. (2023) on agent error detection and correction, and Zhu et al.
(2023) on dynamic retry strategies. These papers treat tool failure as a systems
engineering problem rather than a reasoning problem, and their inclusion
substantially improved coverage of the research question.

### At least one failure

**API rate limiting:** Semantic Scholar returned HTTP 429 errors on every request
during initial testing, disabling half the agent's tools (search_papers,
get_paper_details, get_citations). The server reported this gracefully, but the
agent was reduced to local-only retrieval with no fallback and no persistence.

Fix applied: a disk-based JSON cache (`ss_cache/`) was added to
`src/semantic_scholar.py`. All successful API responses are now written to disk
keyed by URL + parameters. Subsequent identical queries are served from cache
instantly, without a network call or rate-limit wait. Under repeated use — the
typical pattern during a literature review session — the agent now degrades
gracefully even when the API is unreachable.

**Query sensitivity:** Querying "reflexion error correction self-improvement"
returned Self-Refine (Madaan et al., 2023) as the top result instead of Reflexion
(Shinn et al., 2023), which scored barely above threshold at 0.4975. Slight
rephrasing changed which papers appeared — users might not expect this.

This is a fundamental limitation of dense retrieval: semantic embeddings compress
meaning into fixed-size vectors, so paraphrase distance is not zero. The similarity
score range is compressed (useful range ~0.4–0.65), making the threshold parameter
surprisingly sensitive. No code fix was applied here — this is an inherent property
of the embedding model rather than a bug — but the `config.yaml` comment was
updated to document the effective range so future users can set the threshold
with better intuition.

**Chunk boundary artifacts:** With `chunk_size=512` and naive character slicing,
some retrieved passages cut mid-sentence (e.g., ending with "required parameters,
optional"). The agent presents incomplete text that looks complete.

Fix applied: both `src/rag_pipeline.py` and `src/pdf_ingestor.py` now use
sentence-boundary-aware chunking. The new `chunk_text()` splits on sentence
terminators (`.`, `!`, `?`) and greedily fills chunks up to `chunk_size`
characters. The overlap window (`chunk_overlap`, now 128 characters) carries
trailing complete sentences into the next chunk rather than a raw character slice.
Oversized single sentences fall back to whitespace splitting. Retrieved passages
now always end at a sentence boundary.

---

## Part 3: Reflection

### 3.1 Build process

The main design decision was RAG parameters. `chunk_size=512` (~2–3 sentences)
balances context vs. precision. I considered 1024 for more context, but it would
return more irrelevant text alongside relevant portions. The `similarity_threshold`
of 0.3 was deliberately permissive — relevant content scores 0.45–0.65, so 0.3
errs on recall over precision.

The original chunking implementation was character-level, which is what the
assignment scaffolding used. Claude implemented `chunk_text()` and `retrieve()`
correctly on the first try — the sliding window and ChromaDB query pattern are
straightforward. However, the character-level chunking produced mid-sentence cuts
that degraded retrieval quality. After observing this failure, `chunk_text()` was
rewritten in both `rag_pipeline.py` and `pdf_ingestor.py` to split on sentence
boundaries, with `chunk_overlap` raised from 64 to 128 characters to give the
overlap window enough room to carry meaningful sentence context between chunks.

Setup required iteration: the embedding model download timed out once, and
`.mcp.json` needed the Python path updated to point to the venv interpreter.

### 3.2 System prompt engineering

Two prompt variants were implemented and compared: `default` and `concise`.

**Default** starts with Semantic Scholar, uses multiple tool calls, and produces
prose reviews with thematic grouping. It is well-suited for comprehensive
literature review writing where coverage and narrative flow matter.

**Concise** flips the tool order (local library first), caps total tool calls at
two, and requires bullet-point output with a strict schema (Topic, Key Papers,
Main Themes, Gaps). It produced higher information density per token and had fewer
hallucination opportunities because it made fewer total claims. The tradeoff is
breadth: with only two tool calls, it misses papers the default would surface.

Defining "better" as information density per token, concise wins. For a graduate
student writing a review section who needs narrative and attribution, default is
more useful. The right choice depends on the downstream task.

Two additional prompts — `structured` (five required sections, explicit tool
ordering) and `critical` (skeptical evaluation, evidence-first) — are implemented
in `prompts/templates.py` and selectable via `config.yaml`.

### 3.4 Architecture limitations

Three changes would make this viable for real work:

1. **Caching for Semantic Scholar** — implemented. Results are now persisted to
   `ss_cache/` so the agent operates normally during rate-limiting windows. The
   cache is keyed by URL + parameters; clearing the folder forces fresh API calls.

2. **Sentence-boundary-aware chunking** — implemented. The chunker now respects
   sentence terminators so retrieved passages always contain complete thoughts.
   `chunk_overlap` was raised to 128 characters to carry more context across
   chunk boundaries.

3. **Incremental ingestion** — not implemented. Currently, changing any RAG
   parameter requires deleting and rebuilding the entire ChromaDB collection.
   A production system would hash each document and only re-embed changed files.

The biggest surprise was how sensitive retrieval is to query phrasing. Semantic
embeddings don't handle paraphrasing as well as expected — querying "reflexion
error correction" vs. "agent self-improvement from failure" returned different
top results for the same concept. The similarity score range is also very
compressed (useful range ~0.4–0.65), making the threshold parameter surprisingly
sensitive. A hybrid retrieval approach — combining dense embeddings with sparse
BM25 keyword matching — would likely reduce this sensitivity.
