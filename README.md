# Writeup — Agentic AI Assignment

**Name:** Samuel Meddin
**Date:** 2026-03-30
**Agent built:** Literature review agent (Option A — Semantic Scholar + local PDF RAG)

---

## Part 2: Task Analysis

**Research question:** How do LLM agents handle tool failures and error recovery?

### Depth beyond surface-level search

The local RAG pipeline surfaced passage-level details that a metadata-only search
cannot reach. Querying "tool failure recovery in LLM agents" returned a passage
from Wang et al. (2023) describing how agents "learn to perform self-debugging" —
a claim embedded in the methods section that does not appear in the abstract.
Similarly, the ReAct paper (Yao et al., 2022) yielded two precise failure
statistics directly from the results section: hallucination accounts for 14% of
false positives in chain-of-thought vs. 6% in ReAct, and "non-informative search
results account for 23% of error cases." Both figures are absent from the abstract.
A researcher relying on abstract skimming alone — the typical Semantic Scholar
workflow — would have missed the quantitative comparison entirely and known only
that ReAct "outperforms" CoT, not by how much or why.

Across five local retrieval queries, the pipeline returned 18 passage-level results
that complemented the external search. Of these, roughly a third contained
experiment-level detail (specific numbers, failure mode breakdowns, or ablation
comparisons) that the Semantic Scholar metadata did not include. The remaining
two-thirds overlapped with what the abstract conveyed, meaning the local library
added genuine value for approximately one in three retrieved chunks — a meaningful
but not overwhelming return.

The initial corpus was weighted toward reasoning-based error recovery (Reflexion,
Self-Refine, ReAct) and lacked coverage of the engineering side of the problem:
retry logic, fallback tool selection, and structured exception handling. This gap
reflects both the keyword framing ("error recovery" pulls reasoning papers) and
the original 20-paper corpus which contained no work on practical fault tolerance.
Four papers were added to address this: CRITIC (Gou et al., 2023), ExpeL (Zhao
et al., 2024), Chen et al. (2023) on agent error detection and correction, and
Zhu et al. (2023) on dynamic retry strategies. These papers treat tool failure as
a systems engineering problem rather than a reasoning problem — a distinct and
complementary perspective that the original corpus entirely missed.

### Local retrieval contribution

The local library's primary contribution was depth within known papers, not
breadth across new ones. Semantic Scholar identified the relevant papers; the RAG
pipeline extracted the specific claims and numbers that justified including them.
The most useful retrieved passages came from the methods and experiments sections
of ReAct (Yao et al., 2022) and the Voyager paper (Wang et al., 2023) — both
returned passages that directly addressed error recovery mechanics rather than
high-level framing.

The least useful passages came from survey papers (Wang et al. survey, 2023) where
retrieved chunks tended to be definition-dense but light on concrete evidence.
These passages scored 0.45–0.52 on similarity — above threshold but below the
0.55+ scores where retrieved text was reliably on-topic. This suggests the
similarity threshold could be tuned per-query type: higher for targeted factual
retrieval, lower for exploratory scoping queries.

### At least one failure

**API rate limiting:** Semantic Scholar returned HTTP 429 errors on every request
during the first testing session, disabling three of the four agent tools
(search_papers, get_paper_details, get_citations). The agent was reduced to
local-only retrieval with no fallback mechanism, no retry delay, and no cached
state — effectively cutting the agent's capability in half for the entire session.
This was the most impactful failure because it occurred before any useful results
were collected, not after. The practical consequence was that the external search
phase of the pipeline had to be re-run in a separate session the following day.

**Query sensitivity:** Querying "reflexion error correction self-improvement"
returned Self-Refine (Madaan et al., 2023) as the top result (similarity 0.5821)
instead of Reflexion (Shinn et al., 2023), which scored 0.4975 — barely above the
0.3 threshold and ranked fourth. The two papers address overlapping concepts from
different angles, but a user who expects Reflexion to rank first on that query
would not trust the system's ordering. Rephrasing to "verbal reinforcement learning
agent self-reflection" correctly surfaced Reflexion at rank one (0.6103), but users
should not need to know the paper's framing to retrieve it. This is a fundamental
limitation of dense retrieval: the embedding space does not consistently organize
papers by the concept they *are about* vs. the language they *use* to describe it.

**Chunk boundary artifacts:** With naive character-level chunking at chunk_size=512,
retrieved passages occasionally terminated mid-sentence. One returned chunk ended
with "required parameters, optional" — syntactically a fragment, but presented
without any indication that text was cut. More subtly, some chunks began mid-argument,
providing a conclusion without the premise. Both issues risk the agent citing a
passage whose meaning depends on context that was sliced away.

**Synthesis verdict:** Used as a discovery tool with human verification of cited
passages, the agent is genuinely useful. It surfaces specific numbers and claims
that would take hours to find manually across 20+ papers. Used as a source of
truth without verification, it is unreliable: the rate-limiting failure, retrieval
sensitivity, and chunk artifacts can each independently produce misleading output.
The agent is best framed as an accelerator for a researcher who knows the literature
well enough to recognize a bad retrieval when they see one.

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
