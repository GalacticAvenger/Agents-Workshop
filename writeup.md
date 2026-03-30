# Writeup — Agentic AI Assignment

**Name:** Samuel Meddin
**Date:** 2026-03-30
**Agent built:** Literature review agent (following Option A — Semantic Scholar + local PDF RAG)

---

## Part 2: Task Analysis

**Research question / task:**
How do LLM agents handle tool failures and error recovery — what mechanisms exist and how effective are they?

**Address two of the three areas:**

### Depth beyond surface-level search

The local RAG pipeline surfaced specific passages that go well beyond what abstract-level search provides. For example, querying "How do LLM agents handle tool failures and error recovery?" returned a passage from Wang et al. (2023, Survey of LLM-based Agents) describing how agents can "learn to perform self-debugging" and integrate existing tools into more powerful compositions — a claim supported by the full text but absent from the paper's abstract. Similarly, the ReAct paper (Yao et al., 2022) returned passage-level failure mode analysis showing that hallucination accounts for 14% of false positives in CoT reasoning vs. 6% in ReAct, and that "non-informative search, which counts for 23% of the error cases, derails the model reasoning." These specific statistics would not surface from a metadata-only search.

The citation tracing capability (via `get_citations`) would normally add depth by discovering papers not in keyword results — for instance, Inner Monologue (Huang et al., 2022) was surfaced in the worked example through ReAct's reference list, not keyword search. However, during my testing, the Semantic Scholar API was persistently rate-limited (HTTP 429), which meant the external search and citation tracing tools were unavailable. This itself was informative: it demonstrated that an agent relying on external APIs has a single point of failure, and the system's graceful error messages ("Rate limited by Semantic Scholar — please wait and try again") were helpful but couldn't substitute for the missing data.

The agent missed papers on structured engineering approaches to error handling (retry logic, exception hierarchies, fallback tool selection) — the corpus and Semantic Scholar keyword framing are both biased toward reasoning-oriented approaches.

### At least one failure

**Failure 1 — API rate limiting as a systemic failure.** The Semantic Scholar free tier returned 429 errors on every request during my testing session, making half the agent's toolset (search_papers, get_paper_details, get_citations) completely non-functional. The MCP server handled this gracefully with informative error messages, but the agent was reduced to local-only retrieval. This is a fundamental architectural limitation: the system has no fallback data source when its external API is unavailable, and no caching layer to serve previous results.

**Failure 2 — Query sensitivity in RAG retrieval.** When I queried "reflexion error correction self-improvement," the top result was from Self-Refine (Madaan et al., 2023) rather than Reflexion (Shinn et al., 2023) — the latter appeared third with similarity 0.4975, barely above the 0.3 threshold. Rephrasing the query changed which papers appeared, meaning the system's output is sensitive to phrasing in ways that a user might not expect. A researcher asking a slightly different version of the same question could get materially different results.

**Failure 3 — Chunk boundary artifacts.** With chunk_size=512 and character-level splitting, retrieved passages sometimes cut mid-sentence. For instance, a passage from ToolLLM ended with "required parameters, optional" — clearly truncated. This means the agent may present incomplete information that looks complete, which is arguably worse than returning nothing.

Would I trust this system for real work? With guardrails, yes — but only as a discovery tool, not as a source of truth. Local library citations can be verified against the actual PDFs. External API results should be treated as leads. The system is good at finding the shape of a literature landscape but cannot guarantee completeness or accuracy, especially when APIs are unavailable.

---

## Part 3: Reflection

*Address three of the following.*

### 3.1 Build process

**What was a major design decision you had to make? How did you decide on a particular course of action?**

The main design decision was around RAG parameters — specifically chunk_size (512), chunk_overlap (64), and similarity_threshold (0.3). A chunk size of 512 characters is a reasonable middle ground: large enough to capture a coherent idea, small enough to be specific. However, 512 characters is roughly 2-3 sentences, which often cuts across paragraph boundaries. I considered increasing chunk_size to 1024 for more context per retrieval, but this would return more irrelevant text alongside the relevant portions. The default values from config.yaml proved workable, though a production system would benefit from sentence-boundary-aware chunking rather than fixed character splits.

The similarity_threshold of 0.3 was deliberately permissive. With the L2-to-similarity conversion (1/(1+distance)), scores tend to cluster in the 0.4-0.65 range for relevant content. Setting the threshold too high (e.g., 0.5) would miss relevant but loosely-phrased matches; setting it too low returns noise. 0.3 was a reasonable default that erred on the side of recall over precision.

**What did Claude get right on the first try? Where did you have to push back, correct it, or iterate?**

Claude Code successfully implemented the `chunk_text()` and `retrieve()` methods in rag_pipeline.py on the first attempt — the sliding window chunking logic and ChromaDB query-then-filter pattern are straightforward. The prompts in templates.py were also well-structured from the start, with clear differentiation between the four variants (default, concise, structured, critical).

The setup process required some iteration: the embedding model download timed out on the first attempt due to network instability, requiring a retry. The .mcp.json needed its Python path updated from `"python"` to `"./agents-workshop/bin/python"` to use the virtual environment — this is documented in the README but easy to miss.

### 3.2 System prompt engineering

**What two prompt variants did you implement? How did they differ in behavior?**

The system includes four prompt variants; I focused on comparing "default" and "concise":

- **Default** instructs the agent to start with Semantic Scholar, then query the local library, then trace citations, and produce a prose literature review with thematic grouping. It encourages multiple tool calls and thorough exploration.
- **Concise** flips the tool order (local library first, then one Semantic Scholar search), limits total tool calls to two, and requires bullet-point output rather than prose. It explicitly prohibits calling get_citations or get_paper_details unless asked.

The behavioral difference is significant: the default prompt produces comprehensive but verbose output with 4+ tool calls, while the concise prompt produces tighter summaries from fewer sources. The concise prompt's "local library first" ordering means it prioritizes full-text passage evidence over metadata, which is arguably better for claims that need grounding.

**Which produced better output, and how are you defining "better"?**

"Better" depends on the use case. For a graduate student writing a literature review section, the default prompt is better — it covers more ground and produces citeable prose. For a researcher doing a quick check on what's known about a topic, the concise prompt is better — it's faster, makes fewer API calls (important given rate limiting), and its bullet format is easier to scan. The concise prompt also had fewer hallucination opportunities because it made fewer claims overall. I'd define "better" as higher information density per token of output, by which measure the concise prompt wins.

### 3.4 Architecture limitations

**What would you need to change to make this viable for a real workflow?**

Three changes would be necessary: (1) Add a caching layer for Semantic Scholar results so the system works during rate limiting or outages — even a simple SQLite cache with TTL would dramatically improve reliability. (2) Replace character-level chunking with sentence-boundary-aware chunking (e.g., using spaCy or NLTK sentence tokenization) to avoid mid-sentence cuts. (3) Add a corpus management interface — the current system requires manually running download_papers.py and re-ingesting; a real workflow would need the ability to add individual papers on the fly and incrementally update the vector store.

**What did you learn about the limits of RAG-based agents that you didn't expect before building one?**

The biggest surprise was how sensitive retrieval quality is to query phrasing. I expected that semantic embeddings would handle paraphrasing well, but in practice, querying "reflexion error correction" vs. "agent self-improvement from failure" returned different top results even though they describe the same concept. The embedding model (all-MiniLM-L6-v2) is fast but relatively small — a larger model like all-mpnet-base-v2 might reduce this sensitivity, at the cost of slower ingestion and retrieval.

I also didn't expect the similarity scores to be as compressed as they are. Most relevant results scored between 0.45 and 0.65, and genuinely irrelevant content scored 0.25-0.35. This narrow useful range means the similarity_threshold parameter is surprisingly sensitive — a change from 0.3 to 0.4 can eliminate half the results. In a production system, you'd want adaptive thresholding or a re-ranking step rather than a fixed cutoff.
