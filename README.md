# Writeup — Agentic AI Assignment

**Name:** Samuel Meddin
**Date:** 2026-03-30
**Agent built:** Literature review agent (Option A — Semantic Scholar + local PDF RAG)

---

## Part 2: Task Analysis

**Research question:** How do LLM agents handle tool failures and error recovery?

### Depth beyond surface-level search

The local rag pipeline used passage level detailed that the metadata only search missed. quierying about tool failure recovered gave me a passage from Wang et al. that showed how agents learn to perform self debugging a claim in the full text but not the abstract.tye react paper returned speciic failure stats about haullucinations which wouldnt be available whatsover from just skimming the abstract. 

However the agent seemed to not do well and missed paperes on engineering oriented error handling. the corpus and keyword framing both favored reasoning based approaches over practice ones.

### At least one failure

**API rate limiting:** Semantic Scholar returned 429 errors on every request during testing, disabling half the agent's tools (search_papers, get_paper_details, get_citations). The server handled this gracefully with error messages, but the agent was reduced to local-only retrieval with no fallback or cache.

**Query sensitivity:** Querying "reflexion error correction self-improvement" returned Self-Refine (Madaan et al., 2023) as the top result instead of Reflexion (Shinn et al., 2023), which scored barely above threshold at 0.4975. Slight rephrasing changed which papers appeared — users might not expect this.

**Chunk boundary artifacts:** With chunk_size=512, some retrieved passages cut mid-sentence (e.g., ending with "required parameters, optional"). The agent presents incomplete text that looks complete.

For real work: usable as a discovery tool with guardrails, not as a source of truth. Local citations can be verified; external results are leads to check.

---

## Part 3: Reflection

### 3.1 Build process

the main decision i made was the rag params. i had to figure out the chunk size to have the balance between context and precision. i considered 1024 for more context but it would give me more irrelevant text so i used 512 instead. the similarity threshold of .3 was also deliberately permissive and gave relevant content scores of .45 to .65. 

Claude implemented chunk_text() and retrieve() correctly on the first try. Setup required iteration where the embedding model download timed out once, and .mcp.json needed the Python path updated to use the venv.

### 3.2 System prompt engineering

I compared "default" and "concise" prompts. default start with semantic scholar, uses multiple tool calls, and produce prose reviewes with thematic grouping. conside flips it and limits to two tool calls and forces bullet point output.

the conside prompt had higher information density per token and had fewer hallucinations and the default was better for comprehensive literature review writing. i thought conside was better as it was more information dense.

### 3.4 Architecture limitations

for this to work for real work i would add a caching layer for semantic scholar results so the system works during rate limiting. id also replace character level chunking with sentence boundary aware splitting to make results much better. finally id add the ability to add papers incrementally without full reingestion.

the biggest surprise was how sensitive retrieval is to query phrasing. semantic embedding didnt handle paraphrasing very well querying reflexion error correction vs agent self improvement from failure which returned insanely dif results for the same concept. the similarity score also was super compressed so it made the threshold param super sensitive.
