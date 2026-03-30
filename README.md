# Writeup — Agentic AI Assignment

**Name:** Samuel Meddin
**Date:** 2026-03-30
**Agent built:** Literature review agent (Option A — Semantic Scholar + local PDF RAG)

---

## Part 2: Task Analysis

**Research question:** How do LLM agents handle tool failures and error recovery?

### Depth beyond surface-level search

The thing that stood out most using the local RAG pipeline was that it found stuff
that a normal metadata search would totally miss. Like when I queried "tool failure
recovery in LLM agents" it pulled a passage from Wang et al. (2023) where they
talk about how agents "learn to perform self-debugging" which is actually in the
methods section and not the abstract at all. Same thing with the ReAct paper (Yao
et al., 2022) where I got two specific numbers straight from the results section:
hallucinations account for 14% of false positives in chain-of-thought vs only 6%
in ReAct, and "non-informative search results account for 23% of error cases."
Neither of those numbers shows up in the abstract so if you were just skimming
Semantic Scholar you would have only known ReAct "outperforms" CoT without any
idea of by how much or in what way.

I ran five different local retrieval queries and got back 18 results total. About
a third of them had actual experiment level detail like specific numbers or ablation
comparisons that Semantic Scholar didn't have. The other two thirds were pretty much
just restating what the abstract already said. So the local library was genuinely
useful for maybe one in three chunks which honestly felt meaningful but not as
impressive as I expected going in.

The bigger problem I noticed was that the corpus I started with was way too focused
on reasoning based error recovery, things like Reflexion, Self-Refine, and ReAct.
It basically had nothing on the engineering side of the problem like retry logic or
fallback tools. Part of that was just how the keywords worked since "error recovery"
naturally pulls reasoning papers. To fix this I added four papers: CRITIC (Gou et al.,
2023), ExpeL (Zhao et al., 2024), Chen et al. (2023) on error detection and
correction, and Zhu et al. (2023) on dynamic retry. These all look at tool failure
as a systems problem instead of a reasoning problem which is a pretty different
angle that the original corpus just didnt cover at all.

### Local retrieval contribution

The local library was really more useful for going deeper into papers I already
knew were relevant rather than finding new ones. Semantic Scholar handled the
discovery part and then the RAG pipeline pulled out the specific claims and numbers
that actually backed up why those papers mattered. The most useful stuff came from
the methods and experiments sections of ReAct (Yao et al., 2022) and Voyager (Wang
et al., 2023) since those sections directly talked about error recovery mechanics
instead of just framing the problem at a high level.

The least useful results were from survey papers where the retrieved chunks were
mostly definitions and background with not a lot of concrete evidence. Those chunks
scored around 0.45 to 0.52 on similarity which was above the threshold but below the
0.55+ range where results were actually reliably on-topic. It made me think the
threshold probably should be tuned differently depending on what you are searching
for, higher for specific factual queries and lower when you are just trying to get
a general sense of a topic.

### At least one failure

**API rate limiting:** The first testing session was basically a disaster because
Semantic Scholar returned 429 errors on literally every single request. This knocked
out three of the four agent tools (search_papers, get_paper_details, get_citations)
so the agent was stuck using only local retrieval with no fallback and no way to
cache anything. What made this especially bad was that it happened right at the
start before any useful results had been collected so the whole external search phase
had to be redone in a completely separate session the next day.

**Query sensitivity:** When I queried "reflexion error correction self-improvement"
the top result was Self-Refine (Madaan et al., 2023) with a similarity of 0.5821
and not Reflexion (Shinn et al., 2023) which only scored 0.4975 and came in fourth.
Both papers are about overlapping ideas but if someone expects Reflexion to come up
first for that query they would probably lose trust in the system pretty fast. I
rephrased the query to "verbal reinforcement learning agent self-reflection" and
Reflexion correctly came up first at 0.6103 but you really shouldnt have to know
the exact framing a paper uses just to find it. The embedding space just doesnt
organize things by concept in the way you would hope.

**Chunk boundary artifacts:** With the original character level chunking some
retrieved passages just cut off mid-sentence. One chunk I got back ended with
"required parameters, optional" which is completely meaningless out of context but
looks like a normal result at first glance. Some chunks also started mid-argument
so you would get a conclusion with no setup. Both of those things could easily
cause the agent to cite something in a misleading way.

**Overall verdict:** If you treat it as a discovery tool and verify anything it
cites then its genuinely useful. It can surface specific numbers and claims across
20+ papers way faster than doing it by hand. But if you trust it as a source of
truth without checking then you will run into problems. The rate limiting, the
retrieval sensitivity, and the chunk issues can all independently produce bad output.
Its best thought of as something that speeds up research for someone who already
knows the field well enough to catch when something looks off.

---

## Part 3: Reflection

### 3.1 Build process

The main decision was RAG params for me. I saw really big differences in the accuracy and output based on what I selected.`chunk_size=512` (~2–3 sentences)
balances context vs. precision. I considered 1024 for more context, but it would
return more irrelevant text alongside relevant portions. The `similarity_threshold`
of 0.3 was really permissive so 0.3 was more focused on recall over precision.

Also, the original chunking implementation was character-level. This is what the assignment decription used. Claude implemented `chunk_text()` and `retrieve()`
correctly on the first try and the sliding window and ChromaDB query pattern are
straightforward. I saw though that the character-level chunking produced mid-sentence cuts that really hurt retrieval quality. I then focused on rewriting`chunk_text()` in both `rag_pipeline.py` and `pdf_ingestor.py` to split on sentence
boundaries with `chunk_overlap` raised from 64 to 128 characters to give the
overlap window enough room to have real sentence context between chunks.

Setup required iteration: the embedding model download timed out once and
`.mcp.json` needed the Python path updated to point to the venv interpreter.

### 3.2 System prompt engineering

I had two different types of prompt `default` and `concise`

**Default** used Semantic Scholar then produced sentence and word reviews with thematic grouping. It is well-suited for comprehensive literature review writing where coverage and narrative flow matter.

**Concise** used my local library first and capped tool calls at
two requiring bullet point output with a strict schema (Topic, Key Papers,
Main Themes, Gaps). It had way higher information density per token and had fewer
hallucination opportunities because it had way fewer total claims. The tradeoff was
breadth where with only two tool calls it misses papers the default would surface.

Saying that better was information density per token you'd rather have concise. for a graduate student writing a review section who needs narrative and attribution default is better so really netiher one is necessarily better.

I experimented with two other types of prompts `structured` (five required sections, explicit tool
ordering) and `critical` (skeptical evaluation, evidence-first) which are implemented
in `prompts/templates.py` and selectable via `config.yaml`.

### 3.4 Architecture limitations

There's three ways I thought of to make this even better

1. **Caching for Semantic Scholar** I implemented! Results are now persisted to
   `ss_cache/` so the agent operates normally during rate-limiting windows. The
   cache is keyed by URL + parameters clearing the folder forces fresh API calls.

2. **Sentence-boundary-aware chunking** I implemented! The chunker now respects
   sentence terminators so retrieved passages always contain complete thoughts.
   `chunk_overlap` was raised to 128 characters to carry more context across
   chunk boundaries.

3. **Incremental ingestion** not implemented. When you change any RAG
   parameter it forces you to delete and rebuild the entire ChromaDB collection.
   A production system would hash each document and only re-embed changed files.

The biggest surprise was how sensitive retrieval is to query phrasing. Semantic
embeddings don't handle paraphrasing as well as I had imagined just from my use of already existent agents. Querying "reflexion error correction" vs. "agent self-improvement from failure" returned different top results for the same concept which was kind of wild to me as they are seemingly the same thing. The similarity score range is also super compressed (useful range ~0.4–0.65), making the threshold parameter surprisingly sensitive. A hybrid retrieval approach that combines dense embeddings with sparse BM25 keyword matching would probably reduce this sensitivity.
