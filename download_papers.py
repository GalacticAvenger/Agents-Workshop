"""
download_papers.py — Download the curated PDF corpus from arXiv

This script downloads ~25 papers on LLM agents, RAG, and related topics
into the pdfs/ directory. Run it once before ingesting.

Usage:
    python download_papers.py

"""

import time
import pathlib
import sys
import httpx

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Paper list
# ---------------------------------------------------------------------------

PAPERS = [
    # RAG & retrieval
    ("2005.11401", "Lewis2020_RAG_retrieval_augmented_generation.pdf"),
    ("2312.10997", "Gao2023_RAG_survey.pdf"),
    ("2401.18059", "Sarthi2024_RAPTOR_tree_retrieval.pdf"),
    ("2310.06825", "Asai2023_SelfRAG_learn_to_retrieve.pdf"),

    # Reasoning & agents
    ("2201.11903", "Wei2022_chain_of_thought_prompting.pdf"),
    ("2210.03629", "Yao2022_ReAct_reasoning_acting.pdf"),
    ("2305.10601", "Yao2023_tree_of_thoughts.pdf"),
    ("2303.17651", "Madaan2023_self_refine_iterative.pdf"),
    ("2303.11366", "Shinn2023_reflexion_verbal_rl.pdf"),

    # Tool use
    ("2302.04761", "Schick2023_Toolformer_language_model_tools.pdf"),
    ("2305.15334", "Patil2023_Gorilla_LLM_API_calls.pdf"),
    ("2307.16789", "Qin2023_ToolLLM_tool_use_benchmark.pdf"),

    # Multi-agent & planning
    ("2303.17760", "Li2023_CAMEL_communicative_agents.pdf"),
    ("2308.00352", "Hong2023_MetaGPT_multi_agent_framework.pdf"),
    ("2305.16291", "Wang2023_Voyager_open_ended_agents.pdf"),

    # Surveys & benchmarks
    ("2309.07864", "Wang2023_survey_LLM_based_agents.pdf"),
    ("2308.03688", "Liu2023_AgentBench_evaluation.pdf"),

    # Alignment & safety (context for critical evaluation)
    ("2212.10560", "Bai2022_constitutional_AI.pdf"),

    # Memory & long context
    ("2304.01373", "Park2023_generative_agents_simulation.pdf"),

    # Prompting techniques
    ("2205.01068", "Wang2022_self_consistency_chain_of_thought.pdf"),

    # Engineering-oriented error handling & reliability
    # These address the gap where reasoning-focused papers miss practical
    # fault tolerance: retry logic, fallback tools, and error recovery patterns.
    ("2305.11738", "Gou2023_CRITIC_LLM_tool_interaction_critique.pdf"),
    ("2401.10020", "Zhao2024_expel_experience_learning_agents.pdf"),
    ("2308.09687", "Chen2023_agent_error_detection_correction.pdf"),
    ("2310.05029", "Zhu2023_ghost_LLM_agent_dynamic_retry.pdf"),
]

OUTPUT_DIR = pathlib.Path("pdfs")
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
DELAY_SECONDS = 1.5  # Be polite to arXiv servers


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def download_paper(arxiv_id: str, filename: str, output_dir: pathlib.Path) -> bool:
    """Download a single arXiv PDF. Returns True on success."""
    dest = output_dir / filename
    if dest.exists():
        print(f"  [skip] Already exists: {filename}")
        return True

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    print(f"  Downloading arXiv:{arxiv_id} → {filename}")
    try:
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        size_kb = dest.stat().st_size // 1024
        print(f"  ✓ {size_kb} KB")
        return True
    except httpx.HTTPStatusError as e:
        print(f"  ✗ HTTP {e.response.status_code}: {url}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Downloading {len(PAPERS)} papers to {OUTPUT_DIR}/\n")

    success, skipped, failed = 0, 0, 0
    for i, (arxiv_id, filename) in enumerate(PAPERS):
        dest = OUTPUT_DIR / filename
        if dest.exists():
            skipped += 1
            print(f"[{i+1}/{len(PAPERS)}] [skip] {filename}")
            continue

        print(f"[{i+1}/{len(PAPERS)}]", end=" ")
        ok = download_paper(arxiv_id, filename, OUTPUT_DIR)
        if ok:
            success += 1
        else:
            failed += 1

        # Polite delay between requests (skip after last)
        if i < len(PAPERS) - 1:
            time.sleep(DELAY_SECONDS)

    print(f"\nDone. Downloaded: {success}, Skipped: {skipped}, Failed: {failed}")

    if failed > 0:
        print("\nSome downloads failed (arXiv may be temporarily unavailable).")
        print("Re-run this script to retry failed downloads.")

    print(f"\nNext step: run ingestion to build the vector database:")
    print("    python src/pdf_ingestor.py")


if __name__ == "__main__":
    main()
