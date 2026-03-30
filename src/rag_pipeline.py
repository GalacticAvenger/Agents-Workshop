"""
rag_pipeline.py — RAG retrieval pipeline

This module handles querying the ChromaDB vector database to find relevant
text chunks from the local PDF library. See chunk_text() and retrieve() for
the core logic; both are called by the MCP server via query_library().
"""

import re
import pathlib
import yaml
import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        # Try relative to this file's parent
        config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# RAGPipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Wraps a ChromaDB collection and a sentence-transformer model to provide
    semantic search over ingested PDF chunks.
    """

    def __init__(self, config: dict):
        rag_cfg = config["rag"]

        self.top_k: int = rag_cfg["top_k"]
        self.similarity_threshold: float = rag_cfg["similarity_threshold"]
        self.embedding_model_name: str = rag_cfg["embedding_model"]
        db_path: str = rag_cfg["db_path"]
        collection_name: str = rag_cfg["collection_name"]

        # Load the embedding model
        self.model = SentenceTransformer(self.embedding_model_name)

        # Connect to the persistent ChromaDB collection
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(collection_name)

    @staticmethod
    def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """
        Split *text* into overlapping chunks that respect sentence boundaries.

        Strategy:
          1. Split the text into sentences using punctuation heuristics.
          2. Greedily accumulate sentences into a chunk until adding the next
             sentence would exceed chunk_size characters.
          3. When a chunk is full, start the next chunk by backtracking
             chunk_overlap characters worth of sentences (overlap window).
          4. If a single sentence exceeds chunk_size, it is split on the
             nearest whitespace so no text is ever dropped.

        This avoids the mid-sentence cuts produced by naive character slicing.
        Results are more coherent and produce better embedding quality.

        Args:
            text:          The full document text to chunk.
            chunk_size:    Target maximum character length per chunk.
            chunk_overlap: Approximate character overlap between consecutive
                           chunks (applied at the sentence level).

        Returns:
            List of non-empty string chunks.
        """
        if not text:
            return []

        # ----------------------------------------------------------------
        # Step 1 — sentence tokenisation via regex
        # Splits after .  !  ?  followed by whitespace or end-of-string,
        # keeping the terminator attached to the preceding sentence.
        # ----------------------------------------------------------------
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        # ----------------------------------------------------------------
        # Step 2 — greedy accumulation into chunk_size windows
        # ----------------------------------------------------------------
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_len = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]

            # Edge case: sentence alone exceeds chunk_size — hard-split it
            if len(sentence) > chunk_size:
                # Flush current buffer first
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_len = 0

                # Split the oversized sentence on whitespace
                start = 0
                while start < len(sentence):
                    end = start + chunk_size
                    if end < len(sentence):
                        # Retreat to nearest space to avoid mid-word splits
                        space = sentence.rfind(" ", start, end)
                        if space > start:
                            end = space
                    chunks.append(sentence[start:end])
                    start = end
                i += 1
                continue

            # Will adding this sentence exceed the limit?
            separator = " " if current_sentences else ""
            new_len = current_len + len(separator) + len(sentence)

            if new_len <= chunk_size:
                current_sentences.append(sentence)
                current_len = new_len
                i += 1
            else:
                # Flush the current chunk
                if current_sentences:
                    chunks.append(" ".join(current_sentences))

                # Build the overlap window: include trailing sentences whose
                # combined length fits within chunk_overlap characters
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    candidate = len(s) + (1 if overlap_sentences else 0)
                    if overlap_len + candidate <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += candidate
                    else:
                        break

                current_sentences = overlap_sentences
                current_len = overlap_len
                # Do NOT advance i — reprocess this sentence with the new buffer

        # Flush any remaining sentences
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the most relevant chunks from the PDF library for *query*.

        Steps:
          1. Encode *query* into an embedding vector.
          2. Query ChromaDB for the top self.top_k nearest chunks.
          3. Convert L2 distances to similarity scores: 1 / (1 + distance).
          4. Filter out results below self.similarity_threshold.
          5. Return results sorted by similarity descending, each as a dict with
             keys: text, source, chunk_index, similarity.
        """
        # Step 1: encode the query
        query_embedding = self.model.encode(query).tolist()

        # Step 2: query ChromaDB for the top-k nearest chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Steps 3–4: convert distances to similarities and filter by threshold
        output = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1 / (1 + distance), 4)
            if similarity >= self.similarity_threshold:
                output.append({
                    "text": text,
                    "source": meta["source"],
                    "chunk_index": meta["chunk_index"],
                    "similarity": similarity,
                })

        # Step 5: sort by similarity descending
        output.sort(key=lambda x: x["similarity"], reverse=True)
        return output


# ---------------------------------------------------------------------------
# Convenience function used by the MCP server
# ---------------------------------------------------------------------------

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return a cached RAGPipeline instance (lazy-loaded on first call)."""
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = RAGPipeline(config)
    return _pipeline


def query_library(query: str) -> list[dict]:
    """Top-level function called by the MCP server tool."""
    return get_pipeline().retrieve(query)
