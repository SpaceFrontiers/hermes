#!/usr/bin/env python3
"""
Generate benchmark data for dense and sparse vector indexes.

Downloads MS MARCO passages (or uses cached), generates:
- Dense embeddings using Qwen/Qwen3-Embedding-0.6B (Matryoshka-capable)
- Sparse vectors using naver/splade-v3

Usage:
    python generate_benchmark_data.py --num-docs 100000 --num-queries 1000

Output files:
    - dense_embeddings.bin: Dense vectors for documents
    - dense_queries.bin: Dense vectors for queries
    - sparse_embeddings.bin: Sparse vectors for documents
    - sparse_queries.bin: Sparse vectors for queries
    - ground_truth_dense.bin: Ground truth for dense search
    - ground_truth_sparse.bin: Ground truth for sparse search

Requires:
    pip install sentence-transformers torch transformers datasets numpy tqdm
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    print("Please install: pip install sentence-transformers torch transformers")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install datasets")
    sys.exit(1)


# ============================================================================
# Corpus Loading
# ============================================================================


def load_msmarco_passages(
    num_docs: int, cache_dir: Optional[Path] = None
) -> tuple[list[str], list[str]]:
    """Load MS MARCO passages dataset."""
    print(f"Loading MS MARCO passages (requesting {num_docs} documents)...")

    # Load the passage corpus
    dataset = load_dataset(
        "microsoft/ms_marco",
        "v1.1",
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )

    # Extract passages - MS MARCO has nested structure
    passages = []
    seen = set()

    for item in tqdm(dataset, desc="Extracting passages"):
        if len(passages) >= num_docs:
            break
        # Each item has 'passages' which is a dict with 'passage_text' list
        if "passages" in item and "passage_text" in item["passages"]:
            for text in item["passages"]["passage_text"]:
                if text and text not in seen:
                    seen.add(text)
                    passages.append(text)
                    if len(passages) >= num_docs:
                        break

    print(f"Loaded {len(passages)} unique passages")
    return passages


def load_msmarco_queries(
    num_queries: int, cache_dir: Optional[Path] = None
) -> list[str]:
    """Load MS MARCO queries."""
    print(f"Loading MS MARCO queries (requesting {num_queries})...")

    dataset = load_dataset(
        "microsoft/ms_marco",
        "v1.1",
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )

    queries = []
    seen = set()

    for item in tqdm(dataset, desc="Extracting queries"):
        if len(queries) >= num_queries:
            break
        query = item.get("query", "")
        if query and query not in seen:
            seen.add(query)
            queries.append(query)

    print(f"Loaded {len(queries)} unique queries")
    return queries


# ============================================================================
# Dense Embeddings (Qwen3-Embedding)
# ============================================================================


def generate_dense_embeddings(
    texts: list[str],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    target_dim: int = 128,
    batch_size: int = 32,
    is_query: bool = False,
) -> np.ndarray:
    """Generate dense embeddings using Qwen3-Embedding with Matryoshka truncation."""
    print(f"Loading dense model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Add query prefix for queries (important for asymmetric models)
    if is_query:
        texts = [f"query: {t}" for t in texts]

    print(f"Generating dense embeddings for {len(texts)} texts...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Dense encoding"):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(
            batch, normalize_embeddings=True, show_progress_bar=False
        )
        all_embeddings.append(embeddings)

    embeddings = np.vstack(all_embeddings)

    # Matryoshka truncation
    if embeddings.shape[1] > target_dim:
        print(
            f"Truncating from {embeddings.shape[1]} to {target_dim} dimensions (Matryoshka)"
        )
        embeddings = embeddings[:, :target_dim]
        # Re-normalize after truncation
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

    return embeddings.astype(np.float32)


# ============================================================================
# Sparse Embeddings (SPLADE-v3)
# ============================================================================


class SpladeEncoder:
    """SPLADE-v3 sparse encoder."""

    def __init__(self, model_name: str = "naver/splade-v3"):
        print(f"Loading SPLADE model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.model = self.model.to("mps")
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"SPLADE model loaded on {self.device}")

    def encode(
        self, texts: list[str], batch_size: int = 32
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encode texts to sparse vectors.

        Returns list of (indices, values) tuples.
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="SPLADE encoding"):
            batch = texts[i : i + batch_size]
            batch_results = self._encode_batch(batch)
            results.extend(batch_results)

        return results

    def _encode_batch(self, texts: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encode a batch of texts."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # SPLADE uses log(1 + ReLU(x)) aggregated over tokens
            logits = outputs.logits
            # Max pooling over sequence length with attention mask
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            logits = logits * attention_mask
            # Apply SPLADE activation: log(1 + ReLU(x))
            splade_rep = torch.log1p(torch.relu(logits))
            # Max pool over sequence
            splade_rep = torch.max(splade_rep, dim=1).values

        results = []
        for rep in splade_rep:
            # Get non-zero indices and values
            rep_cpu = rep.cpu().numpy()
            nonzero_mask = rep_cpu > 0.0
            indices = np.where(nonzero_mask)[0].astype(np.uint32)
            values = rep_cpu[nonzero_mask].astype(np.float32)

            # Sort by index for consistent ordering
            sort_idx = np.argsort(indices)
            indices = indices[sort_idx]
            values = values[sort_idx]

            results.append((indices, values))

        return results


def generate_sparse_embeddings(
    texts: list[str],
    model_name: str = "naver/splade-v3",
    batch_size: int = 32,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate sparse embeddings using SPLADE-v3."""
    encoder = SpladeEncoder(model_name)
    return encoder.encode(texts, batch_size)


# ============================================================================
# Ground Truth Computation
# ============================================================================


def compute_dense_ground_truth(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    k: int = 100,
) -> np.ndarray:
    """Compute ground truth for dense vectors using brute force."""
    print(f"Computing dense ground truth (k={k})...")

    num_queries = query_embeddings.shape[0]
    ground_truth = np.zeros((num_queries, k), dtype=np.uint32)

    for i in tqdm(range(num_queries), desc="Dense ground truth"):
        # Compute dot product (vectors are normalized, so this is cosine similarity)
        scores = np.dot(doc_embeddings, query_embeddings[i])
        # Get top-k indices
        top_k = np.argsort(-scores)[:k]
        ground_truth[i] = top_k

    return ground_truth


def compute_sparse_ground_truth(
    query_vectors: list[tuple[np.ndarray, np.ndarray]],
    doc_vectors: list[tuple[np.ndarray, np.ndarray]],
    k: int = 100,
) -> np.ndarray:
    """Compute ground truth for sparse vectors using brute force dot product."""
    print(f"Computing sparse ground truth (k={k})...")

    num_queries = len(query_vectors)
    num_docs = len(doc_vectors)
    ground_truth = np.zeros((num_queries, k), dtype=np.uint32)

    for qi in tqdm(range(num_queries), desc="Sparse ground truth"):
        q_indices, q_values = query_vectors[qi]
        scores = np.zeros(num_docs, dtype=np.float32)

        for di in range(num_docs):
            d_indices, d_values = doc_vectors[di]
            # Compute sparse dot product
            score = sparse_dot(q_indices, q_values, d_indices, d_values)
            scores[di] = score

        # Get top-k indices
        top_k = np.argsort(-scores)[:k]
        ground_truth[qi] = top_k

    return ground_truth


def sparse_dot(
    indices1: np.ndarray,
    values1: np.ndarray,
    indices2: np.ndarray,
    values2: np.ndarray,
) -> float:
    """Compute dot product of two sparse vectors."""
    i, j = 0, 0
    result = 0.0

    while i < len(indices1) and j < len(indices2):
        if indices1[i] == indices2[j]:
            result += values1[i] * values2[j]
            i += 1
            j += 1
        elif indices1[i] < indices2[j]:
            i += 1
        else:
            j += 1

    return result


# ============================================================================
# File I/O
# ============================================================================


def save_dense_embeddings(embeddings: np.ndarray, path: Path):
    """Save dense embeddings in binary format.

    Format: num_vectors (u32), dim (u32), vectors (f32 * num_vectors * dim)
    """
    num_vectors, dim = embeddings.shape

    with open(path, "wb") as f:
        f.write(struct.pack("<II", num_vectors, dim))
        f.write(embeddings.tobytes())

    print(f"Saved {num_vectors} dense vectors (dim={dim}) to {path}")
    print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")


def save_sparse_embeddings(vectors: list[tuple[np.ndarray, np.ndarray]], path: Path):
    """Save sparse embeddings in binary format.

    Format:
    - num_vectors: u32
    - For each vector:
        - num_nonzero: u32
        - indices: [u32; num_nonzero]
        - values: [f32; num_nonzero]
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(vectors)))

        total_nnz = 0
        for indices, values in vectors:
            num_nonzero = len(indices)
            total_nnz += num_nonzero
            f.write(struct.pack("<I", num_nonzero))
            f.write(indices.astype(np.uint32).tobytes())
            f.write(values.astype(np.float32).tobytes())

    avg_nnz = total_nnz / len(vectors) if vectors else 0
    print(f"Saved {len(vectors)} sparse vectors to {path}")
    print(f"  Avg non-zeros: {avg_nnz:.1f}")
    print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")


def save_ground_truth(ground_truth: np.ndarray, path: Path):
    """Save ground truth in binary format.

    Format: num_queries (u32), k (u32), indices (u32 * num_queries * k)
    """
    num_queries, k = ground_truth.shape

    with open(path, "wb") as f:
        f.write(struct.pack("<II", num_queries, k))
        f.write(ground_truth.astype(np.uint32).tobytes())

    print(f"Saved ground truth ({num_queries} queries, k={k}) to {path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark data for vector indexes"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("benchmark_data"),
        help="Output directory",
    )
    parser.add_argument(
        "--num-docs",
        "-n",
        type=int,
        default=100000,
        help="Number of documents to index",
    )
    parser.add_argument(
        "--num-queries", "-q", type=int, default=1000, help="Number of queries"
    )
    parser.add_argument(
        "--dense-dim",
        "-d",
        type=int,
        default=128,
        help="Dense embedding dimension (Matryoshka truncation)",
    )
    parser.add_argument(
        "--k", type=int, default=100, help="Number of ground truth neighbors"
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Dense embedding model",
    )
    parser.add_argument(
        "--sparse-model",
        type=str,
        default="naver/splade-v3",
        help="Sparse embedding model",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for encoding"
    )
    parser.add_argument(
        "--skip-dense", action="store_true", help="Skip dense embedding generation"
    )
    parser.add_argument(
        "--skip-sparse", action="store_true", help="Skip sparse embedding generation"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Cache directory for datasets"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    passages = load_msmarco_passages(args.num_docs, args.cache_dir)
    queries = load_msmarco_queries(args.num_queries, args.cache_dir)

    # Ensure we have enough data
    if len(passages) < args.num_docs:
        print(f"Warning: Only got {len(passages)} passages (requested {args.num_docs})")
    if len(queries) < args.num_queries:
        print(
            f"Warning: Only got {len(queries)} queries (requested {args.num_queries})"
        )

    # Generate dense embeddings
    if not args.skip_dense:
        print("\n" + "=" * 60)
        print("Generating Dense Embeddings")
        print("=" * 60)

        doc_embeddings = generate_dense_embeddings(
            passages,
            model_name=args.dense_model,
            target_dim=args.dense_dim,
            batch_size=args.batch_size,
            is_query=False,
        )
        save_dense_embeddings(doc_embeddings, args.output_dir / "dense_embeddings.bin")

        query_embeddings = generate_dense_embeddings(
            queries,
            model_name=args.dense_model,
            target_dim=args.dense_dim,
            batch_size=args.batch_size,
            is_query=True,
        )
        save_dense_embeddings(query_embeddings, args.output_dir / "dense_queries.bin")

        # Compute ground truth
        ground_truth = compute_dense_ground_truth(
            query_embeddings, doc_embeddings, args.k
        )
        save_ground_truth(ground_truth, args.output_dir / "ground_truth_dense.bin")

    # Generate sparse embeddings
    if not args.skip_sparse:
        print("\n" + "=" * 60)
        print("Generating Sparse Embeddings (SPLADE)")
        print("=" * 60)

        doc_sparse = generate_sparse_embeddings(
            passages,
            model_name=args.sparse_model,
            batch_size=args.batch_size,
        )
        save_sparse_embeddings(doc_sparse, args.output_dir / "sparse_embeddings.bin")

        query_sparse = generate_sparse_embeddings(
            queries,
            model_name=args.sparse_model,
            batch_size=args.batch_size,
        )
        save_sparse_embeddings(query_sparse, args.output_dir / "sparse_queries.bin")

        # Compute ground truth
        ground_truth = compute_sparse_ground_truth(query_sparse, doc_sparse, args.k)
        save_ground_truth(ground_truth, args.output_dir / "ground_truth_sparse.bin")

    print("\n" + "=" * 60)
    print("Benchmark data generation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nTo run benchmarks:")
    print(f"  BENCHMARK_DATA={args.output_dir} cargo bench --bench large_benchmark")


if __name__ == "__main__":
    main()
