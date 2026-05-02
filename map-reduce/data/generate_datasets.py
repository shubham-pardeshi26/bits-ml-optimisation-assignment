"""
generate_datasets.py — Generate all synthetic datasets for the map-reduce examples.

Usage
-----
    python map-reduce/data/generate_datasets.py --output map-reduce/data --seed 42

Outputs
-------
    word_count.txt   10,000 lines of synthetic sentences (Zipf vocabulary)
    sales.csv        50,000 transaction rows
    documents.txt    1,000 documents (doc_id<TAB>content)
    matrix_a.txt     64×64 float matrix (whitespace-delimited)
    matrix_b.txt     64×64 float matrix (whitespace-delimited)
    gradients.txt    32 lines: worker_id<TAB>layer<TAB>grad_values (4w × 8l)

All generation uses numpy + stdlib only — no extra dependencies.
"""

import argparse
import os

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORDS_200 = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "great", "between", "need",
    "large", "often", "hand", "high", "place", "hold", "turn", "move", "live",
    "system", "data", "model", "train", "loss", "batch", "epoch", "layer",
    "weight", "gradient", "learn", "network", "deep", "parallel", "reduce",
    "map", "shuffle", "worker", "node", "cluster", "compute", "memory",
    "tensor", "vector", "matrix", "kernel", "process", "thread", "async",
    "sync", "allreduce", "broadcast", "scatter", "gather", "ring", "tree",
    "latency", "bandwidth", "throughput", "scale", "shard", "partition",
    "replica", "checkpoint", "optimizer", "scheduler", "backprop", "forward",
    "accuracy", "precision", "recall", "metric", "evaluate", "deploy",
    "inference", "pipeline", "stream", "buffer", "queue", "channel", "lock",
    "mutex", "semaphore", "barrier", "fence", "spawn", "fork", "join",
    "pool", "task", "job", "stage", "phase", "round", "iteration", "step",
    "bucket", "chunk", "block", "slab", "cache", "page", "frame", "heap",
    "stack", "register", "cycle", "clock", "device", "driver", "runtime",
    "compiler", "linker", "loader", "symbol", "binary", "object", "file",
    "directory", "path", "socket", "packet", "segment", "protocol", "port",
    "address", "header", "payload", "checksum", "hash", "index", "token",
]

REGIONS = ["North", "South", "East", "West", "Central"]
PRODUCTS = ["WidgetA", "WidgetB", "Gadget", "Doohickey", "Thingamajig",
            "Gizmo", "Device", "Module", "Component", "Unit"]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_word_count(rng: np.random.Generator, n_lines: int = 10_000) -> list[str]:
    """Synthetic sentences from a 200-word Zipf-distributed vocabulary."""
    vocab = np.array(WORDS_200)
    # Zipf weights: word i has weight ~ 1/(i+1)
    weights = 1.0 / (np.arange(len(vocab)) + 1)
    weights /= weights.sum()

    lines = []
    for _ in range(n_lines):
        n_words = int(rng.integers(5, 20))
        chosen  = rng.choice(vocab, size=n_words, p=weights)
        lines.append(" ".join(chosen))
    return lines


def gen_sales(rng: np.random.Generator, n_rows: int = 50_000) -> list[str]:
    """CSV rows: transaction_id,region,product,quantity,unit_price"""
    rows = ["transaction_id,region,product,quantity,unit_price"]
    for i in range(n_rows):
        region     = rng.choice(REGIONS)
        product    = rng.choice(PRODUCTS)
        quantity   = int(rng.integers(1, 51))
        unit_price = round(float(rng.uniform(1.0, 500.0)), 2)
        rows.append(f"{i},{region},{product},{quantity},{unit_price:.2f}")
    return rows


def gen_documents(rng: np.random.Generator, n_docs: int = 1_000) -> list[str]:
    """doc_id<TAB>content — 20-100 words each from the same vocabulary."""
    vocab   = np.array(WORDS_200)
    weights = 1.0 / (np.arange(len(vocab)) + 1)
    weights /= weights.sum()

    lines = []
    for doc_id in range(n_docs):
        n_words = int(rng.integers(20, 101))
        words   = rng.choice(vocab, size=n_words, p=weights)
        lines.append(f"{doc_id}\t{' '.join(words)}")
    return lines


def gen_matrix(rng: np.random.Generator, size: int = 64) -> list[str]:
    """Whitespace-delimited float matrix."""
    mat = rng.random((size, size)).astype(np.float32)
    return [" ".join(f"{v:.6f}" for v in row) for row in mat]


def gen_gradients(
    rng: np.random.Generator,
    n_workers: int = 4,
    n_layers:  int = 8,
    grad_dim:  int = 16,
) -> list[str]:
    """
    worker_id<TAB>layer<TAB>space-separated float gradients
    Simulates each worker computing slightly different gradients.
    """
    lines = []
    # True gradient for each layer
    true_grads = rng.standard_normal((n_layers, grad_dim)).astype(np.float32)
    for w in range(n_workers):
        for l in range(n_layers):
            noise = rng.standard_normal(grad_dim).astype(np.float32) * 0.1
            g     = true_grads[l] + noise
            grad_str = " ".join(f"{v:.8f}" for v in g)
            lines.append(f"{w}\tlayer_{l}\t{grad_str}")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets")
    parser.add_argument("--output", default="map-reduce/data",
                        help="Output directory (default: map-reduce/data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = {
        "word_count.txt":  ("\n".join(gen_word_count(rng)),
                            "10,000 lines, Zipf vocabulary"),
        "sales.csv":       ("\n".join(gen_sales(rng)),
                            "50,000 transaction rows"),
        "documents.txt":   ("\n".join(gen_documents(rng)),
                            "1,000 documents (doc_id<TAB>content)"),
        "matrix_a.txt":    ("\n".join(gen_matrix(rng)),
                            "64×64 float matrix A"),
        "matrix_b.txt":    ("\n".join(gen_matrix(rng)),
                            "64×64 float matrix B"),
        "gradients.txt":   ("\n".join(gen_gradients(rng)),
                            "4 workers × 8 layers, gradient vectors"),
    }

    for fname, (content, desc) in files.items():
        path = os.path.join(args.output, fname)
        with open(path, "w") as f:
            f.write(content + "\n")
        n_lines = content.count("\n") + 1
        print(f"  wrote {path:45s}  ({n_lines:>6,} lines)  — {desc}")

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()
