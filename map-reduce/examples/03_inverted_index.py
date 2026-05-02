"""
03_inverted_index.py — Build a word-to-document inverted index.

Concepts demonstrated
---------------------
  - Set-valued reducers (deduplication via sorted(set(...)))
  - Map phase is the most expensive step (tokenization of every doc)
  - Optional word lookup after indexing

Usage
-----
    python map-reduce/examples/03_inverted_index.py
    python map-reduce/examples/03_inverted_index.py --engine parallel --lookup gradient
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import ParallelMapReduce, SequentialMapReduce

# ---------------------------------------------------------------------------
# Mapper / Reducer  (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def mapper(line: str):
    """doc_id<TAB>content  ->  (word, doc_id) pairs."""
    parts = line.split("\t", 1)
    if len(parts) != 2:
        return
    doc_id, content = parts
    doc_id = int(doc_id.strip())
    seen = set()
    for word in content.lower().split():
        word = word.strip(".,!?;:\"'()[]{}")
        if word and word not in seen:
            seen.add(word)
            yield (word, doc_id)


def reducer(key: str, values):
    """Collect doc_ids; deduplicate and sort."""
    yield (key, sorted(set(values)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MapReduce inverted index")
    parser.add_argument("--engine",  choices=["sequential", "parallel"],
                        default="sequential")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lookup",  default=None,
                        help="Word to look up in the built index")
    parser.add_argument("--data",    default="map-reduce/data/documents.txt")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[error] Dataset not found: {data_path}")
        print("  Run: python map-reduce/data/generate_datasets.py --output map-reduce/data")
        sys.exit(1)

    lines = data_path.read_text().splitlines()
    print(f"\n=== Inverted Index ({args.engine}) ===")
    print(f"Input: {len(lines):,} documents\n")

    engine = (ParallelMapReduce(num_workers=args.workers)
              if args.engine == "parallel" else SequentialMapReduce())

    results = engine.run(lines, mapper, reducer, verbose=True)

    # Convert to dict for lookup
    index = {word: doc_ids for word, doc_ids in results}

    print(f"\nIndex built: {len(index):,} unique words across {len(lines):,} docs")

    # Sample: show 5 entries with most postings
    top5 = sorted(index.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    print("\nTop 5 words by document frequency:")
    print(f"  {'Word':<20} {'Doc freq':>8}  {'Sample doc IDs'}")
    print(f"  {'-'*20} {'-'*8}  {'-'*30}")
    for word, doc_ids in top5:
        sample = str(doc_ids[:8])[1:-1] + ("..." if len(doc_ids) > 8 else "")
        print(f"  {word:<20} {len(doc_ids):>8,}  [{sample}]")

    # Optional lookup
    if args.lookup:
        word = args.lookup.lower()
        if word in index:
            doc_ids = index[word]
            print(f"\nLookup '{word}': found in {len(doc_ids):,} documents")
            print(f"  Doc IDs: {doc_ids[:20]}{'...' if len(doc_ids)>20 else ''}")
        else:
            print(f"\nLookup '{word}': not found in index")


if __name__ == "__main__":
    main()
