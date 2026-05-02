"""
02_sales_aggregation.py — Group-by aggregation with composite keys.

Concepts demonstrated
---------------------
  - Composite tuple keys: (region, product)
  - Multi-job chaining (three separate MapReduce passes)
  - ASCII results table output

Jobs
----
  Job 1 — Total revenue per (region, product)
  Job 2 — Transaction count per region
  Job 3 — Average order value per region

Usage
-----
    python map-reduce/examples/02_sales_aggregation.py
    python map-reduce/examples/02_sales_aggregation.py --engine parallel
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import ParallelMapReduce, SequentialMapReduce

# ---------------------------------------------------------------------------
# Job 1: Total revenue per (region, product)
# ---------------------------------------------------------------------------

def revenue_mapper(row: str):
    """CSV row -> ((region, product), revenue)"""
    parts = row.split(",")
    if len(parts) != 5 or parts[0] == "transaction_id":
        return
    _, region, product, quantity, unit_price = parts
    revenue = int(quantity) * float(unit_price)
    yield ((region.strip(), product.strip()), revenue)


def revenue_reducer(key, values):
    yield (key, round(sum(values), 2))


# ---------------------------------------------------------------------------
# Job 2: Transaction count per region
# ---------------------------------------------------------------------------

def tx_count_mapper(row: str):
    parts = row.split(",")
    if len(parts) != 5 or parts[0] == "transaction_id":
        return
    region = parts[1].strip()
    yield (region, 1)


def tx_count_combiner(key, values):
    yield sum(values)


def tx_count_reducer(key, values):
    yield (key, sum(values))


# ---------------------------------------------------------------------------
# Job 3: Average order value per region
# ---------------------------------------------------------------------------

def avg_order_mapper(row: str):
    parts = row.split(",")
    if len(parts) != 5 or parts[0] == "transaction_id":
        return
    region     = parts[1].strip()
    quantity   = int(parts[3])
    unit_price = float(parts[4])
    order_val  = quantity * unit_price
    yield (region, order_val)


def avg_order_reducer(key, values):
    vals = list(values)
    yield (key, round(sum(vals) / len(vals), 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MapReduce sales aggregation")
    parser.add_argument("--engine", choices=["sequential", "parallel"],
                        default="sequential")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data", default="map-reduce/data/sales.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[error] Dataset not found: {data_path}")
        print("  Run: python map-reduce/data/generate_datasets.py --output map-reduce/data")
        sys.exit(1)

    rows = data_path.read_text().splitlines()
    print(f"\n=== Sales Aggregation ({args.engine}) ===")
    print(f"Input: {len(rows)-1:,} transactions\n")

    engine = (ParallelMapReduce(num_workers=args.workers)
              if args.engine == "parallel" else SequentialMapReduce())

    # --- Job 1 ---
    print("Job 1: Revenue per (region, product)")
    revenue_results = engine.run(rows, revenue_mapper, revenue_reducer, verbose=True)
    print()

    # --- Job 2 ---
    print("Job 2: Transaction count per region")
    tx_results = engine.run(rows, tx_count_mapper, tx_count_reducer,
                            combiner=tx_count_combiner, verbose=True)
    print()

    # --- Job 3 ---
    print("Job 3: Average order value per region")
    avg_results = engine.run(rows, avg_order_mapper, avg_order_reducer, verbose=True)
    print()

    # --- ASCII table: top 15 revenue pairs ---
    top15 = sorted(revenue_results, key=lambda kv: kv[1][1], reverse=True)[:15]
    print("Top 15 (region, product) by revenue:")
    print(f"  {'Region':<10} {'Product':<14} {'Revenue':>12}")
    print(f"  {'-'*10} {'-'*14} {'-'*12}")
    for (region, product), rev in top15:
        print(f"  {region:<10} {product:<14} {rev:>12,.2f}")

    print("\nTransaction counts per region:")
    print(f"  {'Region':<10} {'Tx Count':>10}")
    print(f"  {'-'*10} {'-'*10}")
    for region, count in sorted(tx_results, key=lambda kv: kv[1][1], reverse=True):
        print(f"  {region:<10} {count:>10,}")

    print("\nAverage order value per region:")
    print(f"  {'Region':<10} {'Avg Order':>12}")
    print(f"  {'-'*10} {'-'*12}")
    for region, avg in sorted(avg_results, key=lambda kv: kv[1][1], reverse=True):
        print(f"  {region:<10} {avg:>12,.2f}")


if __name__ == "__main__":
    main()
