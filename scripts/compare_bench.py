"""Compare two ``bench_matrix.py`` reports and summarize reference/candidate ratio."""

import argparse
import json
import statistics
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    index = {}
    for case in data["cases"]:
        key = (case["dtype"], case["B"], case["V"], case["K"])
        index[key] = case
    return data, index


def percentile(values, fraction):
    values = sorted(values)
    n = len(values)
    return values[max(0, int(fraction * n) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ref_data, ref = load(args.reference)
    cand_data, cand = load(args.candidate)
    keys = sorted(set(ref) & set(cand), key=lambda k: (k[1], k[2], k[3], k[0]))

    rows = []
    for key in keys:
        ref_us = ref[key]["device_ms"]["median"] * 1000.0
        cand_us = cand[key]["device_ms"]["median"] * 1000.0
        rows.append(
            {
                "dtype": key[0],
                "B": key[1],
                "V": key[2],
                "K": key[3],
                "reference_us": ref_us,
                "candidate_us": cand_us,
                "ratio": ref_us / cand_us,
            }
        )

    ratios = [row["ratio"] for row in rows]
    summary = {
        "cases": len(rows),
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "ratio_min": min(ratios),
        "ratio_p10": percentile(ratios, 0.10),
        "ratio_median": statistics.median(ratios),
        "ratio_mean": statistics.fmean(ratios),
        "ratio_p90": percentile(ratios, 0.90),
        "ratio_max": max(ratios),
        "lt_0_8": sum(1 for r in ratios if r < 0.8),
        "between_0_8_and_1_0": sum(1 for r in ratios if 0.8 <= r < 1.0),
        "ge_1_0": sum(1 for r in ratios if r >= 1.0),
        "rows": rows,
    }

    print(
        f"cases={summary['cases']} "
        f"min={summary['ratio_min']:.3f} "
        f"p10={summary['ratio_p10']:.3f} "
        f"median={summary['ratio_median']:.3f} "
        f"mean={summary['ratio_mean']:.3f} "
        f"p90={summary['ratio_p90']:.3f} "
        f"max={summary['ratio_max']:.3f}"
    )
    print(
        f"ratio<0.8={summary['lt_0_8']} "
        f"0.8<=ratio<1.0={summary['between_0_8_and_1_0']} "
        f"ratio>=1.0={summary['ge_1_0']}"
    )
    print()
    print(f"{'dtype':8s} {'B':>4s} {'V':>7s} {'K':>3s} "
          f"{'ref_us':>9s} {'cand_us':>9s} {'ratio':>7s}")
    for row in rows:
        print(
            f"{row['dtype']:8s} {row['B']:4d} {row['V']:7d} {row['K']:3d} "
            f"{row['reference_us']:9.2f} {row['candidate_us']:9.2f} "
            f"{row['ratio']:7.3f}"
        )

    worst = sorted(rows, key=lambda row: row["ratio"])[:5]
    best = sorted(rows, key=lambda row: row["ratio"], reverse=True)[:5]
    print()
    print("worst 5:")
    for row in worst:
        print(
            f"  {row['dtype']} B={row['B']} V={row['V']} K={row['K']} "
            f"ratio={row['ratio']:.3f}"
        )
    print("best 5:")
    for row in best:
        print(
            f"  {row['dtype']} B={row['B']} V={row['V']} K={row['K']} "
            f"ratio={row['ratio']:.3f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
