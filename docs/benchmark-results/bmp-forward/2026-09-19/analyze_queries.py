"""Summarize query_matrix.py output without pooling cold and warm passes."""

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def distribution(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def read_bytes(counters):
    if "io.stat" not in counters:
        return None
    return sum(
        int(item.split("=")[1])
        for line in counters.get("io.stat", "").splitlines()
        for item in line.split()[1:]
        if item.startswith("rbytes=")
    )


def io_delta(before, after):
    start, end = read_bytes(before), read_bytes(after)
    return end - start if start is not None and end is not None else None


def main():
    runs = json.loads(Path(sys.argv[1]).read_text())
    warm_limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    groups = defaultdict(list)
    pairs = defaultdict(list)
    for run in runs:
        budget = run["label"].split("-")[0]
        pairs[run["label"].rsplit("-", 1)[0]].append(run["hits_sha256"])
        for phase in ["warm"] if budget == "warm" else ["first", "repeat"]:
            times = [
                row
                for row in run["times"]
                if (
                    row["pass"] > 0
                    if phase == "warm"
                    else row["pass"] == (phase == "repeat")
                )
                and (
                    budget != "warm" or warm_limit is None or row["query"] < warm_limit
                )
            ]
            groups[(budget, run["mode"], run["depth"], run["variant"], phase)].append(
                (run, times)
            )
    assert all(len(hashes) == 2 and hashes[0] == hashes[1] for hashes in pairs.values())
    summary = []
    for key, group in groups.items():
        budget, mode, depth, variant, phase = key
        times = [row for _, rows in group for row in rows]
        summary.append(
            {
                "budget": budget,
                "mode": mode,
                "depth": depth,
                "variant": variant,
                "phase": phase,
                "trials": len(group),
                "total_ms": distribution([row["total_ms"] for row in times]),
                "backfill_ms": distribution([row["backfill_ms"] for row in times]),
                "retrieval_ms": distribution([row["retrieval_ms"] for row in times]),
                "candidate_count": distribution([row["candidates"] for row in times]),
                "sequential_qps": 1000
                / statistics.mean(row["total_ms"] for row in times),
                "per_run": [
                    {
                        "label": run["label"],
                        "open_ms": run["open_ms"],
                        "phase_mean_ms": statistics.mean(
                            row["total_ms"] for row in rows
                        ),
                        "maxrss_mib": run["usage_after"]["maxrss_kib"] / 1024,
                        "major_faults": run["usage_after"]["major_faults"]
                        - run["usage_before"]["major_faults"],
                        "minor_faults": run["usage_after"]["minor_faults"]
                        - run["usage_before"]["minor_faults"],
                        "whole_run_disk_read_bytes": io_delta(
                            run["cgroup_before"], run["cgroup_after"]
                        )
                        if budget != "warm"
                        else None,
                        "cgroup_peak_bytes": int(run["cgroup_after"]["memory.peak"])
                        if budget != "warm"
                        else None,
                        "cgroup_events": run["cgroup_after"]["memory.events"]
                        if budget != "warm"
                        else None,
                        "hits_sha256": run["hits_sha256"],
                    }
                    for run, rows in group
                ],
            }
        )
    output = {
        "warm_query_limit": warm_limit,
        "runs": len(runs),
        "matching_pairs": len(pairs),
        "results": summary,
    }
    Path(sys.argv[2]).write_text(json.dumps(output, indent=2) + "\n")
    for r in summary:
        if r["phase"] == "first":
            continue
        t = r["total_ms"]
        print(
            f"{r['budget']:5} {r['mode']:9} {r['depth']:4} {r['variant']:3} "
            f"mean={t['mean']:.2f} p50={t['p50']:.2f} p95={t['p95']:.2f} "
            f"p99={t['p99']:.2f} backfill={r['backfill_ms']['mean']:.2f}"
        )


if __name__ == "__main__":
    main()
