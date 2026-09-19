"""Summarize paired accumulator measurements without pooling cold/warm passes."""

import json
import math
import statistics
import sys
from pathlib import Path

runs = json.loads(Path(sys.argv[1]).read_text())
groups = {}
pairs = {}
for run in runs:
    key = (run["budget"], run["mode"], run["depth"], run["trial"])
    pairs.setdefault(key, {})[run["variant"]] = run["hits_sha256"]
    key = (run["budget"], run["mode"], run["depth"], run["variant"])
    groups.setdefault(key, []).append(run)
for key, hashes in pairs.items():
    assert len(hashes) == 2 and len(set(hashes.values())) == 1, key


def distribution(values):
    values = sorted(values)
    return {
        "mean": statistics.mean(values),
        **{f"p{p}": values[math.ceil(len(values) * p / 100) - 1] for p in [50, 95, 99]},
    }


summary = []
for (budget, mode, depth, variant), group in groups.items():
    phases = {}
    for phase in ["first", "repeat"]:
        times = [
            t
            for r in group
            for t in r["times"]
            if (t["pass"] == 0) == (phase == "first")
        ]
        phases[phase] = {
            k: distribution([t[k] for t in times])
            for k in ["retrieval_ms", "backfill_ms", "total_ms"]
        }
        phases[phase]["requests"] = len(times)
        phases[phase]["serial_qps"] = 1000 / phases[phase]["total_ms"]["mean"]
    memory_peaks = (
        [int(r["cgroup_after"]["memory.peak"]) for r in group]
        if budget != "warm"
        else None
    )
    row = {
        "budget": budget,
        "mode": mode,
        "depth": depth,
        "variant": variant,
        "phases": phases,
        "peak_cgroup_bytes": memory_peaks,
        "peak_rss_kib": [r["usage_after"]["maxrss_kib"] for r in group],
    }
    row["query_major_faults"] = [
        r["usage_after"]["major_faults"] - r["usage_before"]["major_faults"]
        for r in group
    ]
    row["query_cpu_s"] = [
        sum(r["usage_after"][k] - r["usage_before"][k] for k in ["user_s", "system_s"])
        for r in group
    ]
    row["oom_events"] = (
        [r["cgroup_after"].get("memory.events") for r in group]
        if budget != "warm"
        else None
    )
    summary.append(row)
out = {
    "completed_runs": len(runs),
    "matched_pairs": len(pairs),
    "timed_requests": sum(len(r["times"]) for r in runs),
    "groups": summary,
}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")
for g in summary:
    p = g["phases"]["repeat"]
    print(
        g["budget"],
        g["mode"],
        g["depth"],
        g["variant"],
        f"total {p['total_ms']['mean']:.2f}/{p['total_ms']['p95']:.2f} backfill {p['backfill_ms']['mean']:.2f} RSS {max(g['peak_rss_kib']) / 1024:.1f} MiB",
    )
