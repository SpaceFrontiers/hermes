import json
import statistics
import sys
from pathlib import Path

for root in map(Path, sys.argv[1:]):
    if not (root / "results.json").exists():
        continue
    result = json.loads((root / "results.json").read_text())
    summary = {}
    for category, rows in result.items():
        summary[category] = {}
        for variant in dict.fromkeys(row["variant"] for row in rows):
            selected = [row for row in rows if row["variant"] == variant]
            summary[category][variant] = {
                key: statistics.mean(row[key] for row in selected)
                for key in [
                    "mean_ms",
                    "p50_ms",
                    "p95_ms",
                    "p99_ms",
                    "rss_kib",
                    "open_ms",
                    "major_faults",
                    "fs_input_blocks",
                ]
            }
            summary[category][variant] |= {
                "runs": len(selected),
                "mean_range_ms": [
                    min(row["mean_ms"] for row in selected),
                    max(row["mean_ms"] for row in selected),
                ],
                "hits_identical": all(row["hits_identical"] for row in selected),
            }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(root, json.dumps(summary, indent=2))
