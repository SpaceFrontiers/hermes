#!/usr/bin/env python3
"""Capture query work separately from production latency; compare aligned runs."""

import argparse
import json
import selectors
import subprocess
from pathlib import Path

COMMANDS = ["TOP_10", "TOP_1000", "TOP_100_COUNT", "COUNT"]
LEGACY_FIELDS = [
    "doc_blocks",
    "doc_values",
    "doc_payload_bytes",
    "tf_blocks",
    "tf_values",
    "tf_payload_bytes",
    "position_blocks",
    "position_values",
    "position_payload_bytes",
    "posting_seeks",
    "position_reads",
    "positions_requested",
]


def collect(config_path, output):
    config = json.loads(config_path.read_text())
    queries = [
        json.loads(line)
        for p in config["queries"]
        for line in Path(p).read_text().splitlines()
    ]
    expected = config["expected"]
    if len(queries) != len(expected):
        raise ValueError("expected counts must align with the query list")
    passes = config.get("passes", 2)
    if not 1 <= passes <= 10:
        raise ValueError("passes must be in 1..10")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name, argv in config["engines"].items():
        if not name or Path(name).name != name:
            raise ValueError("engine names must be single path components")
        log_path = output / (name + ".stderr")
        with (
            log_path.open("x") as log,
            (output / (name + "-responses.jsonl")).open("x") as answers,
        ):
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
            )
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    for repeat in range(passes):
                        for command in config.get("commands", COMMANDS):
                            for ordinal, (item, count) in enumerate(
                                zip(queries, expected, strict=True)
                            ):
                                process.stdin.write(
                                    command + "\t" + item["query"] + "\n"
                                )
                                process.stdin.flush()
                                if not selector.select(
                                    config.get("query_timeout_seconds", 120)
                                ):
                                    raise TimeoutError((name, command, item["query"]))
                                answer = process.stdout.readline().strip()
                                wanted = (
                                    str(count)
                                    if command.endswith("COUNT") or command == "VERIFY"
                                    else "1"
                                )
                                if answer != wanted:
                                    raise ValueError(
                                        (name, command, item["query"], answer, wanted)
                                    )
                                answers.write(
                                    json.dumps(
                                        {
                                            "repeat": repeat,
                                            "ordinal": ordinal,
                                            "command": command,
                                            "response": answer,
                                        }
                                    )
                                    + "\n"
                                )
                process.stdin.close()
                if process.wait(timeout=30):
                    raise RuntimeError(f"{name} failed; see {log_path}")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        # Streaming conversion bounds memory and preserves duplicate query strings.
        with log_path.open() as log, (output / (name + ".jsonl")).open("x") as dest:
            records = (
                json.loads(line.split("\t", 1)[1])
                for line in log
                if line.startswith(("QUERY_WORK\t", "TRAVERSAL\t"))
            )
            for repeat in range(passes):
                for command in config.get("commands", COMMANDS):
                    for ordinal, item in enumerate(queries):
                        row = next(records)
                        if (row["command"], row["query"]) != (command, item["query"]):
                            raise ValueError(
                                "diagnostic records do not align with requests"
                            )
                        if isinstance(row["work"], list):
                            row["work"] = dict(
                                zip(LEGACY_FIELDS, row["work"], strict=True)
                            )
                            row["schema_version"] = "legacy-traversal-12"
                        if (
                            row.get("schema_version") == 1
                            and command.startswith("TOP_")
                            and not command.endswith("COUNT")
                            and row["work"].get("segment_runs") != 1
                        ):
                            raise ValueError(
                                "ranked query did not capture its sole segment worker"
                            )
                        if not all(
                            type(v) is int and v >= 0 for v in row["work"].values()
                        ):
                            raise ValueError("invalid diagnostic counter")
                        dest.write(
                            json.dumps(
                                dict(
                                    row,
                                    repeat=repeat,
                                    ordinal=ordinal,
                                    tags=item.get("tags", []),
                                )
                            )
                            + "\n"
                        )
            if next(records, None) is not None:
                raise ValueError("extra diagnostic records")
        print(
            name,
            len(queries),
            "queries;",
            passes,
            "passes; responses verified",
            flush=True,
        )
    (output / "complete").touch()


def compare(baseline_path, candidate_path, output, repeat):
    def read(path):
        return [
            r
            for line in path.read_text().splitlines()
            if (r := json.loads(line))["repeat"] == repeat
        ]

    baseline, candidate = read(baseline_path), read(candidate_path)
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("missing or differently sized query passes")
    groups = {}
    changes = []
    for a, b in zip(baseline, candidate, strict=True):
        if any(a[k] != b[k] for k in ("command", "ordinal", "query", "tags")):
            raise ValueError("query order or families differ")
        shared = a["work"].keys() & b["work"].keys()
        delta = {k: b["work"][k] - a["work"][k] for k in sorted(shared)}
        # Different ranking lengths may change score arithmetic but not total work.
        if all(
            k in a["work"] and k in b["work"]
            for k in ("exact_score_units", "lookup_score_units")
        ):
            for row in (a, b):
                row["work"]["term_score_units"] = (
                    row["work"]["exact_score_units"] + row["work"]["lookup_score_units"]
                )
            shared = shared | {"term_score_units"}
            delta["term_score_units"] = (
                b["work"]["term_score_units"] - a["work"]["term_score_units"]
            )
        changes.append(
            {
                "command": a["command"],
                "ordinal": a["ordinal"],
                "query": a["query"],
                "delta": delta,
            }
        )
        tags = ["all", *a["tags"]]
        for tag in tags:
            key = a["command"] + "/" + tag
            group = groups.setdefault(
                key, {"queries": 0, "baseline": {}, "candidate": {}}
            )
            group["queries"] += 1
            for name, row in (("baseline", a), ("candidate", b)):
                for field in shared:
                    group[name][field] = group[name].get(field, 0) + row["work"][field]
    for group in groups.values():
        group["ratios"] = {
            k: group["candidate"][k] / v if v else None
            for k, v in group["baseline"].items()
        }
    result = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "repeat": repeat,
        "note": "Counters and instrumented segment time, not production latency. Missing counters are not zero. Sums weight heavy queries.",
        "groups": groups,
        "queries": changes,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    capture = sub.add_parser("collect")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    diff = sub.add_parser("compare")
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--candidate", type=Path, required=True)
    diff.add_argument("--output", type=Path, required=True)
    diff.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.action == "collect":
        collect(args.config, args.output)
    else:
        compare(args.baseline, args.candidate, args.output, args.repeat)


if __name__ == "__main__":
    main()
