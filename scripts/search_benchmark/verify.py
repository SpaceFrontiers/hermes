#!/usr/bin/env python3
"""Compare exact counts and check Hermes pruning against exhaustive top-k."""

import argparse
import json
import pathlib
import selectors
import subprocess


def request(process, command, query, timeout=300):
    process.stdin.write(f"{command}\t{query}\n")
    process.stdin.flush()
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise TimeoutError(f"{command}: {query}")
    answer = process.stdout.readline().strip()
    if not answer.isdecimal():
        raise RuntimeError(f"invalid answer {answer!r} to {command}: {query}")
    return int(answer)


def close(process):
    if process.stdin and not process.stdin.closed:
        process.stdin.close()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "hermes",
        "hermes-index",
        "tantivy",
        "tantivy-index",
        "queries",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--hermes-arg",
        action="append",
        default=[],
        help="additional Hermes option (repeat; use --hermes-arg=--option for flags)",
    )
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    commands = [
        [args.hermes, "serve", args.hermes_index, *args.hermes_arg],
        [args.tantivy, args.tantivy_index],
    ]
    processes = []
    logs = []
    mismatches = 0
    try:
        for i, command in enumerate(commands):
            log = output.with_suffix(f".engine-{i}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    bufsize=1,
                )
            )
        with open(args.queries) as queries, output.open("x") as results:
            for number, line in enumerate(queries, 1):
                query = json.loads(line)["query"]
                hermes = request(processes[0], "VERIFY", query)
                hermes_count = request(processes[0], "COUNT", query)
                tantivy = request(processes[1], "COUNT", query)
                mismatches += not (hermes == hermes_count == tantivy)
                results.write(
                    json.dumps(
                        {
                            "query": query,
                            "hermes": hermes,
                            "hermes_count": hermes_count,
                            "tantivy": tantivy,
                        }
                    )
                    + "\n"
                )
                results.flush()
                if number % 100 == 0:
                    print(
                        f"verified {number}; count mismatches={mismatches}", flush=True
                    )
    finally:
        for process in processes:
            close(process)
        for log in logs:
            log.close()
    if mismatches:
        raise SystemExit(f"FAIL: {mismatches} queries have different counts")
    print(
        "PASS: score-free/exhaustive/Tantivy counts agree; pruned/exhaustive Hermes top-k agrees"
    )
