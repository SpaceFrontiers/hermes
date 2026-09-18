#!/usr/bin/env python3
"""Derive every distinct term as a separately labeled supplemental workload."""

import argparse
import json
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("queries")
parser.add_argument("output")
args = parser.parse_args()
with open(args.queries) as source:
    terms = sorted(
        {
            term
            for line in source
            for term in re.findall("[a-z]+", json.loads(line)["query"])
        }
    )
with open(args.output, "x") as output:
    for term in terms:
        output.write(
            json.dumps({"query": term, "tags": ["term", "supplemental"]}) + "\n"
        )
print(f"Wrote all {len(terms)} distinct terms; this is not the official query suite")
