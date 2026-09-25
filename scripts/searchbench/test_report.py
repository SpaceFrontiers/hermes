"""Reports must not present shared-input coverage as count agreement."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from campaign import ENGINES


class ComparisonReportTests(unittest.TestCase):
    def test_successful_queries_with_different_counts_are_labelled_shared_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "agreement.json").write_text(
                json.dumps(
                    [
                        {
                            "query": {"query_class": "high_term", "text": "file"},
                            "include": True,
                            "comparison": "shared-input",
                            "counts_agree": False,
                        }
                    ]
                )
            )
            for engine in ENGINES:
                directory = root / engine
                directory.mkdir()
                (directory / "complete.json").write_text('{"complete": true}')
                (directory / "high_term-TOP_10-c32.json").write_text(
                    json.dumps(
                        {
                            "family": "high_term",
                            "operation": "TOP_10",
                            "clients": 32,
                            "errors": [],
                            "repetitions": [{"qps": 10, "errors": 0}],
                            "median_qps": 10,
                            "memory": {"peak_vmrss_kb": 1024},
                        }
                    )
                )
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("report.py")), str(root)],
                check=True,
            )
            report = (root / "comparison.md").read_text()
            self.assertIn("Shared-input coverage: **1/1 queries**", report)
            self.assertIn("Exact counts agree for **0**", report)
            self.assertNotIn("Count agreement: **1/1", report)
            self.assertNotIn("Text-analysis differences exclude", report)


if __name__ == "__main__":
    unittest.main()
