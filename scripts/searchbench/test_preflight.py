"""Native parsing is separate from operators translated by the HTTP adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import preflight


class NativePreflightTests(unittest.TestCase):
    def test_bare_regex_requires_translation_even_if_native_parser_accepts_it(self):
        for returncode in (0, 1):
            with (
                self.subTest(returncode=returncode),
                patch(
                    "preflight.subprocess.run",
                    return_value=SimpleNamespace(returncode=returncode, stderr=""),
                ),
            ):
                result = preflight.probe("unused", {"class": "regex", "query": "a.*b"})
                self.assertEqual(result["status"], "requires_adapter_translation")
                self.assertEqual(result["native_parser_accepted"], returncode == 0)


if __name__ == "__main__":
    unittest.main()
