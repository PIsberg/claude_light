import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestRetrievalBenchmarkFixtureMode(unittest.TestCase):

    def test_load_fixture_instances_resolves_local_repo_paths(self):
        from tests.benchmarks import benchmark_retrieval as bench

        fixture_path = Path("tests/fixtures/retrieval_cases.json")
        instances = bench.load_fixture_instances(fixture_path)

        self.assertEqual(len(instances), 3)
        self.assertTrue(instances[0]["local_repo_path"].is_dir())
        self.assertEqual(instances[0]["gold_files"], ["src/auth.py"])

    def test_fixture_benchmark_outputs_non_zero_signal(self):
        from tests.benchmarks import benchmark_retrieval as bench

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "retrieval.json"
            cache_dir = Path(tmpdir) / "cache"

            argv = [
                "benchmark_retrieval.py",
                "--fixture", "tests/fixtures/retrieval_cases.json",
                "--cache-dir", str(cache_dir),
                "--output", str(output_path),
            ]
            with patch.object(bench.sys, "argv", argv):
                bench.main()

            results = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(results), 3)
        self.assertTrue(all(item["repo"] == "fixture/offline-demo" for item in results))
        self.assertGreater(
            sum(item["metrics"]["hit_at_k"]["10"] for item in results) / len(results),
            0.0,
        )
        self.assertTrue(all(item["metrics"]["mrr"] > 0.0 for item in results))


class TestRetrievalRegressionCheck(unittest.TestCase):

    def test_check_retrieval_fails_on_zero_signal_baseline(self):
        from tests.linting import check_regression as regression

        baseline = [
            {
                "repo": "fixture/offline-demo",
                "instance_id": "fixture-auth-login",
                "metrics": {
                    "hit_at_k": {"10": 0.0},
                    "recall_at_k": {"10": 0.0},
                },
            }
        ]

        with self.assertRaises(SystemExit):
            regression.check_retrieval(baseline, baseline)

    def test_check_retrieval_fails_on_quality_drop(self):
        from tests.linting import check_regression as regression

        baseline = [
            {
                "repo": "fixture/offline-demo",
                "instance_id": "fixture-auth-login",
                "metrics": {
                    "hit_at_k": {"10": 1.0},
                    "recall_at_k": {"10": 1.0},
                },
            }
        ]
        current = [
            {
                "repo": "fixture/offline-demo",
                "instance_id": "fixture-auth-login",
                "metrics": {
                    "hit_at_k": {"10": 0.0},
                    "recall_at_k": {"10": 0.0},
                },
            }
        ]

        with self.assertRaises(SystemExit):
            regression.check_retrieval(baseline, current)


if __name__ == "__main__":
    unittest.main()
