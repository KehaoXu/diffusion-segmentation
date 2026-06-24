import tempfile
import unittest
from pathlib import Path

from scripts.summarize_results import MetricRecord, parse_ratio, summarize_by_ratio, write_csv, write_markdown


class SummarizeResultsTest(unittest.TestCase):
    def tempdir(self):
        root = Path.cwd() / ".tmp_tests"
        root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_parse_ratio_from_experiment_path(self):
        path = Path("outputs") / "split42" / "atlas+uncond6" / "eval" / "eval_metrics.json"

        self.assertEqual(parse_ratio(path), 0.6)

    def test_summarize_by_ratio_reports_delta_from_baseline(self):
        records = [
            MetricRecord(Path("a"), "split1", 0.0, 0.50, 0.30, 0.60, 0.50, 10),
            MetricRecord(Path("b"), "split2", 0.0, 0.70, 0.50, 0.80, 0.60, 10),
            MetricRecord(Path("c"), "split1", 0.5, 0.80, 0.60, 0.90, 0.70, 10),
        ]

        rows = summarize_by_ratio(records)

        self.assertEqual(rows[0]["ratio"], 0.0)
        self.assertEqual(rows[0]["n_runs"], 2)
        self.assertAlmostEqual(rows[0]["mean_dice"], 0.60)
        self.assertAlmostEqual(rows[0]["delta_dice_vs_real_only"], 0.0)
        self.assertEqual(rows[1]["ratio"], 0.5)
        self.assertAlmostEqual(rows[1]["delta_dice_vs_real_only"], 0.20)

    def test_write_csv_excludes_best_run_column(self):
        rows = [
            {
                "ratio": 0.0,
                "n_runs": 1,
                "mean_dice": 0.5,
                "std_dice": 0.0,
                "delta_dice_vs_real_only": 0.0,
                "mean_iou": 0.3,
                "mean_precision": 0.6,
                "mean_recall": 0.5,
                "best_dice": 0.5,
                "best_run": "split1/atlas+uncond0",
            }
        ]
        with self.tempdir() as tmpdir:
            path = Path(tmpdir) / "summary.csv"

            write_csv(rows, path)

            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("best_run", header)

    def test_write_csv_excludes_runs_column(self):
        rows = [
            {
                "ratio": 0.0,
                "n_runs": 1,
                "mean_dice": 0.5,
                "std_dice": 0.0,
                "delta_dice_vs_real_only": 0.0,
                "mean_iou": 0.3,
                "mean_precision": 0.6,
                "mean_recall": 0.5,
                "best_dice": 0.5,
                "best_run": "split1/atlas+uncond0",
            }
        ]
        with self.tempdir() as tmpdir:
            path = Path(tmpdir) / "summary.csv"

            write_csv(rows, path)

            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("n_runs", header)

    def test_write_markdown_excludes_runs_column(self):
        rows = [
            {
                "ratio": 0.0,
                "n_runs": 1,
                "mean_dice": 0.5,
                "std_dice": 0.0,
                "delta_dice_vs_real_only": 0.0,
                "mean_iou": 0.3,
                "mean_precision": 0.6,
                "mean_recall": 0.5,
                "best_dice": 0.5,
                "best_run": "split1/atlas+uncond0",
            }
        ]
        with self.tempdir() as tmpdir:
            path = Path(tmpdir) / "summary.md"

            write_markdown(rows, path)

            text = path.read_text(encoding="utf-8")
            self.assertNotIn("| Runs |", text)


if __name__ == "__main__":
    unittest.main()
