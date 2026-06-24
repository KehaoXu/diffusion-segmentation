import csv
import tempfile
import unittest
from pathlib import Path

from seg_training.splits import save_split_csv, load_split_csv, split_filename


class SplitCsvTest(unittest.TestCase):
    def tempdir(self):
        root = Path.cwd() / ".tmp_tests"
        root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_split_filename_adds_ratio_and_seed(self):
        result = split_filename(Path("atlas_train_val.csv"), train_ratio=0.8, split_seed=42)

        self.assertEqual(result, Path("atlas_train_val_0.8_42.csv"))

    def test_save_split_csv_writes_compact_atlas_headers_and_filenames(self):
        with self.tempdir() as tmpdir:
            split_path = Path(tmpdir) / "atlas_train_val_0.8_42.csv"
            train_data = [
                {
                    "image": "/scratch/peirong/kxu56/atlas/T1/sub-001.nii.gz",
                    "label": "/scratch/peirong/kxu56/atlas/pathology_maps_segmentation/sub-001.nii.gz",
                }
            ]
            val_data = [
                {
                    "image": "/scratch/peirong/kxu56/atlas/T1/sub-002.nii.gz",
                    "label": "/scratch/peirong/kxu56/atlas/pathology_maps_segmentation/sub-002.nii.gz",
                }
            ]

            save_split_csv(split_path, train_data, val_data)

            with split_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            self.assertEqual(
                rows[0],
                ["split", "image(atlas/T1)", "label(atlas/pathology_maps_segmentation)"],
            )
            self.assertEqual(rows[1], ["train", "sub-001.nii.gz", "sub-001.nii.gz"])
            self.assertEqual(rows[2], ["val", "sub-002.nii.gz", "sub-002.nii.gz"])

    def test_load_split_csv_restores_paths_from_header_base(self):
        with self.tempdir() as tmpdir:
            split_path = Path(tmpdir) / "atlas_train_val_0.8_42.csv"
            split_path.write_text(
                "split,image(atlas/T1),label(atlas/pathology_maps_segmentation)\n"
                "train,sub-001.nii.gz,sub-001.nii.gz\n"
                "val,sub-002.nii.gz,sub-002.nii.gz\n",
                encoding="utf-8",
            )

            train_data, val_data = load_split_csv(split_path, path_prefix=Path("/data"))

            self.assertEqual(
                train_data,
                [
                    {
                        "image": str(Path("/data/atlas/T1/sub-001.nii.gz")),
                        "label": str(Path("/data/atlas/pathology_maps_segmentation/sub-001.nii.gz")),
                    }
                ],
            )
            self.assertEqual(
                val_data,
                [
                    {
                        "image": str(Path("/data/atlas/T1/sub-002.nii.gz")),
                        "label": str(Path("/data/atlas/pathology_maps_segmentation/sub-002.nii.gz")),
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
