import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from src.utils.io_utils import load_pickle_list, write_json, write_pickle, write_rows_csv


class IoUtilsTests(unittest.TestCase):
    def test_load_pickle_list_rejects_non_list_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.pkl"
            with path.open("wb") as f:
                pickle.dump({"not": "a list"}, f)

            with self.assertRaises(TypeError):
                load_pickle_list(path)

    def test_write_helpers_create_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "nested"
            write_pickle(base / "rows.pkl", [{"id": "a"}])
            write_json(base / "report.json", {"ok": True})
            write_rows_csv(base / "rows.csv", [{"id": "a", "value": 1}], ["id", "value"])

            self.assertEqual(load_pickle_list(base / "rows.pkl"), [{"id": "a"}])
            self.assertEqual(json.loads((base / "report.json").read_text()), {"ok": True})
            with (base / "rows.csv").open(newline="", encoding="utf-8") as f:
                self.assertEqual(list(csv.DictReader(f)), [{"id": "a", "value": "1"}])


if __name__ == "__main__":
    unittest.main()
