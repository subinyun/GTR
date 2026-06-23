import unittest

from src.gtr.metrics import label_space


class LabelSpaceTests(unittest.TestCase):
    def test_label_space_collects_statutes(self) -> None:
        rows = [{"statutes": ["A", "B"]}, {"statutes": ["B", "C"]}]
        self.assertEqual(label_space(rows), {"A", "B", "C"})


if __name__ == "__main__":
    unittest.main()
