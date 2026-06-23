import unittest

from src.gtr.metrics import exact_match


class MetricTests(unittest.TestCase):
    def test_exact_match_multilabel_sets(self) -> None:
        gold = [["A", "B"], ["C"]]
        pred = [["B", "A"], ["D"]]
        self.assertEqual(exact_match(gold, pred), 0.5)


if __name__ == "__main__":
    unittest.main()
