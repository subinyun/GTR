import unittest

from src.gtr.coordinates import calibrate_logistic, project


class CoordinateTests(unittest.TestCase):
    def test_project_dot_product(self) -> None:
        self.assertEqual(project([1.0, 2.0], [3.0, 4.0]), 11.0)

    def test_calibrate_logistic_range(self) -> None:
        q = calibrate_logistic(0.0, alpha=1.0, beta=0.0)
        self.assertTrue(0.0 < q < 1.0)
        self.assertEqual(q, 0.5)


if __name__ == "__main__":
    unittest.main()
