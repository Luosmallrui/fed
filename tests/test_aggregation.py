import unittest

import torch

from fedpeft.aggregation import weighted_average


class AggregationTests(unittest.TestCase):
    def test_matches_manual_weighted_average(self):
        values = [torch.tensor([1.0, 2.0]), torch.tensor([5.0, 10.0])]
        result = weighted_average(values, [1, 3])
        self.assertTrue(torch.allclose(result, torch.tensor([4.0, 8.0])))

    def test_missing_sparse_coordinate_is_zero_not_renormalized(self):
        values = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 4.0])]
        result = weighted_average(values, [1, 1])
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()

