import unittest

import torch

from fedpeft.metrics import (
    confusion_matrix,
    metrics_from_confusion,
    primary_metric,
    primary_metric_name,
)


class GluePairMetricTests(unittest.TestCase):
    def test_predictions_flow_through_binary_f1_and_composite(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        predictions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

        matrix = confusion_matrix(predictions, labels, num_labels=2)
        self.assertTrue(torch.equal(matrix, torch.tensor([[2, 1], [2, 3]])))

        values = metrics_from_confusion(matrix)
        self.assertAlmostEqual(values["accuracy"], 5 / 8)
        self.assertAlmostEqual(values["f1"], 2 / 3)
        expected_composite = 0.5 * (5 / 8 + 2 / 3)
        self.assertAlmostEqual(values["glue_score"], expected_composite)
        for task in ("mrpc", "qqp"):
            for split in ("train_holdout", "validation"):
                self.assertAlmostEqual(
                    primary_metric(task, {split: values}),
                    expected_composite,
                )

    def test_binary_f1_and_glue_score_use_positive_class(self):
        values = metrics_from_confusion(torch.tensor([[8, 2], [1, 9]]))
        self.assertAlmostEqual(values["accuracy"], 17 / 20)
        self.assertAlmostEqual(values["f1"], 18 / 21)
        self.assertAlmostEqual(
            values["glue_score"], 0.5 * (17 / 20 + 18 / 21)
        )

    def test_binary_f1_is_zero_without_positive_predictions(self):
        values = metrics_from_confusion(torch.tensor([[10, 0], [5, 0]]))
        self.assertEqual(values["f1"], 0.0)

    def test_mrpc_and_qqp_use_same_primary_for_holdout_and_validation(self):
        values = {"accuracy": 0.8, "f1": 0.6, "macro_f1": 0.7}
        for task in ("mrpc", "qqp"):
            self.assertEqual(primary_metric_name(task), "glue_accuracy_f1_mean")
            self.assertAlmostEqual(
                primary_metric(task, {"train_holdout": values}), 0.7
            )
            self.assertAlmostEqual(
                primary_metric(task, {"validation": values}), 0.7
            )

    def test_sst2_and_mnli_primary_metrics_remain_accuracy_based(self):
        self.assertEqual(primary_metric_name("sst2"), "accuracy")
        self.assertAlmostEqual(
            primary_metric(
                "sst2",
                {"validation": {"accuracy": 0.81, "f1": 0.99}},
            ),
            0.81,
        )
        self.assertAlmostEqual(
            primary_metric(
                "sst2",
                {"train_holdout": {"accuracy": 0.82, "f1": 0.01}},
            ),
            0.82,
        )

        self.assertEqual(
            primary_metric_name("mnli"),
            "mean_matched_mismatched_accuracy",
        )
        self.assertAlmostEqual(
            primary_metric(
                "mnli",
                {
                    "validation_matched": {"accuracy": 0.84},
                    "validation_mismatched": {"accuracy": 0.78},
                },
            ),
            0.81,
        )
        self.assertAlmostEqual(
            primary_metric(
                "mnli",
                {"train_holdout": {"accuracy": 0.83, "f1": 0.01}},
            ),
            0.83,
        )


if __name__ == "__main__":
    unittest.main()
