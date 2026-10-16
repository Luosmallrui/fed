import unittest

import numpy as np

from fedpeft.config import SUPPORTED_TASKS
from fedpeft.data import TASK_FIELDS, TASK_NUM_LABELS, TASK_VALIDATION_SPLITS, create_dirichlet_partition
from fedpeft.metrics import primary_metric, primary_metric_name
from fedpeft.utils import sha256_arrays


def reference_partition(labels, num_clients, alpha, holdout_fraction, min_client_examples, seed, max_attempts=1000):
    """The partition code before grouped partitions were added, kept to pin the GLUE partitions."""
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    allocations = None
    for _ in range(max_attempts):
        candidate = [[] for _ in range(num_clients)]
        for label in classes:
            class_indices = np.flatnonzero(labels == label)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(class_indices), proportions)
            cursor = 0
            for client_id, count in enumerate(counts):
                candidate[client_id].extend(class_indices[cursor : cursor + count].tolist())
                cursor += count
        if min(len(items) for items in candidate) >= min_client_examples:
            allocations = candidate
            break
    train, holdout = [], []
    for items in allocations:
        values = np.asarray(items, dtype=np.int64)
        rng.shuffle(values)
        n_holdout = int(round(len(values) * holdout_fraction))
        n_holdout = min(max(n_holdout, 1 if holdout_fraction > 0 else 0), len(values) - 1)
        holdout.append(np.sort(values[:n_holdout]))
        train.append(np.sort(values[n_holdout:]))
    return sha256_arrays([array for pair in zip(train, holdout) for array in pair])


class WebTaskTests(unittest.TestCase):
    def test_tasks_are_registered(self):
        for task, labels in (("banking77", 77), ("moderation", 2)):
            self.assertIn(task, SUPPORTED_TASKS)
            self.assertEqual(TASK_NUM_LABELS[task], labels)
            self.assertEqual(TASK_FIELDS[task], ("text", None))
        self.assertEqual(TASK_VALIDATION_SPLITS["moderation"], ["validation_civil", "validation_twitter"])

    def test_ungrouped_partition_is_unchanged(self):
        rng = np.random.default_rng(0)
        for seed, classes in ((1001, 3), (1002, 2), (42, 77)):
            labels = rng.integers(0, classes, size=4000)
            partition = create_dirichlet_partition(labels, 10, 0.3, 0.1, 24, seed)
            self.assertEqual(partition.digest, reference_partition(labels, 10, 0.3, 0.1, 24, seed))

    def test_grouped_partition_keeps_each_client_on_one_platform(self):
        rng = np.random.default_rng(1)
        groups = np.repeat([0, 1], [3000, 1000])
        labels = (rng.random(4000) < np.where(groups == 0, 0.08, 0.33)).astype(np.int64)
        partition = create_dirichlet_partition(labels, 10, 0.3, 0.1, 24, 1001, groups=groups)
        seen = np.concatenate([np.concatenate(pair) for pair in zip(partition.train, partition.holdout)])
        self.assertEqual(sorted(seen.tolist()), list(range(4000)))
        for client, (train, holdout) in enumerate(zip(partition.train, partition.holdout)):
            members = np.concatenate([train, holdout])
            self.assertTrue(np.all(groups[members] == (0 if client < 5 else 1)), client)
        again = create_dirichlet_partition(labels, 10, 0.3, 0.1, 24, 1001, groups=groups)
        self.assertEqual(partition.digest, again.digest)
        with self.assertRaisesRegex(ValueError, "divided evenly"):
            create_dirichlet_partition(labels, 9, 0.3, 0.1, 24, 1001, groups=groups)

    def test_budget_scope_keeps_the_message_rule_and_moves_the_head_out_of_the_adapter_rule(self):
        import math
        from fedpeft.engine import _wire_caps

        message = {"communication": {}}
        adapter = {"communication": {"budget_scope": "adapter"}}
        for ref, head, ratio in ((589_837, 3_093, 0.1), (589_837, 118_439, 0.05), (589_837, 3_093, 0.2)):
            expected = math.floor((ref + head) * ratio)
            self.assertEqual(_wire_caps(message, ref, head, ratio), (expected, expected - head))
            self.assertEqual(_wire_caps(adapter, ref, head, ratio), (math.floor(ref * ratio) + head, math.floor(ref * ratio)))
        self.assertLess(_wire_caps(message, 589_837, 118_439, 0.1)[1], 0)  # why banking77 needs the adapter scope

    def test_moderation_metric_weights_platforms_equally(self):
        self.assertEqual(primary_metric_name("moderation"), "mean_platform_macro_f1")
        splits = {
            "validation_civil": {"accuracy": 0.95, "macro_f1": 0.70},
            "validation_twitter": {"accuracy": 0.80, "macro_f1": 0.76},
        }
        self.assertAlmostEqual(primary_metric("moderation", splits), 0.73)
        self.assertAlmostEqual(primary_metric("moderation", {"train_holdout": {"accuracy": 0.9, "macro_f1": 0.6}}), 0.6)
        self.assertAlmostEqual(primary_metric("banking77", {"validation": {"accuracy": 0.9, "macro_f1": 0.8}}), 0.9)


if __name__ == "__main__":
    unittest.main()
