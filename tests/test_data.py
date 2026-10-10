import unittest

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from fedpeft.data import (
    TASK_FIELDS,
    TASK_NUM_LABELS,
    TASK_VALIDATION_SPLITS,
    create_dirichlet_partition,
    load_or_create_partition,
)


class PartitionTests(unittest.TestCase):
    def test_mrpc_and_qqp_task_specs_are_registered(self):
        self.assertEqual(TASK_FIELDS["mrpc"], ("sentence1", "sentence2"))
        self.assertEqual(TASK_FIELDS["qqp"], ("question1", "question2"))
        self.assertEqual(TASK_NUM_LABELS["mrpc"], 2)
        self.assertEqual(TASK_NUM_LABELS["qqp"], 2)
        self.assertEqual(TASK_VALIDATION_SPLITS["mrpc"], ["validation"])
        self.assertEqual(TASK_VALIDATION_SPLITS["qqp"], ["validation"])

    def test_partition_is_deterministic_complete_and_disjoint(self):
        labels = np.repeat(np.arange(3), 1000)
        first = create_dirichlet_partition(labels, 8, 0.3, 0.1, 20, 401)
        second = create_dirichlet_partition(labels, 8, 0.3, 0.1, 20, 401)
        third = create_dirichlet_partition(labels, 8, 0.3, 0.1, 20, 402)
        self.assertEqual(first.digest, second.digest)
        self.assertNotEqual(first.digest, third.digest)
        merged = np.concatenate([x for pair in zip(first.train, first.holdout) for x in pair])
        self.assertEqual(len(merged), len(labels))
        self.assertEqual(len(np.unique(merged)), len(labels))

    def test_cached_partition_rejects_changed_labels_or_parameters(self):
        labels = np.repeat(np.arange(2), 100)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "partition.npz"
            load_or_create_partition(labels, path, 4, 0.3, 0.1, 10, 401)
            changed = labels.copy()
            changed[0] = 1
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_or_create_partition(changed, path, 4, 0.3, 0.1, 10, 401)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_or_create_partition(labels, path, 4, 0.3, 0.2, 10, 401)


if __name__ == "__main__":
    unittest.main()
