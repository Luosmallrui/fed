import copy
import unittest

import torch

from fedpeft.diagnostics import DIAGNOSTIC_RECIPE_VERSION, SparseDiagnostics
from fedpeft.wire import decode_packet, encode_sparse


NAMES = ["layer0.lora_B", "layer1.lora_B"]
SHAPES = [(2, 2), (2, 2)]
NUMELS = [4, 4]


def packet_values(corrected: torch.Tensor, indices: list[int]):
    index = torch.tensor(indices, dtype=torch.long)
    packet = encode_sparse(index, corrected.index_select(0, index), corrected.numel())
    transmitted = decode_packet(packet)
    residual = corrected.float() - transmitted
    return packet, transmitted, residual


def complete_round(tracker: SparseDiagnostics, round_index: int):
    if round_index == 1:
        corrected_clients = [
            torch.tensor([4.0, 0.0, 0.0, 0.0, 1.0e-12, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0]),
        ]
        supports = [[0, 4], [1, 5]]
    else:
        corrected_clients = [
            torch.tensor([1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -4.0, 0.0]),
        ]
        supports = [[0, 1], [1, 6]]

    decoded_clients = []
    for client_id, (corrected, support) in enumerate(zip(corrected_clients, supports)):
        packet, transmitted, residual = packet_values(corrected, support)
        tracker.observe_client(
            round_index,
            client_id,
            packet,
            raw_update=corrected,
            corrected_update=corrected,
            transmitted_update=transmitted,
            next_residual=residual,
        )
        decoded_clients.append(transmitted)

    aggregate = 0.5 * (decoded_clients[0] + decoded_clients[1])
    server_support = [0, 5] if round_index == 1 else [0, 1]
    packet, transmitted, residual = packet_values(aggregate, server_support)
    tracker.observe_server(
        round_index,
        packet,
        raw_update=aggregate,
        corrected_update=aggregate,
        transmitted_update=transmitted,
        next_residual=residual,
    )
    return tracker.finish_round(
        round_index,
        decoded_clients,
        [1.0, 1.0],
        aggregate,
        transmitted,
    )


class SparseDiagnosticTests(unittest.TestCase):
    def build(self) -> SparseDiagnostics:
        return SparseDiagnostics(NAMES, SHAPES, NUMELS, rank=2, num_clients=2)

    def test_actual_packet_support_drives_tensor_rank_and_zero_counts(self):
        tracker = self.build()
        output = complete_round(tracker, 1)
        client0 = output["clients"]["per_client"][0]["support"]
        self.assertEqual(output["recipe_version"], DIAGNOSTIC_RECIPE_VERSION)
        self.assertEqual(client0["selected_count"], 2)
        self.assertEqual(
            client0["selected_by_tensor"],
            {"layer0.lora_B": 1, "layer1.lora_B": 1},
        )
        self.assertEqual(client0["selected_by_rank"], [2, 0])
        self.assertEqual(client0["selected_values_rounded_to_zero"], 1)
        self.assertEqual(client0["zero_support_tensor_count"], 0)
        self.assertAlmostEqual(client0["cumulative_coverage"]["overall"], 0.25)
        self.assertEqual(client0["selection_age"]["overall"]["max"], 1.0)
        self.assertIsNone(client0["adjacent_support_jaccard"]["overall"])
        self.assertEqual(output["clients"]["selected_count_total"], 4)
        self.assertEqual(output["clients"]["selected_by_rank_total"], [2, 2])
        self.assertEqual(output["server"]["support"]["selected_count"], 2)
        self.assertIsNone(output["updates"]["consecutive_aggregate_cosine"])
        self.assertIsNone(output["updates"]["consecutive_transmitted_cosine"])

    def test_checkpoint_restore_reproduces_age_jaccard_cosine_and_ema(self):
        uninterrupted = self.build()
        complete_round(uninterrupted, 1)
        state = uninterrupted.state_dict()

        resumed = self.build()
        resumed.load_state_dict(state)
        expected = complete_round(uninterrupted, 2)
        actual = complete_round(resumed, 2)
        self.assertEqual(actual, expected)
        client0 = actual["clients"]["per_client"][0]["support"]
        self.assertAlmostEqual(client0["adjacent_support_jaccard"]["overall"], 1 / 3)
        self.assertAlmostEqual(client0["cumulative_coverage"]["overall"], 3 / 8)
        self.assertEqual(client0["selection_age"]["overall"]["max"], 2.0)
        self.assertIsNotNone(actual["updates"]["consecutive_aggregate_cosine"])
        self.assertIsNotNone(actual["updates"]["consecutive_transmitted_cosine"])

    def test_checkpoint_rejects_wrong_recipe_layout_and_missing_state(self):
        tracker = self.build()
        complete_round(tracker, 1)
        state = tracker.state_dict()

        wrong_recipe = copy.deepcopy(state)
        wrong_recipe["recipe_version"] = "v2"
        with self.assertRaisesRegex(ValueError, "recipe version"):
            self.build().load_state_dict(wrong_recipe)

        wrong_layout = copy.deepcopy(state)
        wrong_layout["layout_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "layout differs"):
            self.build().load_state_dict(wrong_layout)

        missing = copy.deepcopy(state)
        del missing["server_support"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.build().load_state_dict(missing)

    def test_midround_checkpoint_and_incomplete_round_are_rejected(self):
        tracker = self.build()
        corrected = torch.arange(8, dtype=torch.float32)
        packet, transmitted, residual = packet_values(corrected, [6, 7])
        tracker.observe_client(
            1,
            0,
            packet,
            corrected,
            corrected,
            transmitted,
            residual,
        )
        with self.assertRaisesRegex(RuntimeError, "mid-round"):
            tracker.state_dict()
        with self.assertRaisesRegex(ValueError, "missing one or more client"):
            tracker.finish_round(1, [transmitted, transmitted], [1, 1], transmitted, transmitted)

    def test_diagnostics_do_not_mutate_wire_payload(self):
        tracker = self.build()
        corrected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        packet, transmitted, residual = packet_values(corrected, [0, 5])
        payload = packet.payload
        tracker.observe_client(1, 0, packet, corrected, corrected, transmitted, residual)
        self.assertEqual(packet.payload, payload)
        self.assertEqual(packet.nbytes, 21 + 2 * 6)

    def test_zero_vectors_use_frozen_finite_conventions(self):
        tracker = SparseDiagnostics(
            ["layer0.lora_B"], [(2, 2)], [4], rank=2, num_clients=1
        )
        zero = torch.zeros(4)
        packet, transmitted, residual = packet_values(zero, [0, 1])
        tracker.observe_client(1, 0, packet, zero, zero, transmitted, residual)
        tracker.observe_server(1, packet, zero, zero, transmitted, residual)
        output = tracker.finish_round(1, [transmitted], [1], zero, transmitted)
        self.assertEqual(output["updates"]["aggregate_to_transmitted_cosine"], 0.0)
        self.assertEqual(output["updates"]["client_cancellation_ratio"], 1.0)
        self.assertEqual(
            output["server"]["residual"]["overall"]["residual_to_update_ratio"],
            0.0,
        )
        self.assertEqual(
            output["server"]["residual"]["overall"]["drained_residual_energy"],
            1.0,
        )

    def test_layout_requires_canonical_lora_b_rank_axis(self):
        with self.assertRaisesRegex(ValueError, "canonical order"):
            SparseDiagnostics(
                ["z.lora_B", "a.lora_B"], [(2, 2), (2, 2)], [4, 4], 2, 1
            )
        with self.assertRaisesRegex(ValueError, "non-LoRA-B"):
            SparseDiagnostics(["a.lora_A"], [(2, 2)], [4], 2, 1)
        with self.assertRaisesRegex(ValueError, "final dimension"):
            SparseDiagnostics(["a.lora_B"], [(2, 3)], [6], 2, 1)


if __name__ == "__main__":
    unittest.main()
