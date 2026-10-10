import math
import unittest

import torch

from fedpeft.compression import (
    BlockTopKCompressor,
    ErrorFeedbackCompressor,
    TopKCompressor,
    largest_remainder_quotas,
)
from fedpeft.wire import decode_packet, decode_sparse_indices, encode_dense, sparse_k_for_cap


class WireTests(unittest.TestCase):
    def test_dense_roundtrip_uses_fp16_wire(self):
        vector = torch.tensor([0.1, -2.0, 3.25], dtype=torch.float32)
        packet = encode_dense(vector)
        decoded = decode_packet(packet)
        self.assertEqual(packet.nbytes, 13 + 2 * vector.numel())
        self.assertTrue(torch.equal(decoded, vector.half().float()))

    def test_sparse_cap_and_deterministic_ties(self):
        vector = torch.tensor([1.0, -1.0, 1.0, 0.5])
        cap = 21 + 2 * 6
        compressor = TopKCompressor(vector.numel(), cap)
        result = compressor.compress(vector)
        self.assertLessEqual(result.packet.nbytes, cap)
        decoded = decode_packet(result.packet)
        self.assertTrue(torch.equal(decoded, result.reconstructed.cpu()))
        self.assertEqual(decoded.nonzero().flatten().tolist(), [0, 1])
        self.assertEqual(decode_sparse_indices(result.packet).tolist(), [0, 1])

    def test_sparse_support_preserves_selected_values_that_round_to_zero(self):
        vector = torch.tensor([1.0, 1.0e-12, 0.0])
        compressor = TopKCompressor(vector.numel(), 21 + 2 * 6)
        result = compressor.compress(vector)
        self.assertEqual(decode_sparse_indices(result.packet).tolist(), [0, 1])
        self.assertEqual(result.reconstructed.nonzero().flatten().tolist(), [0])

    def test_error_feedback_conservation_after_quantization(self):
        vector = torch.linspace(-1, 1, 100)
        compressor = ErrorFeedbackCompressor(100, 21 + 10 * 6)
        residual = torch.zeros_like(vector)
        result, next_residual = compressor.compress_with_residual(vector, residual)
        self.assertTrue(torch.equal(vector + residual, result.reconstructed + next_residual))

    def test_bidirectional_error_feedback_conserves_joint_ab_vector(self):
        compressor = ErrorFeedbackCompressor(12, 21 + 4 * 6)
        client_residuals = [torch.zeros(12), torch.zeros(12)]
        client_inputs = [torch.zeros(12), torch.zeros(12)]
        client_transmitted = [torch.zeros(12), torch.zeros(12)]
        server_residual = torch.zeros(12)
        server_inputs = torch.zeros(12)
        server_transmitted = torch.zeros(12)
        rounds = (
            (
                torch.tensor([8.0, 0, 0, 7.0, 0, 0, -6.0, 0, 0, -5.0, 0, 0]),
                torch.tensor([0, 4.0, 0, 0, 3.0, 0, 0, -2.0, 0, 0, -1.0, 0]),
            ),
            (
                torch.tensor([0, 0, 2.0, 0, 0, 1.0, 0, 0, -4.0, 0, 0, -3.0]),
                torch.tensor([-2.0, 0, 0, 0, -3.0, 0, 1.0, 0, 0, 0, 4.0, 0]),
            ),
        )

        for deltas in rounds:
            decoded = []
            for client_id, delta in enumerate(deltas):
                result, client_residuals[client_id] = compressor.compress_with_residual(
                    delta, client_residuals[client_id]
                )
                self.assertLessEqual(result.packet.nbytes, compressor.byte_cap)
                client_inputs[client_id] += delta
                client_transmitted[client_id] += result.reconstructed
                self.assertTrue(
                    torch.allclose(
                        client_inputs[client_id],
                        client_transmitted[client_id] + client_residuals[client_id],
                    )
                )
                decoded.append(result.reconstructed)

            aggregate = 0.5 * (decoded[0] + decoded[1])
            downlink, server_residual = compressor.compress_with_residual(
                aggregate, server_residual
            )
            self.assertLessEqual(downlink.packet.nbytes, compressor.byte_cap)
            server_inputs += aggregate
            server_transmitted += downlink.reconstructed
            self.assertTrue(
                torch.allclose(server_inputs, server_transmitted + server_residual)
            )

    def test_k_solver_never_exceeds_cap(self):
        for cap in (27, 81, 1019):
            k = sparse_k_for_cap(10_000, cap)
            self.assertLessEqual(21 + 6 * k, cap)

    def test_full_lora_exact_wire_caps_for_roberta_qv(self):
        adapter_numel = 294_912
        cases = (
            (1_538, 0.1, 59_292, 9_363, 59_288),
            (1_538, 0.2, 118_585, 19_245, 118_580),
            (2_307, 0.1, 59_446, 9_133, 59_446),
            (2_307, 0.2, 118_892, 19_040, 118_888),
        )
        for head_numel, ratio, expected_cap, expected_k, expected_message in cases:
            reference = (
                encode_dense(torch.zeros(adapter_numel)).nbytes
                + encode_dense(torch.zeros(head_numel)).nbytes
            )
            cap = math.floor(reference * ratio)
            head_bytes = 13 + 2 * head_numel
            adapter_cap = cap - head_bytes
            k = sparse_k_for_cap(adapter_numel, adapter_cap)
            message = 21 + 6 * k + head_bytes
            self.assertEqual(cap, expected_cap)
            self.assertEqual(k, expected_k)
            self.assertEqual(message, expected_message)
            self.assertLessEqual(message, cap)
            self.assertLess(cap - message, 6)

    def test_block_quota_rule_is_exact_capacity_aware_and_deterministic(self):
        self.assertEqual(largest_remainder_quotas([1, 2, 7], 2), (0, 1, 1))
        self.assertEqual(largest_remainder_quotas([1, 2, 7], 3), (1, 1, 1))
        self.assertEqual(largest_remainder_quotas([1, 2, 7], 5), (1, 1, 3))
        self.assertEqual(largest_remainder_quotas([1, 2, 7], 7), (1, 2, 4))
        self.assertEqual(largest_remainder_quotas([1, 2, 7], 10), (1, 2, 7))
        # Equal fractional remainders are resolved by canonical block index.
        self.assertEqual(largest_remainder_quotas([4, 4, 4], 5), (2, 2, 1))
        with self.assertRaises(ValueError):
            largest_remainder_quotas([], 1)
        with self.assertRaises(ValueError):
            largest_remainder_quotas([2, 0], 1)
        with self.assertRaises(ValueError):
            largest_remainder_quotas([2, 3], 6)

    def test_block_topk_uses_local_stable_ties_and_one_sparse_packet(self):
        vector = torch.tensor([1.0, -1.0, 1.0, 0.5, -0.5, 0.5])
        compressor = BlockTopKCompressor([3, 3], 21 + 4 * 6)
        result = compressor.compress(vector)
        self.assertEqual(compressor.k, 4)
        self.assertEqual(compressor.quotas, (2, 2))
        self.assertEqual(result.packet.kind, "sparse")
        self.assertEqual(result.packet.transmitted_numel, compressor.k)
        self.assertEqual(result.packet.nbytes, 21 + 6 * compressor.k)
        self.assertEqual(decode_sparse_indices(result.packet).tolist(), [0, 1, 3, 4])
        self.assertTrue(torch.equal(decode_packet(result.packet), result.reconstructed))

    def test_block_topk_selected_fp16_zero_still_counts(self):
        vector = torch.tensor([1.0e-12, 0.0, 2.0, 1.0])
        compressor = BlockTopKCompressor([2, 2], 21 + 2 * 6)
        result = compressor.compress(vector)
        self.assertEqual(decode_sparse_indices(result.packet).tolist(), [0, 2])
        self.assertEqual(result.packet.transmitted_numel, 2)
        self.assertEqual(result.reconstructed.nonzero().flatten().tolist(), [2])

    def test_block_topk_has_same_k_as_global_topk_and_conserves_ef(self):
        cap = 21 + 5 * 6 + 4
        global_compressor = TopKCompressor(12, cap)
        block_compressor = BlockTopKCompressor([3, 4, 5], cap)
        self.assertEqual(block_compressor.k, global_compressor.k)
        self.assertEqual(block_compressor.quotas, (1, 2, 2))
        residual = torch.zeros(12)
        cumulative_input = torch.zeros(12)
        cumulative_transmitted = torch.zeros(12)
        for vector in (torch.linspace(-1, 1, 12), torch.linspace(1, -2, 12)):
            corrected = vector + residual
            result = block_compressor.compress(corrected)
            residual = corrected - result.reconstructed
            cumulative_input += vector
            cumulative_transmitted += result.reconstructed
            self.assertTrue(
                torch.allclose(cumulative_input, cumulative_transmitted + residual)
            )

    def test_roberta_block_quotas_preserve_frozen_exact_wire(self):
        adapter_numel = 147_456
        block_numels = [6_144] * 24
        cases = (
            (1_538, 0.1, 9_363, [391] * 3 + [390] * 21, 59_288),
            (1_538, 0.2, 19_245, [802] * 21 + [801] * 3, 118_580),
            (2_307, 0.1, 9_133, [381] * 13 + [380] * 11, 59_446),
            (2_307, 0.2, 19_040, [794] * 8 + [793] * 16, 118_888),
        )
        full_lora_adapter_numel = 294_912
        for head_numel, ratio, expected_k, expected_quotas, expected_message in cases:
            reference = (
                encode_dense(torch.zeros(full_lora_adapter_numel)).nbytes
                + encode_dense(torch.zeros(head_numel)).nbytes
            )
            message_cap = math.floor(reference * ratio)
            head_bytes = 13 + 2 * head_numel
            compressor = BlockTopKCompressor(block_numels, message_cap - head_bytes)
            self.assertEqual(compressor.logical_numel, adapter_numel)
            self.assertEqual(compressor.k, expected_k)
            self.assertEqual(compressor.quotas, tuple(expected_quotas))
            self.assertEqual(21 + 6 * compressor.k + head_bytes, expected_message)


if __name__ == "__main__":
    unittest.main()
