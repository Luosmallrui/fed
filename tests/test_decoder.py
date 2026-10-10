import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2Model

from fedpeft.adapters import DecoderLastTokenClassifier, FixedALinear, build_model, is_decoder_model
from fedpeft.data import DECODER_TEMPLATES, TASK_FIELDS, is_decoder_tokenizer


def tiny_qwen(directory: Path) -> Path:
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    Qwen2Model(config).save_pretrained(directory)
    return directory


class DecoderClassifierTests(unittest.TestCase):
    def test_templates_cover_every_task_and_tokenizer_detection(self):
        self.assertEqual(set(DECODER_TEMPLATES), set(TASK_FIELDS))
        for task, (first, second) in TASK_FIELDS.items():
            fields = DECODER_TEMPLATES[task].count("{")
            self.assertEqual(fields, 1 if second is None else 2, task)
        self.assertTrue(is_decoder_tokenizer(SimpleNamespace(cls_token=None, sep_token=None)))
        self.assertFalse(is_decoder_tokenizer(SimpleNamespace(cls_token="<s>", sep_token="</s>")))

    def test_last_token_head_reads_the_final_non_padding_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = tiny_qwen(Path(tmp))
            self.assertTrue(is_decoder_model(str(path)))
            model = DecoderLastTokenClassifier(str(path), 3, torch.float32).eval()
            ids = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 10, 11, 12]])
            mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
            out = model(labels=torch.tensor([0, 2]), input_ids=ids, attention_mask=mask, token_type_ids=torch.zeros_like(ids))
            hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            expected = model.classifier(torch.stack([hidden[0, 2], hidden[1, 4]]))
            self.assertTrue(torch.allclose(out.logits, expected, atol=1e-6))
            self.assertTrue(torch.isfinite(out.loss))
            # padding after the last real token does not change its logits (causal attention)
            short = model(input_ids=ids[:1, :3], attention_mask=mask[:1, :3])
            self.assertTrue(torch.allclose(short.logits, out.logits[:1], atol=1e-5))

    def test_build_model_targets_decoder_q_and_v(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = tiny_qwen(Path(tmp) / "qwen")
            bundle = build_model(
                model_path=str(path),
                num_labels=3,
                method="random_a_ef",
                rank=4,
                alpha=8.0,
                dropout=0.0,
                target_suffixes=["self_attn.q_proj", "self_attn.v_proj"],
                basis_seed=1,
                lora_xs_init_std=1e-5,
                spectral_cache_path=str(Path(tmp) / "spectral.pt"),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            fixed = [name for name, module in bundle.model.named_modules() if isinstance(module, FixedALinear)]
            self.assertEqual(len(fixed), 4)
            self.assertTrue(all(name.endswith(("q_proj", "v_proj")) for name in fixed))
            trainable = [name for name, p in bundle.model.named_parameters() if p.requires_grad]
            self.assertTrue(all(name.startswith("classifier.") or "lora_B" in name for name in trainable))
            v_proj = bundle.model.encoder.layers[0].self_attn.v_proj
            self.assertEqual(tuple(v_proj.fixed_A.shape), (4, 32))


if __name__ == "__main__":
    unittest.main()
