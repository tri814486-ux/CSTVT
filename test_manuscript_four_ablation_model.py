"""CPU-only checks for isolated manuscript ablations.

Run: python test_manuscript_four_ablation_model.py
"""
import hashlib
from pathlib import Path
import unittest

import torch
from torch import nn

try:
    from .manuscript_primary_model import ManuscriptCSTVT, ManuscriptSTBlock
    from .manuscript_four_ablation_model import build_manuscript_ablation, _IsolatedAblationBlock
except ImportError:
    from manuscript_primary_model import ManuscriptCSTVT, ManuscriptSTBlock
    from manuscript_four_ablation_model import build_manuscript_ablation, _IsolatedAblationBlock


SETTINGS = dict(image_size=32, channels=(4, 8, 12, 16), depth=2,
                heads=2, grid_side=2, iterations=2, expansion=2, num_classes=5)
SOURCE_SHA256 = "aeddb94b644441cd3401c235b8c40f7daebf26ee294ee31fbb9adab1f36bbd85"


class _ZeroOutput(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _RecordingZeroAttention(_ZeroOutput):
    def forward(self, x):
        self.observed = x.detach().clone()
        return super().forward(x)


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_source_unchanged(self):
        source = Path(__file__).with_name("manuscript_primary_model.py")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), SOURCE_SHA256)

    def test_exact_shared_initialization_and_parameter_removals(self):
        models = {key: build_manuscript_ablation(key, seed=42, **SETTINGS)
                  for key in ("full", "vit", "no_cpe", "no_dwconv")}
        full = models["full"].state_dict()
        for key, model in models.items():
            state = model.state_dict()
            self.assertFalse(set(state) - set(full))
            for name, value in state.items():
                self.assertTrue(torch.equal(value, full[name]), (key, name))
            removed = set(full) - set(state)
            expected = ({name for name in full if ".cpe." in name} if key == "no_cpe"
                        else {name for name in full if ".ffn.3." in name} if key == "no_dwconv"
                        else set())
            self.assertEqual(removed, expected)
            self.assertIn("absolute_position", state)

    def test_full_matches_authoritative_original(self):
        torch.manual_seed(42)
        original = ManuscriptCSTVT(**SETTINGS).eval()
        full = build_manuscript_ablation("full", seed=42, **SETTINGS).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(original(images), full(images)))

    def test_cpe_and_dwconv_are_independent(self):
        vit = build_manuscript_ablation("vit", seed=42, **SETTINGS)
        no_cpe = build_manuscript_ablation("no_cpe", seed=42, **SETTINGS)
        no_dwconv = build_manuscript_ablation("no_dwconv", seed=42, **SETTINGS)
        for block in vit.blocks:
            self.assertIsInstance(block.cpe, nn.Conv2d)
            self.assertIsNone(block.sts)
            self.assertIsInstance(block.ffn[3], nn.Conv2d)
        for block in no_cpe.blocks:
            self.assertIsNone(block.cpe)
            self.assertIsNotNone(block.sts)
            self.assertIsInstance(block.ffn[3], nn.Conv2d)
        for block in no_dwconv.blocks:
            self.assertIsInstance(block.cpe, nn.Conv2d)
            self.assertIsNotNone(block.sts)
            self.assertEqual([type(layer) for layer in block.ffn],
                             [nn.BatchNorm2d, nn.Conv2d, nn.GELU, nn.Identity, nn.Conv2d])

    def test_no_cpe_does_not_double_features(self):
        original = ManuscriptSTBlock(width=16, heads=2, grid_side=2)
        block = _IsolatedAblationBlock(original, remove_cpe=True)
        block.mhsa = _ZeroOutput()
        block.ffn = _ZeroOutput()
        features, cls = torch.randn(2, 16, 8, 8), torch.randn(2, 1, 16)
        visual_out, cls_out = block(features, cls)
        self.assertTrue(torch.equal(visual_out, features))
        self.assertTrue(torch.equal(cls_out, cls))

    def test_vit_attends_all_normalized_visual_tokens(self):
        model = build_manuscript_ablation("vit", seed=42, **SETTINGS)
        block = model.blocks[0]
        recorder = _RecordingZeroAttention()
        block.mhsa = recorder
        features, cls = torch.randn(2, 16, 8, 8), torch.randn(2, 1, 16)
        block(features, cls)
        enhanced = features + block.cpe(features)
        expected = block.norm(enhanced.permute(0, 2, 3, 1)).reshape(2, 64, 16)
        self.assertEqual(tuple(recorder.observed.shape), (2, 65, 16))
        self.assertTrue(torch.equal(recorder.observed[:, :1], cls))
        self.assertTrue(torch.equal(recorder.observed[:, 1:], expected))

    def test_output_gradients_and_original_last_ffn_semantics(self):
        images = torch.randn(2, 3, 32, 32)
        for key in ("full", "vit", "no_cpe", "no_dwconv"):
            model = build_manuscript_ablation(key, seed=42, **SETTINGS).train()
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (2, 5))
            self.assertTrue(torch.isfinite(logits).all())
            nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
            self.assertIsNotNone(model.absolute_position.grad)
            self.assertGreater(float(model.absolute_position.grad.abs().sum()), 0)
            self.assertGreater(float(model.blocks[0].ffn[1].weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.embedding[0][0].weight.grad.abs().sum()), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all()
                                for p in model.parameters() if p.grad is not None))
            self.assertIsNone(model.blocks[-1].ffn[1].weight.grad)

    def test_train_eval_mode_and_bn_statistics(self):
        images = torch.randn(2, 3, 32, 32)
        for key in ("full", "vit", "no_cpe", "no_dwconv"):
            model = build_manuscript_ablation(key, seed=42, **SETTINGS).eval()
            bn = model.embedding[0][1]
            before = bn.running_mean.clone()
            with torch.no_grad():
                first, second = model(images), model(images)
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(before, bn.running_mean))
            self.assertFalse(any(module.training for module in model.modules()))
            model.train()
            model(images)
            self.assertTrue(all(module.training for module in model.modules()))
            self.assertFalse(torch.equal(before, bn.running_mean))

    def test_seed_scope_and_invalid_key(self):
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        build_manuscript_ablation("full", seed=42, **SETTINGS)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaises(ValueError):
            build_manuscript_ablation("incorrect", **SETTINGS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
