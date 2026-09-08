"""Small correctness checks for the maths, storage and model reconstruction."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from manual_compression import (
    activation_parameters, quantize_activation, quantize_weight, pack_codes, unpack_codes,
    compress_weights, calibrate_activations, activation_plan, add_activation_quantization,
    activation_storage, save_compressed, load_compressed,
)


def tiny_model(config=None):
    # Includes a zero channel, depthwise conv, BN and first/last layers.
    torch.manual_seed(12)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU6(),
        nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False), nn.BatchNorm2d(4), nn.ReLU6(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).eval()
    with torch.no_grad():
        model[1].running_mean.copy_(torch.tensor([0.4, -0.2, 0.8, 0.1]))
        model[1].running_var.copy_(torch.tensor([0.3, 1.4, 0.2, 0.9]))
        model[3].weight[0].zero_()
    return model


class CompressionChecks(unittest.TestCase):
    def test_packing_with_known_bytes_and_padding(self):
        self.assertEqual(pack_codes(np.array([1, 2, 15]), 4), bytes([0x21, 0x0F]))
        for bits in (2, 4, 8):
            for length in (0, 1, 3, 7, 19, 1001):
                values = np.arange(length) % (2 ** bits)
                packed = pack_codes(values, bits)
                self.assertEqual(len(packed), (length * bits + 7) // 8)
                np.testing.assert_array_equal(unpack_codes(packed, bits, length), values)
        with self.assertRaises(ValueError):
            pack_codes(np.array([16]), 4)

    def test_weight_error_bound_and_zero_channel(self):
        weights = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.19, 0.77], [-0.04, 0.0, 0.03]])
        for bits in (2, 4, 8):
            restored, record = quantize_weight(weights, bits)
            self.assertTrue(torch.isfinite(restored).all())
            self.assertTrue(torch.equal(restored[0], weights[0]))
            error = (restored - weights).abs()
            self.assertTrue(torch.all(error <= record["scale"][:, None] / 2 + 1e-7))
        restored, record = quantize_weight(weights, 32)
        self.assertTrue(torch.equal(restored, weights))
        self.assertIsNone(record)

    def test_activation_zero_clipping_and_bypass(self):
        for low, high in ((0, 0), (2, 2), (-2, -2), (-3, 4), (0, 6)):
            scale, zero_point = activation_parameters(low, high, 4)
            values = torch.tensor([-100.0, 0.0, 100.0])
            result = quantize_activation(values, 4, scale, zero_point)
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(result[1].item(), 0)
            self.assertAlmostEqual(result[0].item(), -zero_point * scale, places=5)
            self.assertAlmostEqual(result[-1].item(), (15 - zero_point) * scale, places=5)
        self.assertIs(quantize_activation(values, 32, 1, 0), values)
        with self.assertRaises(ValueError):
            activation_parameters(float("nan"), 1, 4)

    def test_fold_and_uncompressed_bypass_preserve_predictions(self):
        baseline = tiny_model()
        before = {name: tensor.clone() for name, tensor in baseline.state_dict().items()}
        images = torch.randn(5, 3, 8, 8)
        bypass, _ = compress_weights(baseline, 32, fold_bn=False)
        torch.testing.assert_close(bypass(images), baseline(images), rtol=0, atol=0)
        folded, _ = compress_weights(baseline, 32, fold_bn=True)
        torch.testing.assert_close(folded(images), baseline(images), rtol=1e-5, atol=1e-6)
        for name, tensor in baseline.state_dict().items():
            self.assertTrue(torch.equal(before[name], tensor))

    def test_file_roundtrip_and_activation_accounting(self):
        baseline = tiny_model()
        images = torch.randn(9, 3, 8, 8)
        loader = DataLoader(TensorDataset(images, torch.zeros(9)), batch_size=4)
        for fold_bn in (False, True):
            for bits in (2, 4, 8, 32):
                model, records = compress_weights(baseline, bits, keep_first_last=True, fold_bn=fold_bn)
                self.assertNotIn("0.weight", records)
                self.assertNotIn("8.weight", records)
                ranges, seen = calibrate_activations(model, loader)
                self.assertEqual(seen, 9)
                self.assertTrue(all(not layer._forward_pre_hooks for layer in model.modules()))
                plan = activation_plan(model, ranges, bits, keep_first_last=True)
                sizes = activation_storage(plan)
                self.assertEqual(sizes["activation_fp32_bytes_per_image"], (3 * 8 * 8 + 4 * 8 * 8 + 4) * 4)
                self.assertEqual(sizes["activation_metadata_bytes"], 0 if bits == 32 else 8)
                add_activation_quantization(model, plan)
                with self.assertRaises(ValueError):
                    add_activation_quantization(model, plan)
                meta = {"model_config": {}, "fold_bn": fold_bn, "activations": plan,
                        "original_state_bytes": sum(t.numel() * t.element_size() for t in baseline.state_dict().values())}
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "test.mq"
                    summary = save_compressed(path, model, records, meta)
                    self.assertEqual(path.stat().st_size, summary["header_bytes"] + summary["tensor_payload_bytes"])
                    restored, _ = load_compressed(path, tiny_model)
                    torch.testing.assert_close(restored(images), model(images), rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
