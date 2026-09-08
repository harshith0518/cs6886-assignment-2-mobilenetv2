"""Our own quantization maths and bit packing. No ready-made compression APIs."""
import copy
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Reading behind the scale/round idea: https://arxiv.org/abs/1712.05877
# Hooks: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html


def check_bits(bits):
    if bits not in (2, 4, 8, 32):
        raise ValueError("Use 2, 4, 8 or 32 bits. 32 means leave it in FP32.")


def quantize_weight(weight, bits):
    # Give each output channel its own ruler, useful for depthwise convolutions.
    check_bits(bits)
    if bits == 32:
        return weight.clone(), None
    weight = weight.detach().float()
    limit = 2 ** (bits - 1) - 1
    channel_max = weight.reshape(weight.shape[0], -1).abs().amax(dim=1)
    scale = torch.where(channel_max > 0, channel_max / limit, torch.ones_like(channel_max))
    shape = (-1,) + (1,) * (weight.ndim - 1)
    signed = torch.round(weight / scale.view(shape)).clamp(-limit, limit)
    restored = signed * scale.view(shape)
    # Shift signed codes to nonnegative values before packing them into bytes.
    record = {"bits": bits, "codes": (signed + limit).to(torch.uint8).cpu(),
              "scale": scale.cpu()}
    return restored, record


def activation_parameters(low, high, bits):
    check_bits(bits)
    if bits == 32:
        raise ValueError("FP32 activations don't need a scale.")
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("Activation range must be finite and ordered.")
    # Include zero, even when all observed values have the same sign.
    low, high = min(low, 0.0), max(high, 0.0)
    limit = 2 ** bits - 1
    scale = (high - low) / limit if high > low else 1.0
    zero_point = max(0, min(limit, round(-low / scale)))
    return scale, zero_point


def quantize_activation(values, bits, scale, zero_point):
    # Like replacing a long decimal with the nearest mark on a short ruler.
    check_bits(bits)
    if bits == 32:
        return values
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be positive and finite.")
    codes = torch.round(values / scale + zero_point).clamp(0, 2 ** bits - 1)
    return (codes - zero_point) * scale


def pack_codes(codes, bits):
    # At 4 bits, two numbers share one byte. At 2 bits, four share it.
    check_bits(bits)
    if bits == 32:
        raise ValueError("Pack integer codes, not FP32 values.")
    values = np.asarray(codes).reshape(-1)
    if np.any(values < 0) or np.any(values >= 2 ** bits) or np.any(values != np.floor(values)):
        raise ValueError("A code is outside the chosen bit range.")
    values = values.astype(np.uint8)
    per_byte = 8 // bits
    packed = np.zeros((len(values) + per_byte - 1) // per_byte, dtype=np.uint8)
    for slot in range(per_byte):
        part = values[slot::per_byte]
        packed[:len(part)] |= part << (slot * bits)
    return packed.tobytes()


def unpack_codes(data, bits, count):
    check_bits(bits)
    if bits == 32 or count < 0:
        raise ValueError("Invalid packed tensor description.")
    per_byte = 8 // bits
    if len(data) != (count + per_byte - 1) // per_byte:
        raise ValueError("Packed tensor length doesn't match its shape.")
    packed = np.frombuffer(data, dtype=np.uint8)
    values = np.empty(len(packed) * per_byte, dtype=np.uint8)
    for slot in range(per_byte):
        values[slot::per_byte] = (packed >> (slot * bits)) & (2 ** bits - 1)
    return values[:count].copy()


@torch.no_grad()
def fold_batch_norm(model):
    # In eval mode BN is just multiply + add. Merge that into the previous conv.
    if model.training:
        raise ValueError("BatchNorm folding is only for an evaluation model.")
    for child in model.children():
        fold_batch_norm(child)
    if isinstance(model, nn.Sequential):
        children = list(model.named_children())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
                continue
            factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
            conv.weight.mul_(factor.reshape(-1, 1, 1, 1))
            conv.bias = nn.Parameter((bias - bn.running_mean) * factor + bn.bias)
            model._modules[bn_name] = nn.Identity()
    return model


def compression_layers(model):
    return {name: layer for name, layer in model.named_modules()
            if isinstance(layer, (nn.Conv2d, nn.Linear))}


def layer_bits(model, bits, keep_first_last):
    check_bits(bits)
    names = list(compression_layers(model))
    protected = {names[0], names[-1]} if keep_first_last and names else set()
    return {name: 32 if name in protected else bits for name in names}


@torch.no_grad()
def compress_weights(baseline, bits=8, keep_first_last=True, fold_bn=True):
    # Work on a copy. The baseline and the running training job stay separate.
    model = copy.deepcopy(baseline).eval().cpu()
    if fold_bn:
        fold_batch_norm(model)
    choices = layer_bits(model, bits, keep_first_last)
    records = {}
    for name, layer in compression_layers(model).items():
        restored, record = quantize_weight(layer.weight, choices[name])
        layer.weight.copy_(restored)
        if record is not None:
            records[name + ".weight"] = record
    return model, records


@torch.inference_mode()
def calibrate_activations(model, loader):
    # Observe TRAINING images without augmentation. Validation/test never set ranges.
    if model.training:
        raise ValueError("Call model.eval() before calibration.")
    ranges, handles = {}, []

    def observer(name):
        def observe(layer, inputs):
            values = inputs[0]
            shape = list(values.shape[1:])
            old = ranges.get(name, {"min": float("inf"), "max": -float("inf"), "shape": shape})
            if old["shape"] != shape:
                raise ValueError("Use a fixed image shape for this activation estimate.")
            old.update(min=min(old["min"], values.min().item()),
                       max=max(old["max"], values.max().item()))
            ranges[name] = old
        return observe

    for name, layer in compression_layers(model).items():
        handles.append(layer.register_forward_pre_hook(observer(name)))
    seen = 0
    try:
        device = next(model.parameters()).device
        for images, _ in loader:
            model(images.to(device))
            seen += len(images)
    finally:
        for handle in handles:
            handle.remove()
    if not seen:
        raise ValueError("Calibration needs at least one training image.")
    return ranges, seen


def activation_plan(model, ranges, bits=8, keep_first_last=True):
    plan = {}
    for name, chosen_bits in layer_bits(model, bits, keep_first_last).items():
        item = {**ranges[name], "bits": chosen_bits}
        if chosen_bits != 32:
            item["scale"], item["zero_point"] = activation_parameters(item["min"], item["max"], chosen_bits)
        plan[name] = item
    return plan


def add_activation_quantization(model, plan):
    # Quantize inputs to Conv/Linear layers. Maths still runs in floating point.
    if hasattr(model, "_compression_handles"):
        raise ValueError("Activation quantization is already attached to this model.")

    def hook(item):
        def apply(layer, inputs):
            rounded = quantize_activation(inputs[0], item["bits"], item["scale"], item["zero_point"])
            return (rounded,) + inputs[1:]
        return apply

    handles = []
    for name, layer in compression_layers(model).items():
        if plan[name]["bits"] != 32:
            handles.append(layer.register_forward_pre_hook(hook(plan[name])))
    model._compression_handles = handles
    return model


def activation_storage(plan):
    # Sum the selected input tensors for ONE image; this is not peak GPU memory.
    original, packed, metadata = 0, 0, 0
    for item in plan.values():
        count = math.prod(item["shape"])
        original += count * 4
        packed += (count * item["bits"] + 7) // 8
        if item["bits"] != 32:
            metadata += 8  # one FP32 scale + one int32 zero point per location
    return {"activation_fp32_bytes_per_image": original,
            "activation_packed_bytes_per_image": packed,
            "activation_metadata_bytes": metadata,
            "activation_compression_ratio": original / (packed + metadata),
            "activation_measurement": "sum of Conv/Linear input occurrences per image; packed estimate, not peak memory"}


def save_compressed(path, model, weight_records, metadata):
    # A tiny custom file: magic + JSON header + packed weights/scales/raw tensors.
    # No zip/compression library. The header itself is counted in the file size.
    entries, pieces, offset = [], [], 0
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().numpy()
        entry = {"name": name, "shape": list(array.shape), "offset": offset}
        if name in weight_records:
            record = weight_records[name]
            codes = pack_codes(record["codes"].numpy(), record["bits"])
            scales = record["scale"].numpy().astype("<f4").tobytes()
            piece = codes + scales
            entry.update(bits=record["bits"], codes_bytes=len(codes), scale_count=len(record["scale"]))
        else:
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
            piece = array.tobytes()
            entry.update(bits=32, dtype=array.dtype.str)
        entry["bytes"] = len(piece)
        entries.append(entry)
        pieces.append(piece)
        offset += len(piece)
    header = json.dumps({"format_version": 1, "metadata": metadata, "tensors": entries},
                        separators=(",", ":"), allow_nan=False).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(b"MQ01" + struct.pack("<Q", len(header)) + header)
        for piece in pieces:
            stream.write(piece)
    temporary.replace(path)
    total = path.stat().st_size
    return {"model_file_bytes": total, "model_size_mb": total / 1e6,
            "tensor_payload_bytes": offset, "header_bytes": 12 + len(header),
            "weight_compression_ratio": metadata["original_state_bytes"] / total}


def load_compressed(path, model_builder):
    # Rebuild the model from the packed file alone, then restore activation hooks.
    with Path(path).open("rb") as stream:
        if stream.read(4) != b"MQ01":
            raise ValueError("This isn't one of our compressed files.")
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
        payload = stream.read()
    if header["format_version"] != 1:
        raise ValueError("Unknown compression format version.")
    metadata = header["metadata"]
    model = model_builder(metadata["model_config"]).eval().cpu()
    if metadata["fold_bn"]:
        fold_batch_norm(model)
    state = {}
    for entry in header["tensors"]:
        start = entry["offset"]
        piece = payload[start:start + entry["bytes"]]
        if len(piece) != entry["bytes"]:
            raise ValueError("Compressed file ended halfway through a tensor.")
        if entry["bits"] != 32:
            count = math.prod(entry["shape"])
            codes = unpack_codes(piece[:entry["codes_bytes"]], entry["bits"], count)
            scales = np.frombuffer(piece[entry["codes_bytes"]:], dtype="<f4").copy()
            if len(scales) != entry["scale_count"]:
                raise ValueError("Missing weight scales.")
            signed = torch.from_numpy(codes.reshape(entry["shape"])).float() - (2 ** (entry["bits"] - 1) - 1)
            shape = (-1,) + (1,) * (len(entry["shape"]) - 1)
            state[entry["name"]] = signed * torch.from_numpy(scales).view(shape)
        else:
            array = np.frombuffer(piece, dtype=entry["dtype"]).reshape(entry["shape"]).copy()
            state[entry["name"]] = torch.from_numpy(array)
    model.load_state_dict(state, strict=True)
    add_activation_quantization(model, metadata["activations"])
    return model, metadata
