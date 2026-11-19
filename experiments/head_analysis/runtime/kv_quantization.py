"""V3-B grouped K/V quantization; integer payloads, FP32 statistics, BF16 decode."""

from dataclasses import dataclass
import struct
from typing import Iterator

import torch


GROUP_SIZE = 32
PROTECTED_TOKENS = 64


def quantized_prompt_range(prompt_length: int) -> tuple[int, int]:
    rows = (prompt_length - 2 * PROTECTED_TOKENS) // GROUP_SIZE * GROUP_SIZE
    if rows <= 0:
        raise ValueError("Prompt must contain at least one complete 32-row middle group")
    return PROTECTED_TOKENS, PROTECTED_TOKENS + rows


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    if bits not in (4, 6, 8) or codes.dtype != torch.uint8:
        raise ValueError("codes must be uint8 and bit width must be 4, 6 or 8")
    flat = codes.flatten()
    if bits == 8:
        return flat.clone()
    if torch.any(flat >= 2**bits):
        raise ValueError(f"{bits}-bit codes must be in [0, {2**bits - 1}]")
    if bits == 6:
        count = flat.numel()
        flat = torch.cat((flat, flat.new_zeros((-count) % 4))).reshape(-1, 4).int()
        words = flat[:, 0] | (flat[:, 1] << 6) | (flat[:, 2] << 12) | (flat[:, 3] << 18)
        return (
            torch.stack((words & 255, (words >> 8) & 255, (words >> 16) & 255), -1)
            .flatten()
            .to(torch.uint8)[: (count * 6 + 7) // 8]
        )
    if flat.numel() % 2:
        flat = torch.cat((flat, flat.new_zeros(1)))
    return flat[::2] | (flat[1::2] << 4)


def unpack_codes(payload: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    if bits not in (4, 6, 8) or payload.dtype != torch.uint8 or payload.numel() != (count * bits + 7) // 8:
        raise ValueError("invalid packed code length, dtype or bit width")
    if bits == 8:
        return payload.flatten().clone()
    if bits == 6:
        flat = payload.flatten()
        groups = torch.cat((flat, flat.new_zeros((-flat.numel()) % 3))).reshape(-1, 3).int()
        words = groups[:, 0] | (groups[:, 1] << 8) | (groups[:, 2] << 16)
        return (
            torch.stack(tuple((words >> shift) & 63 for shift in (0, 6, 12, 18)), -1).flatten().to(torch.uint8)[:count]
        )
    return torch.stack((payload & 15, payload >> 4), dim=-1).flatten()[:count]


def _groups(x: torch.Tensor, side: str) -> torch.Tensor:
    if side == "key":
        return x.reshape(-1, GROUP_SIZE, x.shape[-1]).transpose(1, 2).reshape(-1, GROUP_SIZE)
    return x.reshape(-1, GROUP_SIZE)


def _rows(groups: torch.Tensor, side: str, dim: int) -> torch.Tensor:
    if side == "key":
        return groups.reshape(-1, dim, GROUP_SIZE).transpose(1, 2).reshape(-1, dim)
    return groups.reshape(-1, dim)


@dataclass(frozen=True)
class QuantizedHead:
    """One side of one Head, in token-major packed-code order; no protected rows."""

    side: str
    bits: int
    rows: int
    dim: int
    codes: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor

    @property
    def payload_nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in (self.codes, self.minimum, self.scale))

    def decode_blocks(self, block_rows: int = 1024) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield FP32 restored blocks; callers choose the final Attention dtype."""
        _check_block_rows(block_rows)
        for start in range(0, self.rows, block_rows):
            end = min(start + block_rows, self.rows)
            first, last = start * self.dim, end * self.dim
            payload = self.codes[first * self.bits // 8 : last * self.bits // 8]
            q = unpack_codes(payload, self.bits, last - first).reshape(-1, self.dim)
            a, b = first // GROUP_SIZE, last // GROUP_SIZE
            restored = self.minimum[a:b, None] + self.scale[a:b, None] * _groups(q, self.side).float()
            yield start, _rows(restored, self.side, self.dim)

    def decode(self, dtype: torch.dtype = torch.bfloat16, block_rows: int = 1024) -> torch.Tensor:
        result = torch.empty((self.rows, self.dim), dtype=dtype, device=self.codes.device)
        for start, restored in self.decode_blocks(block_rows):
            result[start : start + restored.shape[0]] = restored.to(dtype)
        return result


def _check_block_rows(block_rows: int) -> None:
    if block_rows <= 0 or block_rows % GROUP_SIZE:
        raise ValueError("block_rows must be a positive multiple of 32")


def quantize_head(x: torch.Tensor, side: str, bits: int, block_rows: int = 1024) -> QuantizedHead:
    """Quantize [middle_tokens,D]; K groups along tokens, V along features."""
    if side not in ("key", "value") or bits not in (4, 6, 8):
        raise ValueError("expected key/value and 4/8 bits (or the 6-bit budget control)")
    if x.ndim != 2 or min(x.shape) <= 0 or any(n % GROUP_SIZE for n in x.shape):
        raise ValueError("middle token count and Head dimension must be positive multiples of 32")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("quantization input must contain finite floating-point values")
    _check_block_rows(block_rows)
    codes, minima, scales = [], [], []
    for start in range(0, x.shape[0], block_rows):
        groups = _groups(x[start : start + block_rows].float(), side)
        minimum, maximum = groups.amin(-1), groups.amax(-1)
        scale = (maximum - minimum) / (2**bits - 1)
        # Constant groups use scale=0 and code=0, without perturbing other groups.
        divisor = torch.where(scale == 0, 1.0, scale)
        q = ((groups - minimum[:, None]) / divisor[:, None]).round().clamp(0, 2**bits - 1).to(torch.uint8)
        codes.append(pack_codes(_rows(q, side, x.shape[1]), bits))
        minima.append(minimum)
        scales.append(scale)
    return QuantizedHead(side, bits, *x.shape, torch.cat(codes), torch.cat(minima), torch.cat(scales))


def quantize_and_restore(x: torch.Tensor, side: str, bits: int, block_rows: int = 1024):
    """Shared actual decode and FP64 error sums for local and model interventions."""
    encoded = quantize_head(x, side, bits, block_rows)
    restored = torch.empty_like(x)
    errors = {"squared_error": 0.0, "fp32_squared_error": 0.0, "original_squared_sum": 0.0}
    for start, decoded in encoded.decode_blocks(block_rows):
        end = start + decoded.shape[0]
        actual = decoded.to(x.dtype)
        restored[start:end] = actual
        reference = x[start:end].double()
        errors["original_squared_sum"] += reference.square().sum().item()
        errors["squared_error"] += (actual.double() - reference).square().sum().item()
        errors["fp32_squared_error"] += (decoded.double() - reference).square().sum().item()
    return restored, encoded, errors


def quantization_byte_ledger(
    prompt_length: int, generated_count: int, num_layers: int, kv_heads: int, head_dim: int, bits: int | None
) -> dict[str, int]:
    """Section 21.6 layout budget, separate from the shadow runner's GPU memory."""
    start, end = quantized_prompt_range(prompt_length)
    if generated_count < 0 or min(num_layers, kv_heads, head_dim) <= 0 or head_dim % GROUP_SIZE:
        raise ValueError("invalid Cache dimensions")
    if bits not in (None, 4, 6, 8):
        raise ValueError("expected bypass or 4/6/8 bits")
    rows, length = end - start, prompt_length + generated_count
    dense = 4 * length * num_layers * kv_heads * head_dim
    codes = rows * head_dim * bits // 8 if bits else 0
    statistics = 8 * rows * head_dim // GROUP_SIZE if bits else 0
    unquantized_side = 2 * head_dim * (length - rows) if bits else 2 * head_dim * length
    data = dense - 2 * length * head_dim + codes + statistics + unquantized_side
    # At D,Q multiples of 32, every code/statistic/BF16 stream is already 64-byte aligned.
    return {
        "dense_kv_data_bytes": dense,
        "quantized_tokens": rows if bits else 0,
        "code_bytes": codes,
        "statistic_bytes": statistics,
        "unquantized_target_side_bytes": unquantized_side,
        "request_kv_data_bytes": data,
        "request_metadata_bytes": 8,
        "request_header_padding_bytes": 56,
        "data_padding_bytes": 0,
        "model_metadata_bytes": 24 if bits else 0,
        "model_padding_bytes": 40 if bits else 0,
        "model_bytes": 64 if bits else 0,
        "request_bytes": data + 64,
        "total_bytes": data + 64 + (64 if bits else 0),
        "net_saved_bytes": dense - data - (64 if bits else 0),
    }


@dataclass(frozen=True)
class HeadBitPlan:
    """One side: uniform, legacy 4/8 bitmap, or one balanced 4/8 pair per layer.

    The 24-byte descriptor and optional bitmap share the existing 64-byte
    alignment contract. No Calib scores are needed by the decoder.
    """

    side: str
    bits: tuple[tuple[int, ...], ...]
    head_dim: int

    def __post_init__(self):
        if (
            self.side not in ("key", "value")
            or not self.bits
            or not self.bits[0]
            or self.head_dim <= 0
            or self.head_dim % GROUP_SIZE
            or any(len(row) != len(self.bits[0]) for row in self.bits)
        ):
            raise ValueError("invalid Head bit plan geometry/side")
        values = {int(b) for row in self.bits for b in row}
        if not values <= {4, 6, 8}:
            raise ValueError("plan must be uniform or use only 4/8-bit Heads")
        if 6 in values and len(values) != 1:
            if len(self.bits[0]) > 16 or any((row.count(4), row.count(8)) not in ((0, 0), (1, 1)) for row in self.bits):
                raise ValueError("a paired plan allows at most one 4-bit and one 8-bit Head per layer")

    @property
    def uniform_bits(self):
        values = {int(b) for row in self.bits for b in row}
        return next(iter(values)) if len(values) == 1 else 0

    @property
    def paired(self):
        return not self.uniform_bits and any(6 in row for row in self.bits)

    @property
    def metadata_bytes(self):
        stride = 1 if self.paired else (len(self.bits[0]) + 7) // 8
        return 24 + (0 if self.uniform_bits else len(self.bits) * stride)

    def to_bytes(self):
        flags = (self.side == "value") | (self.uniform_bits << 8) | (2 if self.paired else 0)
        header = struct.pack("<4s5I", b"HQB1", len(self.bits), len(self.bits[0]), self.head_dim, GROUP_SIZE, flags)
        bitmap = bytearray(self.metadata_bytes - len(header))
        if self.paired:
            heads = len(self.bits[0])
            for layer, row in enumerate(self.bits):
                if 4 in row:
                    d, u = row.index(4), row.index(8)
                    bitmap[layer] = 1 + d * (heads - 1) + u - (u > d)
        elif not self.uniform_bits:
            stride = (len(self.bits[0]) + 7) // 8
            for layer, row in enumerate(self.bits):
                for head, bits in enumerate(row):
                    if bits == 8:
                        bitmap[layer * stride + head // 8] |= 1 << (head % 8)
        data = header + bitmap
        return data + bytes((-len(data)) % 64)

    @classmethod
    def from_bytes(cls, data):
        magic, layers, heads, dim, group, flags = struct.unpack("<4s5I", data[:24])
        if magic != b"HQB1" or group != GROUP_SIZE or flags & ~0xFF03:
            raise ValueError("unknown Head bit plan format")
        uniform, side = flags >> 8, "value" if flags & 1 else "key"
        stride = (heads + 7) // 8
        if flags & 2:
            if uniform or heads < 2:
                raise ValueError("paired descriptors require at least two Heads and no uniform flag")
            rows = []
            for layer in range(layers):
                code, row = data[24 + layer], [6] * heads
                if code:
                    if code > heads * (heads - 1):
                        raise ValueError("paired Head code is outside the layer")
                    d, u = divmod(code - 1, heads - 1)
                    row[d], row[u + (u >= d)] = 4, 8
                rows.append(tuple(row))
            bits = tuple(rows)
        else:
            bits = tuple(
                tuple(
                    uniform or (8 if data[24 + layer * stride + head // 8] & (1 << (head % 8)) else 4)
                    for head in range(heads)
                )
                for layer in range(layers)
            )
        plan = cls(side, bits, dim)
        if plan.to_bytes() != data:
            raise ValueError("Head bit plan length/padding differs from its descriptor")
        return plan


def cache_request_header(prompt_length, generated_count):
    return struct.pack("<II", prompt_length, generated_count) + bytes(56)


def head_plan_byte_ledger(plan: HeadBitPlan, prompt_length: int, generated_count: int):
    """Full K+V footprint: one encoded side, the other dense; real layout bytes."""
    layers, heads = len(plan.bits), len(plan.bits[0])
    dense = quantization_byte_ledger(prompt_length, generated_count, layers, heads, plan.head_dim, None)
    start, end = quantized_prompt_range(prompt_length)
    rows, length, dim = end - start, prompt_length + generated_count, plan.head_dim
    codes = rows * dim * sum(sum(row) for row in plan.bits) // 8
    statistics = layers * heads * rows * dim // GROUP_SIZE * 8
    protected = 2 * (length - rows) * layers * heads * dim
    other_side = 2 * length * layers * heads * dim
    data = codes + statistics + protected + other_side
    model_bytes = len(plan.to_bytes())
    request_metadata = len(cache_request_header(prompt_length, generated_count))
    return {
        "dense_kv_data_bytes": dense["dense_kv_data_bytes"],
        "code_bytes": codes,
        "statistic_bytes": statistics,
        "protected_residual_decode_bytes": protected,
        "dense_other_side_bytes": other_side,
        "request_kv_data_bytes": data,
        "request_metadata_bytes": 8,
        "request_header_padding_bytes": request_metadata - 8,
        "data_padding_bytes": 0,
        "model_metadata_bytes": plan.metadata_bytes,
        "model_padding_bytes": model_bytes - plan.metadata_bytes,
        "model_bytes": model_bytes,
        "request_bytes": data + request_metadata,
        "total_bytes": data + request_metadata + model_bytes,
        "net_saved_bytes": dense["request_bytes"] - (data + request_metadata + model_bytes),
    }
