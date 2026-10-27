"""Exact B1 attention statistics from one ephemeral Probe callback.

The collector intentionally receives Q/K only through
``AttentionProbeContext`` and returns compact CPU arrays.  It neither owns
model hooks nor writes files.  The implementation accepts the current B1 v1
contract only: batch size one, continuous causal positions, no additive mask,
and standard contiguous GQA grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from experiments.head_analysis.runtime.attention_probe import AttentionProbeContext
from kvpress.context import KVPhase


UINT32_SENTINEL: Final[int] = int(np.iinfo(np.uint32).max)
DISTANCE_BIN_LOWER_BOUNDS: Final[tuple[int, ...]] = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "entropy",
    "entropy_normalized",
    "top_mass_16",
    "top_mass_64",
    "top_mass_256",
    "top_mass_1024",
    "sink_mass_4",
    "recent_mass_64",
    "prompt_mass",
    "generated_mass",
    "mean_distance",
    "distance_mass_0",
    "distance_mass_1",
    "distance_mass_2_3",
    "distance_mass_4_7",
    "distance_mass_8_15",
    "distance_mass_16_31",
    "distance_mass_32_63",
    "distance_mass_64_127",
    "distance_mass_128_255",
    "distance_mass_256_511",
    "distance_mass_512_1023",
    "distance_mass_1024_2047",
    "distance_mass_2048_4095",
    "distance_mass_4096_8191",
    "distance_mass_8192_16383",
    "distance_mass_16384_32767",
    "distance_mass_ge_32768",
)


class AttentionCollectionError(RuntimeError):
    """Raised when an observation does not meet the B1 statistical contract."""


@dataclass(frozen=True)
class AttentionCollectorConfig:
    """Frozen B1 online-statistics parameters.

    ``key_block_size`` changes temporary-memory usage only; it must not change
    the resulting full-candidate probability definition.
    """

    key_block_size: int = 1024
    prefill_window: int = 64
    token_topk: int = 64
    save_token_topk: bool = True
    page_size: int = 16
    page_topk: int = 32
    sink_tokens: int = 4
    recent_tokens: int = 64

    def __post_init__(self) -> None:
        for name in (
            "key_block_size",
            "prefill_window",
            "token_topk",
            "page_size",
            "page_topk",
            "sink_tokens",
            "recent_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CollectedAttention:
    """Compact CPU output of a single layer's B1 observation.

    The leading axis is selected Prefill query rows or Decode steps (one row
    under the current standard Decode contract).  No field retains a GPU Q/K
    tensor after :meth:`AttentionStatsCollector.collect` returns.
    """

    sample_id: str
    phase: KVPhase
    layer_id: int
    forward_id: int
    decode_step: int | None
    query_positions: np.ndarray
    visible_token_count: np.ndarray
    scalar_features: np.ndarray
    token_topk_positions: np.ndarray | None
    page_topk_ids_all_visible: np.ndarray | None
    page_topk_ids_prompt_only: np.ndarray | None
    page_candidate_count_all_visible: np.ndarray | None
    page_candidate_count_prompt_only: np.ndarray | None
    page_scope_mass: np.ndarray | None
    page_top32_coverage: np.ndarray | None
    page_scope_valid: np.ndarray | None


class AttentionStatsCollector:
    """Compute B1 statistics using a two-pass, key-blocked exact softmax."""

    def __init__(self, config: AttentionCollectorConfig | None = None) -> None:
        self.config = config or AttentionCollectorConfig()

    def collect(self, context: AttentionProbeContext) -> CollectedAttention:
        """Collect one layer event and immediately release all GPU temporaries."""

        self._validate_context(context)
        with torch.no_grad():
            return self._collect_no_grad(context)

    def _collect_no_grad(self, context: AttentionProbeContext) -> CollectedAttention:
        query = context.query
        key = context.key
        _, query_heads, query_length, head_dim = query.shape
        _, kv_heads, key_length, _ = key.shape
        group_size = query_heads // kv_heads
        device = query.device

        all_query_positions = context.query_positions.to(device=device, dtype=torch.long)
        selected_indices = self._selected_query_indices(context.phase, query_length, device)
        query_positions = all_query_positions.index_select(0, selected_indices)
        selected_queries = query[0].index_select(1, selected_indices).to(dtype=torch.float32)
        selected_length = int(selected_queries.shape[1])
        visible_token_count = query_positions + 1
        kv_for_query = torch.div(
            torch.arange(query_heads, device=device, dtype=torch.long), group_size, rounding_mode="floor"
        )
        key_positions = torch.arange(key_length, device=device, dtype=torch.long)
        scaling = float(context.scaling)

        log_z = torch.full((query_heads, selected_length), -torch.inf, device=device, dtype=torch.float32)
        for start in range(0, key_length, self.config.key_block_size):
            end = min(start + self.config.key_block_size, key_length)
            logits, visible, _, _ = self._block_logits(
                selected_queries,
                key,
                kv_for_query,
                query_positions,
                key_positions[start:end],
                start,
                end,
                scaling,
            )
            block_lse = torch.logsumexp(logits.masked_fill(~visible.unsqueeze(0), -torch.inf), dim=-1)
            log_z = torch.logaddexp(log_z, block_lse)

        if not torch.isfinite(log_z).all():
            raise AttentionCollectionError("at least one query head has no visible key")

        entropy = torch.zeros_like(log_z)
        sink_mass = torch.zeros_like(log_z)
        recent_mass = torch.zeros_like(log_z)
        prompt_mass = torch.zeros_like(log_z)
        generated_mass = torch.zeros_like(log_z)
        mean_distance = torch.zeros_like(log_z)
        distance_mass = torch.zeros(
            (query_heads, selected_length, len(DISTANCE_BIN_LOWER_BOUNDS)),
            device=device,
            dtype=torch.float32,
        )

        ranking_k = max(1024, self.config.token_topk)
        token_values = torch.full(
            (query_heads, selected_length, ranking_k), -torch.inf, device=device, dtype=torch.float32
        )
        token_ids = torch.full(
            (query_heads, selected_length, ranking_k), UINT32_SENTINEL, device=device, dtype=torch.long
        )

        page_all: torch.Tensor | None = None
        page_prompt: torch.Tensor | None = None
        if context.phase is KVPhase.DECODE:
            page_count = (key_length + self.config.page_size - 1) // self.config.page_size
            page_all = torch.zeros((kv_heads, selected_length, page_count), device=device, dtype=torch.float32)
            page_prompt = torch.zeros_like(page_all)

        for start in range(0, key_length, self.config.key_block_size):
            end = min(start + self.config.key_block_size, key_length)
            logits, visible, block_positions, _ = self._block_logits(
                selected_queries,
                key,
                kv_for_query,
                query_positions,
                key_positions[start:end],
                start,
                end,
                scaling,
            )
            log_probabilities = logits - log_z.unsqueeze(-1)
            probabilities = torch.where(
                visible.unsqueeze(0), torch.exp(log_probabilities), torch.zeros_like(log_probabilities)
            )
            entropy += torch.where(
                visible.unsqueeze(0), -probabilities * log_probabilities, torch.zeros_like(probabilities)
            ).sum(dim=-1)

            block_positions_by_query = block_positions.unsqueeze(0).expand(selected_length, -1)
            distances = query_positions.unsqueeze(1) - block_positions_by_query
            sink_mask = block_positions < self.config.sink_tokens
            recent_mask = block_positions_by_query >= (query_positions.unsqueeze(1) - self.config.recent_tokens + 1)
            prompt_mask = block_positions < context.prompt_length
            generated_mask = ~prompt_mask
            sink_mass += (probabilities * sink_mask.view(1, 1, -1)).sum(dim=-1)
            recent_mass += (probabilities * recent_mask.unsqueeze(0)).sum(dim=-1)
            prompt_mass += (probabilities * prompt_mask.view(1, 1, -1)).sum(dim=-1)
            generated_mass += (probabilities * generated_mask.view(1, 1, -1)).sum(dim=-1)
            mean_distance += (probabilities * distances.unsqueeze(0).to(dtype=torch.float32)).sum(dim=-1)

            distance_index = torch.bucketize(
                distances,
                torch.tensor(DISTANCE_BIN_LOWER_BOUNDS[1:], device=device, dtype=torch.long),
                right=True,
            )
            for bin_index in range(len(DISTANCE_BIN_LOWER_BOUNDS)):
                distance_mass[:, :, bin_index] += (probabilities * (distance_index == bin_index).unsqueeze(0)).sum(
                    dim=-1
                )

            candidate_values = torch.where(
                visible.unsqueeze(0), probabilities, torch.full_like(probabilities, -torch.inf)
            )
            candidate_ids = block_positions.view(1, 1, -1).expand(query_heads, selected_length, -1)
            token_values, token_ids = self._merge_topk(token_values, token_ids, candidate_values, candidate_ids)

            if page_all is not None and page_prompt is not None:
                grouped_probabilities = probabilities.reshape(kv_heads, group_size, selected_length, end - start).mean(
                    dim=1
                )
                page_ids = torch.div(block_positions, self.config.page_size, rounding_mode="floor")
                scatter_ids = page_ids.view(1, 1, -1).expand(kv_heads, selected_length, -1)
                page_all.scatter_add_(2, scatter_ids, grouped_probabilities)
                prompt_probabilities = grouped_probabilities * prompt_mask.view(1, 1, -1)
                page_prompt.scatter_add_(2, scatter_ids, prompt_probabilities)

        normalized_entropy = torch.where(
            visible_token_count.view(1, -1) > 1,
            entropy / torch.log(visible_token_count.view(1, -1).to(dtype=torch.float32)),
            torch.zeros_like(entropy),
        )
        top_mass = [token_values[:, :, :k].clamp_min(0).sum(dim=-1) for k in (16, 64, 256, 1024)]
        scalar_features = torch.cat(
            (
                entropy.unsqueeze(-1),
                normalized_entropy.unsqueeze(-1),
                *(value.unsqueeze(-1) for value in top_mass),
                sink_mass.unsqueeze(-1),
                recent_mass.unsqueeze(-1),
                prompt_mass.unsqueeze(-1),
                generated_mass.unsqueeze(-1),
                mean_distance.unsqueeze(-1),
                distance_mass,
            ),
            dim=-1,
        )
        if scalar_features.shape[-1] != len(FEATURE_NAMES):
            raise AssertionError("B1 feature schema length changed unexpectedly")

        saved_token_positions = None
        if context.phase is KVPhase.DECODE and self.config.save_token_topk:
            saved_token_positions = self._cpu_uint32(token_ids[:, :, : self.config.token_topk].permute(1, 0, 2))

        page_outputs = self._page_outputs(page_all, page_prompt, visible_token_count, context.prompt_length)
        return CollectedAttention(
            sample_id=context.sample_id,
            phase=context.phase,
            layer_id=context.layer_id,
            forward_id=context.forward_id,
            decode_step=context.decode_step,
            query_positions=self._cpu_uint32(query_positions),
            visible_token_count=self._cpu_uint32(visible_token_count),
            scalar_features=self._cpu_float32(scalar_features.permute(1, 0, 2)),
            token_topk_positions=saved_token_positions,
            **page_outputs,
        )

    def _page_outputs(
        self,
        page_all: torch.Tensor | None,
        page_prompt: torch.Tensor | None,
        visible_token_count: torch.Tensor,
        prompt_length: int,
    ) -> dict[str, np.ndarray | None]:
        if page_all is None or page_prompt is None:
            return {
                "page_topk_ids_all_visible": None,
                "page_topk_ids_prompt_only": None,
                "page_candidate_count_all_visible": None,
                "page_candidate_count_prompt_only": None,
                "page_scope_mass": None,
                "page_top32_coverage": None,
                "page_scope_valid": None,
            }

        all_candidates = torch.div(
            visible_token_count + self.config.page_size - 1, self.config.page_size, rounding_mode="floor"
        )
        prompt_candidates = torch.full_like(
            all_candidates, (prompt_length + self.config.page_size - 1) // self.config.page_size
        )
        all_ids, all_mass, all_coverage, all_valid = self._rank_pages(page_all, all_candidates)
        prompt_ids, prompt_mass, prompt_coverage, prompt_valid = self._rank_pages(page_prompt, prompt_candidates)
        return {
            "page_topk_ids_all_visible": self._cpu_uint32(all_ids),
            "page_topk_ids_prompt_only": self._cpu_uint32(prompt_ids),
            "page_candidate_count_all_visible": self._cpu_uint32(all_candidates),
            "page_candidate_count_prompt_only": self._cpu_uint32(prompt_candidates),
            "page_scope_mass": self._cpu_float32(torch.stack((all_mass, prompt_mass), dim=-1)),
            "page_top32_coverage": self._cpu_float32(torch.stack((all_coverage, prompt_coverage), dim=-1)),
            "page_scope_valid": torch.stack((all_valid, prompt_valid), dim=-1).cpu().numpy(),
        }

    def _rank_pages(
        self, page_mass: torch.Tensor, candidate_count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank Page IDs by score descending and Page ID ascending for ties."""

        kv_heads, selected_length, page_count = page_mass.shape
        scores = page_mass.permute(1, 0, 2)
        page_ids = torch.arange(page_count, device=page_mass.device, dtype=torch.long)
        candidate_mask = page_ids.view(1, 1, -1) < candidate_count.view(-1, 1, 1)
        values = torch.where(candidate_mask, scores, torch.full_like(scores, -torch.inf))
        ids = page_ids.view(1, 1, -1).expand(selected_length, kv_heads, -1)
        top_values, top_ids = self._stable_topk(values, ids, self.config.page_topk)
        if top_values.shape[-1] < self.config.page_topk:
            padding = self.config.page_topk - top_values.shape[-1]
            top_values = torch.cat(
                (top_values, torch.full((*top_values.shape[:-1], padding), -torch.inf, device=top_values.device)),
                dim=-1,
            )
            top_ids = torch.cat(
                (
                    top_ids,
                    torch.full(
                        (*top_ids.shape[:-1], padding), UINT32_SENTINEL, device=top_ids.device, dtype=top_ids.dtype
                    ),
                ),
                dim=-1,
            )
        scope_mass = torch.where(candidate_mask, scores, torch.zeros_like(scores)).sum(dim=-1)
        valid = scope_mass > 0
        coverage = torch.where(
            valid,
            top_values.clamp_min(0).sum(dim=-1) / scope_mass,
            torch.full_like(scope_mass, torch.nan),
        )
        top_ids = torch.where(
            valid.unsqueeze(-1) & torch.isfinite(top_values),
            top_ids,
            torch.full_like(top_ids, UINT32_SENTINEL),
        )
        return top_ids, scope_mass, coverage, valid

    def _validate_context(self, context: AttentionProbeContext) -> None:
        if context.phase not in (KVPhase.PREFILL, KVPhase.DECODE):
            raise AttentionCollectionError(f"unsupported phase {context.phase!r}")
        if context.attention_mask is not None:
            raise AttentionCollectionError("B1 v1 does not yet implement additive-mask visibility/bias semantics")
        if not context.is_causal:
            raise AttentionCollectionError("B1 v1 requires standard causal visibility")
        if context.scaling is None or not np.isfinite(float(context.scaling)):
            raise AttentionCollectionError("B1 requires a finite attention scaling")
        query, key = context.query, context.key
        if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
            raise AttentionCollectionError("B1 requires Q/K shaped [1, head, sequence, dim]")
        if query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1] != 0:
            raise AttentionCollectionError("B1 requires a contiguous integer GQA query-to-KV mapping")
        if context.query_positions.ndim != 1 or context.query_positions.numel() != query.shape[2]:
            raise AttentionCollectionError("query_positions must have one entry per Q row")
        if context.key_positions.ndim != 1 or context.key_positions.numel() != key.shape[2]:
            raise AttentionCollectionError("key_positions must have one entry per K row")
        expected_key_positions = torch.arange(
            key.shape[2], device=context.key_positions.device, dtype=context.key_positions.dtype
        )
        if not torch.equal(context.key_positions, expected_key_positions):
            raise AttentionCollectionError("B1 v1 requires continuous key positions [0, T-1]")
        expected_query_positions = torch.arange(
            int(context.query_positions[0].item()),
            int(context.query_positions[0].item()) + query.shape[2],
            device=context.query_positions.device,
            dtype=context.query_positions.dtype,
        )
        if not torch.equal(context.query_positions, expected_query_positions):
            raise AttentionCollectionError("B1 v1 requires continuous query positions")
        if int(context.query_positions[-1].item()) >= key.shape[2]:
            raise AttentionCollectionError("a query position has no corresponding current-cache key")
        if context.phase is KVPhase.DECODE and (query.shape[2] != 1 or context.decode_step is None):
            raise AttentionCollectionError("B1 Decode requires exactly one query row and a decode_step")

    def _selected_query_indices(self, phase: KVPhase, query_length: int, device: torch.device) -> torch.Tensor:
        if phase is KVPhase.DECODE:
            return torch.zeros(1, device=device, dtype=torch.long)
        start = max(0, query_length - self.config.prefill_window)
        return torch.arange(start, query_length, device=device, dtype=torch.long)

    @staticmethod
    def _block_logits(
        selected_queries: torch.Tensor,
        key: torch.Tensor,
        kv_for_query: torch.Tensor,
        query_positions: torch.Tensor,
        block_positions: torch.Tensor,
        start: int,
        end: int,
        scaling: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key_block = key[0, :, start:end, :].index_select(0, kv_for_query).to(dtype=torch.float32)
        logits = torch.einsum("hqd,hkd->hqk", selected_queries, key_block) * scaling
        visible = block_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        return logits, visible, block_positions, key_block

    def _merge_topk(
        self,
        current_values: torch.Tensor,
        current_ids: torch.Tensor,
        candidate_values: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat((current_values, candidate_values), dim=-1)
        ids = torch.cat((current_ids, candidate_ids), dim=-1)
        return self._stable_topk(values, ids, current_values.shape[-1])

    @staticmethod
    def _stable_topk(values: torch.Tensor, ids: torch.Tensor, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sort by score descending, then logical ID ascending without float perturbation."""

        id_order = torch.argsort(ids, dim=-1, stable=True)
        values = torch.gather(values, -1, id_order)
        ids = torch.gather(ids, -1, id_order)
        score_order = torch.argsort(values, dim=-1, descending=True, stable=True)
        score_order = score_order[..., : min(limit, values.shape[-1])]
        return torch.gather(values, -1, score_order), torch.gather(ids, -1, score_order)

    @staticmethod
    def _cpu_float32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()

    @staticmethod
    def _cpu_uint32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(device="cpu", dtype=torch.uint32).contiguous().numpy()
