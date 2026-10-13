"""CUDA Graph depth decoding for batched CFG generation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from models.cudagraph.depth_decoder_graph import DepthDecoderGraph


class BatchDepthDecoder:
    """Replace eager per-codebook depth decoding with captured graphs.

    The model's eager CFG depth loop reruns the whole codebook prefix twice per
    codebook. The streaming ``DepthDecoderGraph`` decodes with a static KV cache
    and pads each call to a power-of-two bucket. Capture only the active bucket
    on demand, so startup and single-request inference allocate no batch graphs.
    """

    def __init__(self, model: Any, *, max_batch_size: int) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.graph = None
        self.bucket_size = None

    def prepare(self, batch_size: int) -> None:
        """Keep only the current bucket; never preallocate all batch sizes."""
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(f"Unsupported batch size: {batch_size}")
        bucket = 2 * (1 << (batch_size - 1).bit_length())
        if self.graph is not None and self.bucket_size == bucket:
            return
        # Drop the previous graph before allocating its replacement.
        self.graph = None
        self.bucket_size = None
        model = self.model
        depth_decoder = model.depth_decoder
        generation_config = depth_decoder.generation_config
        top_p = generation_config.top_p
        self.graph = DepthDecoderGraph(
            depth_decoder=depth_decoder,
            config=model.config.depth_decoder_config,
            device=str(model.device),
            dtype=next(depth_decoder.model.parameters()).dtype,
            do_sample=bool(generation_config.do_sample),
            top_k=int(generation_config.top_k or 0),
            top_p=1.0 if top_p is None else float(top_p),
            temperature=float(generation_config.temperature),
            num_codebooks=model.config.num_codebooks,
            codec_codebook_size=int(model.config.codec_config.codebook_size),
            batch_size=bucket,
            bucket_sizes=[bucket],
        ).capture()
        self.bucket_size = bucket

    def generate_with_cfg(
        self,
        depth_decoder_input_ids: torch.LongTensor,
        cond_backbone_hidden_state: torch.Tensor,
        uncond_backbone_hidden_state: torch.Tensor,
        cfg_scale: float,
    ) -> torch.LongTensor:
        backbone_hidden = torch.cat(
            [cond_backbone_hidden_state, uncond_backbone_hidden_state], dim=0
        )
        first_codebook = depth_decoder_input_ids[:, 1].repeat(2)
        if self.graph is None:
            raise RuntimeError("Batch depth graph must be prepared before generation")
        codebooks = self.graph.run(
            backbone_hidden, first_codebook, guidance_scale=cfg_scale
        )
        return torch.cat(
            [depth_decoder_input_ids, codebooks.to(depth_decoder_input_ids.dtype)],
            dim=-1,
        )

    @contextmanager
    def bound(self, model: Any) -> Iterator[None]:
        model._depth_decoder_generate_with_cfg = self.generate_with_cfg
        try:
            yield
        finally:
            del model._depth_decoder_generate_with_cfg
