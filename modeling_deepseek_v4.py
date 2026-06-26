"""DeepSeek-V4 modeling code (faithful small-scale replica).

Parameter naming mirrors the official DeepSeek-V4 safetensors index so that
weights can later be transferred / sliced from real V4-Pro / V4-Flash
checkpoints. Top-level layout (flat, no ``model.`` prefix):

    embed.weight
    layers.{i}.attn_norm.weight
    layers.{i}.ffn_norm.weight
    layers.{i}.hc_attn_{base,fn,scale}
    layers.{i}.hc_ffn_{base,fn,scale}
    layers.{i}.attn.{wq_a, wq_b, wkv, wo_a, wo_b, q_norm, kv_norm, attn_sink}
    layers.{i}.attn.compressor.{wkv, wgate, ape, norm}        # CSA / HCA only
    layers.{i}.attn.indexer.{wq_b, weights_proj, compressor.*}# CSA only
    layers.{i}.ffn.gate.{weight, bias}                        # routed MoE
    layers.{i}.ffn.gate.tid2eid                               # hash MoE
    layers.{i}.ffn.experts.{j}.{w1, w2, w3}.weight
    layers.{i}.ffn.shared_experts.{w1, w2, w3}.weight
    norm.weight
    head.weight
    hc_head_{base, fn, scale}
    mtp.{k}.{...}                                              # one per MTP step
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_deepseek_v4 import DeepseekV4Config


# =============================================================================
# Norms, RoPE, utilities
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.weight * x32).to(in_dtype)


def fixed_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm without a learnable scale (used inside mHC)."""
    in_dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps)).to(in_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(seq_len: int, dim: int, base: float, device, dtype):
    if dim <= 0:
        return torch.zeros(seq_len, 0, device=device, dtype=dtype), \
               torch.zeros(seq_len, 0, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       rope_dim: int, positions: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to last `rope_dim` dims of x at given positions.
    x: [..., L, D]; cos/sin: [P, rope_dim]; positions: [L] long.
    """
    if rope_dim <= 0:
        return x
    x_pass, x_rot = x[..., :-rope_dim], x[..., -rope_dim:]
    c = cos[positions]
    s = sin[positions]
    while c.dim() < x_rot.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    x_rot = (x_rot * c) + (_rotate_half(x_rot) * s)
    return torch.cat([x_pass, x_rot], dim=-1)


# =============================================================================
# Manifold-Constrained Hyper-Connections (mHC)
# =============================================================================

class MHC(nn.Module):
    """Manifold-Constrained Hyper-Connections.

    Parameter layout matches official safetensors / kernel.py exactly:
        - {prefix}_fn    [mix_hc, n*d]  dynamic generator (single combined matmul)
        - {prefix}_base  [mix_hc]       static biases
        - {prefix}_scale [3]            three scalar gates (one per pre/post/comb part)
        with mix_hc = (2 + n) * n.

    Math (matches inference/kernel.py:hc_split_sinkhorn_kernel):
        flat   = X.flatten(-2)                    # [B,S,n*d]
        rsqrt  = rsqrt(mean(flat^2) + eps)        # row-wise
        mixes  = (flat @ fn.T) * rsqrt            # [B,S, mix_hc]
        pre[i]    = sigmoid(mixes[:, i]            * scale[0] + base[i])     + eps   for i in [0,n)
        post[i]   = 2 * sigmoid(mixes[:, n+i]      * scale[1] + base[n+i])           for i in [0,n)
        comb_raw  = mixes[:, 2n + j*n + k]         * scale[2] + base[2n+j*n+k]   [n,n]
        comb      = softmax(comb_raw, dim=-1) + eps               # row softmax then +eps
        comb      = comb / (comb.sum(-2, keepdim=True) + eps)     # column normalize
        repeat (sinkhorn_iters - 1) times:
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
    Apply (matches Block.hc_pre / hc_post):
        sublayer_in = sum_i pre[i] * X[i]                          # [B,S,d]
        new_X[i]    = post[i] * F_out + sum_j comb[i,j] * X[j]     # [B,S,n,d]
    """

    def __init__(self, hidden_size: int, n_hc: int, sinkhorn_iters: int = 20,
                 eps: float = 1e-6):
        super().__init__()
        self.d = hidden_size
        self.n = n_hc
        self.iters = sinkhorn_iters
        self.eps = eps
        self.flat = n_hc * hidden_size
        self.mix_hc = (2 + n_hc) * n_hc      # = 24 for n=4

    def split_and_construct(self, mixes: torch.Tensor, base: torch.Tensor,
                            scale: torch.Tensor):
        """mixes: [..., mix_hc]; base: [mix_hc]; scale: [3].
        Returns (pre [...,n], post [...,n], comb [...,n,n]).

        All math is in fp32 (matches official ``with set_dtype(torch.float32)``
        block around hc_*_fn / base / scale params); base/scale may be stored
        in any dtype but are promoted to mixes.dtype for arithmetic.
        """
        n = self.n
        base = base.to(mixes.dtype)
        scale = scale.to(mixes.dtype)
        # Indexing: pre = first n, post = next n, comb = last n*n flattened row-major.
        pre_raw = mixes[..., :n]
        post_raw = mixes[..., n:2 * n]
        comb_raw = mixes[..., 2 * n:].reshape(*mixes.shape[:-1], n, n)

        base_pre = base[:n]
        base_post = base[n:2 * n]
        base_comb = base[2 * n:].view(n, n)

        pre = torch.sigmoid(scale[0] * pre_raw + base_pre) + self.eps
        post = 2.0 * torch.sigmoid(scale[1] * post_raw + base_post)

        comb_pre = scale[2] * comb_raw + base_comb
        # Row-softmax then +eps, then column normalize, then alternating row/col norms.
        comb = F.softmax(comb_pre, dim=-1) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, comb

    def gen_params(self, X: torch.Tensor, base: torch.Tensor, fn: torch.Tensor,
                   scale: torch.Tensor):
        """X: [B,S,n,d]. Returns (pre [B,S,n], post [B,S,n], comb [B,S,n,n]).
        Always computed in fp32 (matches official `with set_dtype(fp32)` for mHC).
        """
        Bsz, S, n, d = X.shape
        flat = X.reshape(Bsz, S, n * d).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(flat, fn.float()) * rsqrt              # [B,S, mix_hc]
        return self.split_and_construct(mixes, base, scale)

    @staticmethod
    def hc_pre(X: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """X: [B,S,n,d], pre: [B,S,n]. Returns [B,S,d]."""
        return torch.sum(pre.unsqueeze(-1).to(X.dtype) * X, dim=-2)

    @staticmethod
    def hc_post(new_x: torch.Tensor, residual: torch.Tensor,
                post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """new_x: [B,S,d], residual: [B,S,n,d], post: [B,S,n], comb: [B,S,n,n].
        out[i] = post[i] * new_x + sum_j comb[i,j] * residual[j]
        """
        post_e = post.unsqueeze(-1).to(new_x.dtype)             # [B,S,n,1]
        comb_e = comb.to(residual.dtype)                        # [B,S,n,n]
        return post_e * new_x.unsqueeze(-2) + torch.matmul(comb_e, residual)

    # --- Head-side mHC: only computes `pre`, no Sinkhorn. ---
    def gen_head_pre(self, X: torch.Tensor, fn: torch.Tensor,
                     base: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """fn: [n, n*d]; base: [n]; scale: [1] or scalar. Returns pre: [B,S,n]."""
        Bsz, S, n, d = X.shape
        flat = X.reshape(Bsz, S, n * d).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(flat, fn.float()) * rsqrt              # [B,S,n]
        scale = scale.float()
        base = base.float()
        s = scale.view(-1)[0] if scale.numel() else 1.0
        return torch.sigmoid(s * mixes + base) + self.eps


# =============================================================================
# Attention sink helper
# =============================================================================

def sink_softmax(logits: torch.Tensor, sink: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with extra learnable per-head sink logit in the denominator.
    logits: [..., H, ..., K]; sink: [H] (broadcast).
    Caller must shape sink so it broadcasts with logits.
    """
    m = logits.amax(dim=dim, keepdim=True)
    m = torch.maximum(m, sink)
    ex = torch.exp(logits - m)
    sink_ex = torch.exp(sink - m)
    return ex / (ex.sum(dim=dim, keepdim=True) + sink_ex)


# =============================================================================
# Inference helpers — ports of inference/model.py top-level utilities
# =============================================================================

from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=128)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Returns ``[bsz, seqlen, window_size]`` LongTensor of indices into the
    sliding-window portion of a unified ``kv_cache`` buffer (positions
    ``[0, window_size)``). Invalid positions are -1.

    Mirrors the official ``inference/model.py:get_window_topk_idxs``:
      - prefill (start_pos == 0): each query t attends to keys
        ``[max(0, t-w+1) .. t]``; we map these onto the (yet-to-be-circular)
        SW buffer where token t is stored at position t (clamped).
      - decode (start_pos > 0, seqlen == 1): the SW buffer is circular; the
        oldest entry sits at ``(start_pos+1) % w``, newest at
        ``start_pos % w``. We list the buffer indices in chronological order.
      - decode-but-not-yet-full (0 < start_pos < window_size-1): partial
        buffer; valid entries [0..start_pos], rest -1.
    """
    if start_pos >= window_size - 1:
        wrap = start_pos % window_size
        # Chronological order: oldest=(wrap+1)%w, newest=wrap. Build by concat.
        matrix = torch.cat(
            [torch.arange(wrap + 1, window_size), torch.arange(0, wrap + 1)],
            dim=0,
        ).long()
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1, dtype=torch.long),
                       (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen, dtype=torch.long).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), dtype=torch.long)
        matrix = torch.where(matrix > base, torch.full_like(matrix, -1), matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


@lru_cache(maxsize=128)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    """For HCA layers (no Lightning Indexer), returns ``[bsz, seqlen, n_compressed]``
    indices into the compressed portion of the unified buffer. The compressed
    portion starts at index ``offset``; query at absolute position p attends to
    blocks ``[0, (p+1) // ratio)``.
    """
    if start_pos > 0:
        n_valid = (start_pos + 1) // ratio
        matrix = torch.arange(n_valid, dtype=torch.long) + offset
    else:
        n_blocks = seqlen // ratio
        matrix = torch.arange(n_blocks, dtype=torch.long).repeat(seqlen, 1)
        valid_count = torch.arange(1, seqlen + 1).unsqueeze(1) // ratio       # [S, 1]
        mask = matrix >= valid_count
        matrix = torch.where(mask, torch.full_like(matrix, -1), matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor,
                topk_idxs: torch.Tensor, scale: float) -> torch.Tensor:
    """PyTorch port of inference/kernel.py:sparse_attn.

    Args:
        q:          ``[B, S, H, c]``
        kv:         ``[B, K_total, c]`` — single shared K/V (MQA), positions are
                    [SW circular buffer (size window) | committed compressed (rest)].
        sink:       ``[H]`` — per-head learnable sink logit.
        topk_idxs:  ``[B, S, K_topk]`` LongTensor — positions to attend in `kv`.
                    -1 marks invalid.
        scale:      ``1 / sqrt(head_dim)``.

    Returns ``[B, S, H, c]``.
    """
    B, S, H, c = q.shape
    K_topk = topk_idxs.size(-1)
    if topk_idxs.dtype != torch.long:
        topk_idxs = topk_idxs.long()
    invalid = topk_idxs < 0
    idx_safe = topk_idxs.clamp(min=0)

    # Gather selected kv per query: [B, S, K_topk, c]
    kv_exp = kv.unsqueeze(1).expand(-1, S, -1, -1)
    kv_sel = torch.gather(kv_exp, 2,
                          idx_safe.unsqueeze(-1).expand(-1, -1, -1, c))

    # Logits [B, S, H, K_topk]
    logits = torch.einsum("bshd,bskd->bshk", q, kv_sel) * scale
    logits = logits.masked_fill(invalid.unsqueeze(2), float("-inf"))

    # Sink-aware softmax
    sink_e = sink.view(1, 1, -1, 1).to(logits.dtype)
    m = logits.amax(dim=-1, keepdim=True)
    m = torch.maximum(m, sink_e)
    ex = torch.exp(logits - m)
    sink_ex = torch.exp(sink_e - m)
    probs = ex / (ex.sum(dim=-1, keepdim=True) + sink_ex)

    # Apply
    return torch.einsum("bshk,bskd->bshd", probs, kv_sel)


# =============================================================================
# Compressor (token-level pooling); shared for HCA, CSA, and indexer keys
# =============================================================================

class Compressor(nn.Module):
    """Compresses every `m` hidden states into one entry via softmax-weighted pool.

    Matches inference/model.py:Compressor exactly:
        - When overlap (compress_ratio == 4): wkv and wgate output 2*head_dim
          (first half = overlap stream, second half = current); ape: [m, 2*head_dim].
          ``overlap_transform`` rearranges [b,nb,m,2c] -> [b,nb,2m,c] before softmax.
        - When non-overlap: outputs head_dim, ape: [m, head_dim], plain softmax pool.

    Inference cache buffers (allocated by ``setup_caches(max_batch_size)``):
        kv_state    [max_b, coff*m, coff*head_dim]   rolling partial-block kv
        score_state [max_b, coff*m, coff*head_dim]   rolling partial-block scores
        kv_cache    [max_b, max_compressed, head_dim] committed compressed entries
                    (NOT allocated by setup_caches — set by parent Attention via alias)

    Tensor names: ``norm.weight``, ``wkv.weight``, ``wgate.weight``, ``ape``.
    """
    def __init__(self, hidden_size: int, head_dim: int, m: int, overlap: bool):
        super().__init__()
        self.m = m
        self.overlap = overlap
        self.head_dim = head_dim
        coff = 2 if overlap else 1
        self.coff = coff
        self.norm = RMSNorm(head_dim)
        self.wkv = nn.Linear(hidden_size, coff * head_dim, bias=False)
        self.wgate = nn.Linear(hidden_size, coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(m, coff * head_dim))
        # Inference buffers — None until setup_caches() is called
        self.kv_state: Optional[torch.Tensor] = None
        self.score_state: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None     # set by parent Attention via alias
        self.rope_cos_compressed: Optional[torch.Tensor] = None
        self.rope_sin_compressed: Optional[torch.Tensor] = None
        self.rope_dim: int = 0

    def setup_caches(self, max_batch_size: int):
        """Allocate the rolling partial-block state buffers."""
        coff, m, d = self.coff, self.m, self.head_dim
        ref = next(iter(self.parameters()))
        device = ref.device
        self.kv_state = torch.zeros(max_batch_size, coff * m, coff * d,
                                    dtype=torch.float32, device=device)
        self.score_state = torch.full((max_batch_size, coff * m, coff * d),
                                      float("-inf"), dtype=torch.float32, device=device)


    @staticmethod
    def _overlap_transform(t: torch.Tensor, head_dim: int, fill_value) -> torch.Tensor:
        """t: [b, nb, m, 2*head_dim]; returns [b, nb, 2*m, head_dim].
        First m positions of each block come from the previous block's overlap-half;
        next m positions come from the current block's current-half.
        """
        b, nb, m, _ = t.shape
        d = head_dim
        out = t.new_full((b, nb, 2 * m, d), fill_value)
        out[:, :, m:] = t[:, :, :, d:]                         # current half
        out[:, 1:, :m] = t[:, :-1, :, :d]                      # prev block's overlap half, shift +1
        return out

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, n, D]. Returns compressed [B, ceil(n/m), head_dim]."""
        Bsz, n, _ = h.shape
        m, d = self.m, self.head_dim
        # Matmul in whatever dtype the wkv/wgate weights live in (bf16 / fp32).
        # We then upcast to fp32 for the softmax-weighted pool (numerical stability).
        param_dtype = self.wkv.weight.dtype
        xx = h.to(param_dtype)
        kv = self.wkv(xx).float()                              # [B, n, coff*d]
        score = self.wgate(xx).float()                         # [B, n, coff*d]
        # Pad to multiple of m
        pad = (m - n % m) % m
        if pad:
            kv = F.pad(kv, (0, 0, 0, pad))
            score = F.pad(score, (0, 0, 0, pad))
        nb = kv.size(1) // m
        kv = kv.view(Bsz, nb, m, -1)                           # [B,nb,m, coff*d]
        score = score.view(Bsz, nb, m, -1) + self.ape.float()  # bias by ape (fp32)

        if self.overlap:
            kv = self._overlap_transform(kv, d, 0.0)           # [B,nb, 2m, d]
            score = self._overlap_transform(score, d, float("-inf"))   # [B,nb, 2m, d]

        # Softmax over the m (or 2m) positions, weighted sum to one entry per block
        kv = (kv * score.softmax(dim=2)).sum(dim=2)            # [B,nb, d]
        return self.norm(kv.to(h.dtype))

    def forward_inference(self, h: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
        """Inference forward — uses kv_state / score_state buffers.

        start_pos == 0 (prefill, h: [B, S, D]):
            Run prefill compression as in ``forward``, but additionally save
            the residual partial-block (and the last complete block's overlap
            half if overlap=True) into ``kv_state`` / ``score_state`` so that
            future single-token decode steps can extend correctly. Writes
            committed compressed entries into ``self.kv_cache[:bsz, :nb]`` and
            returns them.

        start_pos > 0 (decode, h: [B, 1, D]):
            Append the new token's projected (kv, score) into the rolling state
            buffer. If ``(start_pos+1) % m == 0`` a new compressed block is
            committed: pool the rolling state, apply norm, write into
            ``self.kv_cache[:bsz, start_pos // m]`` and return the new block.
            Otherwise return None (no new block this step).
        """
        assert self.kv_state is not None, "Compressor.setup_caches() not called"
        Bsz, S, _ = h.shape
        m, d = self.m, self.head_dim
        param_dtype = self.wkv.weight.dtype
        xx = h.to(param_dtype)
        kv_full = self.wkv(xx).float()                         # [B, S, coff*d]
        score_full = self.wgate(xx).float()                    # [B, S, coff*d]
        ape = self.ape.float()                                 # [m, coff*d]

        if start_pos == 0:
            # ----- prefill ---------------------------------------------------
            cutoff = (S // m) * m
            remainder = S - cutoff
            # Save the carry / partial state for future decode steps:
            if self.overlap and cutoff >= m:
                # state[:m] = last full block (will serve as overlap source)
                self.kv_state[:Bsz, :m] = kv_full[:, cutoff - m: cutoff]
                self.score_state[:Bsz, :m] = score_full[:, cutoff - m: cutoff] + ape
            offset = m if self.overlap else 0
            if remainder > 0:
                # state[offset:offset+remainder] = the residual partial block
                self.kv_state[:Bsz, offset: offset + remainder] = kv_full[:, cutoff:]
                self.score_state[:Bsz, offset: offset + remainder] = score_full[:, cutoff:] + ape[:remainder]
                kv_full = kv_full[:, :cutoff]
                score_full = score_full[:, :cutoff]
            if cutoff == 0:
                # Nothing to commit yet
                return None
            nb = cutoff // m
            kv_b = kv_full.view(Bsz, nb, m, -1)
            score_b = score_full.view(Bsz, nb, m, -1) + ape
            if self.overlap:
                kv_b = self._overlap_transform(kv_b, d, 0.0)
                score_b = self._overlap_transform(score_b, d, float("-inf"))
            committed = (kv_b * score_b.softmax(dim=2)).sum(dim=2)   # [B, nb, d]
            committed = self.norm(committed.to(h.dtype))
            # Apply RoPE on the rope-dim slice of compressed entries
            if self.rope_dim > 0 and self.rope_cos_compressed is not None:
                comp_pos = (torch.arange(nb, device=h.device) * m + (m - 1)).clamp(
                    max=self.rope_cos_compressed.size(0) - 1
                )
                committed = apply_partial_rope(committed,
                                               self.rope_cos_compressed,
                                               self.rope_sin_compressed,
                                               self.rope_dim, comp_pos)
            # Write into the parent attention's compressed-cache slice
            if self.kv_cache is not None:
                self.kv_cache[:Bsz, :nb] = committed.to(self.kv_cache.dtype)
            return committed

        # ----- decode (S == 1) -----------------------------------------------
        assert S == 1, f"decode step expects S=1, got {S}"
        kv = kv_full.squeeze(1)                                # [B, coff*d]
        score = score_full.squeeze(1) + ape[start_pos % m]     # [B, coff*d]
        should_commit = (start_pos + 1) % m == 0

        if self.overlap:
            # state[:m] = previous full-block overlap source (set on commit shift)
            # state[m:] = current block accumulator
            self.kv_state[:Bsz, m + start_pos % m] = kv
            self.score_state[:Bsz, m + start_pos % m] = score
            if should_commit:
                kv_state_now = torch.cat(
                    [self.kv_state[:Bsz, :m, :d],     # overlap half from prev block
                     self.kv_state[:Bsz, m:, d:]],    # current half from accumulator
                    dim=1,
                )
                score_state_now = torch.cat(
                    [self.score_state[:Bsz, :m, :d],
                     self.score_state[:Bsz, m:, d:]],
                    dim=1,
                )
                committed = (kv_state_now * score_state_now.softmax(dim=1)).sum(
                    dim=1, keepdim=True
                )                                              # [B, 1, d]
                # Roll: current accumulator becomes the new "previous" carry
                self.kv_state[:Bsz, :m] = self.kv_state[:Bsz, m:]
                self.score_state[:Bsz, :m] = self.score_state[:Bsz, m:]
        else:
            self.kv_state[:Bsz, start_pos % m] = kv
            self.score_state[:Bsz, start_pos % m] = score
            if should_commit:
                committed = (self.kv_state[:Bsz] * self.score_state[:Bsz].softmax(dim=1)).sum(
                    dim=1, keepdim=True
                )

        if not should_commit:
            return None

        committed = self.norm(committed.to(h.dtype))
        # RoPE at the LAST position of the just-committed block — matches the
        # existing prefill convention (`block_idx * m + (m-1)`). For start_pos
        # == 27, m == 4: block 6 covers [24..27] -> RoPE at position 27.
        if self.rope_dim > 0 and self.rope_cos_compressed is not None:
            block_pos = start_pos                            # = block_idx * m + (m-1)
            comp_pos = torch.tensor([block_pos], device=h.device, dtype=torch.long).clamp(
                max=self.rope_cos_compressed.size(0) - 1
            )
            committed = apply_partial_rope(committed,
                                           self.rope_cos_compressed,
                                           self.rope_sin_compressed,
                                           self.rope_dim, comp_pos)
        if self.kv_cache is not None:
            self.kv_cache[:Bsz, start_pos // m] = committed.squeeze(1).to(self.kv_cache.dtype)
        return committed


# =============================================================================
# Lightning Indexer
# =============================================================================

class LightningIndexer(nn.Module):
    """
    Names:
        indexer.compressor.*       (separate Compressor for indexer keys)
        indexer.wq_b.weight        (q-up from shared cQ -> H_I * head_dim)
        indexer.weights_proj.weight (per-head weight w_t,h)
    """
    def __init__(self, hidden_size: int, q_lora_rank: int,
                 index_n_heads: int, index_head_dim: int,
                 m: int, overlap: bool):
        super().__init__()
        self.m = m
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.compressor = Compressor(hidden_size, index_head_dim, m, overlap=overlap)
        self.wq_b = nn.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.weights_proj = nn.Linear(hidden_size, index_n_heads, bias=False)
        # Score scaling: softmax_scale * 1/sqrt(n_heads), as in inference/model.py
        self.score_scale = (index_head_dim ** -0.5) * (index_n_heads ** -0.5)
        # Inference cache (allocated by setup_caches)
        self.kv_cache: Optional[torch.Tensor] = None
        self.index_topk: int = 0

    def setup_caches(self, max_batch_size: int, max_seq_len: int,
                     rope_cos, rope_sin, rope_cos_c, rope_sin_c):
        """Allocate the indexer-key cache (committed indexer compressed keys)
        and chain to our compressor."""
        ref = next(iter(self.parameters()))
        device, dtype = ref.device, ref.dtype
        n_blocks = max_seq_len // self.m
        self.kv_cache = torch.zeros(max_batch_size, n_blocks, self.head_dim,
                                    device=device, dtype=dtype)
        self.compressor.setup_caches(max_batch_size)
        # Alias: indexer.compressor writes its committed blocks into our kv_cache
        self.compressor.kv_cache = self.kv_cache
        self.compressor.rope_cos_compressed = rope_cos_c
        self.compressor.rope_sin_compressed = rope_sin_c
        self.compressor.rope_dim = 0   # indexer keys are not RoPE'd in official impl

    def reset_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.zero_()
        if self.compressor.kv_state is not None:
            self.compressor.kv_state.zero_()
            self.compressor.score_state.fill_(float("-inf"))

    def keys(self, h: torch.Tensor) -> torch.Tensor:
        return self.compressor(h)                              # [B, nb, head_dim]

    def select_inference(self, h: torch.Tensor, cQ: torch.Tensor,
                         start_pos: int, offset: int, top_k: int) -> torch.Tensor:
        """Inference top-k block selection. Returns ``[B, S, k]`` indices into
        the unified kv_cache (offset already applied).

        On prefill (start_pos == 0): commits compressed indexer keys via
        ``self.compressor.forward_inference(h, 0)`` — they get written to
        ``self.kv_cache[:bsz, :nb]``.
        On decode (start_pos > 0): updates the rolling state; if a new block
        was committed it lands in ``self.kv_cache[:bsz, start_pos // m]``.
        """
        Bsz, S, _ = h.shape
        m = self.m
        # Update committed indexer keys
        self.compressor.forward_inference(h, start_pos)
        end_pos = start_pos + S
        n_valid = end_pos // m
        if n_valid <= 0:
            return torch.full((Bsz, S, 0), -1, dtype=torch.long, device=h.device)
        K_full = self.kv_cache[:Bsz, :n_valid]                      # [B, n_valid, idx_head_dim]
        qI = self.wq_b(cQ).view(Bsz, S, self.n_heads, self.head_dim)
        wI = self.weights_proj(h) * self.score_scale                # [B, S, H_I]
        qK = torch.einsum("blhd,bsd->blhs", qI, K_full)
        qK = F.relu(qK)
        scores = (wI.unsqueeze(-1) * qK).sum(dim=2)                 # [B, S, n_valid]
        # Unified causal mask: query at absolute position p attends to block s
        # iff (s+1)*m - 1 < p, i.e. block_end < p. Applies to BOTH prefill (each
        # query at position t in [0,S)) and decode (single query at start_pos).
        # In particular, the *just-committed* block during a decode step IS
        # NOT visible to the current query — its end == query position.
        s_idx = torch.arange(n_valid, device=h.device)
        block_end = s_idx * m + (m - 1)                              # [n_valid]
        t_pos = (torch.arange(S, device=h.device) + start_pos).unsqueeze(-1)  # [S, 1]
        mask = block_end.unsqueeze(0) < t_pos                        # [S, n_valid]
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

        k = min(top_k, n_valid)
        idx = scores.topk(k, dim=-1).indices                         # [B, S, k]
        valid = scores.gather(-1, idx) > float("-inf")
        idx = torch.where(valid, idx + offset,
                          torch.full_like(idx, -1))
        return idx

    def select(self, h: torch.Tensor, cQ: torch.Tensor, K: torch.Tensor,
               positions: torch.Tensor, m: int, top_k: int):
        """Returns (idx [B,Lq,k], mask [B,Lq,k] bool)."""
        Bsz, Lq, _ = h.shape
        nb = K.size(1)
        qI = self.wq_b(cQ).view(Bsz, Lq, self.n_heads, self.head_dim)
        wI = self.weights_proj(h) * self.score_scale           # [B,Lq,H_I]

        qK = torch.einsum("blhd,bsd->blhs", qI, K)
        qK = F.relu(qK)
        scores = (wI.unsqueeze(-1) * qK).sum(dim=2)            # [B,Lq,nb]

        # Causal: query at pos t may attend to block s if (s+1)*m - 1 < t  ⇔  s < t/m
        s_idx = torch.arange(nb, device=h.device)
        causal = s_idx.unsqueeze(0) < (positions.unsqueeze(-1) // m)   # [Lq, nb]
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

        k = min(top_k, nb)
        if k <= 0:
            empty = torch.zeros(Bsz, Lq, 0, dtype=torch.long, device=h.device)
            return empty, empty.bool()
        topk = scores.topk(k, dim=-1)
        return topk.indices, torch.isfinite(topk.values)


# =============================================================================
# Attention layer (CSA / HCA / pure sliding-window)
# =============================================================================

class DeepseekV4Cache:
    """Per-model KV cache for incremental decode.

    Each layer keeps a dict with up to three tensors:
      - ``sw_kv``         [B, n_kept, head_dim]    rolling sliding-window K/V
      - ``compressed_kv`` [B, nb, head_dim]        committed compressed entries (CSA/HCA only)
      - ``indexer_keys``  [B, nb, idx_head_dim]    committed indexer keys (CSA only)

    ``seen_tokens`` is the cumulative sequence length consumed so far. The
    next forward expects positions ``[seen_tokens, ..., seen_tokens + S - 1]``.

    Limitation: during decode (``S=1`` calls) the compressed_kv and indexer_keys
    are NOT updated until you re-prefill. For accurate decode beyond
    ``compress_ratio`` (4 for CSA layers, 32+ for HCA) tokens past the prefill
    end, you should re-prefill periodically. This is acceptable for short
    generation and matches the typical inference deployment pattern.
    """
    def __init__(self, num_layers: int):
        self.layers = [{} for _ in range(num_layers)]
        self.seen_tokens = 0

    def reset(self):
        for d in self.layers:
            d.clear()
        self.seen_tokens = 0

    def __len__(self) -> int:
        return len(self.layers)


class DeepseekV4Attention(nn.Module):
    """One attention layer.
    compress_ratio: 0 -> pure SW; small (>0, <16) -> CSA; large (>=16) -> HCA.
    """
    def __init__(self, config: DeepseekV4Config, compress_ratio: int):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        d = config.hidden_size
        H = config.num_attention_heads
        c = config.head_dim
        self.H = H
        self.c = c
        self.q_lora_rank = config.q_lora_rank
        self.o_groups = config.o_groups
        assert H % self.o_groups == 0
        self.heads_per_group = H // self.o_groups
        self.d_g = config.o_lora_rank
        self.rope_dim = config.qk_rope_head_dim
        self.window = config.sliding_window

        if compress_ratio == 0:
            self.mode = "sw"
        elif compress_ratio < 16:
            self.mode = "csa"
        else:
            self.mode = "hca"

        # Query path: low-rank with norm at q_lora; per-head rsqrt-norm applied at use time
        self.wq_a = nn.Linear(d, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, H * c, bias=False)

        # Sliding-window KV (always present, single shared head — MQA)
        self.wkv = nn.Linear(d, c, bias=False)
        self.kv_norm = RMSNorm(c, eps=config.rms_norm_eps)

        # Output projection: per-group wo_a (n_groups separate sub-matrices stored in
        # one Linear; reshape weight to [n_groups, o_lora_rank, heads_per_group*c]
        # at use time and apply via einsum).
        self.wo_a = nn.Linear(self.heads_per_group * c,
                              self.o_groups * self.d_g, bias=False)
        self.wo_b = nn.Linear(self.o_groups * self.d_g, d, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(H))

        if self.mode in ("csa", "hca"):
            self.compressor = Compressor(d, c, compress_ratio, overlap=(self.mode == "csa"))
        if self.mode == "csa":
            self.indexer = LightningIndexer(d, config.q_lora_rank,
                                            config.index_n_heads, config.index_head_dim,
                                            m=compress_ratio, overlap=True)

        # Inference cache: unified buffer
        #   positions [0, window)               -> circular SW K/V
        #   positions [window, window + n_blk)  -> committed compressed entries
        # Allocated by setup_caches(); also wires the compressor's kv_cache as
        # an alias into the compressed slice of this buffer.
        self.kv_cache: Optional[torch.Tensor] = None
        self._max_compressed = 0

    def setup_caches(self, max_batch_size: int, max_seq_len: int,
                     rope_cos, rope_sin, rope_cos_c, rope_sin_c):
        """Allocate the unified kv_cache and wire compressor caches.

        ``max_seq_len`` is used to size the compressed portion (one block per
        ``compress_ratio`` source tokens) and to bound the RoPE table.
        """
        ref = next(iter(self.parameters()))
        device, dtype = ref.device, ref.dtype
        win = self.window
        if self.compress_ratio:
            self._max_compressed = max_seq_len // self.compress_ratio
        else:
            self._max_compressed = 0
        kv_size = win + self._max_compressed
        self.kv_cache = torch.zeros(max_batch_size, kv_size, self.c,
                                    device=device, dtype=dtype)
        if self.mode in ("csa", "hca"):
            self.compressor.setup_caches(max_batch_size)
            # Alias compressor's kv_cache into our compressed slice
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.rope_cos_compressed = rope_cos_c
            self.compressor.rope_sin_compressed = rope_sin_c
            self.compressor.rope_dim = self.rope_dim
        if self.mode == "csa":
            self.indexer.setup_caches(max_batch_size, max_seq_len,
                                      rope_cos, rope_sin, rope_cos_c, rope_sin_c)
        # Cache RoPE tables for use during decode
        self._rope_cos = rope_cos
        self._rope_sin = rope_sin
        self._rope_cos_c = rope_cos_c
        self._rope_sin_c = rope_sin_c

    def reset_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.zero_()
        if self.mode in ("csa", "hca"):
            if self.compressor.kv_state is not None:
                self.compressor.kv_state.zero_()
                self.compressor.score_state.fill_(float("-inf"))
        if self.mode == "csa":
            self.indexer.reset_cache()

    def _output_proj(self, attn_out: torch.Tensor) -> torch.Tensor:
        """attn_out: [B, S, H, c]. Returns [B, S, d].
        Uses per-group wo_a: weight is [n_groups*o_lora, heads_per_group*c]; we
        reshape to [n_groups, o_lora, heads_per_group*c] and apply via einsum
        so each group has its own projection (matching official inference).
        """
        B, S, H, c = attn_out.shape
        out_g = attn_out.reshape(B, S, self.o_groups, self.heads_per_group * c)
        wo_a = self.wo_a.weight.view(self.o_groups, self.d_g, self.heads_per_group * c)
        out = torch.einsum("bsgd,grd->bsgr", out_g, wo_a)      # [B,S,g,d_g]
        out = out.reshape(B, S, self.o_groups * self.d_g)
        return self.wo_b(out)

    def _apply_output_rope(self, out: torch.Tensor, rope_cos, rope_sin,
                           positions) -> torch.Tensor:
        """V4 trick: rotate output by -position so contributions carry relative pos."""
        if self.rope_dim <= 0:
            return out
        cos = rope_cos[positions]                              # [S, rope_dim]
        sin = -rope_sin[positions]                             # negate -> rotate by -i
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        out_pass, out_rot = out[..., :-self.rope_dim], out[..., -self.rope_dim:]
        out_rot = (out_rot * cos) + (_rotate_half(out_rot) * sin)
        return torch.cat([out_pass, out_rot], dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                rope_cos_c: torch.Tensor, rope_sin_c: torch.Tensor,
                pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """x: [B,S,d]; positions: [S] long; pad_mask: [B,S] bool (True=valid) or None."""
        Bsz, S, _ = x.shape
        H, c, m = self.H, self.c, self.compress_ratio

        # Queries: low-rank, latent norm, then per-head no-weight RMSNorm (paper),
        # then partial RoPE on the last `rope_dim` dims.
        cQ = self.q_norm(self.wq_a(x))                         # [B,S,q_lora]
        q = self.wq_b(cQ).view(Bsz, S, H, c)
        # Per-head fixed RMSNorm (no learnable weight) — see inference/model.py
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) +
                            self.config.rms_norm_eps).to(q.dtype)
        q = apply_partial_rope(q.transpose(1, 2), rope_cos, rope_sin,
                               self.rope_dim, positions).transpose(1, 2)
        # q now [B,S,H,c]

        # Sliding-window KV
        kv_sw = self.kv_norm(self.wkv(x))                      # [B,S,c]
        kv_sw = apply_partial_rope(kv_sw, rope_cos, rope_sin, self.rope_dim, positions)

        # Build SW causal+window mask: [S, S] then expand to [B,S,S]
        i = positions.unsqueeze(-1)
        j = positions.unsqueeze(0)
        sw_mask = (j <= i) & (j > i - self.window)             # [S,S]
        sw_mask = sw_mask.unsqueeze(0).expand(Bsz, -1, -1)
        if pad_mask is not None:
            sw_mask = sw_mask & pad_mask.unsqueeze(1)          # mask padded keys

        # ---------------- compressed branch ----------------
        if self.mode in ("csa", "hca"):
            # The compressor has its OWN internal RMSNorm on output — do not
            # re-apply self.kv_norm (that one is for the sliding-window path).
            kv_comp = self.compressor(x)                       # [B,nb,c]
            nb = kv_comp.size(1)
            comp_pos = (torch.arange(nb, device=x.device) * m + (m - 1)).clamp(
                max=rope_cos_c.size(0) - 1
            )
            kv_comp = apply_partial_rope(kv_comp, rope_cos_c, rope_sin_c,
                                         self.rope_dim, comp_pos)
            # Per-query causal mask over compressed blocks
            block_end = torch.arange(nb, device=x.device) * m + (m - 1)
            comp_mask = (block_end.unsqueeze(0) < positions.unsqueeze(-1))   # [S,nb]
            comp_mask = comp_mask.unsqueeze(0).expand(Bsz, -1, -1)           # [B,S,nb]
        else:
            kv_comp = None
            comp_mask = None
            nb = 0

        if self.mode == "csa":
            K_idx = self.indexer.keys(x)                       # [B,nb,idx_head_dim]
            idx, sel_mask = self.indexer.select(x, cQ, K_idx, positions, m,
                                                self.config.index_topk)
            # Gather selected compressed entries for each query
            kk = idx.size(-1)
            if kk == 0:
                kv_sel = kv_comp.new_zeros(Bsz, S, 0, c)
                sel_mask = sel_mask.new_zeros(Bsz, S, 0, dtype=torch.bool)
            else:
                idx_safe = idx.clamp(min=0)
                kv_comp_exp = kv_comp.unsqueeze(1).expand(-1, S, -1, -1)     # [B,S,nb,c]
                kv_sel = torch.gather(
                    kv_comp_exp, 2,
                    idx_safe.unsqueeze(-1).expand(-1, -1, -1, c)
                )                                              # [B,S,kk,c]
        else:
            kv_sel = None
            sel_mask = None
            kk = 0

        # ---------------- core attention ----------------
        scale = 1.0 / math.sqrt(c)
        # Sliding-window logits: einsum over the shared-KV (single head broadcast)
        # q: [B,S,H,c], kv_sw: [B,S,c]
        sw_logits = torch.einsum("bthd,bjd->bthj", q, kv_sw) * scale          # [B,S,H,S]

        if self.mode == "sw":
            mask = sw_mask.unsqueeze(2)                                       # [B,S,1,S]
            logits = sw_logits.masked_fill(~mask, float("-inf"))
            sink = self.attn_sink.view(1, 1, -1, 1)
            probs = sink_softmax(logits, sink, dim=-1)
            kv_v = kv_sw.unsqueeze(1).expand(-1, S, -1, -1)                   # [B,S,S,c] (broadcast read)
            out = torch.einsum("bthj,btjd->bthd", probs, kv_v)                # [B,S,H,c]

        elif self.mode == "hca":
            # logits over [compressed blocks (nb)] + [SW window (S)]
            comp_logits = torch.einsum("bthd,bjd->bthj", q, kv_comp) * scale  # [B,S,H,nb]
            comp_logits = comp_logits.masked_fill(~comp_mask.unsqueeze(2), float("-inf"))
            sw_logits = sw_logits.masked_fill(~sw_mask.unsqueeze(2), float("-inf"))
            logits = torch.cat([comp_logits, sw_logits], dim=-1)              # [B,S,H,nb+S]
            sink = self.attn_sink.view(1, 1, -1, 1)
            probs = sink_softmax(logits, sink, dim=-1)
            p_comp, p_sw = probs.split([nb, S], dim=-1)
            out = (
                torch.einsum("bthj,bjd->bthd", p_comp, kv_comp) +
                torch.einsum("bthj,bjd->bthd", p_sw, kv_sw)
            )                                                                 # [B,S,H,c]

        else:  # csa
            # logits over [selected (kk)] + [SW window (S)]
            sel_logits = torch.einsum("bthd,btjd->bthj", q, kv_sel) * scale   # [B,S,H,kk]
            if kk > 0:
                sel_logits = sel_logits.masked_fill(~sel_mask.unsqueeze(2), float("-inf"))
            sw_logits = sw_logits.masked_fill(~sw_mask.unsqueeze(2), float("-inf"))
            logits = torch.cat([sel_logits, sw_logits], dim=-1)               # [B,S,H,kk+S]
            sink = self.attn_sink.view(1, 1, -1, 1)
            probs = sink_softmax(logits, sink, dim=-1)
            p_sel, p_sw = probs.split([kk, S], dim=-1)
            out = (
                (torch.einsum("bthj,btjd->bthd", p_sel, kv_sel) if kk > 0 else 0) +
                torch.einsum("bthj,bjd->bthd", p_sw, kv_sw)
            )

        # Output RoPE-by-(-i) trick
        out = self._apply_output_rope(out, rope_cos, rope_sin, positions)
        return self._output_proj(out)

    # ----------------------------------------------------------------------
    # Inference forward — uses the unified ``kv_cache`` + sparse_attn helper.
    # Mirrors inference/model.py:Attention.forward(x, start_pos) closely.
    # ----------------------------------------------------------------------
    def forward_inference(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        assert self.kv_cache is not None, "Attention.setup_caches() not called"
        Bsz, S, _ = x.shape
        H, c, m = self.H, self.c, self.compress_ratio
        win = self.window
        rope_dim = self.rope_dim

        # Positions tensor for RoPE on Q and SW K
        positions = torch.arange(start_pos, start_pos + S,
                                 device=x.device, dtype=torch.long).clamp(
            max=self._rope_cos.size(0) - 1
        )

        # Q (with per-head rsqrt-norm, partial RoPE)
        cQ = self.q_norm(self.wq_a(x))
        q = self.wq_b(cQ).view(Bsz, S, H, c)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) +
                            self.config.rms_norm_eps).to(q.dtype)
        q = apply_partial_rope(q.transpose(1, 2), self._rope_cos, self._rope_sin,
                               rope_dim, positions).transpose(1, 2)

        # SW KV for new tokens (with partial RoPE)
        kv_sw = self.kv_norm(self.wkv(x))                              # [B, S, c]
        kv_sw = apply_partial_rope(kv_sw, self._rope_cos, self._rope_sin,
                                   rope_dim, positions)

        # Build topk_idxs (window slot indices, then compressed slot indices)
        topk_w = get_window_topk_idxs(win, Bsz, S, start_pos).to(x.device)
        if m:
            offset = S if start_pos == 0 else win
            if self.mode == "csa":
                topk_c = self.indexer.select_inference(
                    x, cQ, start_pos, offset, self.config.index_topk
                )
            else:  # hca
                topk_c = get_compress_topk_idxs(m, Bsz, S, start_pos, offset).to(x.device)
            topk_idxs = torch.cat([topk_w, topk_c], dim=-1)
        else:
            topk_idxs = topk_w

        if start_pos == 0:
            # ----- prefill ----------------------------------------------------
            # Save SW into circular buffer (last `win` tokens)
            if S <= win:
                self.kv_cache[:Bsz, :S] = kv_sw.to(self.kv_cache.dtype)
            else:
                cutoff = S % win
                last = kv_sw[:, -win:]
                if cutoff > 0:
                    self.kv_cache[:Bsz, cutoff:win] = last[:, :win - cutoff].to(self.kv_cache.dtype)
                    self.kv_cache[:Bsz, :cutoff] = last[:, win - cutoff:].to(self.kv_cache.dtype)
                else:
                    self.kv_cache[:Bsz, :win] = last.to(self.kv_cache.dtype)
            # Build unified prefill KV: [SW (S positions of fresh kv) | compressed]
            if m:
                if self.mode == "csa":
                    # Indexer's compressor was already updated by select_inference;
                    # we still need the main attention compressor.
                    kv_comp = self.compressor.forward_inference(x, 0)
                else:
                    kv_comp = self.compressor.forward_inference(x, 0)
                if kv_comp is not None:
                    kv_unified = torch.cat([kv_sw, kv_comp.to(kv_sw.dtype)], dim=1)
                else:
                    kv_unified = kv_sw
            else:
                kv_unified = kv_sw
            o = sparse_attn(q, kv_unified, self.attn_sink, topk_idxs,
                            scale=1.0 / math.sqrt(c))
        else:
            # ----- decode (single token) --------------------------------------
            # Overwrite circular SW slot
            self.kv_cache[:Bsz, start_pos % win] = kv_sw.squeeze(1).to(self.kv_cache.dtype)
            if m and self.mode in ("csa", "hca"):
                # Compressor may commit a new block (writes via alias into self.kv_cache)
                self.compressor.forward_inference(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:Bsz], self.attn_sink, topk_idxs,
                            scale=1.0 / math.sqrt(c))

        # Output RoPE by -position
        o = self._apply_output_rope(o, self._rope_cos, self._rope_sin, positions)
        return self._output_proj(o)


# =============================================================================
# Clamped SwiGLU expert
# =============================================================================

class SwiGLUExpert(nn.Module):
    """w1 = gate, w3 = up, w2 = down  (matches official naming)."""
    def __init__(self, hidden_size: int, intermediate_size: int, limit: float):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.limit = limit

    def forward(self, x):
        g = self.w1(x)
        u = self.w3(x)
        # V4 SwiGLU clamping: linear in [-limit, limit], gate <= limit
        u = torch.clamp(u, -self.limit, self.limit)
        g = torch.minimum(g, torch.full_like(g, self.limit))
        return self.w2(F.silu(g) * u)


# =============================================================================
# DeepseekMoE (sqrt-softplus routing, aux-loss-free) + Hash variant
# =============================================================================

class MoEGate(nn.Module):
    """Gate parameters matching official ``inference/model.py:Gate``:

    Always present:
        - ``weight`` [n_routed_experts, hidden_size]: produces routing scores
          (sqrt(softplus) by default in V4) for BOTH hash and non-hash layers.
          For hash layers the score still defines per-token expert weights;
          only the *index selection* uses the hash table.

    Conditional:
        - ``bias`` [n_routed_experts] (non-hash only): aux-loss-free routing
          bias added to scores at top-k selection time. Stored as a learnable
          float32 parameter to match the official layout.
        - ``tid2eid`` [vocab_size, top_k] (hash only): non-trainable lookup
          table mapping token-id -> expert indices.
    """
    def __init__(self, hidden_size: int, num_experts: int, vocab_size: int,
                 hash_routing: bool, top_k: int):
        super().__init__()
        self.hash_routing = hash_routing
        # Gate weight is ALWAYS present — used to compute routing scores even
        # in hash-routed layers (only the index selection differs there).
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))
        if hash_routing:
            # tid2eid is non-trainable; matches official (requires_grad=False)
            self.tid2eid = nn.Parameter(
                torch.zeros(vocab_size, top_k, dtype=torch.long),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.zeros(num_experts, dtype=torch.float32))


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: DeepseekV4Config, hash_routing: bool):
        super().__init__()
        self.config = config
        self.hash_routing = hash_routing
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling = config.routed_scaling_factor
        d = config.hidden_size
        inter = config.moe_intermediate_size
        limit = config.swiglu_limit

        self.gate = MoEGate(d, self.num_experts, config.vocab_size,
                            hash_routing=hash_routing, top_k=self.top_k)
        self.experts = nn.ModuleList([
            SwiGLUExpert(d, inter, limit) for _ in range(self.num_experts)
        ])
        if config.n_shared_experts > 0:
            self.shared_experts = SwiGLUExpert(d, inter * config.n_shared_experts, limit)
        else:
            self.shared_experts = None

    def _routed_indices(self, x_flat: torch.Tensor, token_ids_flat: torch.Tensor):
        """Matches inference/model.py:Gate exactly. Hash layers still derive
        weights from the learned gate (only the index selection differs).
        """
        # Score in fp32 for stability, matches official.
        logits = F.linear(x_flat.float(), self.gate.weight.float())     # [N, E]
        if self.config.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        elif self.config.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:  # sqrtsoftplus (V4 default)
            scores = F.softplus(logits).sqrt()
        original_scores = scores

        if self.hash_routing:
            idx = self.gate.tid2eid[token_ids_flat].long()              # [N, K]
        else:
            biased = scores + self.gate.bias.float()
            idx = biased.topk(self.top_k, dim=-1).indices
        weights = original_scores.gather(-1, idx)
        if self.config.scoring_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        weights = weights * self.routed_scaling
        return idx, weights.to(x_flat.dtype)

    def _dispatch_tokens(
        self,
        x_flat: torch.Tensor,
        token_ids_flat: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], dict]:
        """Dispatch Phase:
        1. Router: Computes raw routing scores.
        2. Expert Selection: Identifies top-K expert indices and normalizes weights.
        3. Token Permutation: Stable-sorts tokens by expert destination.
        4. Slicing: Segments grouped tensors into contiguous expert-specific chunks.

        Returns:
            expert_inputs: Contiguous token features mapped to each expert.
            expert_weights: Normalized routing weights mapped to each expert.
            expert_toks: Sequence token index mapping back to original layout.
            metadata: GPU-resident mapping data for future backend optimizations.
        """
        # TODO: Replace stable sort with custom prefix-sum/histogram token bucketing
        # (similar to DeepEP/vLLM binning kernels) to eliminate sorting overhead.

        # 1. Router & Expert Selection
        idx, w = self._routed_indices(x_flat, token_ids_flat)  # [N, K], [N, K]
        flat_idx = idx.reshape(-1)
        flat_w = w.reshape(-1)

        # 2. Token Permutation Setup
        # Stable sort aligns tokens by expert ID while preserving chronological order.
        # Using argsort avoids allocating a temporary tensor for sorted expert IDs.
        sorted_indices = torch.argsort(flat_idx, stable=True)
        sorted_tok = sorted_indices // self.top_k
        sorted_w = flat_w[sorted_indices]

        # Gather contiguous token inputs
        grouped_tok_inputs = x_flat[sorted_tok]

        # Count tokens per expert (routing metadata)
        expert_counts = torch.bincount(flat_idx, minlength=self.num_experts)

        # Segment grouped arrays into expert-specific slices
        # Isolates host-device sync (.tolist() for chunk lengths) to this helper.
        counts_list = expert_counts.tolist()
        expert_inputs = torch.split(grouped_tok_inputs, counts_list, dim=0)
        expert_weights = torch.split(sorted_w, counts_list, dim=0)
        expert_toks = torch.split(sorted_tok, counts_list, dim=0)

        metadata = {
            "expert_counts": expert_counts,
            "counts_list": counts_list,
            "grouped_inputs": grouped_tok_inputs,
        }
        return expert_inputs, expert_weights, expert_toks, metadata

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Mixture-of-Experts execution graph:
        Hidden States -> Dispatch -> Expert Execution -> Merge -> Output
        """
        Bsz, S, D = x.shape
        N = Bsz * S
        x_flat = x.reshape(N, D)
        token_ids_flat = token_ids.reshape(N)

        # ----- 1. DISPATCH PHASE -----
        # Handles routing, token sorting, grouping, and metadata construction.
        expert_inputs, expert_weights, expert_toks, metadata = self._dispatch_tokens(
            x_flat, token_ids_flat
        )
        counts_list = metadata["counts_list"]

        # ----- 2. EXECUTION PHASE -----
        # Computes feed-forward expert MLPs on contiguous sliced inputs.
        # Contains no routing, no sorting, and no scatter/accumulator writes.
        expert_outputs = []
        for e in range(self.num_experts):
            cnt = counts_list[e]
            if cnt == 0:
                expert_outputs.append(None)
                continue
            inp = expert_inputs[e]
            w_e = expert_weights[e]
            y = self.experts[e](inp) * w_e.unsqueeze(-1)
            expert_outputs.append(y)

        # ----- 3. MERGE PHASE -----
        # Combines computed expert outputs back to original token locations.
        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            y = expert_outputs[e]
            if y is not None:
                t = expert_toks[e]
                out.index_add_(0, t, y)

        if self.shared_experts is not None:
            out = out + self.shared_experts(x_flat)
        return out.reshape(Bsz, S, D)





# =============================================================================
# Decoder layer
# =============================================================================

class DeepseekV4Layer(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        compress_ratio = config.compress_ratios[layer_idx]
        is_hash = layer_idx < config.num_hash_layers

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = DeepseekV4Attention(config, compress_ratio)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = DeepseekV4MoE(config, hash_routing=is_hash)

        # mHC: parameter shapes match official ((2+n)*n outputs from a single
        # combined `_fn` matmul; 3 scalar `_scale` gates; mix_hc-sized `_base`).
        # Init: zeros for `_base` (so initial pre/post = sigmoid(0)+eps = 0.5+eps,
        # comb starts as a near-uniform softmax then Sinkhorn-projected); small
        # random for `_fn`; small `_scale`.
        n_hc = config.hc_mult
        mix_hc = (2 + n_hc) * n_hc
        flat = n_hc * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix_hc, flat))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_hc, flat))
        nn.init.normal_(self.hc_attn_fn, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.hc_ffn_fn, mean=0.0, std=config.initializer_range)
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.full((3,), 1e-2))
        self.hc_ffn_scale = nn.Parameter(torch.full((3,), 1e-2))

    def forward(self, X: torch.Tensor, mhc: MHC, token_ids: torch.Tensor,
                positions: torch.Tensor,
                rope_cos, rope_sin, rope_cos_c, rope_sin_c,
                pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Attention sub-block: hc_pre collapses [B,S,n,d] -> [B,S,d] via `pre` weights;
        # hc_post produces [B,S,n,d] = post * new_x + comb @ residual.
        residual = X
        pre, post, comb = mhc.gen_params(X, self.hc_attn_base, self.hc_attn_fn,
                                         self.hc_attn_scale)
        sub_in = MHC.hc_pre(X, pre)
        sub_in = self.attn_norm(sub_in)
        attn_out = self.attn(sub_in, positions, rope_cos, rope_sin,
                             rope_cos_c, rope_sin_c, pad_mask)
        X = MHC.hc_post(attn_out, residual, post, comb)

        # FFN sub-block
        residual = X
        pre, post, comb = mhc.gen_params(X, self.hc_ffn_base, self.hc_ffn_fn,
                                         self.hc_ffn_scale)
        sub_in = MHC.hc_pre(X, pre)
        sub_in = self.ffn_norm(sub_in)
        ffn_out = self.ffn(sub_in, token_ids)
        X = MHC.hc_post(ffn_out, residual, post, comb)
        return X

    def forward_inference(self, X: torch.Tensor, mhc: MHC,
                          token_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Cache-aware forward for incremental decode. Same mHC + sub-block
        structure as ``forward`` but the attention path uses
        ``self.attn.forward_inference(sub_in, start_pos)`` which reads/writes
        the layer's kv_cache + compressor state.
        """
        residual = X
        pre, post, comb = mhc.gen_params(X, self.hc_attn_base, self.hc_attn_fn,
                                         self.hc_attn_scale)
        sub_in = MHC.hc_pre(X, pre)
        sub_in = self.attn_norm(sub_in)
        attn_out = self.attn.forward_inference(sub_in, start_pos)
        X = MHC.hc_post(attn_out, residual, post, comb)

        residual = X
        pre, post, comb = mhc.gen_params(X, self.hc_ffn_base, self.hc_ffn_fn,
                                         self.hc_ffn_scale)
        sub_in = MHC.hc_pre(X, pre)
        sub_in = self.ffn_norm(sub_in)
        ffn_out = self.ffn(sub_in, token_ids)
        X = MHC.hc_post(ffn_out, residual, post, comb)
        return X


# =============================================================================
# MTP module (V3-style single-step)
# =============================================================================

class DeepseekV4MTPModule(nn.Module):
    """One MTP step. Mirrors the official ``MTPBlock`` (which inherits from Block):

    Pre-block:
        e = embed(input_ids); e = enorm(e); X = hnorm(X)
        X = e_proj(e).unsqueeze(2) + h_proj(X)   # broadcast e across hc copies
    Block:
        full hc_pre / attn / hc_post / hc_pre / ffn / hc_post
    Post-block:
        logits = head( hc_head_collapse(X), through final norm + lm_head )

    hc_attn_* and hc_ffn_*  shapes: [(2+n)*n, n*d] / [(2+n)*n] / [3]   (full mHC)
    hc_head_*               shapes: [n, n*d] / [n] / [1]               (pre-only)
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        d = config.hidden_size
        self.enorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.e_proj = nn.Linear(d, d, bias=False)
        self.h_proj = nn.Linear(d, d, bias=False)

        # One transformer block (dense / pure-SW attention)
        self.attn_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.attn = DeepseekV4Attention(config, compress_ratio=0)
        self.ffn_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.ffn = DeepseekV4MoE(config, hash_routing=False)
        self.norm = RMSNorm(d, eps=config.rms_norm_eps)

        n_hc = config.hc_mult
        mix_hc = (2 + n_hc) * n_hc
        flat = n_hc * d
        # Full mHC for attn and ffn sub-blocks
        for prefix in ("hc_attn", "hc_ffn"):
            fn_p = nn.Parameter(torch.zeros(mix_hc, flat))
            nn.init.normal_(fn_p, mean=0.0, std=config.initializer_range)
            setattr(self, f"{prefix}_fn", fn_p)
            setattr(self, f"{prefix}_base", nn.Parameter(torch.zeros(mix_hc)))
            setattr(self, f"{prefix}_scale", nn.Parameter(torch.full((3,), 1e-2)))
        # Pre-only mHC for head collapse
        head_fn = nn.Parameter(torch.zeros(n_hc, flat))
        nn.init.normal_(head_fn, mean=0.0, std=config.initializer_range)
        self.hc_head_fn = head_fn
        self.hc_head_base = nn.Parameter(torch.zeros(n_hc))
        self.hc_head_scale = nn.Parameter(torch.full((1,), 1e-2))

    def forward(self, X: torch.Tensor, embed: nn.Embedding, head: nn.Linear,
                input_ids: torch.Tensor, mhc: MHC, positions: torch.Tensor,
                rope_cos, rope_sin, rope_cos_c, rope_sin_c,
                pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """X: [B,S,n,d] residual stream from main model. Returns logits [B,S,V]."""
        e = embed(input_ids)                                   # [B,S,d]
        e = self.enorm(e)
        Xn = self.hnorm(X)                                      # [B,S,n,d]
        # Mix in next-token embedding broadcast across hc copies
        X = self.e_proj(e).unsqueeze(-2) + self.h_proj(Xn)      # [B,S,n,d]

        # Attention sub-block via full mHC
        residual = X
        pre, post, comb = mhc.gen_params(X, self.hc_attn_base, self.hc_attn_fn,
                                         self.hc_attn_scale)
        sub_in = MHC.hc_pre(X, pre)
        sub_in = self.attn_norm(sub_in)
        attn_out = self.attn(sub_in, positions, rope_cos, rope_sin,
                             rope_cos_c, rope_sin_c, pad_mask)
        X = MHC.hc_post(attn_out, residual, post, comb)

        # FFN sub-block via full mHC
        residual = X
        pre, post, comb = mhc.gen_params(X, self.hc_ffn_base, self.hc_ffn_fn,
                                         self.hc_ffn_scale)
        sub_in = MHC.hc_pre(X, pre)
        sub_in = self.ffn_norm(sub_in)
        ffn_out = self.ffn(sub_in, input_ids)
        X = MHC.hc_post(ffn_out, residual, post, comb)

        # Head: pre-only mHC collapse, then norm, then shared lm_head
        head_pre = mhc.gen_head_pre(X, self.hc_head_fn, self.hc_head_base,
                                    self.hc_head_scale)
        h_out = MHC.hc_pre(X, head_pre)
        h_out = self.norm(h_out)
        return head(h_out)


# =============================================================================
# PreTrainedModel base + top-level classes
# =============================================================================

class DeepseekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = ""           # flat layout, no `model.` prefix
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV4Layer", "DeepseekV4MTPModule"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class DeepseekV4Model(DeepseekV4PreTrainedModel):
    """The base model exposes the same fields as ForCausalLM (flat layout)
    so that names match the official safetensors. We instantiate it as part of
    ForCausalLM rather than wrapping it.
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size,
                                  padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            DeepseekV4Layer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Head-side mHC (collapses [B,S,n_hc,d] residual stream back to [B,S,d])
        # Head-side mHC: ONLY computes the `pre` (collapse hc -> 1) weights,
        # so shapes are [hc, hc*d] / [hc] / [1] (matching official ParallelHead).
        n_hc = config.hc_mult
        flat = n_hc * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.zeros(n_hc, flat))
        nn.init.normal_(self.hc_head_fn, mean=0.0, std=config.initializer_range)
        self.hc_head_base = nn.Parameter(torch.zeros(n_hc))
        self.hc_head_scale = nn.Parameter(torch.full((1,), 1e-2))

        self._mhc = MHC(config.hidden_size, config.hc_mult,
                        sinkhorn_iters=config.hc_sinkhorn_iters,
                        eps=config.rms_norm_eps)
        # MTP modules
        self.mtp = nn.ModuleList([
            DeepseekV4MTPModule(config) for _ in range(config.num_nextn_predict_layers)
        ])
        self.post_init()

    def _build_rope(self, max_len: int, device, dtype):
        rope_dim = self.config.qk_rope_head_dim
        cos, sin = build_rope_cache(max_len, rope_dim, self.config.rope_theta, device, dtype)
        cos_c, sin_c = build_rope_cache(max_len, rope_dim,
                                        self.config.compress_rope_theta, device, dtype)
        return cos, sin, cos_c, sin_c

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                **kwargs) -> BaseModelOutputWithPast:
        Bsz, S = input_ids.shape
        device = input_ids.device
        h = self.embed(input_ids)                                # [B,S,d]
        # Lift into mHC residual stream [B,S,n_hc,d]
        n_hc = self.config.hc_mult
        X = h.unsqueeze(-2).expand(-1, -1, n_hc, -1).contiguous()

        if position_ids is None:
            positions = torch.arange(S, device=device)
        else:
            positions = position_ids[0]

        # Cap RoPE table at S to keep memory bounded (model still supports up to max_position_embeddings)
        rope_cos, rope_sin, rope_cos_c, rope_sin_c = self._build_rope(S, device, h.dtype)

        pad_mask = attention_mask.bool() if attention_mask is not None else None

        for layer in self.layers:
            X = layer(X, self._mhc, input_ids, positions,
                      rope_cos, rope_sin, rope_cos_c, rope_sin_c, pad_mask)

        # Head-side mHC: collapse residual back to [B,S,d] using A_l
        # Head mHC: pre-only collapse hc -> 1, then final norm
        head_pre = self._mhc.gen_head_pre(X, self.hc_head_fn, self.hc_head_base,
                                          self.hc_head_scale)
        h_out = MHC.hc_pre(X, head_pre)
        h_out = self.norm(h_out)
        return BaseModelOutputWithPast(last_hidden_state=h_out)


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel):
    _tied_weights_keys: List[str] = []   # untied (matches V4 config)

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        # Flat layout — instantiate the base model's fields directly on self
        # so safetensors keys come out as `embed.weight`, `layers.0...`, etc.
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size,
                                  padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            DeepseekV4Layer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Head-side mHC: ONLY computes the `pre` (collapse hc -> 1) weights,
        # so shapes are [hc, hc*d] / [hc] / [1] (matching official ParallelHead).
        n_hc = config.hc_mult
        flat = n_hc * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.zeros(n_hc, flat))
        nn.init.normal_(self.hc_head_fn, mean=0.0, std=config.initializer_range)
        self.hc_head_base = nn.Parameter(torch.zeros(n_hc))
        self.hc_head_scale = nn.Parameter(torch.full((1,), 1e-2))

        self._mhc = MHC(config.hidden_size, config.hc_mult,
                        sinkhorn_iters=config.hc_sinkhorn_iters,
                        eps=config.rms_norm_eps)
        self.mtp = nn.ModuleList([
            DeepseekV4MTPModule(config) for _ in range(config.num_nextn_predict_layers)
        ])
        self.post_init()

    # HF auto methods
    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new):
        self.head = new

    # ----------------------------------------------------------------------
    # KV-cache inference path (mirrors inference/model.py:Transformer.forward)
    # ----------------------------------------------------------------------
    def setup_caches(self, max_batch_size: int, max_seq_len: int):
        """Allocate per-layer kv_cache + compressor state buffers and
        precompute RoPE tables sized for ``max_seq_len``. Call once before
        any ``forward_inference`` calls; call ``reset_caches`` between
        independent generation runs."""
        ref = next(iter(self.parameters()))
        device, dtype = ref.device, ref.dtype
        rope_dim = self.config.qk_rope_head_dim
        rope_cos, rope_sin = build_rope_cache(max_seq_len, rope_dim,
                                               self.config.rope_theta,
                                               device, dtype)
        rope_cos_c, rope_sin_c = build_rope_cache(max_seq_len, rope_dim,
                                                   self.config.compress_rope_theta,
                                                   device, dtype)
        self._inf_rope_cos = rope_cos
        self._inf_rope_sin = rope_sin
        self._inf_rope_cos_c = rope_cos_c
        self._inf_rope_sin_c = rope_sin_c
        self._inf_max_batch = max_batch_size
        self._inf_max_seq = max_seq_len
        self._inf_seen = 0
        for layer in self.layers:
            layer.attn.setup_caches(max_batch_size, max_seq_len,
                                     rope_cos, rope_sin, rope_cos_c, rope_sin_c)

    def reset_caches(self):
        self._inf_seen = 0
        for layer in self.layers:
            layer.attn.reset_cache()

    def forward_inference(self, input_ids: torch.Tensor,
                          start_pos: Optional[int] = None) -> torch.Tensor:
        """Cache-aware forward. ``start_pos`` defaults to the cumulative
        ``seen`` count; pass an int to override (or for parallel workers).
        Returns logits ``[B, S, V]``."""
        assert hasattr(self, "_inf_rope_cos"), "setup_caches() not called"
        if start_pos is None:
            start_pos = self._inf_seen
        Bsz, S = input_ids.shape
        h = self.embed(input_ids)
        n_hc = self.config.hc_mult
        X = h.unsqueeze(-2).expand(-1, -1, n_hc, -1).contiguous()
        for layer in self.layers:
            X = layer.forward_inference(X, self._mhc, input_ids, start_pos)
        head_pre = self._mhc.gen_head_pre(X, self.hc_head_fn, self.hc_head_base,
                                          self.hc_head_scale)
        h_out = MHC.hc_pre(X, head_pre)
        h_out = self.norm(h_out)
        logits = self.head(h_out)
        self._inf_seen = start_pos + S
        return logits

    def _backbone(self, input_ids, attention_mask, position_ids):
        """Runs embed -> hc-expand -> N layers and returns BOTH the post-layer
        residual stream X (shape [B,S,n_hc,d], needed by MTP) and the head-collapsed
        hidden state (shape [B,S,d], needed by lm_head).
        """
        Bsz, S = input_ids.shape
        device = input_ids.device
        h = self.embed(input_ids)
        n_hc = self.config.hc_mult
        X = h.unsqueeze(-2).expand(-1, -1, n_hc, -1).contiguous()

        if position_ids is None:
            positions = torch.arange(S, device=device)
        else:
            positions = position_ids[0]

        rope_dim = self.config.qk_rope_head_dim
        rope_cos, rope_sin = build_rope_cache(S, rope_dim, self.config.rope_theta,
                                              device, h.dtype)
        rope_cos_c, rope_sin_c = build_rope_cache(S, rope_dim,
                                                   self.config.compress_rope_theta,
                                                   device, h.dtype)
        pad_mask = attention_mask.bool() if attention_mask is not None else None

        for layer in self.layers:
            X = layer(X, self._mhc, input_ids, positions,
                      rope_cos, rope_sin, rope_cos_c, rope_sin_c, pad_mask)

        # Head mHC: pre-only collapse hc -> 1, then final norm
        head_pre = self._mhc.gen_head_pre(X, self.hc_head_fn, self.hc_head_base,
                                          self.hc_head_scale)
        h_out = MHC.hc_pre(X, head_pre)
        h_out = self.norm(h_out)
        return X, h_out, positions, rope_cos, rope_sin, rope_cos_c, rope_sin_c, pad_mask

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                return_dict: bool = True,
                use_mtp: bool = False,
                **kwargs) -> CausalLMOutputWithPast:
        X, hidden, positions, rc, rs, rcc, rsc, pad_mask = self._backbone(
            input_ids, attention_mask, position_ids
        )
        logits = self.head(hidden)

        loss = None
        mtp_logits_list = []
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # MTP: each step k predicts token at offset +(k+2). Feed embedding of
            # the next token shifted by (k+1) into the MTP module along with the
            # current residual stream.
            for k, mtp in enumerate(self.mtp):
                shift = k + 1
                next_ids = F.pad(input_ids[:, shift:], (0, shift), value=0)
                mtp_logits = mtp(X, self.embed, self.head, next_ids, self._mhc,
                                 positions, rc, rs, rcc, rsc, pad_mask)
                mtp_target = F.pad(labels[:, shift + 1:], (0, shift + 1), value=-100)
                mtp_loss = F.cross_entropy(
                    mtp_logits.view(-1, mtp_logits.size(-1)),
                    mtp_target.view(-1),
                    ignore_index=-100,
                )
                loss = loss + 0.3 * mtp_loss
                mtp_logits_list.append(mtp_logits)

        return CausalLMOutputWithPast(loss=loss, logits=logits)
