from typing import Optional

import torch
import torch.nn as nn


class PerFramePatchAttnPool(nn.Module):
    """
    对每一帧的 patch 特征做 K-token 注意力池化。
    输入形状约定：
        x:    [B, T, P, C]
        mask: [B, T, P]，其中 True 表示 padding（可选）

    输出：
        pooled: [B, T, K, C]
    """

    def __init__(
        self,
        dim: int = 768,
        num_queries: int = 4,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries

        # K 个可学习 query，跨 batch & frame 共享
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # [B, L, C]
        )

        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.latent_queries, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                  # [B, T, P, C]
        mask: Optional[torch.Tensor] = None,  # [B, T, P] (True=padding)
    ) -> torch.Tensor:
        B, T, P, C = x.shape
        assert C == self.dim, f"dim mismatch: {C} vs {self.dim}"

        # 合并 batch & time 维度，每帧单独做 cross-attn
        x_ = x.view(B * T, P, C)          # [B*T, P, C]

        if mask is not None:
            key_padding_mask = mask.view(B * T, P)  # [B*T, P]
        else:
            key_padding_mask = None

        # queries: [K, C] -> [B*T, K, C]
        q = self.latent_queries.unsqueeze(0).expand(B * T, -1, -1)

        # MultiheadAttention: (Q, K, V) 形状均为 [B, L, C]
        pooled, _ = self.attn(
            query=q,                # [B*T, K, C]
            key=x_,                 # [B*T, P, C]
            value=x_,               # [B*T, P, C]
            key_padding_mask=key_padding_mask,  # True 表示忽略
            need_weights=False,
        )  # -> [B*T, K, C]

        pooled = self.out_proj(pooled)   # 线性 + dropout

        # 还原到 [B, T, K, C]
        pooled = pooled.view(B, T, self.num_queries, C)
        return pooled


class ResMLPProjector(nn.Module):
    """
    ResMLP Projector: 将视频特征从 CLIP/LSTV 维度映射到 LLM hidden 维度。
    结构与你现有的 visual_feature_proj 保持一致：
    - 可选输入 LayerNorm
    - Linear(in_dim -> out_dim)
    - Pre-LN 残差 MLP (hidden_dim = 4*out_dim)
    - 输出 LayerNorm
    - MLP 最后一层 zero-init，保证初期近似恒等
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 1536,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_input_ln: bool = True,
        use_output_ln: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim * 4

        self.use_input_ln = use_input_ln
        self.input_ln = nn.LayerNorm(in_dim) if use_input_ln else nn.Identity()
        self.input_proj = nn.Linear(in_dim, out_dim)

        self.ln_mlp = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

        self.output_ln = nn.LayerNorm(out_dim) if use_output_ln else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        # 关键：MLP 最后一层 zero-init
        last_linear: nn.Linear = self.mlp[-2]  # 倒数第二个是 Linear
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, in_dim]
        """
        x = self.input_ln(x)
        x = self.input_proj(x)           # [B, N, out_dim]

        residual = x
        x = self.ln_mlp(x)
        x = self.mlp(x)
        x = x + residual

        x = self.output_ln(x)
        return x


class VideoTimeEmbedding(nn.Module):
    """
    帧级 Video Time Embedding (2fps)，只加在视频 token 上。
    """

    def __init__(
        self,
        max_frames: int = 256,
        dim: int = 1536,
        fps: float = 2.0,
    ):
        super().__init__()
        self.max_frames = max_frames
        self.dim = dim
        self.fps = fps

        self.time_emb = nn.Embedding(max_frames, dim)
        nn.init.normal_(self.time_emb.weight, std=0.02)

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        frame_indices: [B, T] int64，2fps 下的帧索引 0..max_frames-1
        返回:
            vte: [B, T, dim]
        """
        frame_indices = frame_indices.clamp(0, self.max_frames - 1)
        vte = self.time_emb(frame_indices)
        return vte


def apply_vte_and_flatten(
    projected: torch.Tensor,         # [B, T, M, D], M=1+K
    frame_indices: torch.Tensor,     # [B, T]
    vte_module: VideoTimeEmbedding,
    xscale: float = 1.0,
) -> torch.Tensor:
    """
    将 projector 输出与 VTE 相加并展平:
        [B, T, M, D] + [B, T, D] -> [B, T*M, D]
    """
    B, T, M, D = projected.shape
    assert D == vte_module.dim

    # VTE: [B, T, D] -> [B, T, 1, D] -> broadcast 到 M
    vte = vte_module(frame_indices)          # [B, T, D]
    vte = vte.unsqueeze(2).expand(-1, -1, M, -1)  # [B, T, M, D]

    x = (projected + vte) * xscale           # [B, T, M, D]
    x = x.view(B, T * M, D)                  # [B, T*M, D]
    return x


class VideoConditionEncoder(nn.Module):
    """
    输入:
        video_feats: [B, T, 50, 768]
            - 每帧: index 0 为 CLS，1..49 为 spatial patches
        video_len:   [B]，每条样本有效帧数 (<= T)

    内部:
        1. 保留每帧 CLS
        2. 对每帧 49 个 patch 做 K-token 注意力池化
        3. per-frame 拼 CLS + K 个 pooled token -> [B, T, 1+K, 768]
        4. ResMLPProjector: -> [B, T, 1+K, d_model]
        5. 加 VideoTimeEmbedding (按帧共享到 1+K 个 token)，乘 xscale
        6. 展平得到 [B, N_video, d_model]，其中 N_video = T*(1+K)

    输出:
        video_cond_emb: [B, N_video, d_model]
    """

    def __init__(
        self,
        patch_dim: int = 768,
        d_model: int = 1536,
        num_queries_per_frame: int = 4,
        num_heads: int = 8,
        max_frames: int = 256,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.1,
        fps: float = 2.0,
        xscale: float = 1.0,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.d_model = d_model
        self.num_queries_per_frame = num_queries_per_frame
        self.xscale = xscale

        self.pool = PerFramePatchAttnPool(
            dim=patch_dim,
            num_queries=num_queries_per_frame,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.projector = ResMLPProjector(
            in_dim=patch_dim,
            out_dim=d_model,
            hidden_dim=d_model * 4,
            dropout=proj_dropout,
            use_input_ln=True,
            use_output_ln=True,
        )

        self.vte = VideoTimeEmbedding(
            max_frames=max_frames,
            dim=d_model,
            fps=fps,
        )

    def forward(
        self,
        video_feats: torch.Tensor,   # [B, T, 50, 768]
        video_len: torch.Tensor,     # [B]，每条样本有效帧数
    ) -> torch.Tensor:
        B, T, P, C = video_feats.shape
        assert P == 50, f"expect P=50, got {P}"
        assert C == self.patch_dim, f"dim mismatch: {C} vs {self.patch_dim}"

        # 1. 拆 CLS 与 patch：CLS 保留，patch 参与池化
        cls_token = video_feats[:, :, 0:1, :]    # [B, T, 1, C]
        patch_tokens = video_feats[:, :, 1:, :]  # [B, T, 49, C]

        # 2. 构造 patch_mask: 对于 t >= video_len[b] 的帧，49 个 patch 全是 padding
        device = video_feats.device
        frame_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
        valid_mask_frames = frame_indices < video_len.unsqueeze(1)  # [B, T]
        # patch_mask: [B, T, 49]，True = padding
        patch_mask = ~valid_mask_frames.unsqueeze(-1).expand(-1, -1, patch_tokens.size(2))

        # 3. 每帧 49 patch -> K-token 注意力池化
        pooled = self.pool(patch_tokens, mask=patch_mask)  # [B, T, K, C]

        # 4. 拼 CLS + pooled: [B, T, 1+K, C]
        frame_tokens = torch.cat([cls_token, pooled], dim=2)  # M = 1 + K

        # 5. 展平成 [B, T*M, C] 喂给 projector
        B, T, M, C = frame_tokens.shape
        frame_tokens_flat = frame_tokens.view(B, T * M, C)       # [B, T*M, C]

        projected_flat = self.projector(frame_tokens_flat)       # [B, T*M, d_model]
        projected = projected_flat.view(B, T, M, self.d_model)   # [B, T, M, d_model]

        # 6. 加 VTE + xscale，并展平
        video_cond_emb = apply_vte_and_flatten(
            projected=projected,
            frame_indices=frame_indices,
            vte_module=self.vte,
            xscale=self.xscale,
        )  # [B, T*M, d_model]

        return video_cond_emb
