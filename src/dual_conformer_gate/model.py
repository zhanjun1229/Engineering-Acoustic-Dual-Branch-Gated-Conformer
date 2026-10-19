"""Independent Conformer encoders with sample-wise softmax gating."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, ff_mult=4, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.conv_ln = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(d_model),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ff1(self.ln1(x))
        qkv = self.attn_ln(x)
        attn_out, _ = self.attn(qkv, qkv, qkv, need_weights=False)
        x = x + attn_out
        conv_input = self.conv_ln(x).transpose(1, 2)
        x = x + self.conv(conv_input).transpose(1, 2)
        x = x + self.ff2(x)
        return x


class ConformerDualEncoderClassifier(nn.Module):
    """
    双分支独立Conformer + 样本级动态Gate。

    - acoustic分支：MFCC39+LFCC39+CQCC39+GFCC39，156维；
    - industrial分支：IAPD26，26维；
    - 不使用Cross-Attention；
    - 两个分支池化后，Gate根据二者联合表示输出2个Softmax权重；
    - 加权后的两个表示拼接，输入二分类器。
    """

    def __init__(
        self,
        acoustic_input_dim=156,
        industrial_input_dim=26,
        d_model=128,
        num_blocks=2,
        n_heads=4,
        ff_mult=4,
        dropout=0.2,
    ):
        super().__init__()
        self.acoustic_proj = nn.Linear(acoustic_input_dim, d_model)
        self.industrial_proj = nn.Linear(industrial_input_dim, d_model)

        self.acoustic_blocks = nn.Sequential(
            *[
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.industrial_blocks = nn.Sequential(
            *[
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # 每个样本输出 [声学权重, IAPD权重]，Softmax保证两项之和为1。
        gate_hidden = max(16, d_model // 2)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, gate_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 2),
        )

        # 两个分支各输出 d_model 维，拼接后为 2*d_model 维。
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        acoustic: torch.Tensor,
        industrial: torch.Tensor,
        return_gate: bool = False,
    ):
        acoustic = self.acoustic_proj(acoustic)
        industrial = self.industrial_proj(industrial)

        acoustic = self.acoustic_blocks(acoustic).mean(dim=1)
        industrial = self.industrial_blocks(industrial).mean(dim=1)

        gate_input = torch.cat([acoustic, industrial], dim=1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=1)

        acoustic_weighted = acoustic * gate_weights[:, 0:1]
        industrial_weighted = industrial * gate_weights[:, 1:2]
        fused = torch.cat([acoustic_weighted, industrial_weighted], dim=1)
        logits = self.classifier(fused)

        if return_gate:
            return logits, gate_weights
        return logits
