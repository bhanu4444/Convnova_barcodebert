
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


class ConvNovaLongGCB(nn.Module):
    """
    ConvNova Gated Convolution Block.

    This is intentionally kept close to the original
    ConvNova dual-branch gated design.

    The main experimental change in ConvNova-Long is
    the dilation schedule, not additional FFN capacity.
    """

    def __init__(
        self,
        hidden_dim,
        dilation=1,
        kernel_size=9,
        dropout=0.1,
    ):
        super().__init__()

        padding = ((kernel_size - 1) // 2) * dilation

        # Forward branch
        self.norm = nn.LayerNorm(hidden_dim)

        # Reverse-complement / gating branch
        self.rc_norm = nn.LayerNorm(hidden_dim)

        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.gate = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, rc_feat):

        # --------------------------------------------------
        # Forward branch
        # A -> LN -> dilated Conv -> GELU
        # --------------------------------------------------

        a = feat.permute(0, 2, 1)

        a = self.norm(a)

        a = a.permute(0, 2, 1)

        h = self.conv(a)

        h = F.gelu(h)

        # --------------------------------------------------
        # Gating branch
        # B -> LN -> dilated Conv -> sigmoid
        # --------------------------------------------------

        b = rc_feat.permute(0, 2, 1)

        b = self.rc_norm(b)

        b = b.permute(0, 2, 1)

        g = self.gate(b)

        g = torch.sigmoid(g)

        # --------------------------------------------------
        # Dual-branch gated residual update
        # --------------------------------------------------

        h = self.dropout(h)

        feat = feat + h * g

        rc_feat = rc_feat + g

        return feat, rc_feat


class ConvNovaLongModel(nn.Module):
    """
    ConvNova-Long.

    Main differences from the original integrated ConvNova:
        1. Explicit long-range dilation schedule
        2. Full-barcode receptive field
        3. Same nucleotide-level representation
        4. Same reverse-complement pathway
        5. Same MLM interface
    """

    def __init__(
        self,
        hidden_dim=256,
        dropout=0.1,
        alphabet_size=5,
        pretrain=True,
        kernel_size=9,
        dilation_schedule=None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.alphabet_size = alphabet_size
        self.pretrain = pretrain

        if dilation_schedule is None:
            dilation_schedule = [
                1, 1,
                2,
                4,
                8,
                16,
                32,
                16,
                8,
                4,
            ]

        self.dilation_schedule = dilation_schedule
        self.num_layers = len(dilation_schedule)

        # --------------------------------------------------
        # Initial forward / reverse-complement projections
        # --------------------------------------------------

        self.linear = nn.Conv1d(
            alphabet_size,
            hidden_dim,
            kernel_size=9,
            padding=4,
        )

        self.rc_linear = nn.Conv1d(
            alphabet_size,
            hidden_dim,
            kernel_size=9,
            padding=4,
        )

        # --------------------------------------------------
        # Gated convolution blocks
        # --------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                ConvNovaLongGCB(
                    hidden_dim=hidden_dim,
                    dilation=d,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for d in dilation_schedule
            ]
        )

        # --------------------------------------------------
        # Final representation refinement
        # --------------------------------------------------

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # --------------------------------------------------
        # MLM head
        # --------------------------------------------------

        if pretrain:
            self.out_linear = nn.Linear(
                hidden_dim,
                alphabet_size,
            )

        self.dropout = nn.Dropout(dropout)

    # ======================================================
    # Reverse complement
    # ======================================================

    def reverse_complement(self, seq):

        # A C G T N -> T G C A N
        rc_seq = 3 - seq

        n_mask = seq == 4
        rc_seq[n_mask] = 4

        return rc_seq

    # ======================================================
    # Encoder
    # ======================================================

    def encode(self, seq):

        rc_seq = self.reverse_complement(seq)

        seq = F.one_hot(
            seq,
            num_classes=self.alphabet_size,
        ).float()

        rc_seq = F.one_hot(
            rc_seq,
            num_classes=self.alphabet_size,
        ).float()

        feat = seq.permute(0, 2, 1)
        rc_feat = rc_seq.permute(0, 2, 1)

        feat = F.gelu(self.linear(feat))
        rc_feat = F.gelu(self.rc_linear(rc_feat))

        # --------------------------------------------------
        # Long-range GCB stack
        # --------------------------------------------------

        for block in self.blocks:

            feat, rc_feat = block(
                feat,
                rc_feat,
            )

        # --------------------------------------------------
        # Final feature refinement
        # --------------------------------------------------

        feat = feat.permute(0, 2, 1)

        residual = feat

        feat = self.final_norm(feat)
        feat = self.mlp(feat)

        feat = feat + residual

        return feat

    # ======================================================
    # Pretraining forward
    # ======================================================

    def forward(self, seq):

        if self.pretrain:

            mask = seq[1]
            seq = seq[0]

        feat = self.encode(seq)

        if self.pretrain:

            logits = self.out_linear(feat)

            CausalLMOutput = namedtuple(
                "CausalLMOutput",
                ["logits"],
            )

            return (
                CausalLMOutput(
                    logits=(logits, mask)
                ),
                None,
            )

        return feat
