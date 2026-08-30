
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Left-padded 1D convolution.

    Output at position t can only depend on positions <= t.
    With the additional one-position shift used below, the
    representation used for prediction at t excludes x[t].
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class JanusConvNova(nn.Module):
    """
    Janus-style bidirectional ConvNova.

    The same causal ConvNova backbone is applied to:
        1. the forward sequence
        2. the reversed sequence

    The reverse representation is flipped back into the original
    nucleotide order before fusion.

    Each prediction position is constructed from context on the
    opposite sides of the target nucleotide.
    """

    def __init__(
        self,
        hidden_dim=256,
        dropout=0.1,
        num_cnn_stacks=2,
        alphabet_size=5,
        kernel_size=9,
        num_conv1d=5,
        dilation_base=2,
        dilation_max=1024,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.alphabet_size = alphabet_size
        self.num_cnn_stacks = num_cnn_stacks
        self.num_conv1d = num_conv1d
        self.num_layers = num_conv1d * num_cnn_stacks

        # ---------------------------------------------------------
        # Shared input projection
        # ---------------------------------------------------------
        self.input_proj = CausalConv1d(
            alphabet_size,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
        )

        # ---------------------------------------------------------
        # Shared causal convolutional backbone
        # ---------------------------------------------------------
        dilations = [
            1 if i < 2
            else min(dilation_max, dilation_base ** (i - 2))
            for i in range(num_conv1d)
        ]

        self.convs = nn.ModuleList()

        for d in dilations:
            for _ in range(num_cnn_stacks):
                self.convs.append(
                    CausalConv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        dilation=d,
                    )
                )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)]
        )

        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Final shared representation transform
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ---------------------------------------------------------
        # Janus fusion
        # ---------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ---------------------------------------------------------
        # Nucleotide prediction head
        # ---------------------------------------------------------
        self.out_linear = nn.Linear(hidden_dim, alphabet_size)

    def _encode_direction(self, input_ids):
        """
        Encode one direction using the shared causal ConvNova backbone.

        input_ids:
            [B, L]

        returns:
            [B, L, H]
        """

        # A,C,G,T,N -> 0,1,2,3,4
        x = F.one_hot(
            input_ids,
            num_classes=self.alphabet_size,
        ).float()

        # [B,L,5] -> [B,5,L]
        x = x.permute(0, 2, 1)

        x = self.input_proj(x)
        x = F.gelu(x)

        for i in range(self.num_layers):

            residual = x

            # [B,C,L] -> [B,L,C]
            h = x.permute(0, 2, 1)
            h = self.norms[i](h)

            h = self.dropout(h)

            # [B,L,C] -> [B,C,L]
            h = h.permute(0, 2, 1)

            h = self.convs[i](h)
            h = F.gelu(h)

            # residual
            x = h + residual

            # FFN
            h = x.permute(0, 2, 1)
            h = self.ffn[i](h)
            x = x + h.permute(0, 2, 1)

        x = x.permute(0, 2, 1)

        x = self.final_mlp(x)

        return x

    def encode(self, input_ids):
        """
        Return the fused Janus representation.

        input_ids:
            [B,L]

        returns:
            [B,L,H]
        """

        # ---------------------------------------------------------
        # Forward direction
        # ---------------------------------------------------------
        forward = self._encode_direction(input_ids)

        # ---------------------------------------------------------
        # Reverse direction
        # ---------------------------------------------------------
        reversed_input = input_ids.flip(dims=(1,))

        backward = self._encode_direction(reversed_input)

        # Restore original nucleotide ordering
        backward = backward.flip(dims=(1,))

        # ---------------------------------------------------------
        # Important:
        #
        # A causal representation at position t can still contain
        # x[t]. Therefore we shift both directional representations
        # by one position before prediction.
        #
        # Forward[t]  -> information from x[<t]
        # Backward[t] -> information from x[>t]
        # ---------------------------------------------------------

        forward_context = torch.zeros_like(forward)
        forward_context[:, 1:, :] = forward[:, :-1, :]

        backward_context = torch.zeros_like(backward)
        backward_context[:, :-1, :] = backward[:, 1:, :]

        # ---------------------------------------------------------
        # Fuse left and right context
        # ---------------------------------------------------------
        fused = torch.cat(
            [
                forward_context,
                backward_context,
            ],
            dim=-1,
        )

        fused = self.fusion(fused)

        return fused

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        """
        Forward pass.

        If labels are supplied, loss is calculated over all
        nucleotide positions.
        """

        hidden_states = self.encode(input_ids)

        logits = self.out_linear(hidden_states)

        loss = None

        if labels is not None:

            if attention_mask is not None:
                valid = attention_mask.bool()

                loss = F.cross_entropy(
                    logits[valid],
                    labels[valid],
                )
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, self.alphabet_size),
                    labels.reshape(-1),
                )

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "loss": loss,
        }
