import torch
import torch.nn as nn

from barcodebert.convnova_long import ConvNovaLongModel


class ConvNovaLongOutput:

    def __init__(self, logits):
        # Training code requires logits to be a Tensor:
        # [batch, sequence_length, 5]
        if isinstance(logits, (tuple, list)):
            # ConvNovaLongModel may return multiple prediction tensors.
            # The first element is the nucleotide prediction tensor.
            logits = logits[0]

        if not torch.is_tensor(logits):
            raise TypeError(
                f"ConvNovaLongOutput.logits must be a Tensor, "
                f"got {type(logits)}"
            )

        self.logits = logits


class ConvNovaLongWrapper(nn.Module):

    def __init__(self, config):

        super().__init__()

        hidden_dim = getattr(
            config,
            "hidden_dim",
            256,
        )

        dropout = getattr(
            config,
            "dropout",
            0.1,
        )

        self.model = ConvNovaLongModel(
            hidden_dim=hidden_dim,
            dropout=dropout,
            alphabet_size=5,
            pretrain=True,
        )

        self.hidden_size = hidden_dim

    def _prepare_input(self, input_ids):

        # ConvNovaLongModel.encode() performs the one-hot
        # conversion internally.
        #
        # External interface:
        #   [L]    or    [B, L]
        #
        # Internal model input:
        #   integer nucleotide IDs
        #
        # A=0, C=1, G=2, T=3, N=4

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        return input_ids.long()

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):

        outputs, _ = self.model(
            (
                input_ids,
                attention_mask,
            )
        )

        # ConvNovaLongModel.forward() returns:
        #
        # (
        #     CausalLMOutput(
        #         logits=(logits, mask)
        #     ),
        #     None,
        # )
        #
        # BarcodeBERT expects:
        #     out.logits -> Tensor [B, L, 5]

        logits = outputs.logits

        # Remove the auxiliary mask returned by ConvNovaLong.
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if not torch.is_tensor(logits):
            raise TypeError(
                f"ConvNovaLong logits must be a Tensor, "
                f"got {type(logits)}"
            )

        return ConvNovaLongOutput(logits)


    def encode(self, input_ids):

        input_ids = self._prepare_input(input_ids)

        return self.model.encode(
            input_ids
        )
