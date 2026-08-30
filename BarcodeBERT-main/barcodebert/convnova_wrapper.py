
from types import SimpleNamespace
import sys
import torch.nn as nn

sys.path.append("/content/ConvNova-main/convnova")

from src.models.ConvNova.convnova import CNNModel


class ConvNovaOutput:
    def __init__(self, logits=None):
        self.logits = logits


class ConvNovaWrapper(nn.Module):

    def __init__(self, config):
        super().__init__()

        args = SimpleNamespace(
            hidden_dim=getattr(config, "hidden_dim", 256),
            dropout=getattr(config, "dropout", 0.1),
            num_cnn_stacks=getattr(config, "num_cnn_stacks", 2),
        )

        self.model = CNNModel(
            args=args,
            alphabet_size=5,
            pretrain=True,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        outputs, _ = self.model((input_ids, attention_mask))
        return ConvNovaOutput(logits=outputs.logits[0])

    def encode(self, input_ids, attention_mask=None):
        return self.model.encode(input_ids)
