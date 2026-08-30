
import torch.nn as nn

from barcodebert.janus_convnova import JanusConvNova


class JanusConvNovaOutput:
    def __init__(self, logits=None, hidden_states=None, loss=None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.loss = loss


class JanusConvNovaWrapper(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size = getattr(config, "hidden_dim", 256)

        self.model = JanusConvNova(
            hidden_dim=self.hidden_size,
            dropout=getattr(config, "dropout", 0.1),
            num_cnn_stacks=getattr(config, "num_cnn_stacks", 2),
            alphabet_size=5,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return JanusConvNovaOutput(
            logits=outputs["logits"],
            hidden_states=outputs["hidden_states"],
            loss=outputs["loss"],
        )

    def encode(self, input_ids, attention_mask=None):
        return self.model.encode(input_ids)
