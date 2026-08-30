from transformers import BertConfig, BertForTokenClassification

from barcodebert.convnova_wrapper import ConvNovaWrapper
from barcodebert.janus_convnova_wrapper import JanusConvNovaWrapper
from barcodebert.convnova_long_wrapper import ConvNovaLongWrapper

def build_model(config, dataset):
    """
    Build the requested pretraining model.

    Returns
    -------
    model
    metadata
    """

    arch = config.arch.lower()

    ############################################################
    # BarcodeBERT (Transformer)
    ############################################################
    if arch == "transformer":

        bert_config = BertConfig(
            vocab_size=dataset.vocab_size,
            num_hidden_layers=config.n_layers,
            num_attention_heads=config.n_heads,
            num_labels=4 ** config.k_mer,
            output_hidden_states=True,
            max_position_embeddings=max(
                512,
                (1536 + config.stride - 1) // config.stride,
            ),
        )

        model = BertForTokenClassification(bert_config)

        metadata = {
            "type": "transformer",
            "bert_config": bert_config,
        }

        return model, metadata

    ############################################################
    # ConvNova
    ############################################################
    elif arch == "convnova":

        model = ConvNovaWrapper(config)

        metadata = {
            "type": "convnova",
            "bert_config": None,
        }

        return model, metadata

    ############################################################
    # Janus-ConvNova
    ############################################################
    elif arch == "janus_convnova":

        model = JanusConvNovaWrapper(config)

        metadata = {
            "type": "janus_convnova",
            "bert_config": None,
        }

        return model, metadata

    ############################################################
    elif arch == "convnova_long":

        model = ConvNovaLongWrapper(config)

        metadata = {
            "type": "convnova_long",
            "bert_config": None,
        }

        return model, metadata  
    raise ValueError(f"Unknown architecture: {config.arch}")
