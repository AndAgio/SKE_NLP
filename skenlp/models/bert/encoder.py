import copy
from torch import nn
from .layer import BERTLayer


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.num_hidden_layers = config.num_hidden_layers

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_encoder_attention_scores = []
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            all_encoder_attention_scores.append(attention_probs.data)
        return all_encoder_layers, all_encoder_attention_scores
