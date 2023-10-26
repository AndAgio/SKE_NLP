import torch
from torch import nn
from .config import BertConfig
from .encoder import BERTEncoder
from .embeddings import BERTEmbeddings
from .pooler import BERTPooler


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, from_seq_length]
        # So we can broadcast to [batch_size, num_heads, to_seq_length, from_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers, all_encoder_attention_scores = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output, optional_attn_mask=attention_mask)
        return all_encoder_layers, pooled_output, all_encoder_attention_scores, embedding_output
