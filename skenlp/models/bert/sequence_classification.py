import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification
from .bert import BertModel
from skenlp.utils import DumbLogger


class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels, checkpoint=None, logger=None):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.logger = logger if logger is not None else DumbLogger()
        self.name = 'bert'
        self.n_labels = num_labels
        self.config = config

        if not checkpoint:
            self.logger.print_it('No checkpoint found. Loading the transformer pretrained bert-base-uncased...')
            bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
            checkpoint_dir = os.path.join('outs', 'models')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.checkpoint = os.path.join(checkpoint_dir, 'bert-base-uncased.pt')
            model_dict = bert_model.state_dict()
            del model_dict['bert.embeddings.position_ids']
            torch.save(model_dict, self.checkpoint)
        else:
            self.logger.print_it('Checkpoint found. Loading the given pretrained BERT...')
            self.checkpoint = checkpoint

        self.load_checkpoint()

    def load_checkpoint(self):
        loaded = torch.load(self.checkpoint, map_location=torch.device('cpu'))
        self.load_state_dict(torch.load(self.checkpoint, map_location=torch.device('cpu')))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, return_att=False):
        _, pooled_output, all_encoder_attention_scores, embedding_output = self.bert(input_ids, token_type_ids,
                                                                                     attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            try:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            except ValueError:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, torch.unsqueeze(labels, dim=0).float())
            return loss, logits, all_encoder_attention_scores, embedding_output
        elif return_att:
            return logits, all_encoder_attention_scores
        else:
            return logits
