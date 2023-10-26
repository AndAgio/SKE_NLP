import torch

from .base import BaseLocalExplainer


class LayerwiseAttentionTracing(BaseLocalExplainer):
    def __init__(self,
                 tokenizer,
                 model,
                 multi_label=False,
                 threshold=0.,
                 device='cpu',
                 out_dir='outs',
                 logger=None):
        super().__init__(tokenizer, model, multi_label, threshold, device, out_dir, logger)
        self.device = device
        self.explainer_name = 'lat'

    def feed_input(self, sentence):
        tokens = self.tokenizer(sentence)
        outputs, all_encoder_attention_scores = self.model(torch.tensor(tokens['input_ids']).to(self.device),
                                                           return_att=True)
        if self.multi_label:
            predictions = (outputs > self.threshold).float()
            pred_labels = ((predictions == 1).nonzero(as_tuple=True)[1])
        else:
            pred_labels = torch.argmax(torch.softmax(outputs, dim=1)[0])
        return tokens, outputs, pred_labels, all_encoder_attention_scores

    def explain(self, sentence, label=None, sentence_id=0, dataset_config=None,
                visualize=True, verbose=False, get_pred_labels=False):
        tokens, outputs, pred_labels, all_encoder_attention_scores = self.feed_input(sentence)
        # backing out using the quasi-attention
        attention_scores = torch.zeros_like(torch.tensor(tokens['input_ids']), dtype=torch.float).to(self.device)
        # we need to distribution the attention on CLS to each head
        # here, we use grad to do this
        attention_scores[:, 0] = 1.0
        attention_scores = torch.stack(self.model.config.num_attention_heads * [attention_scores],
                                       dim=1).unsqueeze(dim=2)
        for i in reversed(range(self.model.config.num_attention_heads)):
            attention_scores = torch.matmul(attention_scores, all_encoder_attention_scores[i])
        attention_scores = attention_scores.sum(dim=1).squeeze(dim=1).unsqueeze(dim=-1).data
        attention_scores = torch.squeeze(attention_scores)

        words, attention_scores = self.refine_tokens_and_scores(tokens, attention_scores)
        assert len(words) == attention_scores.shape[0]

        if verbose:
            self.logger.print_it('attention_scores: {}'.format(attention_scores))
        if visualize:
            self.visualize(scores=attention_scores,
                           outputs=outputs,
                           pred_labels=pred_labels,
                           label=label,
                           words=words,
                           sentence=sentence,
                           sentence_id=sentence_id,
                           dataset_config=dataset_config)
        if get_pred_labels:
            return words, attention_scores, pred_labels
        else:
            return words, attention_scores
