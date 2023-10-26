import torch
from .hooks import ForwardHook

from .base import BaseLocalExplainer


class GradSensitivity(BaseLocalExplainer):
    def __init__(self,
                 tokenizer,
                 model,
                 multi_label=False,
                 threshold=0.,
                 device='cpu',
                 out_dir='outs',
                 logger=None):
        super().__init__(tokenizer, model, multi_label, threshold, device, out_dir, logger)
        self.hooks = {}
        for name, children_module in self.model.named_modules():
            if name in ['classifier', 'bert.embeddings']:
                self.hooks[name] = ForwardHook(children_module)
        self.device = device
        self.explainer_name = 'gs'
        self.multiply = False
        self.normalisation = 'norm'

    def explain(self, sentence, label=None, sentence_id=0, dataset_config=None,
                visualize=True, verbose=False, get_pred_labels=False):
        tokens, outputs, pred_labels = self.feed_input(sentence)
        gs_score = torch.zeros(outputs.shape).to(self.device)
        gs_score[:, pred_labels] = 1.0
        gs_score = outputs * gs_score
        classifier_out = self.hooks['classifier'].get_hooked_output()
        embedding_output = self.hooks['bert.embeddings'].get_hooked_output()
        if verbose:
            self.logger.print_it('Computing gradient retaining the graph...')
        sensitivity_grads = torch.autograd.grad(classifier_out, embedding_output,
                                                grad_outputs=gs_score,
                                                retain_graph=True)[0]
        if self.multiply:
            sensitivity_grads = sensitivity_grads * embedding_output
        if verbose:
            self.logger.print_it('tokens[\'input_ids\'][0]: {}'.format(tokens['input_ids'][0]))
        if self.normalisation == 'norm':
            sensitivity_grads = torch.norm(sensitivity_grads, dim=-1)
        elif self.normalisation == 'sum':
            sensitivity_grads = torch.sum(sensitivity_grads, dim=-1)
        else:
            raise ValueError('Normalisation operation should be either norm or sum!')
        sensitivity_grads = torch.squeeze(sensitivity_grads)

        words, sensitivity_grads = self.refine_tokens_and_scores(tokens, sensitivity_grads)
        assert len(words) == sensitivity_grads.shape[0]

        if verbose:
            self.logger.print_it('sensitivity_grads: {}'.format(sensitivity_grads))
        if visualize:
            self.visualize(scores=sensitivity_grads,
                           outputs=outputs,
                           pred_labels=pred_labels,
                           label=label,
                           words=words,
                           sentence=sentence,
                           sentence_id=sentence_id,
                           dataset_config=dataset_config)
        if get_pred_labels:
            return words, sensitivity_grads, pred_labels
        else:
            return words, sensitivity_grads
