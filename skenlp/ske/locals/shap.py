import shap
import torch
import numpy as np
import scipy as sp

from .base import BaseLocalExplainer


class ShapLocalExplainer(BaseLocalExplainer):
    def __init__(self,
                 tokenizer,
                 model,
                 multi_label=False,
                 threshold=0.,
                 device='cpu',
                 out_dir='outs',
                 logger=None):
        super().__init__(tokenizer, model, multi_label, threshold, device, out_dir, logger)
        # build an global_explainer using a token masker
        self.explainer_name = 'shap'
        self.explainer = shap.Explainer(self.prediction_func, self.tokenizer)

    # define a prediction function
    def prediction_func(self, x):
        tokens = torch.tensor([self.tokenizer.encode(v,
                                                     padding='max_length',
                                                     max_length=512,
                                                     truncation=True) for v in x]).to(self.device)
        attention_mask = (tokens != 0).type(torch.int64).to(self.device)
        outputs = self.model(tokens).detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores)
        return val

    def explain(self, sentence, label=None, sentence_id=0, dataset_config=None,
                visualize=True, verbose=False, get_pred_labels=False):
        tokens, outputs, pred_labels = self.feed_input(sentence)
        shap_values = self.explainer(sentence, fixed_context=1, silent=not verbose)
        words, shap_values = self.refine_tokens_and_scores(tokens, torch.tensor(shap_values.values[0]))
        assert len(words) == shap_values.shape[0]
        if verbose:
            self.logger.print_it('shap_values: {}'.format(shap_values))
        if visualize:
            self.visualize(scores=shap_values,
                           outputs=outputs,
                           pred_labels=pred_labels,
                           label=label,
                           words=words,
                           sentence=sentence,
                           sentence_id=sentence_id,
                           dataset_config=dataset_config)
        if get_pred_labels:
            return words, shap_values, pred_labels
        else:
            return words, shap_values
