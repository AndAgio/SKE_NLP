import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
from unidecode import unidecode

from .base import BaseLocalExplainer
from .utils import words_from_tokens


class LimeLocalExplainer(BaseLocalExplainer):
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
        self.explainer_name = 'lime'
        self.explainer = LimeTextExplainer()

    def pred_adapter(self, texts: list[str]):
        all_scores = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            encoded_input = self.tokenizer(batch,
                                           return_tensors='pt',
                                           padding=True,
                                           truncation=True,
                                           max_length=self.model.config.max_position_embeddings - 2).to(self.device)
            output = self.model(**encoded_input)
            scores = output.softmax(1).detach().cpu().numpy()
            all_scores.extend(scores)
        return np.array(all_scores)

    def explain(self, sentence, label=None, sentence_id=0, dataset_config=None,
                visualize=True, verbose=False, get_pred_labels=False):
        tokens, outputs, pred_labels = self.feed_input(sentence)
        lime_values = self.explainer.explain_instance(sentence[0],
                                                      self.pred_adapter,
                                                      labels=(i for i in range(self.model.n_labels)),
                                                      num_features=len(tokens['input_ids'][0]))

        words = words_from_tokens(self.tokenizer, tokens)
        lime_words = [unidecode(t[0]) for t in lime_values.as_list(0)]
        words = [word for word in words if word in lime_words]
        np_lime_scores = np.zeros((self.model.n_labels, len(words)))
        for label in range(self.model.n_labels):
            lime_this_label = lime_values.as_list(label)
            lime_this_label = {unidecode(t[0]): t[1] for t in lime_this_label}
            for word_index, word in enumerate(words):
                try:
                    np_lime_scores[label, word_index] = lime_this_label[word]
                except KeyError as error:
                    self.logger.print_it('Couldn\'t find word \"{}\" in'
                                         ' LIME dictionary of values: {}'.format(word, lime_this_label))
                    raise error
        lime_values = torch.transpose(torch.from_numpy(np_lime_scores), 0, 1)
        assert len(words) == lime_values.shape[0]
        if verbose:
            self.logger.print_it('lime_values: {}'.format(lime_values))
        if visualize:
            self.visualize(scores=lime_values,
                           outputs=outputs,
                           pred_labels=pred_labels,
                           label=label,
                           words=words,
                           sentence=sentence,
                           sentence_id=sentence_id,
                           dataset_config=dataset_config)
        if get_pred_labels:
            return words, lime_values, pred_labels
        else:
            return words, lime_values
