import numpy as np
import torch
import pandas as pd

from .base import BaseGlobalExplainer
from skenlp.utils import show_message_with_bar
from skenlp.ske.locals import words_from_tokens


class LemmaExplainerPredictor(BaseGlobalExplainer):
    def __init__(self,
                 tokenizer,
                 local_explanations=None,
                 model=None,
                 loc_explain_mode='gi',
                 aggregator='sum_50',
                 dataset='sst',
                 multi_label=False,
                 threshold=0.,
                 device='cpu',
                 out_dir='outs',
                 logger=None):
        if model is None:
            model = torch.empty((1,))
        super().__init__(tokenizer, model, local_explanations, loc_explain_mode, aggregator,
                         dataset, multi_label, threshold, device, out_dir, logger)
        self.impact_scores = None

    def fit(self, sentences, store: bool = True):
        if self.local_explanations is None:
            self.local_explanations = self.run_local(sentences=sentences, store=store)
        global_explanations = self.get_category_jargon(mode=self.aggregator,
                                                       store=store)
        self.impact_scores = self.reformat_impact_scores(global_explanations)
        return self.impact_scores

    def predict_with_explainer(self, sentence: str, debug: bool = False) -> str:
        assert self.impact_scores is not None
        tokens = self.tokenizer([sentence])
        words = words_from_tokens(tokenizer=self.tokenizer,
                                  tokens=tokens)
        prediction_matrix = pd.DataFrame(columns=words)
        for label in list(self.labels_names_dict.keys()):
            label_name = self.labels_names_dict[label]
            try:
                label_impact_scores = self.impact_scores[label_name]
            except KeyError:
                label_impact_scores = {}
            scores_label = []
            for word in words:
                try:
                    scores_label.append(label_impact_scores[word])
                except KeyError:
                    scores_label.append(0.)
            prediction_matrix = pd.concat([prediction_matrix,
                                           pd.DataFrame.from_records([scores_label],
                                                                     columns=words)])
        if debug:
            self.logger.print_it('prediction_matrix: {}'.format(prediction_matrix))
        prediction_matrix['tot'] = prediction_matrix.sum(axis=1)
        if debug:
            self.logger.print_it('prediction_matrix.iloc[:,-1]: {}'.format(prediction_matrix.iloc[:, -1]))
        prediction_scores = prediction_matrix['tot'].to_numpy()
        prediction = self.labels_names_dict[np.argmax(prediction_scores)]
        if debug:
            self.logger.print_it('prediction_scores: {}'.format(prediction_scores))
            self.logger.print_it('prediction: {}'.format(prediction))
        return prediction

    def predict_with_both(self, sentence: str, label: str = None) -> tuple:
        explainer_prediction = self.predict_with_explainer(sentence=sentence)
        model_prediction = self.predict_with_model(sentence)
        self.logger.print_it('Analysing sentence \"{}\":\n'
                             'True label: {}\n'
                             'Black-box model prediction: {}\n'
                             'Explainer prediction: {}'.format(sentence,
                                                               self.convert_tensor_label_to_list(label)
                                                               if label is not None else 'N/A',
                                                               model_prediction,
                                                               explainer_prediction))
        return explainer_prediction, model_prediction

    def run_both_on_sentences(self, sentences, selected_set='test') -> tuple:
        # Iterate over each sentence in the sentence (rows in the local explanations dataframe)
        model_predictions = []
        explainer_predictions = []
        for index, sentence in enumerate(sentences):
            show_message_with_bar(logger=self.logger,
                                  message='Evaluating model and global_explainer over {} set. '
                                          'Sentence {} out of {}'.format(selected_set.upper(),
                                                                         index + 1,
                                                                         len(sentences)),
                                  index=index,
                                  size=len(sentences))
            model_predictions.append(self.predict_with_model(sentence))
            explainer_predictions.append(self.predict_with_explainer(sentence))
        # Refine model predictions
        if self.multi_label:
            model_predictions = [pred[0] if len(pred) == 1 else
                                 'non-moral' if len(pred) == 0 else
                                 tuple([item for item in pred]) for pred in model_predictions]
        else:
            pass
        return model_predictions, explainer_predictions

    def get_file_name(self) -> str:
        return '{}_{}_{}_lemma_{}.txt'.format(self.model.name, self.loc_explain_mode,
                                              self.aggregator, self.dataset_string)
