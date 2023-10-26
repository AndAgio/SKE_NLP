import os
import time
import pickle
import pandas as pd
import nltk
from nltk.corpus import wordnet31
import numpy as np
from itertools import islice
import torch
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from skenlp.data import format_dataset_name, get_label_names
from skenlp.ske.locals import ShapLocalExplainer, GradSensitivity, GradInput, LayerwiseAttentionTracing, \
    LayerwiseRelevancePropagation, LimeLocalExplainer, HessianLocalExplainer, reformat_impact_scores_dict
from skenlp.utils import format_time, DumbLogger


class BaseGlobalExplainer:
    def __init__(self,
                 tokenizer,
                 model,
                 local_explanations=None,
                 loc_explain_mode='gi',
                 aggregator='sum_50',
                 dataset='sst',
                 multi_label=False,
                 threshold=0.,
                 device='cpu',
                 out_dir='outs',
                 logger=None):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.dataset_string = dataset
        self.dataset_config = format_dataset_name(dataset)
        self.multi_label = multi_label
        self.handle_multi_labels = False
        self.threshold = threshold
        self.device = device
        self.out_dir = out_dir
        self.logger = logger if logger is not None else DumbLogger()
        # Get local explanations depending on what is given
        if isinstance(local_explanations, dict):
            self.local_explanations = local_explanations
        elif isinstance(local_explanations, str):
            self.logger.print_it('Loading local explanations from {}...'.format(local_explanations))
            with open(local_explanations, 'rb') as pickle_file:
                self.local_explanations = pickle.load(pickle_file)
        elif local_explanations is None:
            assert self.model != torch.empty((1,)).to(self.device)
            self.local_explanations = None
        else:
            raise ValueError('Local explanations format \'{}\' is not supported'.format(local_explanations.type()))
        self.loc_explain_mode = loc_explain_mode
        self.local_explainers_options = {'shap': ShapLocalExplainer,
                                         'gs': GradSensitivity,
                                         'gi': GradInput,
                                         'lat': LayerwiseAttentionTracing,
                                         'lrp': LayerwiseRelevancePropagation,
                                         'lime': LimeLocalExplainer,
                                         'hess': HessianLocalExplainer, }
        assert aggregator.split('_')[0] in ['sum', 'abs-sum', 'avg', 'abs-avg']
        self.aggregator = aggregator
        labels_names = get_label_names(self.dataset_config)
        self.labels_names_dict = {i: label_name for i, label_name in enumerate(labels_names)}
        nltk.download('wordnet', quiet=True)
        nltk.download('wordnet31', quiet=True)
        self.lemmatizer = wordnet31
        self.local_explainer = self.define_local_explainer()

    def define_local_explainer(self):
        local_explainer = self.local_explainers_options[self.loc_explain_mode](tokenizer=self.tokenizer,
                                                                               model=self.model,
                                                                               multi_label=True if
                                                                               self.dataset_config['name'] == 'moral'
                                                                               else False,
                                                                               threshold=self.threshold,
                                                                               device=self.device,
                                                                               out_dir=self.out_dir,
                                                                               logger=self.logger)
        return local_explainer

    def convert_tensor_label_to_list(self, label):
        labels_names = get_label_names(self.dataset_config)
        labels_names_dict = {i: label_name for i, label_name in enumerate(labels_names)}
        if self.multi_label:
            return [labels_names_dict[index] for index, item in enumerate(label) if item.cpu().item() == 1]
        else:
            return labels_names_dict[np.argmax(label.cpu().numpy())]

    def predict_with_model(self, sentence: str):
        labels_names = get_label_names(self.dataset_config)
        labels_names_dict = {i: label_name for i, label_name in enumerate(labels_names)}
        if isinstance(self.local_explainer, LayerwiseAttentionTracing):
            _, _, pred_labels, _ = self.local_explainer.feed_input([sentence])
        else:
            _, _, pred_labels = self.local_explainer.feed_input([sentence])
        if self.multi_label:
            return [labels_names_dict[pred.cpu().item()] for pred in pred_labels]
        else:
            return labels_names_dict[pred_labels.cpu().item()]

    def fit(self, sentences, store: bool = True):
        raise NotImplementedError("Global global_explainer object should implement fit method!")

    def predict_with_explainer(self, sentence: str, debug: bool = True) -> str:
        raise NotImplementedError("Global global_explainer object should implement predict_with_explainer method!")

    def predict_with_both(self, sentence: str, label: str = None):
        raise NotImplementedError("Global global_explainer object should implement predict_with_both method!")

    def run_both_on_sentences(self, sentences, selected_set='test'):
        raise NotImplementedError("Global global_explainer object should implement run_both_on_sentences method!")

    def get_category_jargon(self, mode='avg_50', store=True):
        # Define empty dictionary of relevant jargon
        category_relevant_words_dict = {}
        # Extract k value
        top_k = int(mode.split('_')[1])
        t0 = time.time()
        # Iterate over each class
        for label_index, label_name in self.labels_names_dict.items():
            # Get the relevance scores corresponding to the current category
            skimmed_scores = BaseGlobalExplainer.gather_category_scores(self.local_explanations,
                                                                        category=label_name)
            # Convert into lists and get unique words
            words_list, scores_list = self.gather_words_scores_lists(skimmed_scores)
            single_words = set(words_list)
            # Construct dictionary containing scores and counts for each word found
            scores_dict = {word: 0. for word in single_words}
            count_dict = {word: 0. for word in single_words}
            # Aggregate the scores depending on the required approach
            for index in range(len(words_list)):
                scores_dict[words_list[index]] += abs(scores_list[index]) \
                    if 'abs' in mode \
                    else scores_list[index]
                count_dict[words_list[index]] += 1
            if 'avg' in mode:
                for key, _ in scores_dict.items():
                    scores_dict[key] = scores_dict[key] / float(count_dict[key])
            # Sort the words dictionary by scores
            ordered_scores_dict = {k: v for k, v in sorted(scores_dict.items(),
                                                           key=lambda item: item[1],
                                                           reverse=True)}
            # Gather only the top k relevant words
            ordered_scores_items = list(ordered_scores_dict.items())
            top_k_words_dict = dict(ordered_scores_items[:top_k] if top_k != -1 else ordered_scores_items)

            top_k_words = list(top_k_words_dict.keys())
            top_k_words_scores = list(top_k_words_dict.values())
            # Insert category relevant words to the whole dictionary
            category_relevant_words_dict[label_index] = {'label_name': label_name,
                                                         'top_{}_words'.format(top_k): top_k_words,
                                                         'words_scores': top_k_words_scores, }
        self.logger.print_it('Time to convert local explanations into'
                             ' category jargon: {}'.format(format_time(time.time() - t0)))
        if store:
            self.dump_jargon_dict(category_relevant_words_dict)
        return category_relevant_words_dict

    def run_local(self, sentences, store=True):
        t0 = time.time()
        labels_names = get_label_names(self.dataset_config)
        labels_names_dict = {i: label_name for i, label_name in enumerate(labels_names)}
        self.local_explanations = self.local_explainer.explain_dataset(sentences=sentences,
                                                                       labels_names_dict=labels_names_dict,
                                                                       store=store,
                                                                       dataset_name=self.dataset_string)
        self.logger.print_it('Time to explain sentences with local explainer: {}'.format(format_time(time.time() - t0)))
        return self.local_explanations

    def evaluate_global(self, sentences, selected_set='test'):
        labels_names = get_label_names(self.dataset_config)
        # labels_names_dict = {i: label_name for i, label_name in enumerate(labels_names)}
        model_predictions, explainer_predictions = self.run_both_on_sentences(sentences, selected_set=selected_set)
        model_pred_counter = dict(Counter(model_predictions))
        explainer_pred_counter = dict(Counter(explainer_predictions))
        self.logger.print_it('Counter of model predictions: {}'.format(model_pred_counter))
        self.logger.print_it('Counter of global_explainer predictions: {}'.format(explainer_pred_counter))
        # Convert prediction strings to corresponding label
        # names_labels_dict = {label_name: i for i, label_name in enumerate(labels_names)}
        # int_model_predictions = [names_labels_dict[pred] if isinstance(pred, str)
        #                          else -1 #tuple([names_labels_dict[single_pred] for single_pred in pred])
        #                          for pred in model_predictions]
        # int_explainer_predictions = [names_labels_dict[pred] for pred in explainer_predictions]
        # Convert predictions into one-hot encodings
        one_hotter = MultiLabelBinarizer(classes=labels_names)
        if not self.handle_multi_labels:
            model_predictions_one_hot = one_hotter.fit_transform([item if isinstance(item, tuple)
                                                                  else tuple((item,)) for item in model_predictions])
            explainer_predictions_one_hot = one_hotter.fit_transform([item if isinstance(item, tuple)
                                                                      else tuple((item,))
                                                                      for item in explainer_predictions])
        else:
            model_predictions_one_hot = one_hotter.fit_transform([tuple(item.split(' + ')) if ' + ' in item
                                                                  else tuple((item,)) for item in model_predictions])
            explainer_predictions_one_hot = one_hotter.fit_transform([tuple(item.split(' + ')) if ' + ' in item
                                                                      else tuple((item,))
                                                                      for item in explainer_predictions])
        fidelity = accuracy_score(model_predictions_one_hot, explainer_predictions_one_hot) * 100
        fidelity_f1 = f1_score(model_predictions_one_hot, explainer_predictions_one_hot, average='weighted') * 100
        self.logger.print_it('Fidelity over the {} set: {:.3f}%'.format(selected_set.upper(), fidelity))
        self.logger.print_it('Fidelity F1 over the {} set: {:.3f}%'.format(selected_set.upper(), fidelity_f1))
        txt_folder = os.path.join(self.out_dir, 'explanations', 'theories', 'texts')
        if not os.path.exists(txt_folder):
            os.makedirs(txt_folder)
        txt_file = os.path.join(txt_folder, self.get_file_name())
        self.logger.print_it('Appending results to theory file in {}...'.format(txt_file))
        message = '\n\n===========================================================\n' \
                  'Performance over {} set\n' \
                  'Counter of model predictions: {}\n' \
                  'Counter of global_explainer predictions: {}\n' \
                  'Fidelity = {:.3f}%\n' \
                  'Fidelity F1 = {:.3f}%'.format(selected_set.upper(),
                                                 model_pred_counter,
                                                 explainer_pred_counter,
                                                 fidelity,
                                                 fidelity_f1)
        try:
            with open(txt_file, 'a') as opened_file:
                opened_file.write(message)
        except FileNotFoundError:
            with open(txt_file, 'w') as opened_file:
                opened_file.write(message)

    def get_file_name(self):
        raise NotImplementedError('Global SKE class should implement \'get_file_name\' method!')

    def dump_jargon_dict(self, jargon_dict):
        pkl_folder = os.path.join(self.out_dir, 'explanations', 'pickles', 'globals')
        if not os.path.exists(pkl_folder):
            os.makedirs(pkl_folder)
        pkl_file = os.path.join(pkl_folder, '{}_{}_{}_{}.pkl'.format(self.model.name,
                                                                     self.loc_explain_mode,
                                                                     self.aggregator,
                                                                     self.dataset_string))
        self.logger.print_it('Storing pickled explanations of the whole dataset to {}...'.format(pkl_file))
        with open(pkl_file, 'wb') as opened_file:
            pickle.dump(jargon_dict, opened_file)

    @staticmethod
    def gather_category_scores(local_explanations, category):
        local_explanations_df = pd.DataFrame.from_dict(local_explanations)
        category_scores_df = local_explanations_df[local_explanations_df['category'] == category]
        return category_scores_df

    def gather_words_scores_lists(self, skimmed_scores):
        words = list(skimmed_scores.to_dict()['words'].values())
        words = [self.lemmatizer.morphy(word) if self.lemmatizer.morphy(word) is not None else word
                 for sentence in words for word in sentence]
        scores = list(skimmed_scores.to_dict()['scores'].values())
        scores = [score for scrs in scores for score in scrs.tolist()]
        return words, scores

    @staticmethod
    def reformat_impact_scores(data: dict) -> dict:

        def take(n, iterable):
            return list(islice(iterable, n))

        data = reformat_impact_scores_dict(data)

        for key in list(data.keys()):
            if len(take(2, data[key].items())) == 0:
                del data[key]
        return data
