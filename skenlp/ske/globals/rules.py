import os
import time
import pickle
import pandas as pd
import numpy as np
import nltk
import torch
import math

from .base import BaseGlobalExplainer
from skenlp.utils import format_time, show_message_with_bar
from skenlp.metrics import theory_complexity, theory_length, theory_cumbersomeness, theory_spread
from .cart import Cart
from .logic import readable_theory


class RuleExplainerPredictor(BaseGlobalExplainer):
    def __init__(self,
                 tokenizer,
                 local_explanations=None,
                 model=None,
                 loc_explain_mode='gi',
                 aggregator='sum_50',
                 explainer='cart',
                 max_skipgrams_length=2,
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
        assert explainer in ['cart']
        self.explainer = explainer
        self.max_skipgrams_length = max_skipgrams_length
        self.valuable_tokens = None
        self.glob_expl_predictor = None

    def fit(self, sentences, store: bool = True, validate: bool = False):
        if self.local_explanations is None:
            self.local_explanations = self.run_local(sentences=sentences,
                                                     store=store)
        else:
            pass
        jargons = self.get_category_jargon(mode=self.aggregator,
                                           store=store)
        lemmas = [lemma for category_index in jargons.keys()
                  for lemma in jargons[category_index]['top_{}_words'.format(self.aggregator.split('_')[-1])]
                  if lemma not in [',', '\'', '_', ';', '#', '&', '\\']
                  ]
        lemmas = list(set(lemmas))
        dataframe = self.define_dataframe_from_lemmas(lemmas, self.local_explanations, store=store)
        refined_dataframe = dataframe.drop(columns=['sentence'])
        refined_dataframe.columns = refined_dataframe.columns.map(RuleExplainerPredictor.tuple_to_string_col)
        self.valuable_tokens = list(refined_dataframe.columns)[:-1]
        assert len(self.valuable_tokens) == len(list(set(self.valuable_tokens)))
        self.logger.print_it('The number of valuable tokens found is {}'.format(len(self.valuable_tokens)))
        global_explanations = self.extract_theory_from_dataframe(refined_dataframe, store=store)
        if validate:
            _, _ = self.run_both_on_sentences(sentences, selected_set='training')
        return global_explanations

    def run_both_on_sentences(self, sentences, selected_set='test'):
        # Construct empty dataframe to speed up the method
        dataframe = pd.DataFrame(index=range(len(sentences)), columns=self.valuable_tokens)
        # Iterate over each sentence in the sentence (rows in the local explanations dataframe)
        model_predictions = []
        for index, sentence in enumerate(sentences):
            show_message_with_bar(logger=self.logger,
                                  message='Evaluating model and global_explainer over {} set. '
                                          'Sentence {} out of {}'.format(selected_set.upper(),
                                                                         index + 1, len(sentences)),
                                  index=index,
                                  size=len(sentences))
            model_predictions.append(self.predict_with_model(sentence))
            # Convert sentence to dataframe row and append it
            dataframe = self.convert_sentence_to_dataframe_row(sentence=sentence,
                                                               dataframe=dataframe,
                                                               index=index)
        assert list(dataframe.columns) == self.valuable_tokens
        assert len(list(dataframe.columns)) == len(list(set(dataframe.columns)))
        self.logger.set_logger_newline()
        explainer_predictions = self.glob_expl_predictor.predict(dataframe)
        # Refine model predictions
        if self.multi_label:
            model_predictions = [pred[0] if len(pred) == 1 else
                                 'non-moral' if len(pred) == 0 else
                                 tuple([item for item in pred]) for pred in model_predictions]
        else:
            pass
        return model_predictions, explainer_predictions

    def predict_with_explainer(self, sentence: str, explain: bool = True):
        assert self.glob_expl_predictor
        # Construct empty dataframe to speed up the method
        dataframe = pd.DataFrame(index=[0], columns=self.valuable_tokens)
        dataframe = self.convert_sentence_to_dataframe_row(sentence=sentence,
                                                           dataframe=dataframe,
                                                           index=0)
        assert list(dataframe.columns) == self.valuable_tokens
        assert len(list(dataframe.columns)) == len(list(set(dataframe.columns)))
        activated_rules, explainer_prediction = self.glob_expl_predictor.predict_and_explain(dataframe)
        if explain:
            return explainer_prediction, activated_rules
        else:
            return explainer_prediction

    def predict_with_both(self, sentence: str, label: str = None):
        explainer_prediction, activated_rules = self.predict_with_explainer(sentence=sentence,
                                                                            explain=True)
        model_prediction = self.predict_with_model(sentence)
        self.logger.print_it('Analysing sentence \"{}\":\n'
                             'True label: {}\n'
                             'Black-box model prediction: {}\n'
                             'Explainer prediction: {}\n'
                             'Activated rules are:\n{}'.format(sentence,
                                                               self.convert_tensor_label_to_list(label)
                                                               if label is not None else 'N/A',
                                                               model_prediction,
                                                               explainer_prediction,
                                                               activated_rules))
        return explainer_prediction, model_prediction, activated_rules

    @staticmethod
    def skipgrams_from_sentence(words, max_length=3):
        # Construct list of skipgrams in the sentence that is at hand
        all_sequences_in_sentence = list(words.copy())
        max_length = max_length
        for length in range(2, max_length + 1):
            sequences_in_sentence = [tup for tup in list(nltk.skipgrams(words, length, 2))
                                     if all([isinstance(item, str) for item in tup])]
            all_sequences_in_sentence += sequences_in_sentence
        return all_sequences_in_sentence

    def convert_sentence_to_dataframe_row(self, sentence: str, dataframe: pd.DataFrame, index: int = 0):
        # Lemmatize words in order to match correctly the relevant jargons
        words = [self.lemmatizer.morphy(word)
                 if self.lemmatizer.morphy(word) is not None
                 else word for word in self.tokenizer.tokenize(sentence)]
        # Construct list of skipgrams in the sentence that is at hand
        all_sequences_in_sentence = RuleExplainerPredictor.skipgrams_from_sentence(words,
                                                                                   max_length=self.max_skipgrams_length)
        all_sequences_in_sentence = [' + '.join(t) if isinstance(t, tuple)
                                     else t for t in all_sequences_in_sentence]
        # Construct an empty row which will be added to the dataframe
        row = [1 if seq in all_sequences_in_sentence else 0 for seq in self.valuable_tokens]
        dataframe.iloc[index] = {self.valuable_tokens[i]: row[i] for i in range(len(self.valuable_tokens))}
        return dataframe

    def define_dataframe_from_lemmas(self, lemmas, local_explanations, store=True):
        self.logger.print_it('Converting local explanations and lemmas into dataframe to extract global rules.'
                             ' This may take a while...')
        t0 = time.time()
        # Remove nan from the local explanations dataframe (useful when dealing with shap explanations)
        filtered_dictionary = {'words': [],
                               'scores': [],
                               'category': [],
                               'model_prediction': []}
        for i in range(len(local_explanations['words'])):
            elem = local_explanations['model_prediction'][i]
            if isinstance(elem, str) or not np.isnan(elem):
                filtered_dictionary['words'].append(local_explanations['words'][i])
                filtered_dictionary['scores'].append(local_explanations['scores'][i])
                filtered_dictionary['category'].append(local_explanations['category'][i])
                filtered_dictionary['model_prediction'].append(local_explanations['model_prediction'][i])
        local_explanations = filtered_dictionary.copy()
        # Define the columns for an empty dataframe using single relevant words
        columns = ['sentence'] + list(lemmas) + ['model_prediction']
        columns_to_add = []
        indices_of_columns_to_add = {}
        # Construct empty dataframe to speed up the method
        dataframe = pd.DataFrame(index=range(len(local_explanations['words'])), columns=columns)
        # Iterate over each sentence in the sentence (rows in the local explanations dataframe)
        for index in range(len(local_explanations['words'])):
            # Lemmatize words in order to match correctly the relevant jargons
            words = local_explanations['words'][index]
            words = [self.lemmatizer.morphy(word)
                     if self.lemmatizer.morphy(word) is not None
                     else word for word in words]
            all_sequences_in_sentence = RuleExplainerPredictor. \
                skipgrams_from_sentence(words,
                                        max_length=self.max_skipgrams_length)
            # Construct an empty row which will be added to the dataframe
            row_single_sentence = []
            # Iterate over all skipgrams of the sentence to check if it is constructed from relevant lemmas or not
            for ind, seq in enumerate(all_sequences_in_sentence):
                # If we're actually analysing a skipgram then check that all its components are relevant lemmas...
                if isinstance(seq, tuple):
                    can_be = True
                    for word in seq:
                        if word not in lemmas:
                            # ...if they are not, we discard this skipgram and we do not add it to the dataframe.
                            can_be = False
                            break
                    if can_be:
                        # If the skipgram is relevant add it to the columns to consider and remember the row index
                        if seq not in columns_to_add:
                            columns_to_add.append(seq)
                            indices_of_columns_to_add[seq] = [index]
                        else:
                            indices_of_columns_to_add[seq].append(index)
                else:
                    # If we're not analysing a skipgram but rather a single lemma,
                    # we construct directly the row of lemmas to add it to the dataframe
                    row_single_sentence.append(list(lemmas).index(seq) + 1
                                               if seq in list(lemmas) else 0)
            # Add row of single lemmas to the dataframe
            row = [1 if i in row_single_sentence else 0 for i in range(len(columns))]
            row[0] = words
            row[-1] = local_explanations['model_prediction'][index]
            dataframe.iloc[index] = {columns[i]: row[i] for i in range(len(columns))}
        # Append skipgrams columns to the dataframe
        new_columns = {tupla: [1 if index in indices_of_columns_to_add[tupla] else 0 for index in
                               range(len(local_explanations['words']))] for tupla in columns_to_add}
        assert len(list(new_columns.keys())) == len(list(set(list(new_columns.keys()))))
        tupled_tokens = list(new_columns.keys())
        if len(tupled_tokens) > 0:
            max_tuple_length = max([len(tup) for tup in tupled_tokens])
            min_tuple_length = min([len(tup) for tup in tupled_tokens])
            for tuple_length in range(min_tuple_length, max_tuple_length + 1):
                tupled_tokens_with_this_length = [tup for tup in tupled_tokens if len(tup) == tuple_length]
                dict_to_add = {tupla: new_columns[tupla] for tupla in tupled_tokens_with_this_length}
                dataframe = pd.concat((dataframe, pd.DataFrame(dict_to_add)), axis=1)
        # Refine the columns ordering
        column_to_move = dataframe.pop('model_prediction')
        dataframe.insert(len(dataframe.columns), 'model_prediction', column_to_move)
        if self.handle_multi_labels:
            # Convert multi-label rows to single rows
            # self.logger.print_it('Dataframe rows before merging duplicate rows: {}'.format(len(dataframe.index)))
            iterator = 0
            sentence_col = dataframe.columns.get_loc('sentence')
            model_pred_col = dataframe.columns.get_loc('model_prediction')
            dataframe['model_prediction'] = dataframe['model_prediction'].astype(object)
            while iterator < len(dataframe.index) - 1:
                if dataframe.iloc[iterator, sentence_col] != dataframe.iloc[iterator + 1, sentence_col]:
                    iterator += 1
                else:
                    if isinstance(dataframe.iloc[iterator, model_pred_col], str):
                        dataframe.iat[iterator, model_pred_col] = [dataframe.iloc[iterator, model_pred_col],
                                                                   dataframe.iloc[iterator + 1, model_pred_col]]
                    else:
                        dataframe.iat[iterator, model_pred_col].append(dataframe.iloc[iterator + 1, model_pred_col])
                    dataframe = dataframe.drop(iterator + 1, axis=0)
            dataframe['model_prediction'] = dataframe.apply(RuleExplainerPredictor.tuple_to_string_row,
                                                            axis=1)
            # self.logger.print_it('Dataframe rows after merging duplicate rows: {}'.format(len(dataframe.index)))
            used_labels = list(set([tuple(item) if isinstance(item, list)
                                    else item for item in dataframe['model_prediction']]))
            self.logger.print_it('Model predictions used as labels: {}'.format(used_labels))
        self.logger.print_it('Time to convert local explanations and '
                             'relevant lemmas to dataframe: {}'.format(format_time(time.time() - t0)))
        if store:
            self.dump_dataframe(dataframe)
            self.store_df_as_csv(dataframe)
        return dataframe

    @staticmethod
    def tuple_to_string_row(row):
        if isinstance(row['model_prediction'], list):
            return ' + '.join(row['model_prediction'])
        else:
            return row['model_prediction']

    @staticmethod
    def tuple_to_string_col(col):
        if isinstance(col, tuple):
            return ' + '.join(col)
        else:
            return col

    def extract_theory_from_dataframe(self, dataframe, store=True, verbose=False):
        t0 = time.time()
        selected_extractor = self.explainer
        k = int(self.aggregator.split('_')[-1])
        if k != -1:
            max_depth = 5 * math.ceil(len(self.valuable_tokens) / (k*len(self.labels_names_dict.keys())))
        else:
            max_depth = math.ceil(len(self.valuable_tokens) / (100*len(self.labels_names_dict.keys())))
        # max_depth = math.ceil(len(self.valuable_tokens) / (10*len(self.labels_names_dict.keys())))
        # max_depth = None
        # max_depth = 50
        self.logger.print_it('Using {} to extract global theory. '
                             'This may take a while...'.format(selected_extractor.upper() +
                                                               ' with max_depth=' + str(max_depth)
                                                               if selected_extractor == 'cart'
                                                               else selected_extractor.upper()))
        self.glob_expl_predictor = {'cart': Cart(simplify=True, max_depth=max_depth),}[selected_extractor]
        extracted_theory = self.glob_expl_predictor.extract(dataframe)
        self.glob_expl_predictor.set_valuable_tokens(self.valuable_tokens)
        if verbose:
            self.logger.print_it('\nExtracted rules:\n{}\n'.format(readable_theory(extracted_theory)))
        if store:
            self.dump_theory(extracted_theory, max_depth=max_depth)
            self.dump_predictor(self.glob_expl_predictor)
        self.logger.print_it('Time to extract the global theory from the dataframe: {}'.format(format_time(time.time() - t0)))
        return extracted_theory

    def dump_theory(self, theory, max_depth=None):
        theory_read = readable_theory(theory)
        txt_folder = os.path.join(self.out_dir, 'explanations', 'theories', 'texts')
        if not os.path.exists(txt_folder):
            os.makedirs(txt_folder)
        txt_file = os.path.join(txt_folder, '{}_{}_{}_{}_{}.txt'.format(self.model.name,
                                                                        self.loc_explain_mode,
                                                                        self.explainer,
                                                                        self.aggregator,
                                                                        self.dataset_string))
        self.logger.print_it('Storing readable theory to {}...'.format(txt_file))
        with open(txt_file, 'w') as opened_file:
            opened_file.write(theory_read)
            opened_file.write('\n\n===========================================================\n')
            if max_depth is not None:
                opened_file.write('Depth = {}\n\n'.format(max_depth))
            opened_file.write('THEORY COMPLEXITY:\n')
            opened_file.write('Complexity = {}\n'.format(theory_complexity(theory)))
            opened_file.write('Length = {}\n'.format(theory_length(theory)))
            opened_file.write('Cumbersomeness = {}\n'.format(theory_cumbersomeness(theory)))
            opened_file.write('Spread = {}\n'.format(theory_spread(theory)))

    def dump_predictor(self, predictor):
        pkl_folder = os.path.join(self.out_dir, 'explanations', 'theories', 'predictors')
        if not os.path.exists(pkl_folder):
            os.makedirs(pkl_folder)
        pkl_file = os.path.join(pkl_folder, '{}_{}_{}_{}_{}.pkl'.format(self.model.name,
                                                                        self.loc_explain_mode,
                                                                        self.explainer,
                                                                        self.aggregator,
                                                                        self.dataset_string))
        self.logger.print_it('Storing pickled dataframe to {}...'.format(pkl_file))
        with open(pkl_file, 'wb') as opened_file:
            pickle.dump(predictor, opened_file)

    def store_df_as_csv(self, dataframe):
        csv_folder = os.path.join(self.out_dir, 'explanations', 'csvs', 'dataframes')
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
        csv_file = os.path.join(csv_folder, '{}_{}_{}.csv'.format(self.model.name,
                                                                  self.loc_explain_mode,
                                                                  self.dataset_string))
        self.logger.print_it('Storing csv of dataframe to {}...'.format(csv_file))
        dataframe.to_csv(csv_file)

    def dump_dataframe(self, dataframe):
        pkl_folder = os.path.join(self.out_dir, 'explanations', 'pickles', 'dataframes')
        if not os.path.exists(pkl_folder):
            os.makedirs(pkl_folder)
        pkl_file = os.path.join(pkl_folder, '{}_{}_{}.pkl'.format(self.model.name,
                                                                  self.loc_explain_mode,
                                                                  self.dataset_string))
        self.logger.print_it('Storing pickled dataframe to {}...'.format(pkl_file))
        with open(pkl_file, 'wb') as opened_file:
            pickle.dump(dataframe, opened_file)

    def get_file_name(self) -> str:
        return '{}_{}_{}_{}_{}.txt'.format(self.model.name, self.loc_explain_mode,
                                           self.aggregator, self.explainer, self.dataset_string)
