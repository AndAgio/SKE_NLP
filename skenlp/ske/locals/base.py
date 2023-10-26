import os

import numpy as np
import torch
import pickle
import pandas as pd

from skenlp.data import get_label_names
from skenlp.utils import text_visualization, VisualizationDataRecord, DumbLogger, show_message_with_bar


class BaseLocalExplainer:
    def __init__(self,
                 tokenizer,
                 model,
                 multi_label=False,
                 threshold=0.,
                 device='cpu',
                 out_dir='outs',
                 logger=None):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.multi_label = multi_label
        self.threshold = threshold
        self.device = device
        self.out_dir = out_dir
        self.logger = logger if logger is not None else DumbLogger()
        self.explainer_name = None

    def visualize_and_store_scores(self, sentence, score_vis, sentence_id=0, dataset_config=None):
        html_obj = text_visualization(score_vis)
        try:
            html_dir = os.path.join(self.out_dir, 'explanations',
                                    dataset_config['name'], dataset_config['subset'],
                                    self.explainer_name)
        except KeyError:
            html_dir = os.path.join(self.out_dir, 'explanations', dataset_config['name'], self.explainer_name)
        if not os.path.exists(html_dir):
            os.makedirs(html_dir)
        html_file = os.path.join(html_dir, 'sentence_{}.html'.format(sentence_id))
        with open(html_file, "w") as file:
            file.write(html_obj)

    def visualize(self, scores, outputs, pred_labels, label, words, sentence, sentence_id=0, dataset_config=None):
        if label is None:
            raise ValueError('If visualization is required then label should not be None!')
        label_names = get_label_names(dataset_config)
        if self.multi_label:
            predictions = (outputs > self.threshold).float()
            pred_labels = ((predictions == 1).nonzero(as_tuple=True)[1])
            pred_probs = outputs[:, pred_labels].detach().cpu().numpy()[0]
            label = ((label == 1).nonzero(as_tuple=True)[0])
            label = [lab.detach().cpu().item() for lab in label]
            label = [label_names[lab] for lab in label]
            pred_labels = [lab.detach().cpu().item() for lab in pred_labels]
            pred_labels = [label_names[lab] for lab in pred_labels]
        else:
            pred_probs = torch.softmax(outputs, dim=1)[0][pred_labels].item()
            label = label_names[label[0] if isinstance(label, list) else label]
            pred_labels = label_names[pred_labels]
        score_vis = [VisualizationDataRecord(scores,
                                             pred_probs,
                                             pred_labels,
                                             label,
                                             scores.sum(),
                                             words, )]
        self.visualize_and_store_scores(sentence=sentence,
                                        score_vis=score_vis,
                                        sentence_id=sentence_id,
                                        dataset_config=dataset_config)

    def feed_input(self, sentence):
        tokens = self.tokenizer(sentence)
        outputs = self.model(torch.tensor(tokens['input_ids']).to(self.device))
        if self.multi_label:
            predictions = (outputs > self.threshold).float()
            pred_labels = ((predictions == 1).nonzero(as_tuple=True)[1])
        else:
            pred_labels = torch.argmax(torch.softmax(outputs, dim=1)[0])
        return tokens, outputs, pred_labels

    def get_name(self):
        return self.explainer_name

    def refine_tokens_and_scores(self, tokens, scores):
        # Remove first [CLS] and last [SEP] token from the sentence
        for key, _ in tokens.items():
            tokens[key][0] = tokens[key][0][1:-1]
        scores = scores[1:-1]

        words = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
        indices_to_merge = [i for i, word in enumerate(words) if word.startswith("##")]

        if not len(indices_to_merge) == 0:
            for iterator in range(len(indices_to_merge)):
                index_to_merge = indices_to_merge[iterator]
                words[index_to_merge - 1] = words[index_to_merge - 1] + words[index_to_merge].strip('#')
                del words[index_to_merge]
                scores[index_to_merge - 1] += scores[index_to_merge]
                scores = torch.cat([scores[:index_to_merge], scores[index_to_merge + 1:]])
                indices_to_merge = [i - 1 if i >= index_to_merge else i for i in indices_to_merge]
            for index_to_merge in set(indices_to_merge):
                scores[index_to_merge] /= indices_to_merge.count(index_to_merge) + 1
        return words, scores

    def explain(self, sentence, label=None, sentence_id=0, dataset_config=None,
                visualize=True, verbose=False, get_pred_labels=False):
        raise NotImplementedError('Explain method should be implemented in each local global_explainer!')

    def explain_dataset(self, sentences, labels_names_dict, store=True, dataset_name=None):
        explanations = {'words': [],
                        'scores': [],
                        'category': [],
                        'model_prediction': []}
        for sentence_index, sentence in enumerate(sentences):
            if len(self.tokenizer([sentence])['input_ids'][0]) >= 512:
                continue
            show_message_with_bar(logger=self.logger,
                                message='Explaining sentence {} out of {} | n tokens: {} |'.format(
                                        sentence_index + 1,
                                        len(sentences),
                                        len(self.tokenizer([sentence])['input_ids'][0])),
                                index=sentence_index,
                                size=len(sentences))
            try:
                words, score, pred_labels = self.explain([sentence], visualize=False, verbose=False, get_pred_labels=True)
            except (ValueError, RuntimeError) as error:
                # self.logger.print_it('Explainer {} threw error over sentence: {}\n'
                #                      'Sentence length: {}\n'
                #                      'Error:\n{}\n'.format(self.explainer_name.upper(),
                #                                          sentence,
                #                                          len(self.tokenizer([sentence])['input_ids'][0]),
                #                                          error))
                continue
            if not self.multi_label:
                pred_labels = [pred_labels.cpu().item()]
            else:
                pred_labels = [pred.cpu().item() for _, pred in enumerate(pred_labels)]
            pred_labels_names = [labels_names_dict[pred] for _, pred in enumerate(pred_labels)]
            if self.explainer_name in ['shap', 'hess', 'lime']:
                for cat_index, cat_name in labels_names_dict.items():
                    explanations['words'].append(words)
                    explanations['scores'].append(score.detach().cpu().numpy()[:, cat_index])
                    explanations['category'].append(cat_name)
                    explanations['model_prediction'].append(cat_name
                                                            if cat_name in pred_labels_names
                                                            else np.nan)
            else:
                for _, pred in enumerate(pred_labels):
                    explanations['words'].append(words)
                    explanations['scores'].append(score.detach().cpu().numpy())
                    explanations['category'].append(labels_names_dict[pred])
                    explanations['model_prediction'].append(labels_names_dict[pred])
        self.logger.set_logger_newline()
        if store:
            self.dump_explanation_object(explanations, dataset_name)
            self.explanation_to_csv(explanations, dataset_name)
        return explanations

    def dump_explanation_object(self, explanations, dataset_name):
        pkl_folder = os.path.join(self.out_dir, 'explanations', 'pickles', 'locals')
        if not os.path.exists(pkl_folder):
            os.makedirs(pkl_folder)
        pkl_file = os.path.join(pkl_folder, '{}_{}_{}.pkl'.format(self.model.name,
                                                                  self.explainer_name,
                                                                  dataset_name))
        self.logger.print_it('Storing pickled explanations of the whole dataset to {}...'.format(pkl_file))
        with open(pkl_file, 'wb') as opened_file:
            pickle.dump(explanations, opened_file)

    def explanation_to_csv(self, explanations, dataset_name):
        csv_folder = os.path.join(self.out_dir, 'explanations', 'csvs', 'locals')
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
        csv_file = os.path.join(csv_folder, '{}_{}_{}.csv'.format(self.model.name,
                                                                  self.explainer_name,
                                                                  dataset_name))
        self.logger.print_it('Storing explanations of the whole dataset to csv format into file {}...'.format(csv_file))
        pd.DataFrame.from_dict(explanations).to_csv(csv_file)
