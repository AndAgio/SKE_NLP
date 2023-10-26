import os
import time
from transformers import AutoTokenizer
from typing import Union

from skenlp.data import format_dataset_name, get_dataset, check_train_test_datasets
from skenlp.utils import select_device, format_time, DumbLogger, SmartLogger
from skenlp.ske.globals import RuleExplainerPredictor, LemmaExplainerPredictor


class Explainer:
    def __init__(self,
                 model,
                 device,
                 train_dataset: str = 'sst',
                 test_dataset: str = 'sst',
                 threshold: float = 0,
                 local_exp_file_available: bool = False,
                 loc_explain_mode: str = 'shap',
                 aggregator: str = 'sum_50',
                 global_explainer: str = 'cart',
                 max_skipgrams_length: int = 2,
                 run_mode: str = 'glob',
                 logger: Union[DumbLogger, SmartLogger] = None,
                 out_folder: str = 'outs'):
        self.model = model
        self.threshold = threshold
        self.logger = logger if logger is not None else DumbLogger()
        self.out_folder = out_folder
        self.device = select_device(device=device,
                                    batch_size=1,
                                    logger=self.logger)

        self.model.to(self.device)

        if not check_train_test_datasets(train_dataset, test_dataset):
            raise ValueError('Datasets \"{}\" and \"{}\" are not compatible for train and test!'.format(train_dataset,
                                                                                                        test_dataset))
        self.train_dataset_string = train_dataset
        self.test_dataset_string = test_dataset
        self.train_dataset_config = format_dataset_name(train_dataset)
        self.test_dataset_config = format_dataset_name(test_dataset)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.train_dataset = get_dataset(self.train_dataset_config,
                                         tokenizer=self.tokenizer,
                                         split='train',
                                         logger=self.logger)
        self.test_dataset = get_dataset(self.test_dataset_config,
                                        tokenizer=self.tokenizer,
                                        split='test',
                                        logger=self.logger)
        self.loc_explain_mode = loc_explain_mode
        assert aggregator.split('_')[0] in ['sum', 'abs-sum', 'avg', 'abs-avg']
        self.aggregator = aggregator
        assert global_explainer in ['cart', 'lemma']
        self.global_explainer = global_explainer
        self.max_skipgrams_length = max_skipgrams_length
        assert run_mode in ['loc', 'glob']
        self.run_mode = run_mode
        self.local_exp_file_available = local_exp_file_available
        if self.local_exp_file_available:
            self.local_exp_file = os.path.join(self.out_folder, 'explanations', 'pickles', 'locals',
                                               '{}_{}_{}.pkl'.format(self.model.name,
                                                                     self.loc_explain_mode,
                                                                     self.train_dataset_string))
        else:
            self.local_exp_file = None
        self.global_explainer = self.define_global_explainer()

    def define_global_explainer(self):
        if self.global_explainer in ['cart']:
            return RuleExplainerPredictor(tokenizer=self.tokenizer, model=self.model,
                                          local_explanations=self.local_exp_file,
                                          loc_explain_mode=self.loc_explain_mode,
                                          aggregator=self.aggregator, explainer=self.global_explainer,
                                          max_skipgrams_length=self.max_skipgrams_length,
                                          dataset=self.train_dataset_string,
                                          multi_label=True if self.train_dataset_config['name'] == 'moral' else False,
                                          threshold=self.threshold, device=self.device, out_dir=self.out_folder,
                                          logger=self.logger)
        elif self.global_explainer in ['lemma']:
            return LemmaExplainerPredictor(tokenizer=self.tokenizer, model=self.model,
                                           local_explanations=self.local_exp_file,
                                           loc_explain_mode=self.loc_explain_mode,
                                           aggregator=self.aggregator,
                                           dataset=self.train_dataset_string,
                                           multi_label=True if self.train_dataset_config['name'] == 'moral' else False,
                                           threshold=self.threshold, device=self.device, out_dir=self.out_folder,
                                           logger=self.logger)
        else:
            raise ValueError('Explainer mode \'{}\' not supported!'.format(self.global_explainer))

    def run(self, split='train', store=True):
        if self.run_mode == 'loc':
            self.run_local(split=split, store=store)
        elif self.run_mode == 'glob':
            self.fit_global(split=split, store=store)
            self.evaluate_global()
            self.global_explainer.predict_with_both(sentence=self.get_dataset_sentences(split='test')[0],
                                                    label=self.get_dataset_labels(split='test')[0])
        else:
            raise ValueError('Running mode {} is not available!'.format(self.run_mode))

    def run_local(self, split='train', store=True):
        t0 = time.time()
        texts = self.get_dataset_sentences(split=split)
        self.global_explainer.run_local(sentences=texts,
                                        store=store)

    def fit_global(self, split='train', store=True):
        t0 = time.time()
        texts = self.get_dataset_sentences(split=split)
        self.global_explainer.fit(sentences=texts,
                                  store=store)
        self.logger.print_it(
            'Time to run global global_explainer over sentences: {}'.format(format_time(time.time() - t0)))

    def evaluate_global(self):
        test_texts = self.get_dataset_sentences(split='test')
        self.global_explainer.evaluate_global(sentences=test_texts)

    def get_dataset_sentences(self, split='train'):
        dataset = self.get_dataset_split(split=split)
        texts = []
        for index, sample in enumerate(dataset):
            texts.append(sample['text'])
        return texts

    def get_dataset_labels(self, split='train'):
        dataset = self.get_dataset_split(split=split)
        labels = []
        for index, sample in enumerate(dataset):
            labels.append(sample['labels'])
        return labels

    def get_dataset_split(self, split='train'):
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError('Split {} is not valid!'.format(split))
        dataset.to(self.device)
        return dataset
