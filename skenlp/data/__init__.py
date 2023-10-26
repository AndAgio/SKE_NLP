import torch
from .moral import MoralDataset, MFTCDataset
from .sst import SstDataset  # get_sst_dataset
from .imdb import ImdbDataset  # get_imdb_dataset
from .ag_news import AGNewsDataset
from .sms import SmsDataset
from .trec import TrecDataset
from .youtube import YoutubeDataset
from .utils import get_label_names, format_dataset_name

from skei4nlp.utils import DumbLogger


def get_dataset(dataset_config: dict,
                tokenizer,
                split: str = 'train',
                logger=None):
    assert split in ['train', 'test']
    if logger is None:
        logger = DumbLogger()
    # Gather datasets depending on the choice
    if dataset_config['name'] == 'imdb':
        return ImdbDataset(tokenizer=tokenizer,
                           split=split,
                           logger=logger)
    elif dataset_config['name'] == 'sst':
        return SstDataset(tokenizer=tokenizer,
                          split='train' if split == 'train' else 'dev',
                          logger=logger)
    elif dataset_config['name'] in ['agnews', 'sms', 'trec', 'youtube']:
        dataset_classes_dict = {'agnews': AGNewsDataset,
                                'sms': SmsDataset,
                                'trec': TrecDataset,
                                'youtube': YoutubeDataset}
        return dataset_classes_dict[dataset_config['name']](tokenizer=tokenizer,
                                                            split=split,
                                                            logger=logger)
    elif dataset_config['name'] == 'moral':
        return MoralDataset(tokenizer=tokenizer,
                            data=dataset_config['subset'],
                            use_foundations=dataset_config['foundations'],
                            split=split,
                            logger=logger)
    else:
        raise ValueError('Dataset {} not valid!'.format(dataset_config['name']))


def check_train_test_datasets(train_dataset: str,
                              test_dataset: str) -> bool:
    if 'moral' in train_dataset:
        if 'moral' in test_dataset and train_dataset.split('_')[-1] == test_dataset.split('_')[-1]:
            return True
        else:
            return False
    else:
        if train_dataset == test_dataset:
            return True
        else:
            return False
