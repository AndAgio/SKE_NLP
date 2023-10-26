import os
import math
import zipfile
import shutil
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from skenlp.utils import download_file_from_google_drive, DumbLogger


class MFTCDataset(Dataset):
    moral_labels = ['care', 'harm',
                    'fairness', 'cheating',
                    'loyalty', 'betrayal',
                    'authority', 'subversion',
                    'purity', 'degradation',
                    'non-moral']
    moral_foundations = ['care', 'fairness', 'loyalty', 'authority', 'purity', 'non-moral']
    # The moral foundations dict with virtue as key and vice as value
    moral_foundations_dict = {'care': 'harm',
                              'fairness': 'cheating',
                              'loyalty': 'betrayal',
                              'authority': 'subversion',
                              'purity': 'degradation',
                              'non-moral': 'non-moral'}
    datalist = ['ALM', 'Baltimore', 'BLM', 'BLM_cropped', 'Davidson', 'Election', 'MeToo', 'MFTC', 'Sandy', 'Together']
    datadict = {'alm': 'ALM',
                'baltimore': 'Baltimore',
                'blm': 'BLM',
                'blmcrop': 'BLM_cropped',
                'davidson': 'Davidson',
                'election': 'Election',
                'metoo': 'MeToo',
                'mftc': 'MFTC',
                'sandy': 'Sandy',
                'together': 'Together'}
    drive_id = None # Unfortunately, the dataset is not publicly available...

    def __init__(self,
                 tokenizer,
                 folder='datasets',
                 data='alm',
                 split='train',
                 use_foundations=False,
                 logger=None):
        self.tokenizer = tokenizer
        self.folder = folder
        self.chosen_data = data
        self.split = split
        self.label_names = MFTCDataset.moral_foundations if use_foundations else MFTCDataset.moral_labels
        self.logger = logger if logger is not None else DumbLogger()
        if not self.check_dataset_files_available():
            self.download_dataset()

        if isinstance(self.chosen_data, str):
            self.chosen_data = [self.chosen_data]
        self.data = None
        for index, ch_data in enumerate(self.chosen_data):
            data_file = os.path.join(self.folder,
                                     'moral',
                                     '{}_{}.csv'.format(MoralDataset.datadict[ch_data], self.split))
            if index == 0:
                self.data = pd.read_csv(data_file).dropna()
            else:
                self.data = self.data.append(pd.read_csv(data_file), ignore_index=True)
        assert self.data is not None

        max_size = self.data.shape[0]

        self.text = self.data['text'].to_list()[:max_size]
        self.ids = self.data['tweet_id'].to_numpy()[:max_size]

        if use_foundations:
            if self.label_names is None:
                self.label_names = MFTCDataset.moral_foundations_dict.keys()

            for l_name in self.label_names:
                if l_name not in MFTCDataset.moral_foundations_dict:
                    raise KeyError(f'Foundation {l_name} does not exist')

            num_rows = min(self.data.shape[0], max_size)
            num_cols = len(self.label_names)
            self.labels = np.empty([num_rows, num_cols], dtype=bool)
            for i, key in enumerate(self.label_names):
                value = MFTCDataset.moral_foundations_dict.get(key)
                self.labels[:, i] = self.data[key].to_numpy()[:max_size] | self.data[value].to_numpy()[:max_size]
            self.labels = self.labels.astype(int)
        else:
            if self.label_names is None:
                self.label_names = [x for x in self.data.columns if x != 'text']

            for l_name in self.label_names:
                if l_name not in self.data.columns:
                    raise KeyError(f'Moral label {l_name} does not exist')

            self.labels = self.data[self.label_names].to_numpy()[:max_size]

        self.logger.print_it('IDs for {} split of moral {} and use_foundations {} are: {}.'
                             ' Total samples: {}'.format(self.split,
                                                         data,
                                                         use_foundations,
                                                         self.ids,
                                                         len(self.ids)))

    def __getitem__(self, index):
        return {'id': self.ids[index],
                'text': self.text[index],
                'labels': self.labels[index]}

    def __len__(self):
        return len(self.text)

    def check_dataset_files_available(self):
        for data in MoralDataset.datalist:
            for split in ['train', 'test']:
                if not os.path.exists(os.path.join(self.folder,
                                                   'moral',
                                                   '{}_{}.csv'.format(data, split))):
                    self.logger.print_it('Didn\'t find file containing {} for MFTC dataset.'.format(data))
                    return False
        self.logger.print_it('Found all files containing data for MFTC dataset. We will use them.')
        return True

    def download_dataset(self):
        # Download zip file if not found...
        self.logger.print_it('Downloading dataset file, this may take a while...')
        tmp_download_folder = os.path.join(os.getcwd(), 'dwn')
        if not os.path.exists(tmp_download_folder):
            os.makedirs(tmp_download_folder)
        download_file_from_google_drive(MFTCDataset.drive_id, os.path.join(tmp_download_folder, 'moral.zip'))
        # Extract zip files from downloaded dataset
        self.logger.print_it('Unzipping file at {}...'.format(os.path.join(tmp_download_folder, 'moral.zip')))
        zf = zipfile.ZipFile(os.path.join(tmp_download_folder, 'moral.zip'), 'r')
        self.logger.print_it('Unzipping dataset...')
        zf.extractall(tmp_download_folder)
        zf.close()
        # Make order into the project folder moving extracted dataset into home and removing temporary download folder
        self.logger.print_it('Moving dataset to clean repo...')
        files = [f.path for f in os.scandir(os.path.join(tmp_download_folder, 'moral'))]
        if not os.path.exists(os.path.join(self.folder, 'moral')):
            os.makedirs(os.path.join(self.folder, 'moral'))
        for file in files:
            shutil.move(file, os.path.join(self.folder, 'moral'))
        shutil.rmtree(tmp_download_folder)
        # split all domains into train and test files
        self.split_csv_train_test()
        self.join_csvs()

    def split_csv_train_test(self):
        np.random.seed(42)
        for data in MoralDataset.datalist[:-1]:
            data_file = os.path.join(self.folder,
                                     'moral',
                                     '{}.csv'.format(data))
            dataset = pd.read_csv(data_file).dropna()
            dataset.rename(columns={'Unnamed: 0': 'tweet_id'}, inplace=True)
            if 'tweet_id' not in dataset.columns:
                dataset.insert(loc=0, column='tweet_id', value=range(0, len(dataset)))
            else:
                pass
            rng = np.random.RandomState()
            train = dataset.sample(frac=0.7, random_state=rng)
            test = dataset.loc[~dataset.index.isin(train.index)]
            train.to_csv(os.path.join(self.folder, 'moral', '{}_train.csv'.format(data)))
            test.to_csv(os.path.join(self.folder, 'moral', '{}_test.csv'.format(data)))

    def join_csvs(self):
        for split in ['_train', '_test', '']:
            dataset = None
            for index, data in enumerate(MoralDataset.datalist[:-1]):
                data_file = os.path.join(self.folder,
                                         'moral',
                                         '{}{}.csv'.format(data,
                                                           split))
                if index == 0:
                    dataset = pd.read_csv(data_file)
                else:
                    dataset = dataset.append(pd.read_csv(data_file), ignore_index=True)
            dataset.to_csv(os.path.join(self.folder, 'moral', '{}{}.csv'.format(MoralDataset.datalist[-1],
                                                                                split)))


class MoralDataset(MFTCDataset):
    def __init__(self,
                 tokenizer,
                 folder='datasets',
                 data='alm',
                 split='train',
                 use_foundations=False,
                 logger=None):
        super().__init__(tokenizer, folder, data, split, use_foundations, logger)

        self.encodings = self.tokenizer(self.text,
                                        truncation=True,
                                        padding=True,
                                        max_length=512,
                                        return_tensors='pt')
        self.labels = torch.tensor(self.labels)
        self.ids = torch.tensor(self.ids)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['id'] = self.ids[idx]
        item['text'] = self.text[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def to(self, device):
        self.encodings = self.encodings.to(device)
        self.labels = self.labels.to(device)

    def get_tokenization(self, string):
        return self.tokenizer(string, return_tensors='pt')
