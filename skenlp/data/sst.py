import numpy as np
import torch
import os
import wget
import zipfile
import shutil
import pandas as pd
from torch.utils.data import Dataset
from skenlp.utils import DumbLogger


class SstDataset(Dataset):
    def __init__(self, tokenizer, folder='datasets', split='train', logger=None):
        self.folder = folder
        self.split = split
        self.logger = logger if logger is not None else DumbLogger()
        if not self.check_dataset_files_available():
            self.download_dataset()
        texts, labels, ids = self.read_sst_split()
        self.texts = texts
        self.encodings = tokenizer(texts,
                                   truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors='pt')
        self.labels = torch.tensor(labels)
        self.ids = torch.tensor(ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['id'] = self.ids[idx]
        item['text'] = self.texts[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def to(self, device):
        self.encodings = self.encodings.to(device)
        self.labels = self.labels.to(device)

    def read_sst_split(self):
        dataframe = pd.read_csv(os.path.join(self.folder,
                                             'sst',
                                             '{}.tsv'.format(self.split)),
                                delimiter='\t').dropna()
        n_samples = dataframe.shape[0]
        try:
            ids = dataframe['index'].to_numpy()[:n_samples]
        except KeyError:
            ids = np.asarray([i for i in range(n_samples)], dtype=np.int)
        texts = dataframe['sentence'].to_list()[:n_samples]
        try:
            labels = dataframe['label'].to_numpy()[:n_samples]
        except KeyError:
            labels = np.empty([n_samples])
        return texts, labels, ids

    def check_dataset_files_available(self):
        for split in ['train', 'dev', 'test']:
            if not os.path.exists(os.path.join(self.folder,
                                               'sst',
                                               '{}.tsv'.format(split))):
                self.logger.print_it('Didn\'t find file containing {} for SST-2 dataset.'.format(split))
                return False
        self.logger.print_it('Found all files containing data for SST-2 dataset. We will use them.')
        return True

    def download_dataset(self):
        url = 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'
        self.logger.print_it('Downloading SST-2 dataset from {}...'.format(url))
        if not os.path.exists(os.path.join(self.folder, 'sst')):
            os.makedirs(os.path.join(self.folder, 'sst'))
        wget.download(url, out=os.path.join(self.folder, 'sst'))
        self.logger.print_it('Extracting SST-2 dataset...')
        with zipfile.ZipFile(os.path.join(self.folder, 'sst', 'SST-2.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.folder, 'sst'))
        source = os.path.join(self.folder, 'sst', 'SST-2')
        destination = os.path.join(self.folder, 'sst')
        allfiles = os.listdir(source)
        for file in allfiles:
            src_path = os.path.join(source, file)
            dst_path = os.path.join(destination, file)
            shutil.move(src_path, dst_path)
        os.rmdir(source)
        os.remove(os.path.join(self.folder, 'sst', 'SST-2.zip'))
