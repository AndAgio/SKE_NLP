import torch
import os
import shutil
import wget
import tarfile
from pathlib import Path
from torch.utils.data import Dataset
from skenlp.utils import DumbLogger


class ImdbDataset(Dataset):
    def __init__(self, tokenizer, folder='datasets', split='train', logger=None):
        self.folder = folder
        self.split = split
        self.logger = logger if logger is not None else DumbLogger()
        if not self.check_dataset_files_available():
            self.download_dataset()
        texts, labels, ids = ImdbDataset.read_imdb_split(os.path.join(folder,
                                                                      'aclImdb',
                                                                      split))
        self.encodings = tokenizer(texts,
                                   truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors='pt')
        self.texts = texts
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

    @staticmethod
    def read_imdb_split(split_dir):
        split_dir = Path(split_dir)
        texts = []
        labels = []
        for label_dir in ["pos", "neg"]:
            for text_file in (split_dir / label_dir).iterdir():
                texts.append(text_file.read_text())
                labels.append(0 if label_dir == "neg" else 1)
        ids = [i for i in range(len(texts))]

        return texts, labels, ids

    def check_dataset_files_available(self):
        for split in ['train', 'test']:
            if not os.path.exists(os.path.join(self.folder,
                                               'aclImdb',
                                               split)):
                self.logger.print_it('Didn\'t find folder containing {} files for IMDB dataset.'.format(split))
                return False
        self.logger.print_it('Found all folders containing data files for IMDB dataset. We will use them.')
        return True

    def download_dataset(self):
        url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        self.logger.print_it('Downloading IMDB dataset from {}...'.format(url))
        if not os.path.exists(os.path.join(self.folder, 'aclImdb')):
            os.makedirs(os.path.join(self.folder, 'aclImdb'))
        wget.download(url, out=os.path.join(self.folder, 'aclImdb'))
        self.logger.print_it('Extracting IMDB dataset...')
        tar = tarfile.open(os.path.join(self.folder, 'aclImdb', 'aclImdb_v1.tar.gz'))
        tar.extractall(path=os.path.join(self.folder, 'aclImdb'))
        tar.close()
        os.remove(os.path.join(self.folder, 'aclImdb', 'aclImdb_v1.tar.gz'))
        source = os.path.join(self.folder, 'aclImdb', 'aclImdb')
        destination = os.path.join(self.folder, 'aclImdb')
        allfiles = os.listdir(source)
        for file in allfiles:
            src_path = os.path.join(source, file)
            dst_path = os.path.join(destination, file)
            shutil.move(src_path, dst_path)
        os.rmdir(source)
