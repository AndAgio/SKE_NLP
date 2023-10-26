import torch
import os
import zipfile
import shutil
import json
from torch.utils.data import Dataset
from skenlp.utils import download_file_from_google_drive, DumbLogger


class WrenchDataset(Dataset):
    def __init__(self, drive_id, tokenizer, folder='datasets', name='sms', split='train', logger=None):
        self.drive_id = drive_id
        self.tokenizer = tokenizer
        self.folder = folder
        self.name = name
        self.split = split
        self.logger = logger if logger is not None else DumbLogger()
        if not self.check_dataset_files_available():
            self.download_dataset()
        texts, labels, ids = self.read_split()
        self.texts = texts
        self.encodings = self.tokenizer(texts,
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

    def get_tokenization(self, string):
        return self.tokenizer(string, return_tensors='pt')

    def read_split(self):
        with open(os.path.join(self.folder, self.name, '{}.json'.format(self.split))) as json_file:
            data = json.load(json_file)
        ids, texts, labels = [], [], []
        for key, item in data.items():
            ids.append(int(key))
            texts.append(item['data']['text'])
            labels.append(item['label'])
        return texts, labels, ids

    def check_dataset_files_available(self):
        for split in ['train', 'valid', 'test']:
            if not os.path.exists(os.path.join(self.folder,
                                               self.name,
                                               '{}.json'.format(split))):
                self.logger.print_it('Didn\'t find file containing {} for {} dataset.'.format(split,
                                                                                              self.name.upper()))
                return False
        self.logger.print_it('Found all files containing data for {} dataset. '
                             'We will use them.'.format(self.name.upper()))
        return True

    def download_dataset(self):
        # Download zip file if not found...
        self.logger.print_it('Downloading {} dataset file, this may take a while...'.format(self.name.upper()))
        tmp_download_folder = os.path.join(os.getcwd(), 'dwn')
        if not os.path.exists(tmp_download_folder):
            os.makedirs(tmp_download_folder)
        download_file_from_google_drive(self.drive_id, os.path.join(tmp_download_folder, '{}.zip'.format(self.name)))
        # Extract zip files from downloaded dataset
        zf = zipfile.ZipFile(os.path.join(tmp_download_folder, '{}.zip'.format(self.name)), 'r')
        self.logger.print_it('Unzipping dataset...')
        zf.extractall(tmp_download_folder)
        zf.close()
        # Make order into the project folder moving extracted dataset into home and removing temporary download folder
        self.logger.print_it('Moving dataset to clean repo...')
        files = [f.path for f in os.scandir(os.path.join(tmp_download_folder, self.name))]
        if not os.path.exists(os.path.join(self.folder, self.name)):
            os.makedirs(os.path.join(self.folder, self.name))
        for file in files:
            shutil.move(file, os.path.join(self.folder, self.name))
        shutil.rmtree(tmp_download_folder)
