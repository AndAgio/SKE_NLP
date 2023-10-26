import requests
import pandas as pd


def combine_datasets(left_out, target_frac=1):
    datasets = ['ALM', 'Baltimore', 'BLM', 'Davidson', 'Election', 'MeToo', 'Sandy']
    train_corpora = [x for x in datasets if x != left_out]
    train_data = pd.concat([pd.read_csv(f'nlp/data/processed/{f}.csv') for f in train_corpora],
                           ignore_index=True).sample(frac=1)
    test_data = pd.read_csv(f'nlp/data/processed/{left_out}.csv').sample(frac=target_frac)
    return train_data, test_data


def get_dataset(datas):
    data = pd.concat([pd.read_csv(f'nlp/data/processed/{f}.csv') for f in datas],
                     ignore_index=True).sample(frac=1)
    return data


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
