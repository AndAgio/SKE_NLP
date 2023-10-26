import os
import time
import warnings
import pickle
import pandas as pd
import nltk
from nltk.corpus import wordnet31
from pyJoules.energy_meter import measure_energy
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from skenlp.models import BertConfig, BertForSequenceClassification
from skenlp.data import get_label_names
from skenlp.bin import Explainer
from skenlp.utils import set_seeds, get_logger, get_options, format_time, str2bool
from skenlp.data import format_dataset_name, get_dataset


def main(arguments, dataset):
    print('=======================================================================================')
    print('Measuring efficiency of over {}'.format(dataset.upper()))
    model = BertForSequenceClassification(BertConfig(),
                                        num_labels=len(get_label_names(dataset)),
                                        checkpoint='{}/models/best_{}_{}.pt'.format(arguments.out_folder,
                                                                                    arguments.model,
                                                                                    dataset),
                                        logger=None)
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    nltk.download('wordnet', quiet=True)
    nltk.download('wordnet31', quiet=True)
    lemmatizer = wordnet31
    test_dataset_config = format_dataset_name(dataset)
    test_dataset = get_dataset(test_dataset_config, tokenizer=tokenizer, split='test', logger=None)

    cpu_domains = [RaplPackageDomain(0)]
    gpu_domains = [RaplPackageDomain(0), NvidiaGPUDomain(1)]
    cpu_meter = EnergyMeter(DeviceFactory.create_devices(cpu_domains))
    gpu_meter = EnergyMeter(DeviceFactory.create_devices(gpu_domains))

    loc_explainers = ['shap', 'lime']
    ks = [50, 250]

    for loc_explainer in loc_explainers:
        for k in ks:
            aggregator = 'avg_{}'.format(k)

            cart_file = os.path.join('outs', 'explanations', 'theories', 'predictors', '{}_{}_cart_{}_{}.pkl'.format(arguments.model, loc_explainer, aggregator, dataset))
            with open(cart_file, 'rb') as f:
                cart = pickle.load(f)
            valuable_tokens = cart.get_valuable_tokens()

            cpu_meter.start(tag='cart')
            predict_with_cart(cart, test_dataset, tokenizer, lemmatizer)
            cpu_meter.stop()
            energy_trace = cpu_meter.get_trace()['cart']
            tot_energy = energy_trace.energy['package_0'] * 1e-6
            print('Processing with CART_{}_{}:'.format(loc_explainer, aggregator))
            print('\tTime = {:.5f} s\n\tEnergy = {} J'.format(energy_trace.duration, tot_energy))
            print('\tAverage time = {:.5f} s\n\tAverage Energy = {} J'.format(energy_trace.duration/len(test_dataset), tot_energy/len(test_dataset)))

    cpu_meter.start(tag='bert')
    predict_with_bert(model, test_dataset, test_dataset_config)
    cpu_meter.stop()
    energy_trace = cpu_meter.get_trace()['bert']
    tot_energy = energy_trace.energy['package_0'] * 1e-6
    print('Processing with BERT:\n\tTime = {:.5f} s\n\tEnergy = {} J'.format(energy_trace.duration, tot_energy))
    print('\tAverage time = {:.5f} s\n\tAverage Energy = {} J'.format(energy_trace.duration/len(test_dataset), tot_energy/len(test_dataset)))

    gpu_meter.start(tag='bert_gpu')
    predict_with_bert_on_gpu(model, test_dataset, test_dataset_config)
    gpu_meter.stop()
    energy_trace = gpu_meter.get_trace()['bert_gpu']
    tot_energy = energy_trace.energy['package_0'] * 1e-6 + energy_trace.energy['nvidia_gpu_1'] * 1e-3
    print('Processing with BERT:\n\tTime = {:.5f} s\n\tEnergy = {} J'.format(energy_trace.duration, tot_energy))
    print('\tAverage time = {:.5f} s\n\tAverage Energy = {} J'.format(energy_trace.duration/len(test_dataset), tot_energy/len(test_dataset)))
    print('=======================================================================================\n\n\n')

def predict_with_cart(cart, test_dataset, tokenizer, lemmatizer):
    valuable_tokens = cart.get_valuable_tokens()
    for index, sample in enumerate(test_dataset):
        print('Processing sample {} out of {}...'.format(index, len(test_dataset)), end='\r')
        sentence = sample['text']
        dataframe = pd.DataFrame(index=[0], columns=valuable_tokens)
        dataframe = convert_sentence_to_dataframe_row(sentence=sentence, lemmatizer=lemmatizer, tokenizer=tokenizer, dataframe=dataframe, valuable_tokens=valuable_tokens, index=0)
        prediction = cart.predict(dataframe)
    print()

def skipgrams_from_sentence(words, max_length=3):
    # Construct list of skipgrams in the sentence that is at hand
    all_sequences_in_sentence = list(words.copy())
    max_length = max_length
    for length in range(2, max_length + 1):
        sequences_in_sentence = [tup for tup in list(nltk.skipgrams(words, length, 2)) if all([isinstance(item, str) for item in tup])]
        all_sequences_in_sentence += sequences_in_sentence
    return all_sequences_in_sentence

def convert_sentence_to_dataframe_row(sentence: str, lemmatizer, tokenizer, dataframe: pd.DataFrame, valuable_tokens, index: int = 0):
    # Lemmatize words in order to match correctly the relevant jargons
    words = [lemmatizer.morphy(word) if lemmatizer.morphy(word) is not None else word for word in tokenizer.tokenize(sentence)]
    # Construct list of skipgrams in the sentence that is at hand
    all_sequences_in_sentence = skipgrams_from_sentence(words, max_length=2)
    all_sequences_in_sentence = [' + '.join(t) if isinstance(t, tuple) else t for t in all_sequences_in_sentence]
    # Construct an empty row which will be added to the dataframe
    row = [1 if seq in all_sequences_in_sentence else 0 for seq in valuable_tokens]
    dataframe.iloc[index] = {valuable_tokens[i]: row[i] for i in range(len(valuable_tokens))}
    return dataframe


def predict_with_bert(model, test_dataset, test_dataset_config):
    test_loader = DataLoader(test_dataset, batch_size=1)
    model.eval()
    pred_threshold = 0.
    for batch_index, batch in enumerate(test_loader):
        print('Processing sample {} out of {}...'.format(batch_index+1, len(test_loader)), end='\r')
        with torch.no_grad():
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            if test_dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
                labels = batch['labels']
            elif test_dataset_config['name'] in ['moral']:
                labels = batch['labels'].float()
            else:
                raise ValueError('Something wrong with dataset labels in evaluate...')
            outputs = model(input_ids, attention_mask=attention_mask)
        if test_dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
            logits = outputs
        elif test_dataset_config['name'] in ['moral']:
            logits = (outputs > pred_threshold).float()
        else:
            raise ValueError('Something wrong with dataset labels in evaluate...')
    print()


def predict_with_bert_on_gpu(model, test_dataset, test_dataset_config):
    device = torch.device('cuda:1')
    model = model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=1)
    model.eval()
    pred_threshold = 0.
    for batch_index, batch in enumerate(test_loader):
        print('Processing sample {} out of {}...'.format(batch_index+1, len(test_loader)), end='\r')
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if test_dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
                labels = batch['labels'].to(device)
            elif test_dataset_config['name'] in ['moral']:
                labels = batch['labels'].to(device).float()
            else:
                raise ValueError('Something wrong with dataset labels in evaluate...')
            outputs = model(input_ids, attention_mask=attention_mask)
        if test_dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
            logits = outputs
        elif test_dataset_config['name'] in ['moral']:
            logits = (outputs > pred_threshold).float()
        else:
            raise ValueError('Something wrong with dataset labels in evaluate...')
    print()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    options = get_options()
    datasets = ['sms', 'youtube', 'trec', 'moral_alm_false', 'moral_blm_false', 'moral_baltimore_false', 'moral_election_false', 'moral_metoo_false', 'moral_sandy_false', 'moral_together_false']
    for dataset in datasets:
        main(options, dataset)
