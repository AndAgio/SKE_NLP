def get_label_names(dataset):
    if isinstance(dataset, str):
        dataset_config = format_dataset_name(dataset)
    elif isinstance(dataset, dict):
        dataset_config = dataset
    else:
        raise ValueError('Dataset should be either string or dictionary!')
    moral_labels = ['care', 'harm',
                    'fairness', 'cheating',
                    'loyalty', 'betrayal',
                    'authority', 'subversion',
                    'purity', 'degradation',
                    'non-moral']
    moral_found_labels = ['care', 'fairness',
                          'loyalty', 'authority',
                          'purity', 'non-moral']
    if dataset_config['name'] in ['sst', 'imdb', 'agnews', 'sms', 'trec', 'youtube']:
        return {'sst': ['negative', 'positive'],
                'imdb': ['negative', 'positive'],
                'agnews': ['world', 'sports', 'business', 'sci/tech'],
                'sms': ['ham', 'spam'],
                'trec': ['abbr', 'enty', 'desc', 'hum', 'loc', 'num'],
                'youtube': ['ham', 'spam'],
                }[dataset_config['name']]
    elif dataset_config['name'] == 'moral' and dataset_config['foundations']:
        return moral_found_labels
    elif dataset_config['name'] == 'moral' and not dataset_config['foundations']:
        return moral_labels
    else:
        raise ValueError('Dataset {} is not valid!'.format(dataset))


def format_dataset_name(dataset: str):
    dataset_config = {'name': dataset.split('_')[0]}
    try:
        dataset_config['subset'] = dataset.split('_')[1].split(',')
        dataset_config['foundations'] = True if dataset.split('_')[2] in ['true', 'True', 't', 'T', 1] else False
    except:
        pass
    dataset_config['full'] = dataset
    return dataset_config
