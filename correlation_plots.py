import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
from statistics import mean
from skenlp.metrics import cos_sim, pearson_corr, dist
from skenlp.dictionaries import OriginalMFD, ExtendedMFD
import warnings

warnings.filterwarnings("ignore")

GLOBAL_K = -1
LOCAL_EXPLAINERS = ['gi', 'gs', 'lat', 'lrp', 'shap', 'hess', 'lime']
GLOBAL_AGGREGATORS = ['abs-avg_{}'.format(GLOBAL_K), 'avg_{}'.format(GLOBAL_K),
                      'abs-sum_{}'.format(GLOBAL_K), 'sum_{}'.format(GLOBAL_K)]
DISTANCES = ['pearson']
DATASETS = ['sms', 'trec', 'youtube', 'moral_alm_false', 'moral_baltimore_false', 'moral_blm_false',
            'moral_election_false', 'moral_metoo_false', 'moral_sandy_false']
PLOTS_DIR = 'outs/plots/extraamas'


def reformat_data(data: dict) -> dict:
    data = {item['label_name']: {item[list(item.keys())[1]][i]: item[list(item.keys())[2]][i]
                                 for i in range(len(item[list(item.keys())[1]]))}
            for _, item in data.items()}

    def take(n, iterable):
        return list(islice(iterable, n))

    for key in list(data.keys()):
        if len(take(2, data[key].items())) == 0:
            del data[key]
    data = normalize_scores(data)
    # print('data: {}'.format(data))
    return data


def normalize_scores(data: dict) -> dict:
    for label in data.keys():
        unnorm_words_and_scores = data[label]
        unnorm_scores = list(unnorm_words_and_scores.values())
        norm_scores = [(float(i) - min(unnorm_scores)) / (max(unnorm_scores) - min(unnorm_scores)) for i in
                       unnorm_scores]
        norm_words_and_scores = {list(unnorm_words_and_scores.keys())[i]: norm_scores[i]
                                 for i in range(len(unnorm_words_and_scores))}
        data[label] = norm_words_and_scores
    return data


def load_and_reformat(approach: str) -> dict:
    pickles_folder = os.path.join('outs', 'explanations', 'pickles', 'globals')
    with open(os.path.join(pickles_folder, '{}.pkl'.format(approach)), 'rb') as pickle_file:
        data = reformat_data(pickle.load(pickle_file))
    return data


def compare_two_over_single_label(data1: dict,
                                  data2: dict,
                                  label: str,
                                  mode: str = 'pearson') -> float:
    try:
        words_dict1 = data1[label]
        words_dict2 = data2[label]
    except AttributeError:
        raise ValueError('Label \'{}\' is not available!'.format(label))
    found_keys = list(set(list(words_dict1.keys()) + list(words_dict2.keys())))
    vec1 = np.asarray([float(words_dict1.get(x, 0)) for x in found_keys], dtype=float)
    vec2 = np.asarray([float(words_dict2.get(x, 0)) for x in found_keys], dtype=float)
    return {'pearson': pearson_corr(vec1, vec2),
            'cosine': cos_sim(vec1, vec2),
            'dist': dist(vec1, vec2)}[mode]


def compare_two(approach1: str = 'bert_gi_abs-avg_{}_moral_alm_true'.format(GLOBAL_K),
                approach2: str = 'bert_shap_abs-avg_{}_moral_alm_true'.format(GLOBAL_K),
                mode: str = 'pearson') -> tuple[float, dict]:
    data1 = load_and_reformat(approach1)
    data2 = load_and_reformat(approach2)
    min_keys = list(data1.keys()) if len(list(data1.keys())) <= len(list(data2.keys())) else list(data2.keys())
    data1 = {key: value for key, value in data1.items() if key in min_keys}
    data2 = {key: value for key, value in data2.items() if key in min_keys}
    metrics = {}
    for key in list(data1.keys()):
        metrics[key] = compare_two_over_single_label(data1, data2, label=key, mode=mode)
    average_metric = mean(list(metrics.values()))
    return average_metric, metrics


def compare_all_explainer_over_same_aggregator(model: str = 'bert',
                                               aggregator: str = 'abs-avg_{}'.format(GLOBAL_K),
                                               dataset: str = 'moral_alm_true',
                                               mode: str = 'pearson'):
    similarity_df = pd.DataFrame(columns=LOCAL_EXPLAINERS)
    for exp1 in LOCAL_EXPLAINERS:
        similarities_exp1_all = {}
        for exp2 in LOCAL_EXPLAINERS:
            similarity_exp1_exp2, _ = compare_two(approach1='{}_{}_{}_{}'.format(model,
                                                                                 exp1,
                                                                                 aggregator,
                                                                                 dataset),
                                                  approach2='{}_{}_{}_{}'.format(model,
                                                                                 exp2,
                                                                                 aggregator,
                                                                                 dataset),
                                                  mode=mode)
            similarities_exp1_all[exp2] = similarity_exp1_exp2
        similarity_df = pd.concat([similarity_df, pd.DataFrame.from_records([similarities_exp1_all])])
    plot_confusion_matrix(similarities=similarity_df,
                          x_classes=LOCAL_EXPLAINERS,
                          y_classes=LOCAL_EXPLAINERS,
                          mode=mode,
                          plot_path=os.path.join(PLOTS_DIR, dataset, 'local_explainers',
                                                 '{}_{}'.format(aggregator, mode)))


def compare_single_explainer_over_all_aggregators(model: str = 'bert',
                                                  explainer: str = 'shap',
                                                  dataset: str = 'moral_alm_true',
                                                  mode: str = 'pearson'):
    similarity_df = pd.DataFrame(columns=GLOBAL_AGGREGATORS)
    for agg1 in GLOBAL_AGGREGATORS:
        similarities_agg1_all = {}
        for agg2 in GLOBAL_AGGREGATORS:
            similarity_agg1_agg2, _ = compare_two(approach1='{}_{}_{}_{}'.format(model,
                                                                                 explainer,
                                                                                 agg1,
                                                                                 dataset),
                                                  approach2='{}_{}_{}_{}'.format(model,
                                                                                 explainer,
                                                                                 agg2,
                                                                                 dataset),
                                                  mode=mode)
            similarities_agg1_all[agg2] = similarity_agg1_agg2
        similarity_df = pd.concat([similarity_df, pd.DataFrame.from_records([similarities_agg1_all])])
    # print(similarity_df)
    plot_confusion_matrix(similarities=similarity_df,
                          x_classes=[agg.rsplit('_', 1)[0] for agg in GLOBAL_AGGREGATORS],
                          y_classes=[agg.rsplit('_', 1)[0] for agg in GLOBAL_AGGREGATORS],
                          mode=mode,
                          plot_path=os.path.join(PLOTS_DIR, dataset,
                                                 'aggregators', '{}_{}'.format(explainer, mode)))


def plot_confusion_matrix(similarities: pd.DataFrame,
                          x_classes: list[str],
                          y_classes: list[str],
                          mode: str = 'pearson',
                          cmap=plt.cm.RdBu,
                          plot_path: str = 'plot',
                          transpose: bool = False):
    similarities = np.asarray(similarities)
    if transpose:
        similarities = np.transpose(similarities)
        tmp = x_classes
        x_classes = y_classes
        y_classes = tmp
    print('similarities: {}'.format(similarities))
    if mode in ['pearson', 'cosine']:
        plt.imshow(similarities, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
        thresh = 0.
    else:
        plt.imshow(similarities, interpolation='nearest', cmap=cmap)
        thresh = np.max(similarities) / 2.
    plt.colorbar()
    plt.xticks(np.arange(len(x_classes)), [cls.upper() for cls in x_classes], rotation=45)
    plt.yticks(np.arange(len(y_classes)), [cls.upper() for cls in y_classes])

    for i in range(similarities.shape[0]):
        for j in range(similarities.shape[1]):
            plt.text(j, i, '{0:.2f}'.format(similarities[i, j]),
                     horizontalalignment='center',
                     verticalalignment='center', fontsize=7,
                     color='red' if similarities[i, j] > thresh else 'blue')

    plt.tight_layout()
    if not os.path.exists(os.sep.join(plot_path.split(os.sep)[:-1])):
        os.makedirs(os.sep.join(plot_path.split(os.sep)[:-1]))
    print('Saving plot to {}...'.format(plot_path))
    plt.savefig('{}.pdf'.format(plot_path))
    plt.close()


def compare_against_mfd_over_single_label(data: dict,
                                          mfd_dict: dict,
                                          label: str,
                                          mode: str = 'pearson') -> float:
    try:
        words_dict1 = data[label]
    except AttributeError:
        raise ValueError('Label \'{}\' is not available!'.format(label))
    found_keys = list(set(list(words_dict1.keys()) + list(mfd_dict.keys())))
    vec1 = np.asarray([float(words_dict1.get(x, 0)) for x in found_keys], dtype=float)
    vec2 = np.asarray([float(mfd_dict.get(x, 0)) for x in found_keys], dtype=float)
    return {'pearson': pearson_corr(vec1, vec2),
            'cosine': cos_sim(vec1, vec2),
            'dist': dist(vec1, vec2)}[mode]


def compare_against_mfd(approach: str = 'bert_gi_abs-avg_{}_moral_alm_true'.format(GLOBAL_K),
                        chosen_dict: str = 'original',
                        mode: str = 'pearson') -> tuple[float, dict]:
    data1 = load_and_reformat(approach)
    assert chosen_dict in ['original', 'extended']
    words_mfd = OriginalMFD().get_dictionary() if chosen_dict == 'original' else ExtendedMFD().get_dictionary()
    min_keys = list(set(list(data1.keys())).intersection(list(words_mfd.keys())))
    data1 = {key: value for key, value in data1.items() if key in min_keys}
    data2 = {key: value for key, value in words_mfd.items() if key in min_keys}
    metrics = {}
    for key in list(data1.keys()):
        metrics[key] = compare_two_over_single_label(data1, data2, label=key, mode=mode)
    print('metrics: {}'.format(metrics))
    average_metric = mean(list(metrics.values()))
    return average_metric, metrics


def compare_all_explainer_against_mfd(model: str = 'bert',
                                      aggregator: str = 'abs-avg_{}'.format(GLOBAL_K),
                                      dataset: str = 'moral_alm_true',
                                      mode: str = 'pearson'):
    similarity_df = pd.DataFrame(columns=['original', 'extended'])
    for exp in LOCAL_EXPLAINERS:
        similarities_exp_all = {}
        for chosen_dict in ['original', 'extended']:
            similarity_exp1_cd, _ = compare_against_mfd(approach='{}_{}_{}_{}'.format(model,
                                                                                      exp,
                                                                                      aggregator,
                                                                                      dataset),
                                                        chosen_dict=chosen_dict,
                                                        mode=mode)
            similarities_exp_all[chosen_dict] = similarity_exp1_cd
        similarity_df = pd.concat([similarity_df, pd.DataFrame.from_records([similarities_exp_all])])
    plot_confusion_matrix(similarities=similarity_df,
                          x_classes=['original', 'extended'],
                          y_classes=LOCAL_EXPLAINERS,
                          mode=mode,
                          plot_path=os.path.join(PLOTS_DIR, dataset, 'mfds',
                                                 '{}_{}'.format(aggregator, mode)),
                          transpose=True)


def main():
    compare_two('bert_lrp_abs-avg_{}_moral_alm_true'.format(GLOBAL_K),
                'bert_shap_abs-avg_{}_moral_alm_true'.format(GLOBAL_K),
                'pearson')
    for dataset in DATASETS:
        print('Processing dataset: {}'.format(dataset))
        for aggregator in GLOBAL_AGGREGATORS:
            for distance in DISTANCES:
                compare_all_explainer_over_same_aggregator(aggregator=aggregator, mode=distance, dataset=dataset)
        #         compare_all_explainer_against_mfd(aggregator=aggregator, mode=distance, dataset=dataset)
        # for explainer in LOCAL_EXPLAINERS:
        #     for distance in DISTANCES:
        #         compare_single_explainer_over_all_aggregators(explainer=explainer, mode=distance, dataset=dataset)


if __name__ == '__main__':
    main()
