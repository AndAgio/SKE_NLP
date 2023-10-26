import numpy as np
import pandas as pd
import statistics
from decimal import Decimal
from sklearn.metrics import classification_report

from .time import format_time


def classification(y_true, y_predicted, output_dict=False, target_names=None):
    print('y_true shape: {}'.format(y_true.shape))
    print('y_predicted shape: {}'.format(y_predicted.shape))
    if y_true.shape != y_predicted.shape:
        y_predicted = np.argmax(y_predicted, axis=1)
    # in case of a single target name add its counterpart to the list
    if target_names is not None and len(target_names) == 1:
        label = target_names[0]
        target_names = [f'non-{label}', label]

    return classification_report(y_true, y_predicted, output_dict=output_dict, target_names=target_names)


def print_message(logger, epoch, tot_epochs, index, size, loss,
                  metrics, time=None, mode='train'):
    message = '| Epoch: {}/{} |'.format(epoch, tot_epochs)
    bar_length = 10
    progress = float(index) / float(size)
    if progress >= 1.:
        progress = 1
    block = int(round(bar_length * progress))
    message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
    if mode == 'train':
        message += '| TRAIN: '
    elif mode == 'eval':
        message += '| VALID: '
    else:
        raise ValueError('Mode should be either train or eval!')
    if loss is not None:
        message += 'loss={:.3f} '.format(loss)
    if metrics is not None:
        metrics_message = ''
        for metric_name, metric_value in metrics.items():
            metrics_message += '{}={:.2f}% '.format(metric_name,
                                                    metric_value * float(100))
        message += metrics_message
    if time is not None:
        message += '| time: {} '.format(format_time(time))
    message += ''
    message += '|'
    # print('\n\n' + log_function.__class__.__name__ + '\n\n')
    # if log_function.__class__.__name__ == 'builtin_function_or_method':
    #     log_function(message, end='\r')
    # elif log_function.__class__.__name__ == 'method':
    #     log_function(message)
    # else:
    #     raise ValueError('Not a valid log function!')
    logger.print_it_same_line(message)

def show_message_with_bar(logger, message, index, size):
    text = '|'
    bar_length = 10
    progress = float(index) / float(size) if float(index) / float(size) < 1 else 1
    block = int(round(bar_length * progress))
    text += '[{}]'.format('=' * block + ' ' * (bar_length - block))
    text += '| {}'.format(message)
    logger.print_it_same_line(text)


# Summarize the f1 results in a table
def f1_results(f1_scores, labels):
    values = np.array([[label, round(Decimal(statistics.mean(f1_scores[label])), 2),
                        round(Decimal(statistics.stdev(f1_scores[label])), 2)] for label in
                       labels])
    classification_table = pd.DataFrame(values, columns=['Labels', 'Mean', 'SD'])
    print(classification_table)
    mean_f1 = round(Decimal(statistics.mean(classification_table["Mean"])), 2)
    return mean_f1


# Print all results of the experiment.
def print_results(f1_scores_source, f1_scores_target, f1_scores_source_cf, labels):
    print("Classification table on source")
    f1_source = f1_results(f1_scores_source, labels)
    print("Classification table on target")
    f1_target = f1_results(f1_scores_target, labels)

    print(f'\nAverage F1 Source: {f1_source}')
    print(f'\nAverage F1 Target: {f1_target}')

    print(f'\nAverage Micro F1 Source: {statistics.mean(f1_scores_source["micro avg"])}')
    print(f'Average Macro F1 Source: {statistics.mean(f1_scores_source["macro avg"])}')
    print(f'Average Weighted F1 Source: {statistics.mean(f1_scores_source["weighted avg"])}')

    print(f'\nAverage Micro F1 Target: {statistics.mean(f1_scores_target["micro avg"])}')
    print(f'Average Macro F1 Target: {statistics.mean(f1_scores_target["macro avg"])}')
    print(f'Average Weighted F1 Target: {statistics.mean(f1_scores_target["weighted avg"])}')

    print(f'\nAverage Micro F1 Source CF: {statistics.mean(f1_scores_source_cf["micro avg"])}')
    print(f'Average Macro F1 Source CF: {statistics.mean(f1_scores_source_cf["macro avg"])}')
    print(f'Average Weighted F1 Source CF: {statistics.mean(f1_scores_source_cf["weighted avg"])}')
