from sklearn.metrics import accuracy_score
from .utils import format_predictions


class Accuracy:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute(self, y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        if self.dataset in ['imdb', 'sst', 'agnews', 'sms', 'trec', 'youtube']:
            y_true, y_pred = format_predictions(y_true, y_pred)
            # Compute score and return it
            score = accuracy_score(y_true=y_true,
                                   y_pred=y_pred)
        elif self.dataset in ['moral']:
            score = accuracy_score(y_true=y_true,
                                   y_pred=y_pred)
        else:
            raise ValueError('Dataset {} is not valid!'.format(self.dataset))
        return score
