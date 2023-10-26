from sklearn.metrics import f1_score
from .utils import format_predictions


class F1:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute(self, y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        if self.dataset in ['imdb', 'sst', 'agnews', 'sms', 'trec', 'youtube']:
            y_true, y_pred = format_predictions(y_true, y_pred)
            # Compute score and return it
            score = f1_score(y_true=y_true,
                             y_pred=y_pred,
                             average='weighted' if self.dataset in ['agnews', 'trec'] else 'binary')
        elif self.dataset in ['moral']:
            score = f1_score(y_true=y_true,
                             y_pred=y_pred,
                             average='weighted')
        else:
            raise ValueError('Dataset {} is not valid!'.format(self.dataset))
        return score
