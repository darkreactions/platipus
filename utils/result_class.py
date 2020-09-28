
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                              precision_score, recall_score)
from collections import defaultdict, namedtuple
import pandas as pd


class Results:
    def __init__(self, al=False, category='H', total_sets=1,
                 model_name=None, model_setting=None,
                 selection='random'):
        self.al = al
        self.category = category
        self.total_sets = total_sets
        self.model_name = model_name
        self.model_setting = model_setting

        self.selection = selection
        self.metric_names = ['bcr', 'accuracy', 'precision', 'recall',
                             'y_true', 'y_pred']
        self.metrics = {}

        for name in self.metric_names:
            self.metrics[name] = defaultdict(list)

    def add_result(self, y_true, y_pred, set_id=0, amine=None):
        self.metrics['y_true'][(set_id, amine)].append(y_true)
        self.metrics['y_pred'][(set_id, amine)].append(y_pred)
        self.metrics['accuracy'][(set_id, amine)].append(
            accuracy_score(y_true, y_pred))
        self.metrics['precision'][(set_id, amine)].append(
            precision_score(y_true, y_pred, zero_division=1))
        self.metrics['recall'][(set_id, amine)].append(
            recall_score(y_true, y_pred, zero_division=1))
        self.metrics['bcr'][(set_id, amine)].append(
            balanced_accuracy_score(y_true, y_pred))

    def get_avg(self, metric_name='bcr'):
        if metric_name in self.metrics:
            # print(self.metrics[metric_name])
            df = pd.DataFrame(self.metrics[metric_name])
            mean = df.mean(axis=0, skipna=True).to_numpy()
            if self.al:
                return mean
            else:
                return mean[0]
