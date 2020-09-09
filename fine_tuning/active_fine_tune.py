import sys
from collections import defaultdict
import pickle
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from modAL.models import ActiveLearner

import numpy as np

from utils.dataset_class import DataSet, Setting
from utils.result_class import Results


def knn_params():

    ft_params = {
        'n_neighbors': [i for i in range(1, 10)],
        'leaf_size': [i for i in range(1, 51)],
        'p': [i for i in range(1, 4)]
    }
    return ft_params


def lr_params():
    class_weights = [{0: i, 1: 1.0 - i}
                     for i in np.linspace(.05, .95, num=50)]
    class_weights.append('balanced')
    class_weights.append(None)

    ft_params = {
        'penalty': ['l2', 'none'],
        'dual': [False],
        'tol': [1e-4, 1e-5],
        'C': [.1 * i for i in range(1, 3)],
        'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [4000, 5000, 6000, 7000, 9000],
        'class_weight': class_weights
    }
    return ft_params


def dt_params():
    class_weights = [{0: i, 1: 1.0 - i} for i in np.linspace(.05, .95, num=50)]
    class_weights.append('balanced')
    class_weights.append(None)

    max_depths = [i for i in range(9, 26)]
    max_depths.append(None)

    ft_params = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': max_depths,
        'min_samples_split': [i for i in range(2, 11)],
        'min_samples_leaf': [i for i in range(1, 4)],
        'class_weight': class_weights
    }
    return ft_params


def gbc_params():
    ft_params = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 200, 500, 1000],
        'criterion': ['friedman_mse', 'mse', 'mae'],
        'max_depth': [i for i in range(1, 9)],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 5, 10],
        'ccp_alpha': [.1 * i for i in range(1)]
    }
    return ft_params


def rf_params():
    class_weights = [{0: i, 1: 1.0 - i} for i in np.linspace(.05, .95, num=50)]
    class_weights.append('balanced')
    class_weights.append(None)

    ft_params = {
        'n_estimators': [100, 200, 500, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [i for i in range(1, 9)],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True],
        'min_samples_leaf': [i for i in range(1, 6)],
        'min_samples_split': [i for i in range(2, 11)],
        'ccp_alpha': [.1 * i for i in range(1)],
        'class_weight': class_weights
    }
    return ft_params


def svc_params():
    class_weights = [{0: i, 1: 1.0-i} for i in np.linspace(.1, .9, num=9)]
    class_weights.append('balanced')
    class_weights.append(None)

    ft_params = {
        'kernel': ['poly', 'sigmoid', 'rbf'],
        'C': [.01, .1, 1, 10, 100],
        'degree': [i for i in range(1, 6)],
        'gamma': ['auto', 'scale'],
        'tol': [.001, .01, .1, 1],
        'decision_function_shape': ['ovo'],
        'break_ties': [False],
        'class_weight': class_weights,
        'probability': [True],
    }
    return ft_params


def get_combos(ft_params):
    combinations = []
    keys, values = zip(*ft_params.items())
    for bundle in itertools.product(*values):
        combinations.append(dict(zip(keys, bundle)))
    return combinations


def model_decoder(model_name):
    sk_models = {
        'knn': (KNeighborsClassifier, get_combos(knn_params())),
        'svc': (SVC, get_combos(svc_params())),
        'rf': (RandomForestClassifier, get_combos(rf_params())),
        'lr': (LogisticRegression, get_combos(lr_params())),
        'gbc': (GradientBoostingClassifier, get_combos(gbc_params())),
        'dt': (DecisionTreeClassifier, get_combos(dt_params()))
    }
    return sk_models[model_name]


if __name__ == '__main__':
    all_results = []
    cat, selection, model_name = sys.argv[1:]

    dataset = pickle.load(open('./data/full_frozen_dataset.pkl', 'rb'))

    # categories = ['H', 'Hkx', 'kx']
    # al_categories = ['ALHk', 'ALk']
    # selection = ['random', 'success']
    sk_model, combinations = model_decoder(model_name)

    for combo in combinations:
        result = Results(al=True, category=cat, total_sets=dataset.num_draws,
                         model_name=model_name, model_setting=combo,
                         selection=selection)
        for set_id in range(dataset.num_draws):
            data = dataset.get_dataset(cat, set_id, selection)
            if data is None:
                print(
                    f'Dataset not found for {cat} {selection} {model_name}. Set: {set_id}. Combination: {combo}')
                continue
            model = sk_model(**combo)

            amines = list(data.keys())
            for a in range(len(amines)):
                amine = amines[a]
                d = data[amine]
                try:
                    learner = ActiveLearner(
                        estimator=model, X_training=d['x_t'], y_training=d['y_t'])
                    X_pool = np.array(d['x_vrem'], copy=True)
                    y_pool = np.array(d['y_vrem'], copy=True)
                    for x in range(10):
                        query_index, query_instance = learner.query(X_pool)
                        X, y = X_pool[query_index].reshape(
                            1, -1), y_pool[query_index].reshape(1, )
                        learner.teach(X=X, y=y)
                        X_pool, y_pool = np.delete(
                            X_pool, query_index, axis=0), np.delete(y_pool, query_index)
                        y_pred = learner.predict(d['x_v'])
                        result.add_result(d['y_v'], y_pred, set_id, amine)

                except Exception as e:
                    print(f'{cat} {selection} {model_name} : {e}')
                    continue

            all_results.append(result)

        pickle.dump(all_results, open(
            f'./results/{cat}_{selection}_{model_name}_results.pkl', 'wb'))
