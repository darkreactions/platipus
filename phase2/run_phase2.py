import ast
import pickle

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from modAL.models import ActiveLearner

from utils.dataset_class import *
from utils.result_class import Results

phase2_data = pickle.load(open('./data/phase2_dataset.pkl', 'rb'))


def model_decoder(model_name):
    sk_models = {
        'knn': KNeighborsClassifier,
        'svc': SVC,
        'rf': RandomForestClassifier,
        'lr': LogisticRegression,
        'gbc': GradientBoostingClassifier,
        'dt': DecisionTreeClassifier, 
    }
    return sk_models[model_name] if model_name in sk_models else None


def run_model(model_obj, combo, cat, selection, model_name):
    result = Results(al=True, category=cat, total_sets=phase2_data.num_draws,
                     model_name=model_name, model_setting=combo,
                     selection=selection)
    for set_id in range(phase2_data.num_draws):
        data = phase2_data.get_dataset(cat, set_id, selection)
        if data is None:
            print(
                f'Dataset not found for {cat} {selection} {model_name}. Set: {set_id}. Combination: {combo}')
            continue
        model = model_obj(**combo)

        amines = list(data.keys())
        for a in range(len(amines)):
            amine = amines[a]
            d = data[amine]
            try:
                if 'AL' in cat:
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
                else:
                    model.fit(d['x_t'], d['y_t'])
                    y_pred = model.predict(d['x_v'])
                    result.add_result(d['y_v'], y_pred, set_id, amine)

            except Exception as e:
                print(f'{cat} {selection} {model_name} : {e}')
                continue
    print(result.get_avg())
    with open(f'./phase2/{cat}_{selection}_{model_name}_results.pkl', 'wb') as f:
        pickle.dump(result, f)


model_list = pd.read_csv('./phase2/non_meta_local.csv')
for i, row in model_list.iterrows():
    model = model_decoder(str(row['Model name']))
    if model is not None:
        model_name = row['Model name']
        model_setting = ast.literal_eval(row['Setting'])
        data_category = row['Category']
        data_selection = row['Selection']
        print(f'Running {model_name} on {data_category} {data_selection}')
        run_model(model, model_setting, data_category, data_selection, model_name)

