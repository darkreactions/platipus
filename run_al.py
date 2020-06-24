from models.non_meta import KNN
import models.meta.main as platipus
from pathlib import Path
from utils.plot import plot_metrics_graph
from utils import read_pickle, write_pickle
from model_params import common_params, knn_params, meta_train, meta_test


if __name__ == '__main__':
    #platipus_train_params = {**common_params, **meta_params, **meta_train}
    #params = platipus.initialize(["PLATIPUS"], platipus_train_params)
    # platipus.main(params)

    #platipus_test_params = {**common_params, **meta_params, **meta_test}
    #params = platipus.initialize(["PLATIPUS"], platipus_test_params)
    # platipus.main(params)

    """
    KNN_params = {**common_params, **knn_params}
    KNN.run_model(KNN_params)

    knn_params['neighbors'] = 1
    knn_params['model_name'] = 'Knn-1'
    KNN1_params = {**common_params, **knn_params}
    KNN.run_model(KNN1_params)
    """

    cv_stats = read_pickle(common_params['stats_path'])
    amines = cv_stats['Knn-2']['amine']
    print(cv_stats.keys())

    models_to_plot = ['PLATIPUS', 'Knn-1']
    for i, amine in enumerate(amines):
        plot_metrics_graph(96, cv_stats, './results',
                           amine=amine, amine_index=i, models=models_to_plot)
