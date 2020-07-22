from pathlib import Path
import os
import sys
import time
from mpi4py import MPI

from models.non_meta import RandomForest, KNN, SVM, DecisionTree, LogisticRegression, GradientBoosting
from utils.plot import plot_all_graphs
from utils import read_pickle, write_pickle, define_non_meta_model_name, run_non_meta_model, find_avg_metrics
from model_params import common_params, knn_params, svm_params, randomforest_params, logisticregression_params, \
    decisiontree_params, gradientboosting_params, meta_train, meta_test


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #rank = sys.argv[1]
    print(f'My rank is: {rank}')
    #print(f'My rank is : {rank}', file=sys.stderr)

    # Set up the results directory
    if rank == 0:
        results_folder = './results'

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            print('No folder for results storage found')
            print('Make folder to store results at')
        else:
            print('Found existing folder. All results will be stored at')
        print(results_folder)

        # Append the data to cv_stats or overwrite the current results
        overwrite = common_params['cv_stats_overwrite']
        cv_stats_dst = common_params['stats_path']
        if os.path.exists(cv_stats_dst) and overwrite:
            print('Overwriting the current cv_stats.pkl')
            os.remove(cv_stats_dst)
    else:
        time.sleep(5)

    categories = ['category_3', 'category_4_i', 'category_5_i']

    orig_stdout = sys.stdout
    f = open(f'./results/out_{categories[rank]}.txt', 'w')
    sys.stdout = f

    base_model = SVM
    model_params = svm_params
    # for category in categories:
    category = categories[rank]

    run_non_meta_model(
        base_model,
        common_params,
        model_params,
        category
    )

    sys.stdout = orig_stdout
    f.close()
