import sys
from collections import defaultdict
import pickle
import itertools

import numpy as np

from utils.dataset_class import DataSet, Setting
from utils.result_class import Results

if __name__ == '__main__':
    all_results = []
    cat, selection, model_name = sys.argv[1:]

    dataset = pickle.load(open('./data/full_frozen_dataset.pkl', 'rb'))

    result = Results(al=True, category=cat, total_sets=dataset.num_draws,
                     model_name=model_name, model_setting=combo,
                     selection=selection)
