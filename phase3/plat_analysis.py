from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pickle
import sklearn
from datetime import datetime, timedelta
import torch
import pandas as pd

from utils.dataset_class import Phase3DataSet, Setting
from phase3.scripts.custom_pipeline import get_model_pkl, dump_model_pkl
from models.meta.platipus_class import Platipus
from hpc_scripts.param_generator import get_all_params
from models.meta.init_params import init_params

results = {}
results[0] = {10022:1, 1441:0, 6127:0, 9716:0, 8205:0, 7177:0, 17269:1, 15434:0, 10040:1, 9958:0}
results[1] = {17833:0, 9275:0, 6470:0, 2378:1, 6890:0, 17171:0, 16771:0, 14533:0, 5180:1, 13862:0}

def run_analysis():
    phase3dataset = pickle.load(open('../data/phase3_dataset.pkl', 'rb'))
    dataset = {}
    dataset[0] = phase3dataset.get_dataset('metaALHk', 0, 'random')['JMXLWMIFDJCGBV-UHFFFAOYSA-N']
    dataset[1] = phase3dataset.get_dataset('metaALHk', 1, 'random')['JMXLWMIFDJCGBV-UHFFFAOYSA-N']

    return tuple(**dataset)


    