from utils.dataset_class import DataSet, Setting
import pandas as pd
import pickle

dataset = pickle.load(open('./data/full_frozen_dataset.pkl', 'rb'))
dataset.print_config()
