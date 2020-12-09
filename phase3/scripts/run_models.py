import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from modAL.models import ActiveLearner

from .custom_pipeline import get_concs, get_stateset, scale_stateset, get_model_pkl, dump_model_pkl
from utils.dataset_class import Phase3DataSet, Setting
from .al_lab_process import get_al_result_sheet, write_result_sheet

SCOPES = 'https://www.googleapis.com/auth/spreadsheets'


def get_sheet_data(sheet_id, sheet_name, sheet_range):
    """
        This is to get chemical data for stateset generation
    """
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=f'{sheet_name}!{sheet_range}').execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        return values

def prepare_stateset(sheet_id=None):
    data = [['null', 'null'], ['1.37', 'milliliter'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['0.9', 'milliliter'], ['0.7641', 'gram'], ['0.2927', 'gram'], ['0.63', 'milliliter'], ['null', 'null'], ['1.07', 'milliliter'], ['0.5222', 'gram'], ['1.06', 'milliliter'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['1.21', 'milliliter'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['1.21', 'milliliter']]
    if sheet_id:
        data = get_sheet_data(sheet_id, 'Entry', 'D17:E48')
    
    chemical_num = 3
    reagent_num = None
    reagent_data = {}
    for i in range(len(data)):
        if i%5 == 0:
            if reagent_num == None:
                reagent_num = 0
            else:
                reagent_num += 1
        if i%5 != 0:
            #print(f'{i} : {(i-1)%4}')
            if data[i][0] != 'null':
                chemical_num = (i%5) - 1
                key = f'_raw_reagent_{reagent_num}_chemicals_{chemical_num}_actual_amount'
                #print(key, data[i][0])
                reagent_data[key] = float(data[i][0])

    # "Fix" reagent 5
    delete_key = '_raw_reagent_5_chemicals_0_actual_amount'
    if delete_key in reagent_data:
        reagent_data['_raw_reagent_6_chemicals_0_actual_amount'] += reagent_data[delete_key] 
        reagent_data.pop(delete_key, None)
    

    chemical_amounts = pd.read_csv('./phase3/extra_data/al_ingredients.csv')
    for chemical in reagent_data:
        chemical_amounts.loc[0, chemical] = reagent_data[chemical]
    
    stateset_template = pd.read_csv('./phase3/extra_data/Me2NH2I_10uL_stateset.link.csv')
    ss_reagent_vols = stateset_template[[col for col in stateset_template.columns if 'Reagent' in col]]
    chemdf_dict = {'HC': pd.read_csv('./phase3/extra_data/chemical_inventory.csv', index_col='InChI Key (ID)')}
    phase3_training = pickle.load(open('./phase3/phase3_dataset.pkl', 'rb'))

    summed_mmol, summed_molarity = get_concs(chemical_amounts, chemdf_dict, ss_reagent_vols)
    summed_mmol.index.rename('name')
    summed_molarity.index.rename('name')
    stateset = get_stateset(summed_mmol, summed_molarity)
    x_stateset = scale_stateset(stateset, phase3_training.scaler)

    stateset.to_csv(f'./phase3/statesets/stateset_{datetime.today().date()}.csv')
    np.savetxt(f'./phase3/statesets/scaled_stateset_{datetime.today().date()}.csv', x_stateset, delimiter=",")

def load_data(model_name):
    phase3_training = pickle.load(open('./phase3/phase3_dataset.pkl', 'rb'))
    
    training_dataset = {}
    model_list = {}
    iteration = {}
    query_index = {}
    query_instances = {}
    x_stateset = {}
    rvol_raw = {}
    ss_reagent_vols = {}

    for set_id in range(2):
        training_dataset[set_id] = phase3_training.get_dataset('ALHk', set_id, 'random')['JMXLWMIFDJCGBV-UHFFFAOYSA-N']
        model, current_iteration = get_model_pkl(model_name, set_id)
        model_list[set_id] = model
        iteration[set_id] = current_iteration
        query_index[set_id] = pickle.load(Path(f'./phase3/{model_name}/q_idx_set{set_id}.pkl').open('rb'))
        print(query_index)
        query_instances[set_id] = pickle.load(Path(f'./phase3/{model_name}/q_inst_set{set_id}.pkl').open('rb'))
        x_stateset[set_id] = np.genfromtxt(f'./phase3/statesets/scaled_stateset_{datetime.today().date()}.csv', delimiter=',')
        rvol_raw[set_id] =  pd.read_csv('./phase3/extra_data/Me2NH2I_10uL_stateset.link.csv')
        ss_reagent_vols[set_id] = rvol_raw[set_id][[col for col in rvol_raw[set_id].columns if 'Reagent' in col]]
        print(query_index[set_id])
        ss_reagent_vols[set_id].drop(ss_reagent_vols[set_id].index[query_index[set_id]], inplace=True)
        x_stateset[set_id] = np.delete(x_stateset[set_id], query_index[set_id], 0)

    return training_dataset, model_list, iteration, query_index, query_instances, x_stateset, ss_reagent_vols


def update_dt():
    sheet_range = ['A2:H2', 'A3:H3']
    vial_ids = ['A7', 'B7']
    training_set, model_list, iteration, query_idx, query_inst, x_stateset, ss_reagent_vols = load_data('dt')
    
    for set_id in range(2):
        current_round = iteration[set_id] + 2 # zero index + next round 
        print(f'Current round {current_round}')
        current_result = get_al_result_sheet(sheet_name=f'Round {current_round}')
        row = current_result[current_result['name']==f'DT{set_id}'].iloc[0]
        score = 1 if row['Crystal Score'] == 4 else 0
        idx = int(row['Index'])
        exp_location = ss_reagent_vols[set_id].index.get_loc(idx)
        instance = x_stateset[set_id][exp_location]
        model_list[set_id].teach([instance], [score])
        print(f'Teaching dt set{set_id} instance {idx} with score of {score}')

        # Add instance, dump instance and model to disk 
        ss_reagent_vols[set_id].drop(ss_reagent_vols[set_id].index[exp_location], inplace=True)
        x_stateset[set_id] = np.delete(x_stateset[set_id], exp_location, 0)
        query_idx[set_id].append(idx)
        query_inst[set_id].append(instance)

        
        dump_model_pkl(model_list[set_id], 'dt', set_id, current_round-1)
        pickle.dump(query_idx[set_id], Path(f'./phase3/dt/q_idx_set{set_id}.pkl').open('wb'))
        pickle.dump(query_inst[set_id], Path(f'./phase3/dt/q_inst_set{set_id}.pkl').open('wb'))
        

        # Next prediction
        query_location, query_instance = model_list[set_id].query(x_stateset[set_id])
        vol_row = ss_reagent_vols[set_id].iloc[query_location]
        next_idx = vol_row.index[0]
        row = [f'DT{set_id}', int(next_idx), vial_ids[set_id], int(vol_row['Reagent1 (ul)']), 
                int(vol_row['Reagent2 (ul)']), int(vol_row['Reagent3 (ul)']), 
                int(vol_row['Reagent7 (ul)'])/2,int(vol_row['Reagent7 (ul)'])/2]
        print(row)
        write_result_sheet(sheet_range[set_id], [row], sheet_name=f'Round {current_round+1}')
        



def update_knn():
    sheet_range = ['A4:H4', 'A5:H5']
    vial_ids = ['C7', 'D7']
    training_set, model_list, iteration, query_idx, query_inst, x_stateset, ss_reagent_vols = load_data('knn')
    for set_id in range(2):
        current_round = iteration[set_id] + 2 # zero index + next round 
        current_result = get_al_result_sheet(sheet_name=f'Round {current_round}')
        row = current_result[current_result['name']==f'KNN{set_id}'].iloc[0]
        score = 1 if row['Crystal Score'] == 4 else 0
        idx = int(row['Index'])
        exp_location = ss_reagent_vols[set_id].index.get_loc(idx)
        instance = x_stateset[set_id][exp_location]
        model_list[set_id].teach([instance], [score])
        print(f'Teaching knn set{set_id} instance {idx} with score of {score}')

        # Add instance, dump instance and model to disk 
        ss_reagent_vols[set_id].drop(ss_reagent_vols[set_id].index[exp_location], inplace=True)
        x_stateset[set_id] = np.delete(x_stateset[set_id], exp_location, 0)
        print(f'Full index: {query_idx}')
        query_idx[set_id].append(idx)
        query_inst[set_id].append(instance)
        
        
        dump_model_pkl(model_list[set_id], 'knn', set_id, current_round-1)
        pickle.dump(query_idx[set_id], Path(f'./phase3/knn/q_idx_set{set_id}.pkl').open('wb'))
        pickle.dump(query_inst[set_id], Path(f'./phase3/knn/q_inst_set{set_id}.pkl').open('wb'))
        

        # Next prediction
        query_location, query_instance = model_list[set_id].query(x_stateset[set_id])
        vol_row = ss_reagent_vols[set_id].iloc[query_location]
        next_idx = vol_row.index[0]
        row = [f'KNN{set_id}', int(next_idx), vial_ids[set_id], int(vol_row['Reagent1 (ul)']), 
               int(vol_row['Reagent2 (ul)']), int(vol_row['Reagent3 (ul)']), 
               int(vol_row['Reagent7 (ul)'])/2,int(vol_row['Reagent7 (ul)'])/2]
        print(row)
        write_result_sheet(sheet_range[set_id], [row], sheet_name=f'Round {current_round+1}')
        

def reset_q_index(indices, stateset, model, set_id):
    instances = []
    for idx in indices:
        instance = stateset[idx]
        instances.append(instance)
    pickle.dump(indices, Path(f'./phase3/{model}/q_idx_set{set_id}.pkl').open('wb'))
    pickle.dump(instances, Path(f'./phase3/{model}/q_inst_set{set_id}.pkl').open('wb'))

def reset_dt():
    phase3_training = pickle.load(open('./phase3/phase3_dataset.pkl', 'rb'))
    dataset = {}
    dt = {}
    learner = {}
    training_set, model_list, iteration, query_idx, query_inst, x_stateset, ss_reagent_vols = load_data('dt')
    query_idx = {
        0: [47, 11, 0, 9279, 9555, 2, 242],
        1: [47, 8895,8896, 19082, 19083, 2, 3]
    }
    results = {
        0: [0,0,0,0,1,0,0],
        1: [0,0,0,0,0,0,0]
    }
    for i in range(2):
        dataset[i] = phase3_training.get_dataset('ALHk', i, 'random')['JMXLWMIFDJCGBV-UHFFFAOYSA-N']
        dt[i] = DecisionTreeClassifier(**{'criterion': 'gini', 'splitter': 'best', 'max_depth': 9, 'min_samples_split': 6, 'min_samples_leaf': 3, 'class_weight': {0: 0.2520408163265306, 1: 0.7479591836734694}})
        learner[i] = ActiveLearner(estimator=dt[i], X_training=dataset[i]['x_t'], y_training=dataset[i]['y_t'])
        for x,idx in enumerate(query_idx[i]):
            print(f'Index: {idx} for model {i}')
            instance = x_stateset[i][idx]
            learner[i].teach([instance], [results[i][x]])
        pickle.dump(learner[i], open(f'./phase3/dt/dt_set{i}_it5_20201207-190739.pkl', 'wb'))

def reset_knn():
    phase3_training = pickle.load(open('./phase3/phase3_dataset.pkl', 'rb'))
    dataset = {}
    knn = {}
    learner = {}
    
    training_set, model_list, iteration, query_idx, query_inst, x_stateset, ss_reagent_vols = load_data('knn')
    query_idx = {
        0: [0, 1, 2, 3, 4, 5],
        1: [0, 1, 2, 3, 4, 5],
    }
    results = {
        0: [0,0,0,0,0,0],
        1: [0,0,0,0,0,0],
    }
    for i in range(2):
        dataset[i] = phase3_training.get_dataset('ALHk', i, 'random')['JMXLWMIFDJCGBV-UHFFFAOYSA-N']
        knn[i] = KNeighborsClassifier(**{'n_neighbors': 1, 'leaf_size': 1, 'p': 1})
        learner[i] = ActiveLearner(estimator=knn[i], X_training=dataset[i]['x_t'], y_training=dataset[i]['y_t'])
        for x,idx in enumerate(query_idx[i]):
            print(f'Index: {idx} for model {i}')
            instance = x_stateset[i][idx]
            learner[i].teach([instance], [results[i][x]])
        # pickle.dump(learner[i], open(f'./phase3/knn/knn_set{i}_it5_20201207-190739.pkl', 'wb'))
        # knn_set1_it6_20201208-122421
        pickle.dump(learner[i], open(f'./phase3/knn/knn_set{i}_it5_20201207-190739.pkl', 'wb'))
        pickle.dump(query_idx[i], open(f'./phase3/knn/q_idx_set{i}.pkl', 'wb'))

if __name__=='__main__':
    #prepare_stateset('1Lkwf2kLhPQedMoVBTGwaQWXERj5WPkEOgYxVmCrPOMg')
    #update_knn()
    #training_set, model_list, iteration, query_idx, query_inst, x_stateset, ss_reagent_vols = load_data('knn')
    #reset_q_index([0, 1, 2, 3, 4, 5, 6,], x_stateset[1], 'knn', 1)
    update_dt()
    # reset_knn()
    # reset_dt()