import pandas as pd
from .compound_ingredient import CompoundIngredient
# from expworkup.ingredients.pipeline import one_compound_ingredient
import numpy as np
# from itertools import product
import re
from pathlib import Path
import pickle
from datetime import datetime

def get_mmol_df(reagent_volumes_df, reagents, chemical_count):
    mmol_data = {}
    for i, reagent_col_name in enumerate(reagent_volumes_df.columns):
        if reagents[i]:
            # (M / L  * volume (uL) * (1L / 1000mL) * (1mL / 1000uL) * (1000mmol / 1mol) = mmol 
            concs = np.array(reagents[i].solud_conc)
            vols = reagent_volumes_df[reagent_col_name].values / 1000
            for j, conc in enumerate(concs):
                col_name = f'_reagent_{i}_chemicals_{j}_mmol'
                mmol_data[col_name] = vols*conc
    mmol_df = pd.DataFrame(data=mmol_data)   
    return mmol_df


def get_experimental_run_lab(run_filename):
    lab = 'HC'
    lab_pat = re.compile(f'_({lab})($|.json$)')
    labname = lab_pat.search(run_filename.strip()) #returns if match
    if labname:
        return labname.group(1)

    raise RuntimeError(f'{run_filename} does not specify a supported lab')


def one_compound_ingredient(one_ingredient_series_static, compound_ingredient_label, chemdf_dict):
    one_ingredient_series = one_ingredient_series_static.copy()
    experiment_uid = one_ingredient_series.pop('name')
    compound_ingredient_label_uid = experiment_uid + '_' + compound_ingredient_label.split('_', 1)[1]

    experiment_lab = get_experimental_run_lab(experiment_uid.rsplit('_', 1)[0])

    chem_df = chemdf_dict[experiment_lab]  # .set_index('InChI Key (ID)')

    if one_ingredient_series.isnull().all():
        return(None)
    else:
        one_ingredient_series.dropna(inplace=True)
        myreagent_object = CompoundIngredient(one_ingredient_series,
                                              compound_ingredient_label_uid,
                                              chem_df)
        return(myreagent_object)


def compound_ingredient_chemical_return(ingredient, chemical_count, compoundingredient_func):
    ordered_conc_list = getattr(ingredient, compoundingredient_func)
    diff = chemical_count-len(ordered_conc_list)
    ordered_conc_list.extend([0]*diff)
    return(pd.Series(ordered_conc_list))

# Get Reagent volumes
"""
state_set = pd.read_csv('./extra_data/Me2NH2I_10uL_stateset.link.csv')
ss_reagent_vols = state_set[[col for col in state_set.columns if 'Reagent' in col]]
chemdf_dict = {'HC': pd.read_csv('./extra_data/chemical_inventory.csv', index_col='InChI Key (ID)')}
all_ingredients_df = pd.read_csv('./extra_data/al_ingredients.csv')
"""

def get_concs(all_ingredients_df, chemdf_dict, ss_reagent_vols):
    reagents = []
    for i in range(10):
        df = all_ingredients_df.filter(regex=f'raw_reagent_{i}')
        df.loc[:, 'name'] = list(all_ingredients_df['name'].values)
        reagent = one_compound_ingredient(df.iloc[0], 'raw_reagent_0', chemdf_dict)
        reagents.append(reagent)
    # Get mmol
    mmol_df = get_mmol_df(ss_reagent_vols,
                          reagents,
                          4)
    # Calculate Molarity
    unique_inchis = set()
    for i, reagent in enumerate(reagents):
        if reagent:
            mmol_temp = mmol_df.filter(regex=f'_reagent_{i}_')
            current_cols = list(mmol_temp.columns)
            column_names = [(f'Reagent {i}', inchi) for inchi in reagent.inchilist]
            unique_inchis.update(reagent.inchilist)
            renamed = {current_cols[i]:column_names[i] for i in range(len(current_cols))}
            mmol_df = mmol_df.rename(columns=renamed)

    mmol_df.columns = pd.MultiIndex.from_tuples(mmol_df.columns, names=['Reagent', 'Inchi'])
    summed_mmol = mmol_df.groupby(axis=1, level=['Inchi']).sum()
    summed_molarity = summed_mmol.mul(1000).div(ss_reagent_vols.sum(axis=1), axis='rows')
    replacement_cols = {
        'BDAGIHXWWSANSR-UHFFFAOYSA-N': '_rxn_M_acid',
        'JMXLWMIFDJCGBV-UHFFFAOYSA-N': '_rxn_M_organic',
        'RQQRAHKHDFPBMC-UHFFFAOYSA-L': '_rxn_M_inorganic',
        'ZMXDDKWLCZADIW-UHFFFAOYSA-N': '_rxn_M_solvent',
    }
    replacement_cols_mmol = {
        'BDAGIHXWWSANSR-UHFFFAOYSA-N': '_stoich_mmol_acid',
        'JMXLWMIFDJCGBV-UHFFFAOYSA-N': '_stoich_mmol_org',
        'RQQRAHKHDFPBMC-UHFFFAOYSA-L': '_stoich_mmol_inorg',
        'ZMXDDKWLCZADIW-UHFFFAOYSA-N': '_stoich_mmol_solv',
    }
    summed_mmol = summed_mmol.rename(columns=replacement_cols_mmol)
    summed_molarity = summed_molarity.rename(columns=replacement_cols)

    summed_molarity.to_csv('./phase3/extra_data/stateset_molarities.csv', index_label='name')
    summed_mmol.to_csv('./phase3/extra_data/stateset_mmol.csv', index_label='name')

    return summed_mmol, summed_molarity


def add_dimethyl_columns(stateset):
    data = [105, 12600, 6.09, 26.11,
            22.94, 3.3, 5.05, 16.67, 2.61, 6.54, 5.88, 116.23, 228.53, 197.63,
            30.9, 69.52, 159.01, 16.61, 0, 1, 0, 46.092, 1, 10, 3, 0,
            'escalate_MathematicaUniformRandom_1', 0, 1, 1, 22.94, 62.9978022]
    columns = ['_rxn_temperature_c', '_rxn_reactiontime_s',
               '_feat_organic_0_avgpol_std', '_feat_organic_0_refractivity_std',
               '_feat_organic_0_maximalprojectionarea_std',
               '_feat_organic_0_maximalprojectionradius_std',
               '_feat_organic_0_maximalprojectionsize_std',
               '_feat_organic_0_minimalprojectionarea_std',
               '_feat_organic_0_minimalprojectionradius_std',
               '_feat_organic_0_minimalprojectionsize_std',
               '_feat_organic_0_molpol_std', '_feat_organic_0_asavdwp_std',
               '_feat_organic_0_asa_std', '_feat_organic_0_asah_std',
               '_feat_organic_0_asap_std', '_feat_organic_0_asa-_std',
               '_feat_organic_0_asa+_std', '_feat_organic_0_protpsa_std',
               '_feat_organic_0_hacceptorcount_std',
               '_feat_organic_0_hdonorcount_std',
               '_feat_organic_0_rotatablebondcount_std',
               '_feat_organic_0_organoammoniummolecularweight_std',
               '_feat_organic_0_atomcount_n_std',
               '_feat_organic_0_bondcount_std',
               '_feat_organic_0_chainatomcount_std',
               '_feat_organic_0_ringatomcount_std', '_raw_modelname',
               '_feat_primaryAmine', '_feat_secondaryAmine', '_rxn_plateEdgeQ',
               '_feat_maxproj_per_N', '_raw_RelativeHumidity']
    
    for i, col in enumerate(columns):
        stateset[col] = [data[i]] * len(stateset.index)
    return stateset


def get_stateset(stateset_mmol, stateset_molarities):
    combined = stateset_mmol.join(stateset_molarities)
    combined['_solv_GBL'] = [0] * len(combined.index)
    combined['_solv_DMSO'] = [0] * len(combined.index)
    combined['_solv_DMF'] = [1] * len(combined.index)
    combined['_rxn_organic-inchikey'] = ['JMXLWMIFDJCGBV-UHFFFAOYSA-N'] * len(combined.index)
    organic = combined['_stoich_mmol_org']
    solvent = combined['_stoich_mmol_solv']
    acid = combined['_stoich_mmol_acid']
    inorganic = combined['_stoich_mmol_inorg']
    liquids = combined['_stoich_mmol_acid'] + combined['_stoich_mmol_solv']
    combined["_stoich_org/solv"] = (combined['_stoich_mmol_org']/combined['_stoich_mmol_solv']).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_inorg/solv"] = (inorganic/solvent).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_acid/solv"] = (acid/solvent).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_org+inorg/solv"] = ((organic + inorganic)/solvent).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_org+inorg+acid/solv"] = ((organic + inorganic + acid)/solvent).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_org/liq"] = (organic/liquids).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_inorg/liq"] = (inorganic/liquids).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_org+inorg/liq"] = ((organic + inorganic)/liquids).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_org/inorg"] = (organic/inorganic).replace(np.inf, 0).replace(np.nan, 0)
    combined["_stoich_acid/inorg"] = (acid/inorganic).replace(np.inf, 0).replace(np.nan, 0)
    combined.to_csv('./phase3/extra_data/final_data.csv', index_label='name')
    combined = pd.read_csv('./phase3/extra_data/final_data.csv', index_col='name')
    combined = add_dimethyl_columns(combined)

    return combined

def scale_stateset(stateset, scaler):
    distribution_header = '_raw_modelname'
    amine_header = '_rxn_organic-inchikey'
    # score_header = '_out_crystalscore'
    name_header = 'name'
    num_cols = ['_rxn_M_acid', '_rxn_M_inorganic', '_rxn_M_organic', '_stoich_mmol_org',
       '_stoich_mmol_inorg', '_stoich_mmol_acid', '_stoich_mmol_solv',
       '_stoich_org/solv', '_stoich_inorg/solv', '_stoich_acid/solv',
       '_stoich_org+inorg/solv', '_stoich_org+inorg+acid/solv',
       '_stoich_org/liq', '_stoich_inorg/liq', '_stoich_org+inorg/liq',
       '_stoich_org/inorg', '_stoich_acid/inorg', '_rxn_temperature_c',
       '_rxn_reactiontime_s', '_feat_organic_0_avgpol_std',
       '_feat_organic_0_refractivity_std',
       '_feat_organic_0_maximalprojectionarea_std',
       '_feat_organic_0_maximalprojectionradius_std',
       '_feat_organic_0_maximalprojectionsize_std',
       '_feat_organic_0_minimalprojectionarea_std',
       '_feat_organic_0_minimalprojectionradius_std',
       '_feat_organic_0_minimalprojectionsize_std',
       '_feat_organic_0_molpol_std', '_feat_organic_0_asavdwp_std',
       '_feat_organic_0_asa_std', '_feat_organic_0_asah_std',
       '_feat_organic_0_asap_std', '_feat_organic_0_asa-_std',
       '_feat_organic_0_asa+_std', '_feat_organic_0_protpsa_std',
       '_feat_organic_0_hacceptorcount_std', '_feat_organic_0_hdonorcount_std',
       '_feat_organic_0_rotatablebondcount_std',
       '_feat_organic_0_organoammoniummolecularweight_std',
       '_feat_organic_0_atomcount_n_std', '_feat_organic_0_bondcount_std',
       '_feat_organic_0_chainatomcount_std',
       '_feat_organic_0_ringatomcount_std', '_feat_maxproj_per_N']
    bool_cols = ['_solv_GBL',
                '_solv_DMSO',
                '_solv_DMF',
                '_feat_primaryAmine',
                '_feat_secondaryAmine',
                '_rxn_plateEdgeQ']
    to_exclude = [distribution_header, amine_header, '_raw_RelativeHumidity', '_rxn_M_solvent']
    ss = stateset.drop(to_exclude+bool_cols, axis=1)
    print(stateset.columns)
    ss = stateset[num_cols]
    print(ss.columns)
    x_test = ss.values
    
    x_test = scaler.transform(x_test)
    x_test_bool = stateset[bool_cols].values

    x_test = np.concatenate([x_test, x_test_bool], axis=1)

    return x_test

def get_model_pkl(model_name, set_num):
    basepath = Path(f'./phase3/{model_name}')
    #.strftime("%Y%m%d-%H%M%S")
    list_of_paths = list(basepath.glob(f'{model_name}_set{set_num}_*'))
    latest_path = max(list_of_paths, key=lambda p: p.stat().st_ctime)
    iteration = int(latest_path.name.split('_')[2][2:])
    print(f'Loading {latest_path} : Iteration {iteration}')
    return pickle.load(latest_path.open('rb')), iteration

def dump_model_pkl(model, model_name, set_num, iteration):
    pickle.dump(model, Path(f'./phase3/{model_name}/{model_name}_set{set_num}_it{iteration}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pkl').open('wb'))