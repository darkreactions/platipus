
import pandas as pd

distribution_header = '_raw_modelname'
amine_header = '_rxn_organic-inchikey'
score_header = '_out_crystalscore'
name_header = 'name'
to_exclude = [score_header, amine_header, name_header]
path ='.\\data\\0050.perovskitedata_DRP.csv'
df = pd.read_csv(path)
if True:
    print('---------- COUNT FOR AMINES ----------')
    print(df[amine_header].value_counts())
    print('---------- COUNT FOR DISTRIBUTIONS ----------')
    print(df[distribution_header].value_counts())
    print('---------- COUNT FOR UNIFORM AMINES ----------')
    print(df[df[distribution_header].str.contains('Uniform')][amine_header].value_counts())