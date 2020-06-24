
import pandas as pd

distribution_header = '_raw_modelname'
amine_header = '_rxn_organic-inchikey'
score_header = '_out_crystalscore'
name_header = 'name'
to_exclude = [score_header, amine_header, name_header]
path ='.\\data\\0050.perovskitedata_DRP.csv'

# These amines have a reaction drawn from a uniform distribution with a successful outcome
viable_amines = ['ZEVRFFCPALTVDN-UHFFFAOYSA-N',
'KFQARYBEAKAXIC-UHFFFAOYSA-N',
'NLJDBTZLVTWXRG-UHFFFAOYSA-N',
'LCTUISCIGMWMAT-UHFFFAOYSA-N',
'JERSPYRKVMAEJY-UHFFFAOYSA-N',
'JMXLWMIFDJCGBV-UHFFFAOYSA-N',
'VAWHFUNJDMQUSB-UHFFFAOYSA-N',
'WGYRINYTHSORGH-UHFFFAOYSA-N',
'FCTHQYIDLRRROX-UHFFFAOYSA-N',
'VNAAUNTYIONOHR-UHFFFAOYSA-N',
'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
'FJFIJIDZQADKEE-UHFFFAOYSA-N',
'XFYICZOIWSBQSK-UHFFFAOYSA-N',
'UMDDLGMCNFAZDX-UHFFFAOYSA-O',
'HBPSMMXRESDUSG-UHFFFAOYSA-N',
'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
'CALQKRVFTWDYDG-UHFFFAOYSA-N',
'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
'BAMDIFIROXTEEM-UHFFFAOYSA-N',
'XZUCBFLUEBDNSJ-UHFFFAOYSA-N']

df = pd.read_csv(path)
if True:
    print('---------- COUNT FOR AMINES ----------')
    print(df[amine_header].value_counts())
    print('---------- COUNT FOR DISTRIBUTIONS ----------')
    print(df[distribution_header].value_counts())
    print('---------- COUNT FOR UNIFORM AMINES ----------')
    print(df[df[distribution_header].str.contains('Uniform')][amine_header].value_counts())
    print('---------- COUNT FOR SUCCESSES ----------')
    print(df[(df[score_header] == 4) & (df[distribution_header].str.contains('Uniform'))][amine_header].value_counts())
    print('---------- COUNT FOR VIABILITYS ----------')
    print(df[(df[amine_header].isin(viable_amines)) & (df[distribution_header].str.contains('Uniform'))][amine_header].value_counts())