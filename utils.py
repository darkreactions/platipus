
import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Load in the chemistry data, it is harder than it sounds
# def load_chem_dataset(k_shot, meta_batch_size=32, num_batches=100, verbose=False, uniform_only=True):
   
#     # Set up various strings corresponding to headers
#     distribution_header = '_raw_modelname'
#     amine_header = '_rxn_organic-inchikey'
#     score_header = '_out_crystalscore'
#     name_header = 'name'
#     to_exclude = [score_header, amine_header, name_header]
#     path ='.\\data\\0050.perovskitedata_DRP.csv'

#     # Successful reaction is defined as having a crystal score of...
#     SUCCESS = 4

#     # Get amine and distribution counts for the data 
#     df = pd.read_csv(path)
#     if verbose:
#         print('---------- COUNT FOR AMINES ----------')
#         print(df[amine_header].value_counts())
#         print('---------- COUNT FOR DISTRIBUTIONS ----------')
#         print(df[distribution_header].value_counts())
#         print('---------- COUNT FOR UNIFORM AMINES ----------')
#         print(df[df[distribution_header].str.contains('Uniform')][amine_header].value_counts())

#     # Set up the 0/1 labels and drop non-uniformly distributed reactions
#     if uniform_only:
#         df = df.loc[df.distribution_header.contains('Uniform')]
#     df[score_header] = [1 if val == SUCCESS else 0 for val in df[score_header].values]
#     amines = df[amine_header].unique().tolist()

#     # Hold out 5 amines for testing, the other 40 are fair game for the cross validation
#     # Will want to specify which 5 soon
#     num_to_holdout = 5
#     hold_out_amines = np.random.choice(amines ,size=num_to_holdout)
#     amines = [a for a in amines if a not in hold_out_amines]

#     batches = []
#     for _ in range(num_batches):
#         # t for train, v for validate (but validate is outer loop, trying to be consistent with the PLATIPUS code)
#         x_t, y_t, x_v, y_v = [], [], [], []

#         for _ in range(meta_batch_size):
#             # Grab the tasks
#             X = df.loc[df.amine_header==np.random.choice(amines)]

#             y = X[score_header].values

#             # Drop these columns from the dataset
#             X = X.drop(to_exclude, axis=1).values

#             # Standardize features since they are not yet standardized in the dataset
#             scaler = StandardScaler()
#             scaler.fit(X)
#             X = scaler.transform(X)

#             spt = np.random.choice(X.shape[0], size=k_shot, replace=False)
#             qry = np.random.choice(X.shape[0], size=k_shot, replace=False)

#             x_t.append(X[spt])
#             y_t.append(y[spt])
#             x_v.append(X[qry])
#             y_v.append(y[qry])

#         batches.append([np.array(x_t),np.array(y_t),np.array(x_v),np.array(y_v)])

#     assessment_batches = []
#     x_t, y_t, x_v, y_v = [], [], [], []

#     for a in hold_out_amines:
#         # grab task
#         X = df.loc[df.amine==a]

#         y = X[score_header].values
#         X = X.drop(to_exclude, axis=1).values
#         spt = np.random.choice(X.shape[0], size=k_shot, replace=False)
#         qry = [i for i in range(len(X)) if i not in spt]
#         if len(qry) <= 5:
#             print ("Warning: minimal testing data for meta-learn assessment")

#         x_s = X[spt]
#         y_s = y[spt]
#         x_q = X[qry]
#         y_q = y[qry]

#         scaler = StandardScaler()
#         scaler.fit(x_s)

#         x_s = scaler.transform(x_s)
#         x_q = scaler.transform(x_q)

#         x_t.append(x_s)
#         y_t.append(y_s)
#         x_v.append(x_q)
#         y_v.append(y_q)

#     assessment_batches = [np.array(x_t), np.array(y_t), np.array(x_v), np.array(y_v)]
#     # batches should have shape [num_batches, 4, meta_batch_size, k_shot, 68/1 - (68 features, 1 class values)]
#     # not perfectly numpy array like, we're using some lists here

#     return batches, assessment_batches, len(df.columns) - len(to_exclude)
    

# Use the data generator to get points for one task (either a sinusoid OR a line)
def get_task_sine_line_data(data_generator, p_sine, num_training_samples, noise_flag=True):
    if (np.random.binomial(n=1, p=p_sine) == 0):
        # Generate sinusoidal data
        # Discard true amplitude and phase
        x, y, _, _ = data_generator.generate_sinusoidal_data(noise_flag=noise_flag)
    else:
        # Generate line data
        # Discard true slope and intercept
        x, y, _, _ = data_generator.generate_line_data(noise_flag=noise_flag)
    
    # Training data and labels
    x_t = x[:num_training_samples]
    y_t = y[:num_training_samples]
    
    # Validation data and labels
    x_v = x[num_training_samples:]
    y_v = y[num_training_samples:]

    return x_t, y_t, x_v, y_v




