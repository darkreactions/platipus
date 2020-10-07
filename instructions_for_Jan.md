## Setting up
1. Download/Clone latest version of `https://github.com/darkreactions/platipus/tree/shekar_rewrite`
2. Install requirements.txt, most important are `modAL` and `scikit-learn`

## Dataset
1. Unzip the dataset for Phase 1 crossvalidation from a file called `./data/full_frozen_dataset.zip`
2. The `full_frozen_dataset.pkl` is a python pickle file.
3. Before unpickling the file make sure that the Dataset class is imported: `from utils.dataset_class import DataSet`
4. Then unpickle the file
    ``` 
    import pickle
    f = open('./data/full_frozen_dataset.zip', 'rb')
    data = pickle.load(f)
    f.close()
    ```
5. This new data variable acts like the DataSet class, so you can call DataSet methods
6. Call data.print_config() to display available datasets for different models

## Models
1. Use modAL and scikit-learn to import models
2. The models hyperparameters are defined in the csv file I will send along with this file 
