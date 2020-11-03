# Dataset Description
This document describes the dataset use for this study

## Full Dataset
The complete and raw dataset of is derived from [Crank #57](raw/0057.perovskitedata_redone_2020-07-02.csv). The raw dataset is processed to use DRP features with a [Mathematica Notebook](raw/2020.07.02_cleanup_for_AL.nb). The resulting dataset is available [here](raw/0057.perovskitedata_DRPFeatures_2020-07-02.csv).

There are total of 46 amines in the raw dataset. Only the experiments that satisfy the following conditions are selected:
1. There exists at least 1 success for an amine
2. The experiment is selected using the `Uniform` sampling method under `_raw_modelname` column in the full dataset

20 out of the 46 amines satisfy these conditions. Their InchI keys are:
```python
viable_amines = [   'ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                    'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                    'NLJDBTZLVTWXRG-UHFFFAOYSA-N',
                    'LCTUISCIGMWMAT-UHFFFAOYSA-N',
                    'JERSPYRKVMAEJY-UHFFFAOYSA-N',
                    'VAWHFUNJDMQUSB-UHFFFAOYSA-N',
                    'WGYRINYTHSORGH-UHFFFAOYSA-N',
                    'VNAAUNTYIONOHR-UHFFFAOYSA-N',
                    'FJFIJIDZQADKEE-UHFFFAOYSA-N',
                    'XFYICZOIWSBQSK-UHFFFAOYSA-N',
                    'KFXBDBPOGBBVMC-UHFFFAOYSA-N',
                    'HBPSMMXRESDUSG-UHFFFAOYSA-N',
                    'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
                    'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
                    'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                    'XZUCBFLUEBDNSJ-UHFFFAOYSA-N',
                    'CALQKRVFTWDYDG-UHFFFAOYSA-N',
                    'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                    'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                    'JMXLWMIFDJCGBV-UHFFFAOYSA-N',]
```

Finally, the outcome `_out_crystalscore` is binarized so that crystal scores of `1` , `2` and `3` are considered failed experiments (i.e `0`) and a score of `4` is considered a success (i.e. `1`)

## Phase 1 Dataset
Phase 1 includes 16 amines that are used for cross validation i.e. For each amine, train the model on the 15 remaining amines and train on the held out amine.

The list of amines that are selected for phase 1 are
```python 
  phase1_amines = [ 'ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                    'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                    'NLJDBTZLVTWXRG-UHFFFAOYSA-N',
                    'LCTUISCIGMWMAT-UHFFFAOYSA-N',
                    'JERSPYRKVMAEJY-UHFFFAOYSA-N',
                    'VAWHFUNJDMQUSB-UHFFFAOYSA-N',
                    'WGYRINYTHSORGH-UHFFFAOYSA-N',
                    'VNAAUNTYIONOHR-UHFFFAOYSA-N',
                    'FJFIJIDZQADKEE-UHFFFAOYSA-N',
                    'XFYICZOIWSBQSK-UHFFFAOYSA-N',
                    'KFXBDBPOGBBVMC-UHFFFAOYSA-N',
                    'HBPSMMXRESDUSG-UHFFFAOYSA-N',
                    'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
                    'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
                    'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                    'XZUCBFLUEBDNSJ-UHFFFAOYSA-N']
```

### Training strategies
Multiple training strategies used to train different models. We define parts of the training data as:
- `H` is the "historical" data collected for different amines. For Phase 1 this would be the training data for 15 amines
- `k` is the number of experiments given to a model from the unseen amine as a "jump start". For this experimental campaign we assign `k = 10`
- `x` is the number of experiments that is requested by an active learning model, aka number of iteration of the AL loop. For this experimental campaign we assign `x = 10`

All models are validated on the entire left out amine including the `k+x` points

Depending on the model the following training datasets are used:
- Standard models: Models that are not part of an active learning loop. Training strategies for this kind of model include:
    1. `H` - Training set with historical data ***only***
    2. `Hkx` - Training set with historical data ***and*** `k+x` randomly selected experiments from the held out amine
    3. `kx` - Training set with `k+x` randomly selected experiments from the held out amine

- Active Learning models: Models that can be integrated into an active learning loop. Training strategies include:
    1. `ALHk` - Training set for AL models with historical data and `k` randomly selected experiments from held out amine
    2. `ALk` - Training set for AL models with `k` randomly selected experiments from held out amine

Performance of different models can vary significantly depending on the `k` jump start points. Therefore, five random draws of `k` points are done and the average performance of the model across all five draws is measured.

Further, for the specific training set of `kx` and `ALk` some models will not train unless there are points that belong to both classes. The dataset being used has a large number of failures leading to an unbalanced set. 

To allow such models to train, there are two "modes" for selecting `k` points. 
1. `random` - Where the `k` points selected are completely random, so there is a possibility of no success being present in the draw
2. `success` - Where the `k` points selected has atleast 1 success present in the draw

### Python Class
To access the dataset as a python class import the `DataSet` and `Settings` classes from utils.dataset_class. Unzip and unpickle `./phase1/full_frozen_dataset.zip`

```python
from utils.dataset_class import DataSet, Setting
dataset = pickle.load(open('./data/phase1/full_frozen_dataset.pkl', 'rb'))
category = 'Hkx' # String indicating the training strategy described above
draw_number = 0  # int indicating the random draw number
selection = 'random' # String indicating k sample selection random/success

data = dataset.get_dataset(category, draw_number, selection)
```

### CSV
To access the dataset as a csv, unzip `phase1_dataset.csv` 

## Phase 2 Dataset
Phase 2 