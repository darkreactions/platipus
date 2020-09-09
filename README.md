# Dark Reactions Project - platipus

## Table of Contents
   * [Introduction](#introduction)
      * [Experimental Plan](#experimental-plan)
   * [Getting Started](#getting-started)
      * [Remotely](#on-a-lab-machine-remotely)
      * [Locally](#on-your-own-device-locally)
   * [Repo Structure](#structure)  
   * [Usage](#usage)
      * [Dataset](#dataset)
      * [Meta Models](#meta-models)
      * [Non-Meta Models](#non-meta-models)
        * [Visualize Decision Trees](#visualize-decision-trees)
      * [Results](#results)
        * [Graphs](#graphs)
        * [Model Performance Statistics](#statistics)
        * [Fine Tuning Results](#fine-tuning-performance-logs)
      * [To Run Non-meta Models](#to-run-the-non-meta-models-based-on-the-categories)
      * [To Fine Tune Non-meta Models](#to-run-fine-tuning-for-the-non-meta-models)
      * [Transfer Files Remotely w/ SCP](#to-scp-results-and-graphs-from-the-remote-lab-machines-to-local-directories)
   * [Built with](#built-with)
   * [Authors and Acknowledgements](#authors-and-acknowledgements)
   * [Future Work](#to-be-continued)

## Introduction
This is the machine learning model repository related to the [Dark Reactions Project](https://darkreactions.haverford.edu/) to learn about the experimental data collected from the chemists and provide insights on the direction of future experiments. On top of the traditional machine learning models, this repository is using meta-learning models such as MAML and PLATIPUS to predict the experimental outcomes more accurately. The use of this repository extends to the interpretability project and the recommendation system as well. 

### Experimental Plan
Currently, we are using "0057.perovskitedata_DRPFeatures_2020-07-02.csv" as the raw historical dataset. Within the dataset, **all the experiments of 16 amines with uniform distribution and at least one successful experiment are selected as historical data**. 

Excluding the distribution header (column name: <span>_raw_modelname</span>), the amine header (column name: <span>_rxn_organic-inchikey</span>), the score header (column name: <span>_out_crystalscore</span>), and the name header (column name: <span>name</span>), the data has in total **51 features**. As for the labels, instead of the four classes the raw dataset provided, there are only **two classes of labels** used to train and validate models: if the experiment has a crystal score of 4, we mark the label as **1, a success**; otherwise we mark the label as **0, a failure**. 

As of August 1, the experimental procedure is as follows:  
   1.  First, establish the definitions of the following categories before generating the corresponding training set variants for the dataset:
   <ul>
      <li> **Category 3**: Models trained on **historical data only with no active learning involved**. In other words, each task/amine-specific is trained on all the experiments of the other 15 amines prior to validation.  
         Models include: kNN, SVM, Decision Tree, Random Forest, Logistic Regression, Gradient Boosting Tree.  
         (May include NN in the future)
      </li>  
      <li> 
         **Category 4.1**: Models trained on **historical data and k+x many amine data with no active learning involved**. In other words, each task/amine-specific is trained on all the experiments of the other 15 amines and k+x many experiments of this amine prior to validation.  
         Models include: kNN, SVM, Decision Tree, Random Forest, Logistic Regression, Gradient Boosting Tree, MAML.
      </li>  
      <li> 
         **Category 4.2**: Models trained on **k+x amine data only with no active learning involved**.  
         Models include: kNN, SVM, Decision Tree, Random Forest, Logistic Regression, Gradient Boosting Tree.  
         **NOTE**: SVM, Logistic Regression and Gradient Boosting Tree are using a slightly different set of data under this category. For more details, see portion on creating the dataset.
      </li>  
      <li> 
         **Category 5.1**: Models trained on **historical data and k many amine data with x many active learning iterations**. In other words, each task/amine-specific is trained on all the experiments of the other 15 amines and k many experiments of this amine before going through active learning x many times.  
         Models include: kNN, SVM, Decision Tree, Random Forest, Logistic Regression, Gradient Boosting Tree, PLATIPUS.
      </li>  
      <li> 
         **Category 5.2**: Models trained on **k amine data only with x many active learning iterations**. In other words, each amine-specific model is first trained with k many experiments of this amine before conducting active learning x many times.    
         Models include: kNN, SVM, Decision Tree, Random Forest, Logistic Regression, Gradient Boosting Tree.  
         **NOTE**: SVM, Logistic Regression and Gradient Boosting Tree are using a slightly different set of data under this category. For more details, see portion on creating the dataset.
      </li>  
   </ul>
      **As of now, k is set to 10 and x is set to 10.**
      
   2. Create the dataset for each amine in the following steps:  
      <ul>
      <li>For <b>training set</b>: 
        <ul>
          <li><b>Category 3</b>: No need for random draws since there's only the historical data from all other 15 amines.</li>
          <li>For all other categories, first do 5 random draws and select k + x experiments from all the amine-specific experiments. Then do another 5 random draws to select the k + x experiments, but this time we require the selected k experiments to have at least 2 successful experiments. This is to provide a jump start for some models due to the low number of successful experiments of some amines. We call the second set of 5 random draws "random draws with success". Keep the selected k and x of each random draw so that all categories can use the same ones. </li>
          <li><b>Category 4.1</b>: there are five different training sets from the 5 regular random draws. Each set consists of all the historical experiments from the other 15 amines <b>plus</b> the selected k+x amine-specific experiment. <b>Each set shares the same k experiments from the corresponding random draws with category 5.1</b>.
          </li>
          <li>
            <b>Category 4.2</b>: there are five different training sets from the 5 random draws with success. Each set consists of <b>only</b> the selected k+x amine-specific experiment. <b>Each set shares the same k experiments from the corresponding random draws with category 5.2</b>.
          </li>
          <li><b>Category 5.1</b>: there are five different training sets from the 5 regular random draws. Each set consists of all the historical experiments from the other 15 amines <b>plus</b> the selected k amine-specific experiment. <b>Each set shares the same k experiments from the corresponding random draws with category 4.1</b>.
          </li>
          <li>
            <b>Category 5.2</b>: there are five different training sets from the 5 random draws with success. Each set consists of <b>only</b> the selected k amine-specific experiment. <b>Each set shares the same k experiments from the corresponding random draws with category 4.2</b>.
          </li>
        </ul>  
      </li>
      <li>For <b>validation set</b>: the validation set consists of all the experiments of this amine, and this is the case across all categories. In other words, the validation data will be all the amine-specific data, and the validation labels will be all the corresponding amine-specific labels.
      </li>
      <li>For <b>active learning pool</b> used by models from category 5.1 and 5.2: the active learning pool consists of all the experiments of this amine except for the selected k experiments from each random draw. 
      </li>
      </ul>  
   3. **_(TO BE UPDATED WITH THE FINALIZED PLAN)_** Fine tune and look for the best hyper-parameter combinations of each categorical model using the above training and validation sets. The validation process is the same for all models, and there's no active learning involved during fine tuing. We then store each configuration's performance and select the one with the best performance afterwards with some post-processing. **For the purpose of our project, the best performance is defined as when a set of hyper-parameter configurations has a BCR score higher than the previous best BCR score minus epsilon and has a higher recall in our implementation. The current epsilon is 0.01.**  
   For more information, see our [hyper-parameter tuning log here](https://docs.google.com/document/d/1HK8sn3cmrVFRjhJ8rYur_7LnbnreADlbcvCWDPGTq6E/edit?usp=sharing).
   4. Train and validate the models using the above training and validation sets. For models with active learning, conduct x=10 active learning iterations for validation. 
   5. Evaluate and plot each categorical model's performance by calculating the average accuracies, precisions, recalls, and balanced classification rates (referred to as BCR) over all runs on the different training set variants of each amine-specific model and of all 16 models. Plot the per-amine success volume versus BCR for the chosen x value as well.  
   6. **CONSULT THE GROUP BEFORE PROCEEDING TO RUNNING ON THE HOLD OUT SET.**
   7. Run all of the above on the holdout set: 3 (out of 4) amines with successes and experiments performed uniformly at random that have been held out since the beginning of this process
   8. Based on the above, choose the best of the below model categories (balancing between the above metrics) to run live:
      * PLATIPUS (active learning loop required)
      * MAML (no active learning involved)
      * Best from category 3 above (no active learning involved)
      * Best from category 4 above (no active learning involved)
      * Best from category 5 above (active learning loop required)
   9. Live experiments will be run on one of the held out amines (the one with an 11% success rate) as well as the 3 additional unknown amines Mansoor has prepared
  10. For each of the 4 test amines: run a single 96 well plate of uniform randomly chosen experiments for a single new (or held out) amine.
  11. Screen the above amines.  Stop evaluating any amines that have 0 success from the 96.
  12. Randomly choose k of the 96 new amine experiments to use as k above.
  13. Train the models from step 8.
  14. Test the fully trained models against the 96 - k experiments for the new amine not yet used for training.  Run all the evaluations from step 6.
  15. Use the fully trained (post-AL-loop) models to predict successes given all possible experiments (challenge problem list).
  16. Rank order any labeled successful experiments based on model-reported confidence so that highest-confidence labels are at the top.  Perform 96/#models experiments in order from this list per model on a single new plate.

## Getting Started

### On a Lab Machine, Remotely

To run the codes on a Haverford CS Department lab machine remotely, either set up your ssh login or connect through Haverford’s VPN before the following steps. For more information, please see [this document](https://docs.google.com/document/d/1uSfLYzD9UnveVdMeRMvp-At2NC1YFqGQfzBOfNT1J14/edit) from the Haverford CS Department.

1. ssh into a lab machine by entering:
   ```sh 
   ssh -Y h205c@<machine-name-here>.cs.haverford.edu 
   ```
   and enter the password for that lab machine.

2. Create your own working directory on the lab machine:
   ```sh 
   mkdir <your-directory-name-here>
   ```
   and go into your directory with commands:
   ```sh
   cd <your-directory-name-here>
   ```
3. Clone the platipus repo to your directory:
   ```sh
   git clone https://github.com/darkreactions/platipus.git
   ```
   and follow the prompts on the terminal.

> The following steps are not necessary if using Ellis, Hoover, Vaughan, or Fried

4. Before creating virtual environments, either use ```venv``` that comes with Python 3.3 or higher, or install conda by [downloading it](https://www.anaconda.com/products/individual#linux) onto your local device, transfer it to the lab machine with commands:
   ```sh
   scp the-directory-of-anaconda-sh-file h205c@<machine-name-here>.cs.haverford.edu:/home/h205c/Downloads
   ```
   go to the Downloads folder on the machine, run:
   ```sh 
   bash <anaconda-file-name>
   ```
   and follow the instructions on the terminal.

5. Create your virtual environment:  
     
   For venv, make sure that you have python 3.6 or higher. Then, run:
   ```sh
   python3 -m venv platipus
   ```
   For anaconda, run:
   ```sh
   conda create -n platipus python=3.8
   ```
6. Activate your virtual environment:  
     
   For venv, run:
   ```sh
   source platipus/bin/activate
   ```
   For anaconda, run:
   ```sh
   conda activate platipus
   ```
7. Install all requirements by running:
   ```sh
   pip install -r requirements.txt
   ```
### On your own device, locally
Follow step 2 to 7 above, with a slightly different step 4: either use venv and make sure your python version is 3.6 or higher, or [download and install anaconda](https://www.anaconda.com/products/individual)

## Structure
```
├── README.md
├── __init__.py
├── data: the folder that contains all the experimental data files
│   ├── DRP chemical experiments csv files
│   ├── 2020.07.02_cleanup_for_AL.nb
│   ├── data_analysis.ipynb
│   ├── model_analysis.ipynb
│   ├── non_meta_data.pkl: the dataset pickle file for all non-meta models.
│   ├── percent.csv
│   ├── selected_k
│   │   ├── CSV Files of the selected experiments from each random draws
│   └── temperature_humidity_logs.csv
├── fine_tune.sh: the bash script to run fine tuning process and log the terminal output for models
│                  except for SVM and GBT. 
├── fine_tune_gbc.sh: the bash script to run fine tuning process and log the terminal output for GBT. 
├── fine_tune_svm.sh: the bash script to run fine tuning process and log the terminal output for SVM. 
├── ft_gbc.py: the python script to fine tune GBT.
├── ft_svm.py: the python script to fine tune SVM.
├── hpc_params.py: the parameters/setting file used to run script on High Performance Computer 
│                  (**Usage to be updated**)
├── hpc_scripts: the folder that contains the script to run on HPC
│   ├── __init__.py
│   ├── hpc_params.py
│   ├── local_meta_test.py
│   ├── maverick.sh
│   └── run_mpi.py
├── maverick.sh: (**Usage to be updated**)
├── multiproc_runner.py: a python script to run models concurrently.
├── model_params.py: The parameters/setting file to run models.
├── models: the folder than contains all the machine learning models
│   ├── __init__.py
│   ├── meta
│   │   ├── FC_net.py
│   │   ├── __init__.py
│   │   ├── core.py:  (**Usage to be updated**)
│   │   ├── main.py:  (**Usage to be updated**)
│   │   ├── maml.py:  (**Usage to be updated**)
│   │   └── platipus.py: (**Usage to be updated**)
│   └── non_meta: the folder that contains all the non-meta machine learning models.
│       ├── DecisionTree.py: Decision tree model with active learning feature.
│       ├── GradientBoosting.py: Gradient boosting tree model with active learning feature.
│       ├── KNN.py: k-nearest neighbors model with active learning feature.
│       ├── LinearSVM.py: Linear SVM classifier model with active learning feature. 
│       │                 This is a mid-stage test model. Please use SVM.py instead.
│       ├── LogisticRegression.py: Logistic regression model with active learning feature. 
│       ├── RandomForest.py: Random forest model with active learning feature.
│       ├── SVC.py: SVM models built using libsvm package. 
│       │           This is a mid-stage test model. Please use SVM.py instead.
│       ├── SVM.py: SVM classifier model with active learning feature.
│       └── __init__.py
├── mpi_run.sh: (**Usage to be updated**)
├── requirements.txt: the required packages to run files in this repo. (See "Packages used")
├── results: the folder that contains all the performance statistics and the corresponding graphs.
│   ├── avg_metrics_all_models.png: the 4-plot graphs of all models. The 4 plots are: accuracy, precision, recall, 
│   │                               and BCR vs. the additional number of points given.
│   ├── by_category: the 4-plot graphs of models in each sub-categories.
│   │   ├── amine_only.png
│   │   ├── amine_only_AL.png
│   │   ├── historical_amine.png
│   │   ├── historical_amine_AL.png
│   │   └── historical_only.png
│   ├── by_model: the 4-plot graphs of models of each type of classifier.
│   │   ├── Decision_Tree.png
│   │   ├── KNN.png
│   │   ├── Logistic_Regression.png
│   │   ├── PLATIPUS.png
│   │   ├── Random_Forest.png
│   │   └── SVM.png
│   ├── H: the 4-plot graphs of models of category 3.
│   │   ├── average_metrics_H.png
│   │   ├── amine-specific model graphs
│   ├── nonAL: the 4-plot graphs of models of category 4.1 and 4.2.
│   │   ├── average_metrics_category_4.png
│   │   ├── amine-specific model graphs
│   ├── AL: the 4-plot graphs of models of category 5.1 and 5.2.
│   │   ├── average_metrics_AL.png
│   │   ├── amine-specific model graphs
│   ├── success_rate: the 2-plot graphs of all models and of models under each category. 
│   │   │             The two plots are: BCR vs. Success Volume and BCR vs. Success Percentage. 
│   │   ├── bcr_against_all.png
│   │   ├── bcr_against_success_H.png
│   │   ├── bcr_against_success_category_4.png
│   │   └── bcr_against_success_AL.png
│   ├── cv_statistics.pkl: the pkl file that contains all the model performance statistics.
│   └── winning_models.png: the 4-plot graph of models with the highest AUC of BCR in each sub-category. 
│                            PLATIPUS is alwasy included.
├── run_al.py: the python script to run models with the model_params.py setting.
├── run_mpi.py
└── utils: the folder than contains all the utility python scripts and functions
    ├── __init__.py
    ├── csv_check.py
    ├── data_generator.py
    ├── dataset.py
    ├── plot.py
    └── utils.py
```

## Usage
### Dataset
In folder ```data```, you will find the pickle file named ```non_meta_data.pkl``` that contains the dataset dictionary. If not, run process_dataset function by either calling it in the utils/dataset.py script, or run any number of models using multiproc_runner.py or run_al.py. For more info, please see the instruction below.  

To load the dictionary in a python terminal or a jupyter notebook, use either the read_pickle function in utils/utils.py, or use the following lines:
```python

path_of_dataset = <the directory where the dataset is located, type=str>

with open(path_of_dataset, "rb") as f:
    dataset = pickle.load(f)
```
The structure of the dataset dictionary is as follows:  
```
dataset 
├── 'full' or 'test'
│    └── 'random' or 'w/_success'
│        └── 'k_x' or 'k_y'
│          └── integer 0 to 4 (given we want 5 random draws)
│            └── amine
│              └── data
│
└── settings as a tuple in the format of
  ('full' / 'test', 'w/_AL' / 'w/o_AL', 'w/_hx' / 'w/o_hx', 'w/_k' / 'w/o_k', 'w/_x' / 'w/o_x', 'random' / 'w/_success')
     └──integer 0 to 4 (given we want 5 random draws, only 0 if querying for category 3)
       └──'x_t', 'y_t', 'x_v', 'y_v', 'all_data', or 'all_labels'
         └──amine
           └──data
```

### Meta Models
**TO BE UPDATED**

### Non-Meta Models
Unlike meta-models that utilize meta-training and meta-learning, non-meta models are the traditional machine learning models that follow the same train-validate-test pipeline. Currently the repository has 6 running non-meta classifier models: kNN, SVM, Decision Tree, Random Forest, Logistic Regression, and Gradient Boosting Tree. All of them are built on top of the base classifier model:```ActiveLearningClassifier```, located in models/non_meta/BaseClassifier.py.

The basic pipeline of running an amine-specific non-meta model is as follows:
1. Identify the machine learning model and the hyper-parameter configuration. 

2. Load the dataset under designated categorical settings.  

> For each random draw:  

3. Train the model with the loaded training set and evaluate using all the experiments of that amine.  

4. If the model does conduct active learning, query the data pool **x=10** times, each time picking the most uncertain point in the pool.  

> After running all random draws

5. Average out model's performance over all the random draws and save the metrics to a dictionary in the repository.  

For more details, see the documentation in the base model's python script and each model's python script.

#### Visualize Decision Trees
To visualize each decision tree, first set the visualize parameter of decisiontree_params dictionary in model_params.py to True. Then when running decision tree models, it will automatically generate the .dot files of each decision tree. To convert it into actual pictures, run the following command:
```sh
dot -Tpng "<dt_file_name>.dot" -o "<desired file name>.png"
```
The file name of the .dot files should be in the format of ```<model_name>_dt_<amine_handle>_<random_draw_id>.dot```.

To complile the graphs in batches, it is recommended to use Jupyter notebook, copy the list of amines from dataset.py, and generate/run commands with a for-loop.

### Results
#### Graphs
See [folder structure](#structure) for more details.

#### Statistics
In this folder, ```cv_statistics.pkl``` contains the dictionary with all the performance information of the models you've run. The current structure of the dictionary is ```{model_name:{metric_name: [metric_value]}}```. To load the dictionary, see similar instructions in the [dataset](#dataset) section.

#### Fine Tuning Performance Logs
If you are fine tuning any models, there will be a bunch of pkl files named in the format of ```ft_<model_name>.pkl```. These pickle files contain the dictionaries with the performance metrics of each configuration tried during the fine tuning stage. The current structure of the dictionary is ```{configuration in string form:{metric_name: metric_value}}```. To load the dictionary, see similar instructions in the [dataset](#dataset) section.

### To run the non-meta models based on the categories
0. Set up all the parameters in model_params.py. Make sure fine_tuning in common_params is set to False.
1. Specify the categories that we want to run the models on in [run_al.py](https://github.com/darkreactions/platipus/blob/0665aed38ba2e2978285511c38f2052ab8f98ff7/run_al.py#L29) or multiproc_runner.py.
2. Define the models that we are want to run in run_al.py file or multiproc_runner.py.
3. Type screen in terminal 
4. Enter
5. Activate your virtual environment:  
     
   For virtualenv:
   ```sh
   source platipus/bin/activate
   ```
   For anaconda:
   ```sh
   conda activate platipus
   ```
6. Run run_al.py file to run models in a single thread:
   ```sh
   python run_al.py
   ```
   
   Run multiproc_runner.py to run models in multi-threads:
   ```sh
   python multiproc_runner.py
   ```

### To run fine tuning for the non-meta models
For models except for SVM and GBT:
1. Change the ‘fine_tuning’ key in the common_params dictionary in model_params.py into True

2. Go to each model's python script to change the hyper-parameters and their ranges. 

3. Specify the categories and models we want to run fine tuning on in run_al.py file 

4. Type screen in terminal and enter  

5. Activate your virtual environment:  
     
   For virtualenv:
   ```sh
   source platipus/bin/activate
   ```
   For anaconda:
   ```sh
   conda activate platipus
   ```
6. Run fine_tune.sh for models except for SVM and GBC:
   ```sh 
   bash fine_tune.sh
   ```
For SVM and GBT:
1. Go into ft_svm.py or ft_gbc.py to change the hyper-parameters and their ranges you'd like to fine tune.

2. Specify the categories to fine tune on in fine_tune_svm.sh / fine_tune_gbc.sh, and change the maximum index in the for loop based on the total number of combinations you'd like to try.

3. Activate your virtual environment:  
     
   For virtualenv:
   ```sh
   source platipus/bin/activate
   ```
   For anaconda:
   ```sh
   conda activate platipus
   ```
4. Run fine_tune_svm.sh for SVM:
   ```sh 
   bash fine_tune_svm.sh
   ```
   Run fine_tune_gbc.sh for GBT:
   ```sh 
   bash fine_tune_gbc.sh
   ```
   
> Requires a different method to run if running on HPC, but we haven’t changed it yet

### To scp results and graphs from the remote lab machines to local directories  
   Run the following command from the local terminal:
   ```sh
   scp h205c@<machine-name-here>.cs.haverford.edu:/home/h205c/<file-full-directory> <local-directory>
   ```
   For example, to transfer the cv_statistics.pkl file from Fried to your Desktop, run
   ```sh
   scp h205c@fried.cs.haverford.edu:/home/h205c/<your-working-directory>/platipus/results/cv_statistics.pkl Desktop
   ```
   and follow the instructions on the terminal.

## Built with
* [Gareth’s DRP repository](https://github.com/gxnicholas/gareth-platipus)
* Packages used

   | Name | Documentation |
   | ----- | ----- |
   | libsvm | https://pypi.org/project/libsvm/ |
   | matplotlib | https://matplotlib.org/contents.html |
   | modAL | https://modal-python.readthedocs.io/en/latest/index.html |
   | numpy | https://numpy.org/doc/stable/ |
   | pandas | https://pandas.pydata.org/docs/user_guide/index.html#user-guide |
   | scikit-learn | https://scikit-learn.org/stable/user_guide.html |
   | torch | https://pytorch.org/docs/stable/index.html |
   | torchvision | Same as above |

## Authors and Acknowledgements
 > In ascending order of the last names 
* Gareth Nicholas [@gxnicholas](https://github.com/gxnicholas)
* Venkateswaran Shekar [@vshekar](https://github.com/vshekar)
* Sharon (Xiaorong) Wang [@sharonwang77](https://github.com/sharonwang77)
* Vincent Yu [@vjmoriarty](https://github.com/vjmoriarty)

## To Be Continued...
- [ ] Make sure PLATIPUS and non-meta models are storing their performance metrics in the same structure to cv_statistics.pkl
- [ ] Fine tune all non-meta models.
- [ ] Post process fine tuning information.
- [ ] Update plot.py to incorporate the current model line-up.  
- [ ] Implement the testing branch of the dataset.  
- [ ] Implement the testing portion of the non-meta models.  
