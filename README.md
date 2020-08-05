# Dark Reactions Project - platipus

## Table of Contents
   * [Introduction](#introduction)
      * [Experimental Plan](#experimental-plan)
   * [Getting Started](#getting-started)
      * [Remotely](#on-a-lab-machine-remotely)
      * [Locally](#on-your-own-device-locally)
   * [Structure](#structure)  
      * [Folder Roadmap](#folder-roadmap)
   * [Usage](#usage)
      * [To Run Non-meta Models](#to-run-the-non-meta-models-based-on-the-categories)
      * [To Fine Tune Non-meta Models](#to-run-fine-tuning-for-the-non-meta-models)
      * [Transfer Files Remotely w/ SCP](#to-scp-results-and-graphs-from-the-remote-lab-machines-to-local-directories)
   * [Built with](#built-with)
   * [Authors and Acknowledgements](#authors-and-acknowledgements)
   * [Future Work](#to-be-continued)

## Introduction
This is the machine learning model repository of [Dark Reactions Project](https://darkreactions.haverford.edu/) to learn about the experimental data collected from the chemists and provide insights on the direction of future experiments. On top of the traditional machine learning models, this repository is using meta-learning models such as MAML and PLATIPUS to predict the experimental outcomes more accurately. The use of this repository extends to the interpretability project and the recommendation system as well. 

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
### Folder Roadmap

### Main Driver Files

### data

### models

### results

How to Interpret

### utils

## Usage
### To run the non-meta models based on the categories
1. Specify the categories that we want to run the models on in the [run_al.py file](https://github.com/darkreactions/platipus/blob/0665aed38ba2e2978285511c38f2052ab8f98ff7/run_al.py#L29)  
2. Define the models that we are want to run in run_al.py file  
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
6. Run run_al.py file 
   ```sh
   python run_al.py
   ```

### To run fine tuning for the non-meta models
1. Change the ‘fine_tuning’ key in the common_params dictionary in model_params.py into True
     
2. Specify the categories and models we want to run fine tuning on in run_al.py file  
  * If it is for SVM model, specify the categories in the parser argument in fine_tuning_svm.sh 
  * If it is for GBC model, specify the categories in the parser argument in fine_tuning_gbc.sh 
3. Type screen in terminal and enter  
4. Activate your virtual environment:  
     
   For virtualenv:
   ```sh
   source platipus/bin/activate
   ```
   For anaconda:
   ```sh
   conda activate platipus
   ```
5. Run fine_tune.sh for models except for SVM and GBC:
   ```sh 
   bash fine_tune.sh
   ```
   Run fine_tune_svm.sh to fine tune SVM:
   ```sh
   bash fine_tune_svm.sh
   ```
   Run fine_tune_gbc.sh to fine tune GBC:
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
* Packages used:

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
- [ ] Implement the testing branch of the dataset.  
- [ ] Implement the testing portion of the non-meta models.  
