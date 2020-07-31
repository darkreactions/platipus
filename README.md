# Dark Reactions Project - platipus

## Introduction
One short paragraph as a summary of the repo
Experimental plan here maybe? Need to add it to somewhere

## Getting started

### On a lab machine, remotely:

To run the codes on a Haverford CS Department lab machine remotely, either set up your ssh login or connect through Haverford’s VPN before the following steps. For more information, please see [this document](https://docs.google.com/document/d/1uSfLYzD9UnveVdMeRMvp-At2NC1YFqGQfzBOfNT1J14/edit) created by the CS Department.

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
### On your own device, locally:
Follow step 2 to 7 above, with a slightly different step 4: either use venv and make sure your python version is 3.6 or higher, or [download and install anaconda](https://www.anaconda.com/products/individual)

## Structure
To be continued...

## Usage
### To run the non-meta models based on the categories:
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

### To run fine tuning for the non-meta models:
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

