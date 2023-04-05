# steps taken to run the code and export yml file

* Git clone the repo mle-training.git - using the command - git clone git@github.com:sravanisasu66/mle-training.git
* If not already present install any python interpreter or Miniconda from the following link : https://docs.conda.io/en/latest/miniconda.html
* From the command line navigate to the path when the git clone repo is present
* Create a virual environment using the yml file using the command - conda env create -f env.yml
* Activate the virtual environment using the command - conda activate mle-dev
* Excute the python code nonstandardcode.py using the command - python nonstandardcode.py

# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python < scriptname.py >
