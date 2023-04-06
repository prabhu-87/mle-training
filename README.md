# steps taken to run the code and export yml file

 -> a new environment mle-dev was created and activated
 -> The nonstandard code was pushed to the git and run using - python3 nonstandardcode.py command
 -> few errors where encountered hence changes to the code were made and then run. 
 -> The environment file was exported using export env.yml command

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
