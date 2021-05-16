# FUZZ-IEEE COMPETITION ON EXPLAINABLE ENERGY PREDICTION

This repository contains the experiments we performed for IEEE CIS Technical Challenge 2021 to compute the monthly energy consumption of 3248 homes, and explain the generated predictions.

# Instructions for Execution

Before executing the run_arima_catboost.sh command, make sure to download the below 
files from: https://drive.google.com/drive/folders/12yNLdFA-PD7QUjCy2kthFWpSac8EU3B4?usp=sharing

1) Make sure the below files are available at the execution path.

* df_train.feather
* weekday_mean_season_imputed.csv
* df_test.feather
* all_temp_boostrap_median.cs

2) Run "./run_catboost.sh" bash script from the project base directory


