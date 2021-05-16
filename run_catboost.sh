#!/bin/bash

# install python packages.
python ./install_packages.py

# run CatBoostModel
python ./Aux_Energy_prediction_Competition.py

python ./Energy_prediction_Competition.py
