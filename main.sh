#!/bin/bash

ENV_NAME="fashionMNIST"

conda env create -n $ENV_NAME -f requirements.yml

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

python model_train.py

conda deactivate

