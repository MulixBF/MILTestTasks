#!/bin/bash


python train.py --experiment_name "configuration_3x3_3x3" --model_configuration conv3x3,conv3x3 --num_epochs 300
python train.py --experiment_name "configuration_3x3_5x5" --model_configuration conv3x3,conv5x5 --num_epochs 300
python train.py --experiment_name "configuration_5x5_3x3" --model_configuration conv5x5,conv3x3 --num_epochs 300
python train.py --experiment_name "configuration_5x5_5x5" --model_configuration conv5x5,conv5x5 --num_epochs 300
python train.py --experiment_name "configuration_random" --num_epochs 300
