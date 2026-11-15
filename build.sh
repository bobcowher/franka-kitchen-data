#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate franka-kitchen-data

python scripts/human_control.py "$@"
