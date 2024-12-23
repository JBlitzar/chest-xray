#!/bin/bash

bash cleanup.sh
tensorboard --logdir=runs &
python train.py