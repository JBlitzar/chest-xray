#!/bin/bash

bash cleanup.sh
tensorboard --logdir=runs 1>/dev/null 2>/dev/null &
python train.py