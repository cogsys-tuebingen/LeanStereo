#!/usr/bin/env bash
set -x
DATAPATH="/data/"
python -W ignore test_model.py --dataset sceneflow \
     --datapath $DATAPATH \
     --testlist ./filenames/sceneflow_test.txt \
     --model leanstereo \
     --loadckpt ./checkpoints/sf/checkpoint_best.ckpt \
     --example_img_freq 100 \
     --loss_type LogL1Loss_v2 \
     --maxdisp 192