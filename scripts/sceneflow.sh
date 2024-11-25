#!/usr/bin/env bash
set -x
DATAPATH="/data/rahim/data/"
python -W ignore main.py --dataset sceneflow \
    --datapath $DATAPATH \
     --trainlist ./filenames/sceneflow_things_train.txt \
     --testlist ./filenames/sceneflow_test.txt \
     --model leanstereo \
     --logdir ./checkpoints/sf/ \
     --batch_size 8 \
     --test_batch_size 8 \
     --loss_type LogL1Loss_v2 \
     --lr 0.0004 \
     --lrepochs 220,240,290,310:2 \
     --epochs 320 \
     --maxdisp 192