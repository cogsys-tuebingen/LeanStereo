#!/usr/bin/env bash
set -x
DATAPATH="/data/rahim/data/Kitti/"
python -W ignore main.py --dataset kitti \
    --datapath $DATAPATH \
    --trainlist ./filenames/kitti12_15_15_train.txt \
    --testlist ./filenames/kitti15_val.txt \
    --model leanstereo \
    --logdir ./checkpoints/kitti15/ \
    --batch_size 16 \
    --test_batch_size 8 \
    --loss_type smoothL1 \
    --lr 0.001 \
    --lrepochs 300:2 \
    --epochs 500 \
    --maxdisp 192 \
    --loadckpt ./checkpoints/sf/checkpoint_best.ckpt