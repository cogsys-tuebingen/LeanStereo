#!/usr/bin/env bash
set -x
DATAPATH="/data/rahim/data/Kitti_2015/"
python save_disp.py --datapath $DATAPATH \
                    --dataset kitti \
                    --testlist ./filenames/kitti15_test.txt \
                    --loadckpt ./checkpoints/kitti15/checkpoint_best.ckpt \
                    --loss_type LogL1Loss_v2
