#!/bin/bash
time=$(date "+%m%d-%H%M")
python train.py \
--client_num 4 \
--c_rounds 1000 \
--round_per_train 100 \
--version ${time}_version \
--device_ids 0,0,0,0 \
--gpus 0 \
--use_multiprocessing False