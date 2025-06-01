#!/bin/bash

gpus=1
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=16

date='2025-05-31'
prefix='genbe'
config='config/bgerr_learning/genbe_config.yaml'

mkdir experiments/learning_background/$prefix
cp $config experiments/learning_background/$prefix

srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/learn_background_error.py --config $config --prefix $prefix --date $date

sleep 2
rm -f batchscript-*
