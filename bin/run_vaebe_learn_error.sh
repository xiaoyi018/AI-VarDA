#!/bin/bash

gpus=2
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=4

PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))

echo $PORT

date='2025-05-31'
prefix='vaebe'
config='config/bgerr_learning/vaebe_config.yaml'

mkdir experiments/learning_background/$prefix
cp $config experiments/learning_background/$prefix

srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/learn_background_error.py --config $config --prefix $prefix --date $date --init_method 'tcp://127.0.0.1:'$PORT --world_size $gpus --per_cpus $cpus

sleep 2
rm -f batchscript-*
