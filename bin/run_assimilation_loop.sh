#!/bin/bash

gpus=1
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=16

date='2025-05-31'

# prefix='3dvar'
# config='config/assimilation_loop/exp_3DVar.yaml'
# mkdir experiments/assimilation/$prefix
# cp $config experiments/assimilation/$prefix
# srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/run_assimilation_loop.py --config $config --prefix $prefix --date $date

# prefix='4dvar'
# config='config/assimilation_loop/exp_4DVar.yaml'
# mkdir experiments/assimilation/$prefix
# cp $config experiments/assimilation/$prefix
# srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/run_assimilation_loop.py --config $config --prefix $prefix --date $date

# prefix='3denvar'
# config='config/assimilation_loop/exp_3DEnVar.yaml'
# mkdir experiments/assimilation/$prefix
# cp $config experiments/assimilation/$prefix
# srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/run_assimilation_loop.py --config $config --prefix $prefix --date $date

# prefix='4denvar'
# config='config/assimilation_loop/exp_4DEnVar.yaml'
# mkdir experiments/assimilation/$prefix
# cp $config experiments/assimilation/$prefix
# srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/run_assimilation_loop.py --config $config --prefix $prefix --date $date

prefix='vae-3dvar'
config='config/assimilation_loop/exp_VAE-3DVar.yaml'
mkdir experiments/assimilation/$prefix
cp $config experiments/assimilation/$prefix
srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/run_assimilation_loop.py --config $config --prefix $prefix --date $date

prefix='vae-4dvar'
config='config/assimilation_loop/exp_VAE-4DVar.yaml'
mkdir experiments/assimilation/$prefix
cp $config experiments/assimilation/$prefix
srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/run_assimilation_loop.py --config $config --prefix $prefix --date $date

prefix='lora-3denvar'
config='config/assimilation_loop/exp_LoRA-3DEnVar.yaml'
mkdir experiments/assimilation/$prefix
cp $config experiments/assimilation/$prefix
srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/run_assimilation_loop.py --config $config --prefix $prefix --date $date

prefix='lora-4denvar'
config='config/assimilation_loop/exp_LoRA-4DEnVar.yaml'
mkdir experiments/assimilation/$prefix
cp $config experiments/assimilation/$prefix
srun --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python scripts/run_assimilation_loop.py --config $config --prefix $prefix --date $date

sleep 2
rm -f batchscript-*
