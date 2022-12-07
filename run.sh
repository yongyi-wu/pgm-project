#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pgm

model=actor_critic
mkdir -p logs/$model
for dataset in $(ls benchmarks/xgraph/config/datasets/)
do
    dataset=${dataset/\.yaml/}
    date > logs/${model}/${dataset}.log
    python -um benchmarks.xgraph.${model} datasets=$dataset explainers=$model &>> logs/${model}/${dataset}.log
    date >> logs/${model}/${dataset}.log
done
