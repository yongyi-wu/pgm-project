#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pgm

mkdir -p logs
for dataset in $(ls benchmarks/xgraph/config/datasets/)
do
    dataset=${dataset/\.yaml/}
    date > logs/${dataset}.log
    python -um benchmarks.xgraph.subgraphx datasets=$dataset explainers=subgraphx &>> logs/${dataset}.log
    date >> logs/${dataset}.log
done
