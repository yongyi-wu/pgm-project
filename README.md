# 10-708 Probabilistic Graphical Models - Project

> This repository is adapted from [https://github.com/divelab/DIG/tree/dig-stable/benchmarks/xgraph](https://github.com/divelab/DIG/tree/dig-stable/benchmarks/xgraph). 


## Preliminaries

### Install dependencies

```shell
conda env create -f requirements.yml
pip install -e .
```

### Download data

```shell
cd benchmarks/xgraph

mkdir -p checkpoints
cd checkpoints
gdown --folder 19krHmYpGDmR4abDB3bz0fVLYUeFmsmR7

mkdir -p datasets
cd datasets
gdown --folder 1dt0aGMBvCEUYzaG00TYu1D03GPO7305z
```

You also need to decapitalize dataset names (`Graph-SST2 -> graph_sst2`, `Graph-SST5 -> graph_sst5`, `Graph-Twitter -> twitter`). 


## Reproduction

Change model name in `run.sh` according to names in `benchmarks/xgraph/config/explainers`. Then, execute the following:

```bash
./run.sh
```

To plot reproduction results, use the `plot.ipynb` notebook. 
