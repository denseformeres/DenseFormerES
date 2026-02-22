# DenseFormerES

This repository contains code for DenseFormerES and DenseFormer.  
The base transformer implementation is also included to allow for comparison.

## Model description

* base: Standard Transformer
* denseformer: DenseFormer
* denseformeres: DenseFormerES

## Data

The OpenWebText2 dataset can be obtained from https://huggingface.co/datasets/segyges/OpenWebText2.

## Setup

To reproduce the environment used for this project, you can create a Conda environment from the provided `environment.yml`.

### Create the Conda Environment

1. Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed.
2. In a terminal, navigate to the root of the repository where `environment.yml` is located.
3. Run the following command to create the environment:

```bash
conda env create -f environment.yml
```

## Changing Path Names

Paths will need to be changed. Those that need to be changed are lines 4 and 5 of `experiments/compare.py`, line 53 of `experiments/config/base.py`, and lines 23 and 32 of `experiments/data/openwebtext2.py`. 

## Getting Started 

Running training is done by executing `experiments/main.py`. The config arguments can be seen in `experiments/config/base.py`. To run evaluation only use `experiments/eval.py`.