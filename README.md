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

## Multiple GPUs

Note that the main branch is only configured to use 1 gpu. We have an additional branch `multi-gpu` that allows the models to be run across 2 gpus on the same node. We manually assign the number of layers per gpu based on the gpus we were working with. We used 2 NVIDIA A40 GPUs (46GB mem). If you would like to change how the layers are assigned to each gpu, you need to change lines 240, 261, and 236 of `experiments/models/base.py` (Transformer), `experiments/models/denseformer.py`, and `experiments/models/denseformeres.py`, respectively.
