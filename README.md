# Training code for ARES neural network.

## Overview

Training code for ARES network from "Geometric Deep Learning of RNA Structure."

## Installation

### Create conda environment

```
conda create -n ares python=3.8 pip
conda activate ares
```

### Install pytorch

Install appropriate versions of torch and attendant libraries.  You need to figure out the appropriate version of cuda you have on your system, and set below.  Currently shown for cuda 10.1:

```
TORCH="1.5.0"
TORCH_VISION="0.6.0" # Version compatible with torch 1.5.0
CUDA="+cu101" # To install on cpu, use "+cpu" for linux, or "" for mac.
pip install torch==${TORCH}${CUDA} torchvision==${TORCH_VISION}${CUDA} -f https://download.pytorch.org/whl/torch_stable.html pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}${CUDA}.html
```

### Install pytorch-lightning, atom3d, and other generic dependencies.

`pip install pytorch-lightning python-dotenv wandb atom3d==v0.2.4`

### Install e3nn

Install the ARES-compatible version of e3nn (note this version is only provided
for compatability, further development should be done using the main e3nn branch):

`pip install git+https://github.com/drorlab/e3nn_ares.git@6682f60c2c4d99375cfa6321dfcdfa87f104cf8b#egg=e3nn`

### Set environment variables

Copy the environment template to set your environment variables, and update your new `.env` as necessary (in particular the ROOT_DIR variable):

`cp .env.template .env`

### Mac specific instructions:

For mac install, it is possible the following packages need to be pinned for compatibility with torch 1.5.0:

```
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.7 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.7 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install tensorboard==2.4.1
pip install tensorflow==2.3.1
pip install pymatgen==2020.10.20
```

## Usage

### Training

To train the model on CPU, you can run the following command:

`python -m ares.train data/lmdbs/train data/lmdbs/val -f lmdb --batch_size=8 --accumulate_grad_batches=2 --learning_rate=0.0005 --max_epochs=5 --num_workers=8`

Note this will run quite slowly.  To run faster, consider using a GPU (see below).

### Inference

To make a prediction, the general format is as follows:

`python -m ares.predict input_dir checkpoint.ckpt output.csv -f [pdb|silent|lmdb] [--nolabels] [--label_dir=score_dir]`

For example, to predict on the two test pdbs included in the repository, using dummy weights:

`python -m ares.predict data/pdbs data/sample_weights.ckpt output.csv -f pdb --nolabels`

For a 100-nucleotide RNA, this should run at 5 s/it on a single CPU.  The expected output in `output.csv` for the above command would be (with possible fluctuation in up to 7th decimal place):

```
pred,id,rms,file_path
6.6142735,S_000028_476.pdb,0,/PATH/TO/ares_release/data/pdbs/S_000028_476.pdb
9.5971880,S_000041_026.pdb,0,/PATH/TO/ares_release/ares_release/data/pdbs/S_000041_026.pdb
```


### Using a GPU

You can enable a gpu with the `--gpus` flag.  It is also recommended to provision additional CPUs with the `--num_workers` flags (more is better). GPU should have at least 12GB of memory.  For example to train:

`python -m ares.train data/lmdbs/train data/lmdbs/val -f lmdb --batch_size=8 --accumulate_grad_batches=2 --learning_rate=0.0005 --max_epochs=5  --gpus=1 --num_workers=8`

And to predict:

`python -m ares.predict data/pdbs data/sample_weights.ckpt output.csv -f pdb --nolabels --gpus=1 --num_workers=8`

For a 100-nucleotide RNA, this should run at 2 it/s on a Titan X Pascal GPU.  


### Create input LMDB dataset

The LMDB data format allows for fast, random access to your data, which is useful for machine learning workloads.  To convert a PDB or silent dataset to the LMDB format, you can run:

`python -m atom3d.datasets input_dir output_lmdb -f [pdb|silent]`

For example, can use the test data to create one:

`python -m atom3d.datasets data/pdbs test_lmdb -f pdb`
