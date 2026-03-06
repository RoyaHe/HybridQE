# HybridQE

This temporary repository is created for the implementation of the paper Hybrid Query Answering over Incomplete Text-Labeled Graph Databases.

## 1. Datasets Preparation

We introduce two benchmarks for the task of hybrid query answering over incomplete semi-structured database, ProdHybridQA and BioHybridQA, which are derived from real-world e-commerce and protein-disease data. The benchmarks will be opensourced later.

# HybridQE

Implementation for hybrid query answering over incomplete text-labeled graph databases.

## Overview
HybridQE supports two datasets (from real-world e-commerce and protein-disease data. The benchmarks will be opensourced later):
- `BioHybridQA` (triple-style graph)
- `ProdHybridQA` (hyperedge-style product-attribute graph)

The codebase provides training and evaluation for:
- `beta` model (`--geo beta`)
- `gamma` model (`--geo gamma`)

## Repository Layout
- `main.py`: training / validation / test entry point
- `models.py`: model definitions
- `dataloader.py`: data loading and query iterators
- `utils.py`: helper functions
- `betae.sh`: sample BetaE runs
- `gamma.sh`: sample GammaE runs

## Requirements
- Python 3.7+ (recommended: 3.8 or 3.9)
- PyTorch 1.7+
- `tqdm`
- `transformers`
- `tensorboardX`

Install dependencies:

```bash
pip install torch tqdm transformers tensorboardX
```


## Dataset Preparation
Pass the dataset root with `--data_directory`. The expected layout is:

```text
<data_directory>/
  prime/
    train.txt
    valid.txt
    test.txt
    train-queries.pkl
    valid-queries.pkl
    test-queries.pkl
    train-answers.pkl
    valid-easy-answers.pkl
    valid-hard-answers.pkl
    test-easy-answers.pkl
    test-hard-answers.pkl
    id2ent.pkl
    id2rel.pkl
    ent_id2title.pkl
    stats.txt
  amazon/
    train.txt
    valid.txt
    test.txt
    train-queries.pkl
    valid-queries.pkl
    test-queries.pkl
    train-answers.pkl
    valid-easy-answers.pkl
    valid-hard-answers.pkl
    test-easy-answers.pkl
    test-hard-answers.pkl
    prod_id2ent.pkl
    attr_id2ent.pkl
    attr_id2rel.pkl
    prodid2title.pkl
    stats.txt
```

## Quick Start

### 1. Train + test on `prime` with BetaE

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
  --graph_type triple \
  --dataset prime \
  --data_directory /path/to/Datasets \
  -n 128 -b 512 -d 800 -g 50 \
  --tasks 1p.2p.3p.2i.3i.1n.2in.3in.2u.3u \
  -lr 0.0001 --max_steps 450001 --cpu_num 0 \
  --geo beta -betam "(1600,2)" \
  --graph_added
```

### 2. Train + test on `amazon` with BetaE

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
  --graph_type hyperedge \
  --dataset amazon \
  --data_directory /path/to/Datasets \
  -n 128 -b 512 -d 800 -g 50 \
  --tasks 1p.2i.3i.1n.2in.3in.2u.3u \
  -lr 0.0001 --max_steps 450001 --cpu_num 0 \
  --geo beta -betam "(1600,2)" \
  --graph_added
```

### 3. Train + test on `amazon` with GammaE

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
  --graph_type hyperedge \
  --dataset amazon \
  --data_directory /path/to/Datasets \
  -n 128 -b 512 -d 800 -g 60 \
  --tasks 1p.2i.3i.1n.2in.3in.2u.3u \
  -lr 0.0001 --max_steps 450001 --cpu_num 0 \
  --geo gamma -gammam "(1600,4)" --drop 0.1 \
  --graph_added
```

## Evaluate from a Checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --dataset amazon \
  --data_directory /path/to/Datasets \
  --geo beta \
  --checkpoint_path /path/to/checkpoint_dir
```

## Notes
- `--graph_type` should be:
  - `triple` for `BioHybridQA`
  - `hyperedge` for `ProdHybridQA`
- `--dataset` should be:
  - `prime` for `BioHybridQA`
  - `amazon` for `ProdHybridQA`
- `stats.txt` is required by `main.py` for dataset statistics.
- Use `--prefix /path/to/log_root` if you want to control log/checkpoint locations.


