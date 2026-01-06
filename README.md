# HybridQE

This temporary repository is created for the implementation of the paper Hybrid Query Answering over Incomplete Text-Labeled Graph Databases.

## 1. Datasets Preparation

We introduce two benchmarks for the task of hybrid query answering over incomplete semi-structured database, ProdHybridQA and BioHybridQA, which are derived from real-world e-commerce and protein-disease data. The hybrid queries can be downloaded via this [link](https://drive.google.com/drive/folders/1EZKTx2YuEnEPXsxm5bMEqvGH8MMKOmjw?usp=drive_link).

## 2. Environment Requirement
- Python 3.7
- PyTorch 1.7
- tqdm

## 3. Reproduce the Results of HybridQE
Here we display the example commands for reproducing the results of HybridQE(BetaE) on BioHybridQA and ProdHybridQA datasets
```
## BioHybridQA
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --do_train --do_test --graph_type 'triple' \
    --dataset='prime' -n 128 -b 512 -d 800 -g 50 --tasks '1p.2p.3p.2i.3i.1n.2in.3in.2u.3u' \
    -lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta -betam "(1600,2)" --graph_added --data_directory "./data"
```
```
## ProdHyperHybrid
CUDA_VISIBLE_DEVICES=3 python main.py --cuda --do_train --do_test --graph_type 'hyperedge' \
    --dataset='amazon' -n 128 -b 512 -d 800 -g 50 --tasks '1p.2i.3i.1n.2in.3in.2u.3u' \
    -lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta -betam "(1600,2)" --graph_added \ --data_directory "./data"
```


