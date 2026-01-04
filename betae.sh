## ProdHyperHybrid
CUDA_VISIBLE_DEVICES=3 python main.py --cuda --do_train --do_test --graph_type 'hyperedge' \
    --dataset='amazon' -n 128 -b 512 -d 800 -g 50 --tasks '1p.2i.3i.1n.2in.3in.2u.3u' \
    -lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta -betam "(1600,2)" --prefix '/workspace/HybridQA/QELM_cross-attention_ov/log/' --graph_added \

## BioTripleHybrid
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --do_train --do_test --graph_type 'triple' \
    --dataset='prime' -n 128 -b 512 -d 800 -g 50 --tasks '1p.2p.3p.2i.3i.1n.2in.3in.2u.3u' \
    -lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta -betam "(1600,2)" --prefix /workspace/HybridQA/QELM_cross-attention_ov/log --graph_added 