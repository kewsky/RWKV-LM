#!/bin/bash

BASE_NAME="model/v6-1B5"
N_LAYER="24"
N_EMBD="2048"
M_BSZ="4" # takes 16G VRAM (reduce this to save VRAM)
LR_INIT="5e-5"
LR_FINAL="5e-5"
GRAD_CP=1 # set to 1 to save VRAM (will be slower)
EPOCH_SAVE=10
CTX_LEN=4096
N_TOKENS=71724510
MAGIC_PRIME=17489
TRAIN_DATA="trainset"

# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case)
# use https://www.dcode.fr/prime-numbers-search

python train.py --load_model "0" --wandb "RWKV-6-sft-mask" --proj_dir $BASE_NAME \
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file $TRAIN_DATA --my_exit_tokens $N_TOKENS --magic_prime $MAGIC_PRIME \
 --num_nodes 1 --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 100 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb 200 \
 --my_testing "x060" --my_qa_mask 1