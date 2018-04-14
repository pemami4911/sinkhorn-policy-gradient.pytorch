#!/bin/bash

N_FEATURES=2
N_NODES=10
COP="mwm2D_$N_NODES"
ARCH='pnac'
TRAIN_SIZE=500000
VAL_SIZE=1000
TEST_SIZE=1000
EMBEDDING_DIM=128
LSTM_DIM=128
RANDOM_SEED=$1
DISABLE_TENSORBOARD='False'
RUN_NUM=$2
RUN_NAME="spg-sl-$RANDOM_SEED$2"
ACTOR_LR=1e-5
BATCH_SIZE=128
N_EPOCHS=10
SINKHORN_ITERS=10
SINKHORN_TAU=0.05
USE_CUDA='True'
CUDA_DEVICE=2
USE_VAL='False'
SAVE_STATS='False'
MAKE_ONLY=3


./train_sl.py --task $COP --n_epochs $N_EPOCHS --train_size $TRAIN_SIZE --val_size $VAL_SIZE --test_size $TEST_SIZE --n_features $N_FEATURES --n_nodes $N_NODES --random_seed $RANDOM_SEED --run_name $RUN_NAME$RUN_NUM --disable_tensorboard $DISABLE_TENSORBOARD --actor_lr $ACTOR_LR --batch_size $BATCH_SIZE --sinkhorn_tau $SINKHORN_TAU --sinkhorn_iters $SINKHORN_ITERS --arch $ARCH --lstm_dim $LSTM_DIM --embedding_dim $EMBEDDING_DIM --use_cuda $USE_CUDA --use_val $USE_VAL --save_stats $SAVE_STATS --make_only $MAKE_ONLY --cuda_device $CUDA_DEVICE
