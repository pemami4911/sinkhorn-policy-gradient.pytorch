#!/bin/bash
N_NODES=25
MAX_N_NODES=50
N_FEATURES=2
COP="mwm2D_$N_NODES"
ACTOR_WORKERS=4
ARCH='matching'
RANDOM_SEED=$1
RUN_NUM=$2
RUN_NAME="$COP-$N_NODES-$RANDOM_SEED$RUN_NUM"
TEST_SIZE=1000
PARALLEL_ENVS=128
N_SAMPLES=10
SINKHORN_TAU=0.05
SINKHORN_ITERS=10
TAU_DECAY=0.9
ANNEALING_ITERS=1
EMBEDDING_DIM=128
RNN_DIM=128
USE_CUDA='True'
CUDA_DEVICE=3
EPOCH_START=0
SAVE_STATS='False'
SAVE_MODEL='False'
BASE_DIR='/data/pemami/spg'
MAKE_ONLY=3
ID=$RANDOM_SEED
DISABLE_TENSORBOARD='True'
ACTOR_LOAD_PATH='trained_models/mwm2D_25/actor-epoch-39.pt'

cd ..
python eval_spg.py --task $COP --arch $ARCH --test_size $TEST_SIZE --n_nodes $N_NODES --n_features $N_FEATURES --random_seed $RANDOM_SEED --run_name $RUN_NAME --disable_tensorboard $DISABLE_TENSORBOARD --_id $ID --sinkhorn_iters $SINKHORN_ITERS --sinkhorn_tau $SINKHORN_TAU --save_stats $SAVE_STATS --embedding_dim $EMBEDDING_DIM --rnn_dim $RNN_DIM --use_cuda $USE_CUDA --save_model $SAVE_MODEL --parallel_envs $PARALLEL_ENVS  --cuda_device $CUDA_DEVICE --base_dir $BASE_DIR --actor_workers $ACTOR_WORKERS --make_only $MAKE_ONLY --max_n_nodes $MAX_N_NODES --annealing_iters $ANNEALING_ITERS --tau_decay $TAU_DECAY --n_samples $N_SAMPLES --load_actor $ACTOR_LOAD_PATH
