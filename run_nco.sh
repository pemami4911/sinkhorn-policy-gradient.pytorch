#!/bin/bash
INPUT_SIZE=10
TASK="mwm2D_$INPUT_SIZE"
N_FEATURES=2
USE_DECODER='False'
EMBEDDING_DIM=128
HIDDEN_DIM=128
PARALLEL_ENVS=128
BATCH_SIZE=128
ACTOR_NET_LR=1e-4
CRITIC_NET_LR=1e-3
ACTOR_LR_DECAY_RATE=0.96
ACTOR_LR_DECAY_STEP=5000
CRITIC_LR_DECAY_RATE=0.96
CRITIC_LR_DECAY_STEP=5000
N_GLIMPSES=1
N_EPOCHS=30
EPOCH_START=0
MAX_GRAD_NORM=1.0
RANDOM_SEED=$1
TRAIN_SIZE=500000
VAL_SIZE=1000
USE_CUDA='True'
DISABLE_TENSORBOARD='False'
USE_TANH=True
CRITIC_BETA=0.8
SAVE_STATS='False'
SAVE_MODEL='False'
CUDA_DEVICE=0
BASE_DIR='.'

if [ $USE_DECODER = 'True'] 
then
    MODEL='PN-AC'
else
    MODEL='AC'
fi

RUN_NAME="$MODEL-$TASK-seed-$RANDOM_SEED$2"

./train_nco.py --task $TASK --train_size $TRAIN_SIZE  --actor_net_lr $ACTOR_NET_LR --critic_net_lr $CRITIC_NET_LR --n_epochs $N_EPOCHS --random_seed $RANDOM_SEED --max_grad_norm $MAX_GRAD_NORM --run_name $RUN_NAME  --epoch_start $EPOCH_START --train_size $TRAIN_SIZE --n_process_blocks $N_PROCESS_BLOCKS --batch_size $BATCH_SIZE --parallel_envs $PARALLEL_ENVS --actor_lr_decay_rate $ACTOR_LR_DECAY_RATE --actor_lr_decay_step $ACTOR_LR_DECAY_STEP --critic_lr_decay_rate $CRITIC_LR_DECAY_RATE --critic_lr_decay_step $CRITIC_LR_DECAY_STEP --embedding_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM --val_size $VAL_SIZE --n_glimpses $N_GLIMPSES --use_cuda $USE_CUDA --disable_tensorboard $DISABLE_TENSORBOARD --reward_scale $REWARD_SCALE --use_tanh $USE_TANH --critic_beta $CRITIC_BETA --input_size $INPUT_SIZE --save_stats $SAVE_STATS --save_model $SAVE_MODEL --n_features $N_FEATURES --use_KT $USE_KT --cuda_device $CUDA_DEVICE --use_decoder $USE_DECODER --base_dir $BASE_DIR
