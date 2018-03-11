#!/bin/bash
MODEL='ddpg'
N_NODES=5
COP="mwm2D_$N_NODES"
ARCH='siamese'
RANDOM_SEED=$1
ALPHA=1.0
RUN_NUM=$2
RUN_NAME="mwm2d-$RANDOM_SEED$RUN_NUM-alpha-$ALPHA-rmsprop-wd-BS-2000"
TRAIN_SIZE=100000
VAL_SIZE=1000
USE_BATCHNORM='False'
PARALLEL_ENVS=128
BATCH_SIZE=2000
N_FEATURES=2
HIDDEN_DIM=300
ACTOR_LR=1e-4 # 3e-5
CRITIC_LR=1e-3 # 7e-4
ACTOR_LR_DECAY_RATE=0.9
CRITIC_LR_DECAY_RATE=0.9
ACTOR_LR_DECAY_STEP=10000
CRITIC_LR_DECAY_STEP=10000
CRITIC_WEIGHT_DECAY=0
N_EPOCHS=75
EPSILON=1.0
EPSILON_DECAY_RATE=0.97
EPSILON_DECAY_STEP=200000
POISSON_LAMBDA=2
POISSON_DECAY_RATE=0.97
POISSON_DECAY_STEP=200000
BUFFER_SIZE=1000000
SINKHORN_TAU=1
SINKHORN_ITERS=5
SIGOPT='False'
ID=$RANDOM_SEED
DISABLE_TENSORBOARD='False'
EMBEDDING_DIM=128
LSTM_HIDDEN=128
USE_CUDA='True'
EPOCH_START=0
SAVE_STATS='False'
SAVE_MODEL='False'
USE_LAYER_NORM='False'
#ACTOR_LOAD_PATH='results/models/mwm2D/spg/siamese/120/actor-epoch-6.pt'
#CRITIC_LOAD_PATH='results/models/mwm2D/spg/siamese/91212/critic-epoch-100.pt'

python train_spg.py --task $COP --arch $ARCH --train_size $TRAIN_SIZE --val_size $VAL_SIZE --batch_size $BATCH_SIZE --n_nodes $N_NODES --n_features $N_FEATURES --hidden_dim $HIDDEN_DIM --random_seed $RANDOM_SEED --run_name $RUN_NAME --disable_tensorboard $DISABLE_TENSORBOARD --actor_lr $ACTOR_LR --critic_lr $CRITIC_LR --n_epochs $N_EPOCHS --poisson_decay_rate $POISSON_DECAY_RATE --poisson_decay_step $POISSON_DECAY_STEP --buffer_size $BUFFER_SIZE --epsilon $EPSILON --epsilon_decay_rate $EPSILON_DECAY_RATE --epsilon_decay_step $EPSILON_DECAY_STEP --sigopt $SIGOPT --_id $ID --sinkhorn_iters $SINKHORN_ITERS --sinkhorn_tau $SINKHORN_TAU --save_stats $SAVE_STATS --embedding_dim $EMBEDDING_DIM --lstm_dim $LSTM_HIDDEN --actor_lr_decay_rate $ACTOR_LR_DECAY_RATE --actor_lr_decay_step $ACTOR_LR_DECAY_STEP --critic_lr_decay_rate $CRITIC_LR_DECAY_RATE --critic_lr_decay_step $CRITIC_LR_DECAY_STEP --poisson_lambda $POISSON_LAMBDA --use_cuda $USE_CUDA --save_model $SAVE_MODEL --parallel_envs $PARALLEL_ENVS --alpha $ALPHA --critic_weight_decay $CRITIC_WEIGHT_DECAY --use_layer_norm $USE_LAYER_NORM

#rss=('44' '22' '33' '89' '55' '66' '77' '88' '99' '1010' '1111' '1212' '1313' '1414' '1515' '1616' '1717' '1818' '1919' '2020')
#for rs in "${rss[@]}"
#do
#    RUN_NAME="mwm2D-RS-$rs-ALPHA-$ALPHA-BS-$BATCH_SIZE-reversed-perm-actor-lr-$ACTOR_LR-my-hungarian"
#    python train_spg.py --task $COP --arch $ARCH --train_size $TRAIN_SIZE --val_size $VAL_SIZE --batch_size $BATCH_SIZE --n_nodes $N_NODES --n_features $N_FEATURES --hidden_dim $HIDDEN_DIM --random_seed $rs --run_name $RUN_NAME --disable_tensorboard $DISABLE_TENSORBOARD --actor_lr $ACTOR_LR --critic_lr $CRITIC_LR --n_epochs $N_EPOCHS --poisson_decay_rate $POISSON_DECAY_RATE --poisson_decay_step $POISSON_DECAY_STEP --buffer_size $BUFFER_SIZE --epsilon $EPSILON --epsilon_decay_rate $EPSILON_DECAY_RATE --epsilon_decay_step $EPSILON_DECAY_STEP --sigopt $SIGOPT --_id $ID --sinkhorn_iters $SINKHORN_ITERS --sinkhorn_tau $SINKHORN_TAU --save_stats $SAVE_STATS --embedding_dim $EMBEDDING_DIM --lstm_dim $LSTM_HIDDEN --actor_lr_decay_rate $ACTOR_LR_DECAY_RATE --actor_lr_decay_step $ACTOR_LR_DECAY_STEP --critic_lr_decay_rate $CRITIC_LR_DECAY_RATE --critic_lr_decay_step $CRITIC_LR_DECAY_STEP --poisson_lambda $POISSON_LAMBDA --use_cuda $USE_CUDA --save_model $SAVE_MODEL --parallel_envs $PARALLEL_ENVS --alpha $ALPHA --critic_weight_decay $CRITIC_WEIGHT_DECAY --use_layer_norm $USE_LAYER_NORM
#done 

#alphas=('0.1' '0.5' '0.9' '1.')
#sinkhorntaus=('1' '0.67' '0.5' '0.33')
#actorlrs=('3e-5' '1e-4')
#for i in "${alphas[@]}"
#do 
#    for j in "${layernorms[@]}"
#    do 
#        for k in "${sinkhorntaus[@]}"
#        do 
#            for l in "${actorlrs[@]}"
#            do
#                runname="mwm2D-RS-$RANDOM_SEED-ALPHA-$i-LAYERNORM-$j-SINKHORN-TAU-$k-ACTOR_LR-$l"
#                python train_spg.py --task $COP --arch $ARCH --train_size $TRAIN_SIZE --val_size $VAL_SIZE --batch_size $BATCH_SIZE --n_nodes $N_NODES --n_features $N_FEATURES --hidden_dim $HIDDEN_DIM --random_seed $RANDOM_SEED --run_name $runname --disable_tensorboard $DISABLE_TENSORBOARD --actor_lr $l --critic_lr $CRITIC_LR --n_epochs $N_EPOCHS --poisson_decay_rate $POISSON_DECAY_RATE --poisson_decay_step $POISSON_DECAY_STEP --buffer_size $BUFFER_SIZE --epsilon $EPSILON --epsilon_decay_rate $EPSILON_DECAY_RATE --epsilon_decay_step $EPSILON_DECAY_STEP --sigopt $SIGOPT --_id $ID --sinkhorn_iters $SINKHORN_ITERS --sinkhorn_tau $k --save_stats $SAVE_STATS --embedding_dim $EMBEDDING_DIM --lstm_dim $LSTM_HIDDEN --actor_lr_decay_rate $ACTOR_LR_DECAY_RATE --actor_lr_decay_step $ACTOR_LR_DECAY_STEP --critic_lr_decay_rate $CRITIC_LR_DECAY_RATE --critic_lr_decay_step $CRITIC_LR_DECAY_STEP --poisson_lambda $POISSON_LAMBDA --use_cuda $USE_CUDA --save_model $SAVE_MODEL --parallel_envs $PARALLEL_ENVS --alpha $i --critic_weight_decay $CRITIC_WEIGHT_DECAY --use_layer_norm $j
#            done
#        done
#    done
#done
