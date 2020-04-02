MODEL_NAME='GG_RNN' # DNN_naive:0/5, DNN :1, DG_RNN:2, GG_DNN:3/6, GG_RNN:4/7
PREPROCESS=0
N_VALID=100
N_TEST=100

EPOCH=10
PRINT_ITER=50
VALID_ITER=250
N_VALID_PRINT=3
LR_DECAY=3
PATIENCE=8

RANK_METHOD=['sfcid','src_max','dst_max'] 
LR=1e-4
OPT='RMSprop'
STATE_DIM=128
EMB_VNF_DIM=32
EMB_NODE_DIM=4
DNN_HIDDEN=128

MAX_N_NODES=50
MAX_GEN_LEN=30

CUDA_VISIBLE_DEVICES=$1 python3 train.py\
    --model_name=$MODEL_NAME --preprocess=$PREPROCESS --n_valid=$N_VALID --n_test=$N_TEST\
    --epoch=$EPOCH --print_iter=$PRINT_ITER --valid_iter=$VALID_ITER\
    --n_valid_print=$N_VALID_PRINT --rank_method=$RANK_METHOD --learning_rate=$LR\
    --opt=$OPT --state_dim=$STATE_DIM --emb_vnf_dim=$EMB_VNF_DIM --emb_node_dim=$EMB_NODE_DIM\
    --max_n_nodes=$MAX_N_NODES --max_gen_len=$MAX_GEN_LEN --lr_decay=$LR_DECAY\
    --patience=$PATIENCE --dnn_hidden=$DNN_HIDDEN
