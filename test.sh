MODEL_NAME='GG_RNN' # DNN_naive:0/5, DNN :1, DG_RNN:2, GG_DNN:3/6, GG_RNN:4/7
USE='test'
DATASET_PATH='../data/testset.csv'
MODEL_PATH='./result/GG_RNN_best.pth'
SAVE_LOG_PATH='./result/GG_RNN_test.txt'

N_VALID_PRINT=100
RANK_METHOD=['sfcid','src_max','dst_max'] 

CUDA_VISIBLE_DEVICES=$1 python3 test.py\
    --model_name=$MODEL_NAME --dataset_path=$DATASET_PATH --model_path=$MODEL_PATH\
    --save_log_path=$SAVE_LOG_PATH --n_valid_print=$N_VALID_PRINT --use=$USE
