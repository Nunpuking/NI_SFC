import topology_settings as tp
import index_dict as idx
from dataset import sfc_dataset
from model import enc_dec_model, DNN_naive
from main import train_main, test_main
from preprocessing import shuffle_dataset

import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.utils import shuffle


def load_placement(raw_placement_dataset_path):
    raw_placement_dataset = np.array(pd.read_csv(raw_placement_dataset_path, skiprows=0))

    placement = {}
    for nodeinfo in raw_placement_dataset:
        arrivaltime = int(nodeinfo[idx.PLACEMENT_RAW_ARRIVALTIME])
        if arrivaltime not in placement.keys():
            placement_id = 0
            placement[arrivaltime] = {}
        node_id = int(nodeinfo[idx.PLACEMENT_RAW_NODEID])-1

        vnf_type = tp.vnf_types[nodeinfo[idx.PLACEMENT_RAW_VNFTYPE]]
        n_inst = int(nodeinfo[idx.PLACEMENT_RAW_NINST])
        placement[arrivaltime][placement_id] = (node_id, vnf_type, n_inst)
        placement_id += 1

    return placement


print("-----Setting the hyperparameters for training------")
parser = argparse.ArgumentParser()
parser.add_argument("--raw_placement_dataset_path", type=str,\
                        default='../data/20190530-nodeinfo.csv')

parser.add_argument("--model_name", type=str, default='')
parser.add_argument("--save_dir", type=str, default='./result/')
parser.add_argument("--trainset_path", type=str, default='../data/trainset.csv')
parser.add_argument("--validset_path", type=str, default='../data/validset.csv')
parser.add_argument("--testset_path", type=str, default='../data/testset.csv')
parser.add_argument("--preprocess", type=int, default=0)
parser.add_argument("--save_model_path", type=str, default='')
parser.add_argument("--save_log_path", type=str, default='')
parser.add_argument("--n_valid", type=int, default=20)
parser.add_argument("--n_test", type=int, default=20)

parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--print_iter", type=int, default=0)
parser.add_argument("--valid_iter", type=int, default=0)
parser.add_argument("--n_valid_print", type=int, default=3)
parser.add_argument("--lr_decay", type=int, default=3)
parser.add_argument("--patience", type=int, default=15)

parser.add_argument("--rank_method", type=str, default=[])
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--opt", type=str, default='RMSprop')
parser.add_argument("--GRU_step", type=int, default=5)
parser.add_argument("--state_dim", type=int, default=16)
parser.add_argument("--emb_vnf_dim", type=int, default=4)
parser.add_argument("--emb_node_dim", type=int, default=1)
parser.add_argument("--max_n_nodes", type=int, default=50)
parser.add_argument("--max_gen_len", type=int, default=30)
parser.add_argument("--annotation_dim", type=int, default=7)

parser.add_argument("--dnn_hidden", type=int, default=80)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")
model_name = args.model_name+'.'+args.rank_method+'.'+args.opt+'.'+\
                str(args.learning_rate)+'.'+str(args.state_dim)+'.'+\
                str(args.emb_vnf_dim)+'.'+str(args.emb_node_dim)
args.save_log_path = args.save_dir + model_name + '.txt'
args.save_model_path = args.save_dir + model_name + '.pth'

print("-----Loading Training Dataset-----")
placement = load_placement(args.raw_placement_dataset_path)

if args.preprocess == 1:
    shuffle_dataset(args.n_valid, args.n_test)

trainset = np.array(pd.read_csv(args.trainset_path, index_col=0))
validset = np.array(pd.read_csv(args.validset_path, index_col=0))
testset = np.array(pd.read_csv(args.testset_path, index_col=0))

train_loader = sfc_dataset(args, trainset, placement)
valid_loader = sfc_dataset(args, validset, placement)
test_loader = sfc_dataset(args, testset, placement)
print("# of trainset : {}".format(train_loader.len()))
print("# of validset : {}".format(valid_loader.len()))
print("# of testset  : {}".format(test_loader.len()))


print("-----Building the Model-----")
if args.model_name == 'GG_RNN' or args.model_name == 'DG_RNN'\
    or args.model_name == 'GG_DNN' or args.model_name == 'DNN':
    model = enc_dec_model(args)
    test_model = enc_dec_model(args)
elif args.model_name == 'DNN_naive':
    model = DNN_naive(args, args.dnn_hidden)
    test_model = DNN_naive(args, args.dnn_hidden)
model.to(args.device)

print("-----Setting the Optimizer-----")
if args.opt == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.opt == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
elif args.opt == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
elif args.opt == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

train_log = train_main(args, train_loader, valid_loader, model, optimizer)

print("Loading Best model for testing")
best_model_path = args.save_model_path + '.best.pth'
tmp_model = torch.load(best_model_path)
state_dict = tmp_model.state_dict()
test_model.load_state_dict(state_dict)
test_model.to(args.device)

use = 'test'
test_main(args, test_model, test_loader, use, train_log)
