import argparse
import numpy as np
import pandas as pd

from topology import Topology
from preproc_label import load_raw_dataset
import index_dict as idx

print("-----Setting the hyperparameters for training------")
parser = argparse.ArgumentParser()
# About topology setting and raw_datasets
parser.add_argument("--topology_name", type=str, default='')
parser.add_argument("--data_dir", type=str, default='')
parser.add_argument("--topo_file", type=str, default='')
parser.add_argument("--middlebox_file", type=str, default='')
parser.add_argument("--sfctypes_file", type=str, default='')
parser.add_argument("--request_file", type=str, default='')
parser.add_argument("--nodeinfo_file", type=str, default='')
parser.add_argument("--label_file", type=str, default='')

# About saving in training
parser.add_argument("--running_mode", type=str, default='')
parser.add_argument("--model_name", type=str, default='')
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--save_model_name", type=str, default='')
parser.add_argument("--n_valid", type=int, default=20)
parser.add_argument("--print_iter", type=int, default=0)
parser.add_argument("--valid_iter", type=int, default=0)
parser.add_argument("--n_valid_print", type=int, default=3)

# About training strategy
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--lr_decay", type=int, default=3)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--opt", type=str, default='RMSprop')

# About model specificatoin
parser.add_argument("--GRU_step", type=int, default=5)
parser.add_argument("--state_dim", type=int, default=16)
parser.add_argument("--emb_vnf_dim", type=int, default=4)
parser.add_argument("--emb_node_dim", type=int, default=1)
parser.add_argument("--max_n_nodes", type=int, default=50)
parser.add_argument("--max_gen_len", type=int, default=30)

parser.add_argument("--dnn_hidden", type=int, default=80)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if args.running_mode == 'train':
    topo_path = args.data_dir + args.topo_file
    middlebox_path = args.data_dir + args.middlebox_file
    sfctypes_path = args.data_dir + args.sfctypes_file
    request_path = args.data_dir + args.request_file
    nodeinfo_path = args.data_dir + args.nodeinfo_file
    label_path = args.data_dir + args.labe_file

    train_topo = Topology(args.topology_name)
    train_topo.Initialize_topology(topo_path, middlebox_path, sfctypes_path)

    
