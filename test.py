import topology_settings as tp
import index_dict as idx
from dataset import sfc_dataset
from model import enc_dec_model, DNN_naive
from main import train_main, test_main

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
parser.add_argument("--model_path", type=str, default='')
parser.add_argument("--save_dir", type=str, default='./result/')
parser.add_argument("--dataset_path", type=str, default='')
parser.add_argument("--save_log_path", type=str, default='')

parser.add_argument("--n_valid_print", type=int, default=3)
parser.add_argument("--rank_method", type=str, default=['sfcid','src_max','dst_max'])
parser.add_argument("--use", type=str, default='test')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")

print("-----Loading Training Dataset-----")
placement = load_placement(args.raw_placement_dataset_path)

dataset_path = args.dataset_path
dataset = pd.read_csv(dataset_path, index_col=0)
dataset = np.array(dataset)

data_loader = sfc_dataset(args, dataset, placement)
print("# of dataset  : {}".format(data_loader.len()))


print("-----Building the Model-----")
test_model = torch.load(args.model_path)
test_model.to(args.device)

test_main(args, test_model, data_loader, use=args.use)
