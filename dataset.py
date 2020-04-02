from topology import Topology
import index_dict as idx

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class sfc_dataset():
    def __init__(self, args, dataset, nodeinfo):
        self.args = args
        self.dataset = dataset
        self.nodeinfo = nodeinfo

    def getitem(self, index, topo):
        package = self.datset[index,0]
        package = package.split('/')
        arrivaltime = package[0]
        package = package[1:]

        n_req = len(package)
        
        for reqline in package:
            tmp_reqline = reqline.split(',')
            reqline = np.zeros([len(tmp_reqline)], dtype=np.int)
            for i, tok in enumerate(tmp_reqline):
                reqline[i] = int(tok)
            request_line = reqline[:idx.EOI]
            label_line = reqline[idx.EOI:]
            
            self.top.set_topology(request_line, arrivaltime=None, placement=None,\
                                 raw_req=False, label=label_line)

        topo.set_topology(reqline=None, arrivaltime=arrivaltime,\
                                placement=self.nodeinfo, raw_req=False, label=None)
        return topo

'''
class sfc_dataset():
    def __init__(self, args, dataset, placement):
        self.args = args
        self.dataset = dataset
        self.placement = placement
        #self.top = Topology(args.topology_name)

    def getitem(self, index):
        self.top.Initialize_topology()

        packet = self.dataset[index,0]
        packet = packet.split('/')
        arrivaltime = packet[0]
        packet = packet[1:]

        n_req = len(packet)

        max_arrivaltime = 0
        for reqline in packet:
            tmp_reqline = reqline.split(',')
            reqline = np.zeros([len(tmp_reqline)], dtype=np.int)
            for i, tok in enumerate(tmp_reqline):
                reqline[i] = int(tok)
            request_line = reqline[:idx.EOI]
            arrivaltime = request_line[idx.REQ_RAW_ARRIVALTIME]
            if arrivaltime > max_arrivaltime:
                max_arrivaltime = arrivaltime
            
            #label_cost = reqline[idx.EOI]
            if len(reqline) == idx.EOI:
                label = None
            else:
                label = reqline[idx.EOI:]
            
            self.top.set_topology(request_line, arrivaltime=None, placement=None, label=label)

        self.top.set_topology(reqline=None, arrivaltime=max_arrivaltime,\
                                placement=self.placement, label=None)

        self.top.ranking(self.args.rank_method)

        sorted_reqs = np.zeros([n_req], dtype=np.int)
        for req_id in self.top.rank.keys():
            rank_idx = self.top.rank[req_id]
            sorted_reqs[rank_idx] = req_id
        
        return self.top, sorted_reqs

    def len(self):
        return len(self.dataset)


def sfc_dataloader(args, dataset, placement):
    dataset = sfc_dataset(args, dataset, placement)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader
'''
