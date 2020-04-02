from topology import Topology
from shortest_path import dijsktra
import index_dict as idx
import argparse
import numpy as np
import pandas as pd
import os

def make_sample(topo, reqid, route, path):
    def make_string(sample):
        result_str = ''
        for tok in sample:
            if len(result_str) == 0:
                result_str += str(tok)
            else:
                result_str += ',' + str(tok)
        return result_str

    arrivaltime, endtime, src, dst, traffic, maxlat, sfcid = topo.reqs[reqid]
    sample = [arrivaltime, endtime, src, dst, traffic, maxlat, sfcid]
    sfcid = topo.reqs[reqid][idx.REQ_SFCID]
    sfc = topo.sfc[sfcid]
    for vnftype in sfc:
        sample.append(route[vnftype])
    sample.append(idx.BOP)
    for nodeid in path:
        sample.append(nodeid)

    sample_str = make_string(sample)
    return sample_str


def load_raw_dataset(path, mode):
    dataset_dict = {}
    raw_dataset = np.array(pd.read_csv(path, skiprows=0))

    for line in raw_dataset:
        arrivaltime = int(line[idx.PLACEMENT_RAW_ARRIVALTIME])
        if arrivaltime not in dataset_dict.keys():
            item_id = 0
            dataset_dict[arrivaltime] = {}
        if mode == 'nodeinfo':
            node_id = int(line[idx.PLACEMENT_RAW_NODEID])-1
            vnf_type = line[idx.PLACEMENT_RAW_VNFTYPE]
            n_inst = int(line[idx.PLACEMENT_RAW_NINST])
            dataset_dict[arrivaltime][item_id] = (node_id, vnf_type, n_inst)
            item_id += 1
        elif mode == 'routeinfo':
            req_id = int(line[idx.ROUTE_RAW_REQID])
            if req_id not in dataset_dict[arrivaltime].keys():
                dataset_dict[arrivaltime][req_id] = {}
            vnf_type = line[idx.ROUTE_RAW_VNFTYPE]
            node_id = int(line[idx.ROUTE_RAW_NODEID])-1
            dataset_dict[arrivaltime][req_id][vnf_type] = node_id
        else:
            print("Wrong Mode : {}".format(mode))
            return 0

    print("Loading {} succeeded".format(path))
    return dataset_dict


parser = argparse.ArgumentParser()
parser.add_argument("--topology_name", type=str, default='')
parser.add_argument("--data_dir", type=str, default='')
parser.add_argument("--topo_file", type=str, default='')
parser.add_argument("--middlebox_file", type=str, default='')
parser.add_argument("--sfctypes_file", type=str, default='')
parser.add_argument("--request_file", type=str, default='')
parser.add_argument("--nodeinfo_file", type=str, default='')
parser.add_argument("--routeinfo_file", type=str, default='')

args = parser.parse_args()

topo_path = args.data_dir + args.topo_file
middlebox_path = args.data_dir + args.middlebox_file
sfctypes_path = args.data_dir + args.sfctypes_file
request_path = args.data_dir + args.request_file
nodeinfo_path = args.data_dir + args.nodeinfo_file
routeinfo_path = args.data_dir + args.routeinfo_file

labeling_topo = Topology(args.topology_name)
labeling_topo.Initialize_topology(topo_path, middlebox_path, sfctypes_path)

nodeinfo_dataset = load_raw_dataset(nodeinfo_path, mode='nodeinfo')
routeinfo_dataset = load_raw_dataset(routeinfo_path, mode='routeinfo')

request_dataset = np.array(pd.read_csv(request_path, skiprows=0))

first=True
dataset = []
for n_req, req_line in enumerate(request_dataset):
    arrivaltime = req_line[idx.REQ_RAW_ARRIVALTIME]
    if first:
        first=False
        labeling_topo.set_topology(reqline=req_line,\
                                    arrivaltime=arrivaltime,\
                                    placement=None,\
                                    raw_req=True)
    else:
        labeling_topo.set_topology(reqline=req_line,\
                                arrivaltime=arrivaltime,\
                                placement=nodeinfo_dataset,\
                                raw_req=True)


        #labeling_topo.print_topology(topology=False,edge=True,vnf=True,req=True)
        route = routeinfo_dataset[arrivaltime]
        package = []
        for req_id in labeling_topo.reqs.keys():
            request = labeling_topo.reqs[req_id]
            src_node = request[idx.REQ_SRC]
            dst_node = request[idx.REQ_DST]
            sfcid = request[idx.REQ_SFCID]

            service_chain = [src_node]
            for vnf_type in labeling_topo.sfc[sfcid]:
                service_chain.append(route[req_id][vnf_type])
            service_chain.append(dst_node)
            
            label_path = []
            for i in range(len(service_chain)-1):
                fromnode = service_chain[i]
                tonode = service_chain[i+1]
                label_path += dijsktra(labeling_topo,fromnode,tonode)[:-1]
            label_path += [tonode]

            label_path_flag, fail_case = labeling_topo.check_label_path_possibility(\
                                        req_id, route[req_id], label_path)
            if label_path_flag is False and fail_case != 3:
                #print("req_id : ", req_id)
                #print("request : ", request)
                #print("route : ", service_chain)
                #print(label_path)
                if fail_case == 1:
                    print("VNF capacities are not enough")
                elif fail_case == 2:
                    print("Edge capacities are not enough")
                else:
                    print("Total delay is over the maxlat")
                raise SyntaxError("Can't make path") 
            else:
                #print("Processed well")
                labeling_topo.update_topology(req_id, route[req_id], label_path)
                sample = make_sample(labeling_topo, req_id, route[req_id], label_path)
                package.append(sample)

        package_sample = str(arrivaltime)
        for sample in package:
            package_sample += '/' + sample
        dataset.append(package_sample)                     
            #print("req_id : ", req_id)
            #print("request : ", request)
            #print("route : ", service_chain)
            #print(label_path)
            #labeling_topo.print_topology(topology=False,edge=True,vnf=True,req=True)
            
        # If check is false --> break
        # else --> save data sample : request, label
        '''
            if n_req == 30:
                labeling_topo.print_topology(topology=False,edge=True,vnf=True,req=True)
                print("req_id : ", req_id)
                print("request : ", request)
                print("service_chain : ", service_chain)
                print(label_path)
                print(label_path_flag)
        '''
            


    
    #labeling_topo.print_topology(topology=True,edge=False,vnf=True,req=True)
    #if n_req == 30:
    #    break
    if n_req % 100 == 0:
        print("{} are processed".format(n_req))

dataset = np.squeeze(np.expand_dims(np.array(dataset), axis=1))
dataset_dir = '../data/20190530-label.csv'
if os.path.exists(dataset_dir):
    os.remove(dataset_dir)
df_label = pd.DataFrame(dataset)
df_label.to_csv(dataset_dir)

print("Done!")
