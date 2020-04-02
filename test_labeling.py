from topology import Topology
import index_dict as idx
import topology_settings as tp
from shortest_path import dijsktra

import argparse
import numpy as np
import pandas as pd
import time
import os

def load_raw_datasets():
    raw_requests_dataset_path = '../data/20190530-requests.csv'
    raw_placement_dataset_path = '../data/20190530-nodeinfo.csv'

    raw_requests_dataset = np.array(pd.read_csv(raw_requests_dataset_path, skiprows=0))
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

    return raw_requests_dataset, placement


def find_optimal_path(top, req_id):
    def make_possible_seq_list(vnfs, src, dst, sfc, traffic):
        n_sfc = len(sfc)
        n_possible = 1
        n_vnf_inst = np.zeros([tp.n_vnfs], dtype=np.int)
        for vnf_type in sfc:
            n_tmp = 0
            vnf_item = vnfs[vnf_type]
            for vnf_inst_id in vnf_item:
                capacity = vnf_item[vnf_inst_id][idx.VNF_CAPACITY]
                if capacity >= traffic:
                    n_tmp += 1
            n_vnf_inst[vnf_type] = n_tmp
            n_possible = n_possible * n_tmp
        if n_possible == 0:
            return np.zeros([1]), False
       
        possible_seq_list = np.zeros([n_possible, n_sfc+2], dtype=np.int)
        possible_seq_list[:,0] = src
        possible_seq_list[:,-1] = dst

        prev_jump_unit = n_possible
        repeat = 1
        sfc_idx = 1
        for vnf_type in sfc:
            jump_unit = int(prev_jump_unit / n_vnf_inst[vnf_type])
            vnf_item = vnfs[vnf_type]
            n_jump = 0
            for r in range(repeat):
                for vnf_inst_id in vnf_item.keys():
                    node_id = vnf_item[vnf_inst_id][idx.VNF_NODEID]
                    capacity = vnf_item[vnf_inst_id][idx.VNF_CAPACITY]
                    if capacity >= traffic:
                        possible_seq_list[jump_unit*n_jump:jump_unit*(n_jump+1),sfc_idx]\
                                                                            = node_id
                        n_jump += 1
            sfc_idx += 1
            repeat = repeat * n_vnf_inst[vnf_type]
            prev_jump_unit = jump_unit

        return possible_seq_list, True

    def compute_seq_cost(top, seq, req_id):
        tmp_path = []
        for i in range(len(seq)-1):
            from_node = seq[i]
            to_node = seq[i+1]
            tmp_path += dijsktra(top, from_node, to_node)[:-1]
        tmp_path += [to_node]
        cost = top.compute_cost(tmp_path, req_id)
        return tmp_path, cost

    src = top.reqs[req_id][idx.REQ_SRC]
    dst = top.reqs[req_id][idx.REQ_DST]
    traffic = top.reqs[req_id][idx.REQ_TRAFFIC]
    sfcid = top.reqs[req_id][idx.REQ_SFCID]
    sfc = tp.sfc_type[sfcid]
    n_sfc = len(sfc)
    possible_seq_list, possible_flag = make_possible_seq_list(top.vnfs, src, dst, sfc, traffic)
    n_possible = len(possible_seq_list)
    if possible_flag == False:
        return [0], 0, False

    best_cost = 99999
    for seq in possible_seq_list:
        tmp_path, cost = compute_seq_cost(top, seq, req_id)
        if cost < best_cost:
            best_path = tmp_path
            best_cost = cost

    return best_path, best_cost, True

        

def generate_label_data(top, req_id, path, cost):
    def make_string(sample):
        result_str = ''
        for tok in sample:
            if len(result_str) == 0:
                result_str += str(tok)
            else:
                result_str += ',' + str(tok)
        return result_str

    arrivaltime = top.reqs[req_id][idx.REQ_ARRIVALTIME]
    duration = top.reqs[req_id][idx.REQ_ENDTIME] - arrivaltime
    src = top.reqs[req_id][idx.REQ_SRC]
    dst = top.reqs[req_id][idx.REQ_DST]
    traffic = top.reqs[req_id][idx.REQ_TRAFFIC]
    maxlat = top.reqs[req_id][idx.REQ_MAXLAT]
    sfcid = top.reqs[req_id][idx.REQ_SFCID]

    sample = [arrivaltime, duration, src, dst, traffic, maxlat, 0, 0, 0, 0, 0, sfcid]
    sample_str = make_string(sample)
    return sample_str

def generate_label_packet(packet, arrivaltime):
    result_str = str(arrivaltime)
    for sample in packet:
        result_str += '/' + sample
    return result_str


def labeling(save, rank_method, f):
    generated_labels = [0] * 9999999
    n_success = 0
    n_fail = 0
    n_overmax = 0
    n_packet = 0
    raw_requests_dataset, placement = load_raw_datasets()

    label_top = Topology()
    label_top.Initialize_topology()
    start = True
    start_time = time.time()
    stacked_avg_time = 0
    total_time = 0

    n_try = 0
    for raw_request_line in raw_requests_dataset:
        n_try += 1
        gen_time_start = time.time()
        arrivaltime = raw_request_line[idx.REQ_RAW_ARRIVALTIME]
        fail_packet_flag = False
        packet_sample = []
        if start == True:
            start = False
            label_top.set_topology(raw_request_line)
        else:
            label_top.set_topology(raw_request_line, arrivaltime, placement)
            
            label_top.ranking(rank_method)
            
            n_req = len(label_top.reqs.keys())

            sorted_reqs = np.zeros([n_req], dtype=np.int)
            for req_id in label_top.rank.keys():
                rank_idx = label_top.rank[req_id]
                sorted_reqs[rank_idx] = req_id

            for rank_idx in range(n_req):
                req_id = sorted_reqs[rank_idx]
                path, cost, flag = find_optimal_path(label_top, req_id)
                store_sample = generate_label_data(label_top, req_id, 0, 0)
                packet_sample.append(store_sample)
                if flag == False:
                    n_fail += 1
                    fail_packet_flag = True

                    #print("Cannot make path for {} at arrivaltime {}".format(\
                    #                                        req_id, arrivaltime))
                else:
                    if cost > label_top.reqs[req_id][idx.REQ_MAXLAT]:
                        n_overmax += 1
                        fail_packet_flag = True
                    else:
                        n_success += 1
                        label_top.update_topology(path, req_id)

        current_time = time.time()
        total_time = current_time - start_time

        if fail_packet_flag == True:
            generated_labels[n_packet] = generate_label_packet(packet_sample, arrivaltime)
            n_packet += 1
            stacked_avg_time += (time.time() - gen_time_start)
            avg_time = stacked_avg_time / n_packet

        if n_try % 50 == 0:
            print("{} rank_method {} success {} overmax {} fail, Avg Time {:.3f}, Total Time : {:.3f}".format(rank_method, n_success, n_overmax, n_fail, avg_time, total_time))


    print("------------Labeling Done---------------")
    print("Method : {}".format(rank_method))
    print("Success : {}, OverMax : {}, Fail : {}".format(\
                                        n_success, n_overmax, n_fail))
    print("Avg Time : {}, Total Time : {}".format(avg_time, total_time))

   

    if save == True:
        method_str = ''
        for method_item in rank_method:
            method_str += method_item

        f.write("Test-{}\t{}\t{}\t{}\n".format(method_str,n_packet,n_overmax,n_fail))
        f.close()
        target_dir = '../data/test_dataset.csv'
        if os.path.exists(target_dir):
            os.remove(target_dir)
            print("Remove exist target file")
        df_label = pd.DataFrame(generated_labels[:n_packet-1])
        df_label.to_csv(target_dir)
        print("Generating Labels Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_method", type=str, default=[])
    args = parser.parse_args()

    log_file_path = '../data/labeling_log.txt'
    #if os.exist.path(log_file_path):
    #    os.remove(log_file_path)

    f = open(log_file_path, 'a')

    save = True
    rank_method = args.rank_method
    labeling(save, rank_method, f)


