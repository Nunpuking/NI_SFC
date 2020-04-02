import index_dict as idx
from shortest_path import dijsktra

import numpy as np
import pandas as pd
import statistics as stat
import torch

class Topology():
    def __init__(self, name):
        self.topology_name = name

    def Initialize_topology(self, topo_path, middlebox_path, sfc_path):
        # (1) Load the topology file, middlebox file, and sfctype file
        # (2) Initialize the topology class based on these files
        # Input Arguments
        # - topo_path : path of topology setting file
        # - middlebox_path : path of middlebox setting file
        # - sfc_path : path of sfctypes setting file
        # Outputs
        # - self.reqs[REQID] : stored requests, dict - (init)
        # - self.vnfs[VNFTYPE] : current VNF placement setting, dict (init)
        # - self.labels[REQID] : labels of each stored requests, dict (init)
        # - self.reqid : ID of new received request, scalar (init)
        # - self.n_nodes = # of nodes on topology, scalar
        # - self.n_edges = # of edges on topology, scalar
        # - self.n_vnfs = # of vnftype, scalar
        # - self.edges[fromnode,tonode] : information of edge, tuple - (delay, capacity)
        # - self.edges_to[nodeid] : possible node to move, list - [nodeid, ...]
        # - self.vnfinfo[VNFTYPE] : information of VNFTYPE, tuple - (delay, unit capacity)
        # - self.sfc[SFCTYPE] : information of SFCTYPE, list - [VNFTYPE1, VNFTYPE2, ...]
        
        topo = pd.read_csv(topo_path, header=None).values.tolist()
        middlebox = pd.read_csv(middlebox_path, header=None).values.tolist()
        sfctypes = pd.read_csv(sfc_path, header=None).values.tolist()

        self.n_nodes, self.n_edges = np.array(topo[0][0].split(), dtype=np.int)

        self.reqs = {}
        self.vnfs = {}
        self.labels = {}
        self.reqid = 1

        self.edges = {}
        self.edges_to = {}
        self.edges_org = {}

        for i in range(self.n_edges):
            fromnode, tonode, capacity, delay =\
                    np.array(topo[i+1+self.n_nodes][0].split(), dtype=np.int)
            self.edges[(fromnode,tonode)] = (delay, capacity)
            self.edges_org[(fromnode,tonode)] = (delay, capacity)
            self.edges[(tonode,fromnode)] = (delay, capacity)
            self.edges_org[(tonode,fromnode)] = (delay, capacity)

            if fromnode not in self.edges_to.keys():
                self.edges_to[fromnode] = []
            if tonode not in self.edges_to.keys():
                self.edges_to[tonode] = []
            self.edges_to[fromnode].append(tonode)
            self.edges_to[tonode].append(fromnode)

        self.vnfinfo = {}
        for line in middlebox:
            vnftype, cpu, delay, capacity, _ = line
            cpu, delay, capacity = int(cpu), int(delay), int(capacity)
            self.vnfinfo[vnftype] = (delay, capacity)
        self.n_vnfs = len(self.vnfinfo.keys())

        self.sfc = {}
        for line in sfctypes:
            sfctype, sfc_len = int(line[idx.SFCTYPE_ID]), int(line[idx.SFC_LENGTH])
            chain = []
            for i in range(sfc_len):
                chain.append(line[idx.SFC_LENGTH+i+1])
            self.sfc[sfctype] = chain


    def set_topology(self, reqline=None, arrivaltime=None, placement=None, raw_req=False, label=None):
        # (1) Initialize the capacity status with placement strategy of given arrivaltime
        # (2) Update the active request list given new request
        # (3) Store the label of request
        # Input
        #   placement : { arrivaltime0 : { placement_id0 : (node_id, vnf_type, n_inst),
        #                                  placement_id1 : (node_id, vnf_type, n_inst),
        #                                  ...
        #                                }
        #                 arrivaltime1 : { ... }
        #                 ...
        #               }
        #
        #   reqline : (arrivaltime, duration, src, dst, traffic, maxlat, sfcid)
        #
        #   arrivaltime : value
        #
        #   label : [route1, route2, ... , BOP token, node0, node1, ...]
        #
        # Output
        #   self.vnfs : { vnf_type0 : { vnf_inst_id0 : (node_id, capacity),
        #                               vnf_inst_id1 : (node_id, capacity),
        #                               ...
        #                             }
        #                 vnf_type1 : { ...}
        #               }
        #
        #   self.reqs : { req_id n  : (arrivaltime, endtime, src, dst, traffic, maxlat, sfcid),
        #                 req_id n+1: ( ... ),
        #                 ...
        #               }
        #

        for edge_key in self.edges.keys():
            self.edges[edge_key] = self.edges_org[edge_key]

        if placement is not None:
            self.vnfs = {}
            vnf_inst_id = {}
            for vnftype in self.vnfinfo.keys():
                    self.vnfs[vnftype] = {}
                    vnf_inst_id[vnftype] = 0

            # Set the Placement Strategy given arrivaltime
            current_placement = placement[arrivaltime]
            for placement_id in current_placement.keys():
                placement_item = current_placement[placement_id]
                node_id = placement_item[idx.PLACEMENT_PRE_NODEID]
                vnf_type = placement_item[idx.PLACEMENT_PRE_VNFTYPE]
                n_inst = placement_item[idx.PLACEMENT_PRE_NINST]
                
                for instance in range(n_inst):
                    vnf_inst_id_tmp = vnf_inst_id[vnf_type]
                    capacity = self.vnfinfo[vnf_type][idx.VNFINFO_CAPACITY]
                    self.vnfs[vnf_type][vnf_inst_id_tmp] = (node_id, capacity)
                    vnf_inst_id[vnf_type] += 1
            
            new_vnfs = {}
            for vnftype in self.vnfs.keys():
                vnf_placement = self.vnfs[vnftype]
                new_vnfs[vnftype] = {}
                for vnf_inst_id in vnf_placement.keys():
                    vnf_item = vnf_placement[vnf_inst_id]
                    nodeid = vnf_item[idx.VNF_NODEID]
                    capacity = vnf_item[idx.VNF_CAPACITY]
                    if nodeid not in new_vnfs[vnftype].keys():
                        new_vnfs[vnftype][nodeid] = (nodeid, capacity)
                    else:
                        new_capacity = new_vnfs[vnftype][nodeid][idx.VNF_CAPACITY] + capacity
                        new_vnfs[vnftype][nodeid] = (nodeid, new_capacity)
            self.vnfs = new_vnfs
                    

        if reqline is not None:
            if raw_req is True:
                # Add the new request and delete the expired requests
                arrivaltime = reqline[idx.REQ_RAW_ARRIVALTIME]
                endtime = arrivaltime + reqline[idx.REQ_RAW_DURATION]
                src = reqline[idx.REQ_RAW_SRC]
                dst = reqline[idx.REQ_RAW_DST]
                traffic = reqline[idx.REQ_RAW_TRAFFIC]
                maxlat = reqline[idx.REQ_RAW_MAXLAT]
                sfcid = reqline[idx.REQ_RAW_SFCID]
            else:
                arrivaltime, endtime, src, dst, traffic, maxlat, sfcid = reqline

            self.reqs[self.reqid] = (arrivaltime, endtime, src, dst, traffic, maxlat, sfcid)
            self.reqid += 1
            
            if label is not None:
                self.label[req_id] = label

            delete_list = []
            for req_id in self.reqs.keys():
                endtime = self.reqs[req_id][idx.REQ_ENDTIME]
                if arrivaltime > endtime:
                    delete_list.append(req_id)
            for req_id in delete_list:
                del self.reqs[req_id]

    def check_label_path_possibility(self, req_id, route, path):
        # Check the possiblity of given label path
        # Input Argument
        # - req_id : ID of the request, scalar
        # - route : nodes set of the VNFs which are requested from the request, dict -
        #       { vnftype1 : nodeid, vnftype2 : nodeid, ... }
        # - path : nodes set of result of the request, list - [node1, node2, ... ]
        fail_case = 0
        possible_flag = True
        request = self.reqs[req_id]
        traffic = request[idx.REQ_TRAFFIC]
        maxlat = request[idx.REQ_MAXLAT]
        sfcid = request[idx.REQ_SFCID]
        for vnftype in self.sfc[sfcid]:
            vnf_possible_flag = False
            vnf_placement = self.vnfs[vnftype]
            route_node = route[vnftype]
            for vnf_inst_id in vnf_placement.keys():
                vnf_nodeid = vnf_placement[vnf_inst_id][idx.VNF_NODEID]
                if route_node == vnf_nodeid and\
                    traffic <= vnf_placement[vnf_inst_id][idx.VNF_CAPACITY]:
                    vnf_possible_flag = True
                    break

            if vnf_possible_flag is False:
                possible_flag = False
                fail_case = 1
                print("{} doesn't have enough capacity for node {}"\
                        .format(vnftype, route_node))
                break

        for i in range(len(path)-1):
            fromnode = path[i]
            tonode = path[i+1]
            if self.edges[(fromnode,tonode)][idx.EDGE_CAPACITY] < traffic:
                possible_flag = False
                fail_case = 2
                print("{} doesn't have enough capacity : {} are left"\
                    .format((fromnode,tonode),self.edges[(fromnode,tonode)][idx.EDGE_CAPACITY]))
                break

        total_delay = self.compute_cost(req_id, path)
        if maxlat < total_delay:
            possible_flag = False
            fail_case = 3
            #print("{} total cost is over the maxlat {}".format(total_delay, maxlat))

        return possible_flag, fail_case
                    

    def update_topology(self, req_id, route, path):
        # Update the capacity status with given path and req_id
        # Input
        # - req_id : ID of the request, scalar
        # - route : nodes set of the VNFs which are requested from the request, dict -
        #       { vnftype1 : nodeid, vnftype2, nodeid, ... }
        # - path : nodes set of result of the request, list - [node1, node2, ... ]
        
        request = self.reqs[req_id]
        traffic = request[idx.REQ_TRAFFIC]
        for vnftype in route.keys():
            vnf_placement = self.vnfs[vnftype]
            for vnf_inst_id in vnf_placement.keys():
                vnf_item = vnf_placement[vnf_inst_id]
                vnf_nodeid = vnf_item[idx.VNF_NODEID]
                route_node = route[vnftype]
                if route_node == vnf_nodeid and\
                    traffic <= vnf_item[idx.VNF_CAPACITY]:
                    new_capacity = vnf_item[idx.VNF_CAPACITY] - traffic
                    self.vnfs[vnftype][vnf_inst_id] = (vnf_nodeid, new_capacity)
                    break

        # Reduce the edge capacity
        for path_id in range(len(path)-1):
            from_node = path[path_id]
            to_node = path[path_id+1]
            (cost, capacity) = self.edges[(from_node,to_node)]
            self.edges[(from_node,to_node)] = (cost, capacity-traffic)
            (cost, capacity) = self.edges[(to_node,from_node)]
            self.edges[(to_node,from_node)] = (cost, capacity-traffic)

                
    def dataload(self): # For generation (no label)
        return 0

    def ranking(self, method):
        # Make ranks of current requests list
        # Output
        #   Rank { req_id a : rank number0,
        #          req_id b : rank number1,
        #          ...
        #        }

        def normalize(costs):
            n_obj = len(costs.keys())
            items = []
            for item_id in costs.keys():
                items.append(costs[item_id])
            mean = stat.mean(items)
            std = stat.stdev(items)
            for item_id in costs.keys():
                costs[item_id] = float(costs[item_id]-mean)/float(std)
            return costs
            
        def cost_sum(cost, tmp_cost):
            result_cost = {}
            for req_id in cost.keys():
                result_cost[req_id] = cost[req_id] + tmp_cost[req_id]
            return result_cost

        def sorting(costs):
            result = {}
            for req_id in costs.keys():
                rank = 0
                value = costs[req_id]
                for tmp_id in costs.keys():
                    if req_id != tmp_id:
                        tmp_value = costs[tmp_id]
                        if value > tmp_value:
                            rank += 1
                        elif value == tmp_value:
                            if req_id > tmp_id:
                                rank += 1
                result[req_id] = rank
            return result

        n_reqs = len(self.reqs.keys())
        cost = {}
        for req_id in self.reqs.keys():
            cost[req_id] = 0

        if 'naive' in method:
            tmp_cost = {}
            for req_id in self.reqs.keys():
                cost_value = req_id
                tmp_cost[req_id] = cost_value
            tmp_cost = normalize(tmp_cost)
            cost = cost_sum(cost, tmp_cost)
        
        if 'traffic_up' in method:
            tmp_cost = {}
            for req_id in self.reqs.keys():
                cost_value = self.reqs[req_id][idx.REQ_TRAFFIC]
                tmp_cost[req_id] = cost_value
            tmp_cost = normalize(tmp_cost)
            cost = cost_sum(cost, tmp_cost)
        
        if 'traffic_down' in method:
            tmp_cost = {}
            for req_id in self.reqs.keys():
                cost_value = -1 * self.reqs[req_id][idx.REQ_TRAFFIC]
                tmp_cost[req_id] = cost_value
            tmp_cost = normalize(tmp_cost)
            cost = cost_sum(cost, tmp_cost)
        
        if 'sfcid' in method:
            tmp_cost = {}
            sfc_rank = [0,1,3,2,0] # 4 > 1 > 3 > 2
            for req_id in self.reqs.keys():
                sfcid = self.reqs[req_id][idx.REQ_SFCID]
                cost_value = sfc_rank[sfcid]
                tmp_cost[req_id] = cost_value
            tmp_cost = normalize(tmp_cost)
            cost = cost_sum(cost, tmp_cost)
        
        if 'src_avg' in method:
            src_cost = {}
            for req_id in self.reqs.keys():
                # Compute Src costs
                total_src_cost = 0
                from_node = self.reqs[req_id][idx.REQ_SRC]
                sfcid = self.reqs[req_id][idx.REQ_SFCID]
                sfc = self.sfc_types[sfcid]
                
                vnf_type = sfc[0]
                vnf_item = self.vnfs[vnf_type]
                possible_vnf_nodes_list = []
                for vnf_inst_id in vnf_item.keys():
                    vnf_node = vnf_item[vnf_inst_id][idx.VNF_NODEID]
                    if vnf_node not in possible_vnf_nodes_list:
                        possible_vnf_nodes_list.append(vnf_node)
                        
                for to_node in possible_vnf_nodes_list:
                    total_src_cost += self.shortest_path[(from_node,to_node)][idx.SP_COST]
                src_cost[req_id] = -1*total_src_cost
            src_cost = normalize(src_cost)
            cost = cost_sum(cost, src_cost)

        if 'dst_avg' in method:
            dst_cost = {}
            for req_id in self.reqs.keys():
                # Compute Dst costs
                total_dst_cost = 0
                to_node = self.reqs[req_id][idx.REQ_DST]
                sfcid = self.reqs[req_id][idx.REQ_SFCID]
                sfc = self.sfc_types[sfcid]

                vnf_type = sfc[-1]
                vnf_item = self.vnfs[vnf_type]
                possible_vnf_nodes_list = []
                for vnf_inst_id in vnf_item.keys():
                    vnf_node = vnf_item[vnf_inst_id][idx.VNF_NODEID]
                    if vnf_node not in possible_vnf_nodes_list:
                        possible_vnf_nodes_list.append(vnf_node)

                for from_node in possible_vnf_nodes_list:
                    total_dst_cost += self.shortest_path[(from_node,to_node)][idx.SP_COST]
                dst_cost[req_id] = -1*total_dst_cost
            dst_cost = normalize(dst_cost)
            cost = cost_sum(cost, dst_cost)
            
        if 'src_max' in method:
            src_cost = {}
            for req_id in self.reqs.keys():
                from_node = self.reqs[req_id][idx.REQ_SRC]
                sfcid = self.reqs[req_id][idx.REQ_SFCID]
                sfc = self.sfc_types[sfcid]

                vnf_type = sfc[0]
                vnf_item = self.vnfs[vnf_type]
                possible_vnf_nodes_list = []
                for vnf_inst_id in vnf_item.keys():
                    vnf_node = vnf_item[vnf_inst_id][idx.VNF_NODEID]
                    if vnf_node not in possible_vnf_nodes_list:
                        possible_vnf_nodes_list.append(vnf_node)

                max_cost = 0
                for to_node in possible_vnf_nodes_list:
                    tmp_src_cost = self.shortest_path[(from_node,to_node)][idx.SP_COST]
                    if tmp_src_cost > max_cost:
                        max_cost = tmp_src_cost
                src_cost[req_id] = -1*max_cost
            src_cost = normalize(src_cost)
            cost = cost_sum(cost, src_cost)

        if 'dst_max' in method:
            dst_cost = {}
            for req_id in self.reqs.keys():
                to_node = self.reqs[req_id][idx.REQ_DST]
                sfcid = self.reqs[req_id][idx.REQ_SFCID]
                sfc = self.sfc_types[sfcid]

                vnf_type = sfc[-1]
                vnf_item = self.vnfs[vnf_type]
                possible_vnf_nodes_list = []
                for vnf_inst_id in vnf_item.keys():
                    vnf_node = vnf_item[vnf_inst_id][idx.VNF_NODEID]
                    if vnf_node not in possible_vnf_nodes_list:
                        possible_vnf_nodes_list.append(vnf_node)

                max_cost = 0
                for from_node in possible_vnf_nodes_list:
                    tmp_dst_cost = self.shortest_path[(from_node,to_node)][idx.SP_COST]
                    if tmp_dst_cost > max_cost:
                        max_cost = tmp_dst_cost
                dst_cost[req_id] = -1*max_cost
            dst_cost = normalize(dst_cost)
            cost = cost_sum(cost, dst_cost)

        if 'maxlat_up' in method:
            tmp_cost = {}
            for req_id in self.reqs.keys():
                cost_value = self.reqs[req_id][idx.REQ_MAXLAT]
                tmp_cost[req_id] = cost_value
            tmp_cost = normalize(tmp_cost)
            cost = cost_sum(cost, tmp_cost)

        if 'maxlat_down' in method:
            tmp_cost = {}
            for req_id in self.reqs.keys():
                cost_value = -1*self.reqs[req_id][idx.REQ_MAXLAT]
                tmp_cost[req_id] = cost_value
            tmp_cost = normalize(tmp_cost)
            cost = cost_sum(cost, tmp_cost)
        
        self.rank = sorting(cost)

    def make_adj_matrix(self, req_id):
        # Make adjacency matrix for model
        def make_adj_element(self, from_node, to_node):
            return 1/self.edges[(from_node,to_node)][idx.EDGE_COST]

        def numpy_softmax(data):
            for i in range(len(data)):
                if data[i] == 0:
                    data[i] = -float('inf')
            exp_a = np.exp(data)
            sum_exp_a = np.sum(exp_a)
            y = exp_a / sum_exp_a
            return y

        traffic = self.reqs[req_id][idx.REQ_TRAFFIC]
        A_in = np.zeros([self.n_nodes,self.n_nodes])
        A_out = np.zeros([self.n_nodes,self.n_nodes])

        for from_node in range(self.n_nodes):
            for to_node in range(self.n_nodes):
                if (from_node,to_node) in self.edges.keys() and\
                    self.edges[(from_node,to_node)][idx.EDGE_CAPACITY] >= traffic:
                    A_out[from_node,to_node] = make_adj_element(self,from_node,to_node)
                if (to_node,from_node) in self.edges.keys() and\
                    self.edges[(to_node,from_node)][idx.EDGE_CAPACITY] >= traffic:
                    A_in[to_node,from_node] = make_adj_element(self,to_node,from_node)

        for i in range(self.n_nodes):
            A_out[:,i] = numpy_softmax(A_out[:,i])
            A_in[:,i] = numpy_softmax(A_in[:,i])
        return A_out, A_in


    def make_annotation_matrix(self, req_id):
        # Make Annotation matrix for model
        traffic = self.reqs[req_id][idx.REQ_TRAFFIC]
        annotation = np.zeros([self.n_nodes,self.n_vnfs+2]) # for src,dst annotation
        src = self.reqs[req_id][idx.REQ_SRC]
        dst = self.reqs[req_id][idx.REQ_DST]
        annotation[src,0] = 1
        annotation[dst,-1] = 1

        for vnf_type in self.vnfs.keys():
            vnf_item = self.vnfs[vnf_type]
            for vnf_inst_id in vnf_item:
                if vnf_item[vnf_inst_id][idx.VNF_CAPACITY] >= traffic:
                    node_id = vnf_item[vnf_inst_id][idx.VNF_NODEID]
                    annotation[node_id,vnf_type+1] = 1
        return annotation


    def make_generation_info(self, req_id, node_id=None, needVNF=None, allVNF=None, train=None):
        # Make needed infomation vectors for model (needVNF, allVNF)
        def make_vnf_nodes_list(vnfs, needvnf_type, traffic):
            nodes_list = {}
            for vnf_inst_id in vnfs[needvnf_type].keys():
                vnf_item = vnfs[needvnf_type][vnf_inst_id]
                if vnf_item[idx.VNF_CAPACITY] >= traffic:
                    nodes_list[vnf_item[idx.VNF_NODEID]] = vnf_inst_id
            return nodes_list

        def update_info(vnfs, needVNF, allVNF, node_id, traffic, sfc):
            if torch.sum(allVNF).item() == 0:
                return needVNF, allVNF, True

            needvnf_type = torch.argmax(needVNF).item()
            vnf_nodes_list = make_vnf_nodes_list(vnfs, needvnf_type, traffic)
            if len(vnf_nodes_list) == 0:
                return 0, 0, False
            while(True):
                if node_id.item() in vnf_nodes_list.keys():
                    allVNF[needvnf_type] = 0
                    needVNF[needvnf_type] = 0
                    if torch.sum(allVNF).item() == 0:
                        break
                    else:
                        needvnf_type = sfc[np.argmax(np.equal(sfc, needvnf_type))+1]
                        needVNF[needvnf_type] = 1
                        vnf_nodes_list = make_vnf_nodes_list(vnfs, needvnf_type, traffic)
                        if len(vnf_nodes_list) == 0:
                            return 0, 0, False
                else:
                    break
            return needVNF, allVNF, True
        
        if train == True:
            label = self.labels[req_id][idx.LABEL_COST+1:]
            len_label = len(label)
            traffic = self.reqs[req_id][idx.REQ_TRAFFIC]
            sfcid = self.reqs[req_id][idx.REQ_SFCID]
            sfc = self.sfc_types[sfcid]
            len_sfc = len(sfc)
            needVNF = torch.from_numpy(np.zeros([len_label-1,self.n_vnfs]))
            allVNF = torch.from_numpy(np.zeros([len_label-1,self.n_vnfs]))

            tmp_needVNF = torch.from_numpy(np.zeros([self.n_vnfs]))
            tmp_allVNF = torch.from_numpy(np.zeros([self.n_vnfs]))
            tmp_needVNF[sfc[0]] = 1
            for vnf_type in sfc:
                tmp_allVNF[vnf_type] = 1

            for node_idx in range(len_label-1):
                node_id = torch.tensor(label[node_idx])
                tmp_needVNF, tmp_allVNF, flag = update_info(self.vnfs,\
                                  tmp_needVNF, tmp_allVNF, node_id, traffic, sfc)
                if flag == False:
                    self.print_topology(vnf=True, req=True)
                    print("req_id : ", req_id)
                    print("label : ", label)
                    print("sfc : ", sfc)
                    print("node_id : ", node_id)
                    print("tmp_needVNF, tmp_allVNF : ", tmp_needVNF, tmp_allVNF)
                    print("needVNF, allVNF : ", needVNF, allVNF)
                    return 0, 0, False
                needVNF[node_idx] = tmp_needVNF
                allVNF[node_idx] = tmp_allVNF

            return needVNF, allVNF, True
        else:
            if needVNF is None and allVNF is None:
                needVNF = np.zeros([self.n_vnfs])
                allVNF = np.zeros([self.n_vnfs])
                sfcid = self.reqs[req_id][idx.REQ_SFCID]
                sfc = self.sfc_types[sfcid]
                
                first_vnf_type = sfc[0]
                needVNF[first_vnf_type] = 1
                for vnf_type in sfc:
                    allVNF[vnf_type] = 1
                return needVNF, allVNF, True
            else:
                traffic = self.reqs[req_id][idx.REQ_TRAFFIC]
                sfcid = self.reqs[req_id][idx.REQ_SFCID]
                sfc = self.sfc_types[sfcid]
                needVNF, allVNF, flag = update_info(self.vnfs,\
                            needVNF, allVNF, node_id, traffic, sfc)
                if flag == False:
                    return 0, 0, False
                return needVNF, allVNF, True


    def compute_cost(self, req_id, path):
        # Compute the total cost of given path
        sfcid = self.reqs[req_id][idx.REQ_SFCID]
        sfc_chain = self.sfc[sfcid]
        total_cost = 0
        for vnftype in sfc_chain:
            total_cost += self.vnfinfo[vnftype][idx.VNFINFO_DELAY]
        
        for path_id in range(len(path)-1):
            from_node = path[path_id]
            to_node = path[path_id+1]
            tmp_cost = self.edges[(from_node, to_node)][idx.EDGE_COST]
            total_cost += tmp_cost
        return total_cost


    def make_mask(self, req_id):
        traffic = self.reqs[req_id][idx.REQ_TRAFFIC]
        mask = np.zeros([self.n_nodes,self.n_nodes])
        for (from_node,to_node) in self.edges.keys():
            if self.edges[(from_node,to_node)][idx.EDGE_CAPACITY] >= traffic:
                mask[from_node,to_node] = 1
        return mask

    
    def print_topology(self,\
                        topology=False,\
                        edge=False,\
                        vnf=False,\
                        req=False):
        if topology == True:
            print("-------------Topology Basic Info.--------------")
            print("Topology Name : {}".format(self.topology_name))
            print("Number of nodes : {}".format(self.n_nodes))
            print()
        if edge == True:
            print("-------------Edges--------------")
            for edge_id in self.edges.keys():
                print("{} edge : {} cost / {} capacity".format(edge_id,\
                                            self.edges[edge_id][idx.EDGE_COST],\
                                            self.edges[edge_id][idx.EDGE_CAPACITY]))
            print()
        if vnf == True:
            print("-------------VNFs--------------")
            for vnf_type in self.vnfs.keys():
                vnf_item = self.vnfs[vnf_type]
                print("{} VNF TYPE".format(vnf_type))
                for vnf_inst_id in vnf_item.keys():
                    print("------{} VNF INST ID : {} node, {} capacity".format(vnf_inst_id,\
                                                vnf_item[vnf_inst_id][idx.VNF_NODEID],\
                                                vnf_item[vnf_inst_id][idx.VNF_CAPACITY]))
            print()
        if req == True:
            print("-------------REQs---------------")
            print("[arrivaltime, endtime, src, dst, traffic, maxlat, sfcid]")
            for req_id in self.reqs.keys():
                print("{} Request : {}".format(req_id, self.reqs[req_id]))
            print()

    def topology_change(self):
        node_importance = np.zeros([self.n_nodes])
        for req_id in self.reqs.keys():
            src = self.reqs[req_id][idx.REQ_SRC]
            dst = self.reqs[req_id][idx.REQ_DST]
            node_importance[src] += 1
            node_importance[dst] += 1

        for vnf_type in self.vnfs.keys():
            vnf_item = self.vnfs[vnf_type]
            for inst_id in vnf_item.keys():
                vnf_node = vnf_item[inst_id][idx.VNF_NODEID]
                node_importance[vnf_node] += 1

        tmp_del_node = np.argwhere(np.equal(0, node_importance)==True)
        del_node = []
        if len(tmp_del_node) != 0:
            del_node = tmp_del_node[0,0]
            for from_node in range(self.n_nodes):
                edge_id = (from_node,del_node)
                if edge_id in self.edges.keys():
                    del self.edges[edge_id]
            for to_node in range(self.n_nodes):
                edge_id = (del_node,to_node)
                if edge_id in self.edges.keys():
                    del self.edges[edge_id]

            for from_node in range(self.n_nodes):
                if del_node in self.edges_to[from_node]:
                    self.edges_to[from_node].remove(del_node)
            #self.edges_to[del_node] = []
            del self.edges_to[del_node]

            # For labeling
            self.shortest_path = {}
            for from_node in range(self.n_nodes):
                for to_node in range(self.n_nodes):
                    if from_node != del_node and to_node != del_node:
                        optimal_path = dijsktra(self, from_node, to_node)
                        self.shortest_path[(from_node,to_node)] = (optimal_path,\
                                                            self.compute_cost(optimal_path))
            del_node = [del_node]

            
        return del_node
