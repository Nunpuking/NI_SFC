import topology_settings as tp
from topology import Topology
import index_dict as idx

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


'''
class beam_search():
    def __init__(self, width, top, src):
        self.path = {}
        self.cost = {}
        for path_id in range(width):
            self.path[path_id] = [src]
            self.cost[path_id] = 10000
        self.new_path = {}
        self.width = width
        self.top = top

    def forward(self, path_id, new_node):
        n_cand = len(self.new_path.keys())
        new_path = self.path[path_id].append(new_node)
        self.new_path[n_cand] = new_path
        new_path_cost = self.top.compute_cost(new_path)

            # Find max cost one
            max_cost = 0
            for path_id in self.path.keys():
                if maxcost < self.cost[path_id]:
                    max_path_id = path_id
                    max_cost = self.cost[path_id]

            # Compare and Replace
            if self.cost[max_path_id] > new_path_cost:
                self.path[max_path_id] = new_path


    def call_node(self, path_id):
'''    


class Embedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(input_dim, embedding_dim).type(torch.float))

    def forward(self, x):
        return torch.matmul(x, self.param)

def position_encoding_init(n_position, emb_dim):
    position_enc = np.array([\
            [pos / np.power(10000, 2*(j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class enc_dec_model(nn.Module):
    def __init__(self, args, topo):
        super().__init__()
        if args.model_name == 'GG_RNN':
            self.encoder_model = 'RNN'
            self.decoder_model = 'RNN'
        elif args.model_name == 'DG_RNN':
            self.encoder_model = 'DNN'
            self.decoder_model = 'RNN'
        elif args.model_name == 'GG_DNN':
            self.encoder_model = 'RNN'
            self.decoder_model = 'DNN'
        elif args.model_name == 'DNN':
            self.encoder_model = 'DNN'
            self.decoder_model = 'DNN'
        else:
            print("Wrong model name..")

        self.n_vnfs = topo.n_vnfs
        self.n_nodes = topo.n_nodes
        self.args = args
        self.hidden_unit = args.dnn_hidden
        self.annotation_dim = self.vnf_dim + 2

        self.anno_emb = Embedding(self.annotation_dim, args.state_dim)
        self.encoder_dnn = nn.Sequential(\
                            nn.Linear(args.state_dim*n_nodes + \
                                      2*n_nodes*n_nodes, self.hidden_unit),
                            nn.ReLU(True), nn.Dropout(),
                            nn.Linear(self.hidden_unit, self.hidden_unit),
                            nn.ReLU(True), nn.Dropout(),
                            nn.Linear(self.hidden_unit, n_nodes*args.state_dim)
                            )
        self.GRUcell = nn.GRUCell(2*args.state_dim, args.state_dim, bias=False)

        self.allVNF_emb = Embedding(n_vnfs, args.emb_vnf_dim)
        self.needVNF_emb = Embedding(n_vnfs, args.emb_vnf_dim)
        self.pos_enc = position_encoding_init(args.max_n_nodes, args.emb_node_dim)
        self.pos_enc = Variable(self.pos_enc).to(args.device)

        self.decoder_dnn = nn.Sequential(\
                            nn.Linear(args.state_dim + 2*args.emb_vnf_dim + args.emb_node_dim,\
                                        self.hidden_unit),
                            nn.ReLU(True), nn.Dropout(),
                            nn.Linear(self.hidden_unit, self.hidden_unit),
                            nn.ReLU(True), nn.Dropout(),
                            nn.Linear(self.hidden_unit, 1)
                            )

        self.decoder_GRUcell = nn.GRUCell(\
                        args.state_dim+2*args.emb_vnf_dim+args.emb_node_dim,\
                        args.state_dim*2)
        self.decoder_out = nn.Sequential(\
                nn.Linear(args.state_dim*2, args.state_dim),\
                nn.ReLU(True),\
                nn.Linear(args.state_dim,1)
                )
        self.decoder_process = nn.Sequential(\
                nn.Linear(args.

        self.softmax = nn.Softmax(dim=0)
        self.criterion = nn.CrossEntropyLoss()


    def encoder(self, A_out, A_in, annotation):
        if self.encoder_model == 'RNN':
            N = self.n_nodes
            h = self.anno_emb(annotation) # <N,E>
            for i in range(self.args.GRU_step):
                a_out = torch.matmul(A_out,h) # <N,E>
                a_in = torch.matmul(A_in,h)   # <N,E>
                a = torch.cat((a_out,a_in),dim=1) # <N,2E>
                h = self.GRUcell(a,h)
            enc_out = h # <N,E>
            return enc_out
        elif self.encoder_model == 'DNN':
            anno_emb = self.anno_emb(annotation)
            anno_emb = anno_emb.view(-1)
            A_out = A_out.view(-1)
            A_in = A_in.view(-1)

            concat_input = torch.cat((anno_emb, A_out, A_in),0).unsqueeze(0)
            # <1, anno_emb + 2*N>
            enc_out = self.encoder_dnn(concat_input)
            enc_out = enc_out.view(self.n_nodes,-1)
            return enc_out


    def decoder(self, enc_out, allVNF, needVNF, from_node, h=None):
        # Input
        #   enc_out : encoded representation of current topology status <N,E>
        #   allVNF : VNF list have to process <n_vnfs>
        #   needVNF : VNF type have to process right now <n_vnfs>
        #   from_node : node_id which agent is currently on <1>

        allVNF_emb = self.allVNF_emb(allVNF.unsqueeze(0)) # <1,E_vnf>
        needVNF_emb = self.needVNF_emb(needVNF.unsqueeze(0)) # <1,E_vnf>

        allVNF_emb = allVNF_emb.repeat(self.n_nodes,1) # <N,E_vnf>
        needVNF_emb = needVNF_emb.repeat(self.n_nodes,1) # <N,E_vnf>

        node_pos_enc = self.pos_enc[from_node].repeat(self.n_nodes,1) # <N,E_node>

        concat_input = torch.cat((enc_out,node_pos_enc,\
                                    allVNF_emb,needVNF_emb),1) # <N,E+E_vnf*2+E_node>

        if self.decoder_model == 'RNN':
            hidden = self.decoder_GRUcell(concat_input,h) # <N,2*E>
            output = self.decoder_out(hidden) # <N,1>
            return output,hidden

        elif self.decoder_model == 'DNN':
            output = self.decoder_dnn(concat_input)
            return output, None

    def forward(self, label, req_id, top):
        device = self.args.device
        A_out, A_in = top.make_adj_matrix(req_id)
        A_out = Variable(torch.from_numpy(A_out).type(torch.float)).to(device)
        A_in = Variable(torch.from_numpy(A_in).type(torch.float)).to(device)

        annotation = top.make_annotation_matrix(req_id)
        annotation = Variable(torch.from_numpy(annotation).type(torch.float)).to(device)

        enc_out = self.encoder(A_out, A_in, annotation)

        # Load Initial vnf information
        needVNF, allVNF, _ = top.make_generation_info(req_id, train=True)
        needVNF = Variable(needVNF.type(torch.float)).to(device)
        allVNF = Variable(allVNF.type(torch.float)).to(device)

        label = Variable(torch.from_numpy(label).type(torch.long)).to(device)
        gen_nodes = label.clone().detach()

        mask = top.make_mask(req_id) # <N,N>
        mask = Variable(torch.from_numpy(mask).type(torch.float)).to(device)

        total_loss = 0
        hidden = None
        for node_idx in range(len(label)-1):
            from_node = label[node_idx]
            label_node = label[node_idx+1]
            output, hidden = self.decoder(enc_out,\
                                            allVNF[node_idx],\
                                            needVNF[node_idx],\
                                            from_node,\
                                            hidden)
            # output : <N,1>, hidden : <N,2*E>


            probs = self.softmax(output) # <N,1>
            probs = torch.mul(probs, torch.transpose(mask[from_node.item()].unsqueeze(0),0,1))

            gen_node = torch.argmax(probs).item()
            gen_nodes[node_idx+1] = gen_node
            
            loss = self.criterion(torch.transpose(output,0,1),\
                                label_node.unsqueeze(0).type(torch.long))

            total_loss += loss

            #needVNF, allVNF, _ = top.make_generation_info(req_id,\
            #                                            node_id=label_node,\
            #                                            needVNF=needVNF,\
            #                                            allVNF=allVNF)
            #needVNF, allVNF = needVNF.detach(), allVNF.detach()


        avg_loss = total_loss.item() / float(len(label)-1)

        return gen_nodes.type(torch.long), avg_loss, total_loss

    def generation(self, req_id, top):
        device = self.args.device
        A_out, A_in = top.make_adj_matrix(req_id)    
        A_out = Variable(torch.from_numpy(A_out).type(torch.float)).to(device)
        A_in = Variable(torch.from_numpy(A_in).type(torch.float)).to(device)

        annotation = top.make_annotation_matrix(req_id)
        annotation = Variable(torch.from_numpy(annotation).type(torch.float)).to(device)

        enc_out = self.encoder(A_out, A_in, annotation)

        needVNF, allVNF, VNF_flag = top.make_generation_info(req_id, train=False)
        if VNF_flag == False:
            return 0, 0, False, idx.FAIL0
        needVNF = Variable(torch.from_numpy(needVNF).type(torch.float)).to(device)
        allVNF = Variable(torch.from_numpy(allVNF).type(torch.float)).to(device)

        from_node = top.reqs[req_id][idx.REQ_SRC]
        from_node = Variable(torch.tensor(from_node).type(torch.long)).to(device)
        dst_node = top.reqs[req_id][idx.REQ_DST]
        
        gen_nodes = Variable(torch.zeros(self.args.max_gen_len).type(torch.long)).to(device)
        gen_nodes[0] = from_node

        mask = top.make_mask(req_id)
        mask = Variable(torch.from_numpy(mask).type(torch.float)).to(device)
        hidden = None
        gen_idx = 1
        while True:
            output, hidden = self.decoder(enc_out,\
                                            allVNF,\
                                            needVNF,\
                                            from_node,\
                                            hidden)

            probs = self.softmax(output)
            probs = torch.mul(probs, torch.transpose(mask[from_node.item()].unsqueeze(0),0,1))

            gen_node = torch.argmax(probs)
            gen_nodes[gen_idx] = gen_node.item()
            gen_idx += 1

            from_node = gen_node

            needVNF, allVNF, VNF_flag = top.make_generation_info(req_id,\
                                                        node_id=gen_node,\
                                                        needVNF=needVNF,\
                                                        allVNF=allVNF)
            if VNF_flag == False:
                return 0, 0, False, idx.FAIL0

            process_flag = True if torch.sum(allVNF) == 0 else False
            if process_flag is True and gen_node.item() == dst_node:
                gen_nodes = gen_nodes[:gen_idx].cpu().numpy()
                maxlat = top.reqs[req_id][idx.REQ_MAXLAT]
                cost = top.compute_cost(gen_nodes, req_id)
                if cost > maxlat:
                    return gen_nodes, cost, True, idx.FAIL2
                else:
                    return gen_nodes, cost, True, 0

            if gen_idx >= self.args.max_gen_len:
                return gen_nodes, 0, False, idx.FAIL1


class DNN_naive(nn.Module):
    def __init__(self, args, hidden_unit):
        super().__init__()
        self.args = args
        self.vnf_dim = tp.n_vnfs
        self.hidden_unit = hidden_unit

        self.allVNF_emb = Embedding(self.vnf_dim, args.emb_vnf_dim)
        self.needVNF_emb = Embedding(self.vnf_dim, args.emb_vnf_dim)
        self.pos_enc = position_encoding_init(args.max_n_nodes, args.emb_node_dim)
        self.pos_enc = Variable(self.pos_enc).to(args.device)

        self.anno_emb = Embedding(args.annotation_dim, args.state_dim)
        
        self.out = nn.Sequential(
                    nn.Linear(args.emb_vnf_dim*2 + args.emb_node_dim + args.state_dim*tp.n_nodes\
                        + 2*tp.n_nodes*tp.n_nodes, hidden_unit*2),
                    nn.ReLU(True), nn.Dropout(),
                    nn.Linear(hidden_unit*2, hidden_unit*2), nn.ReLU(True), nn.Dropout(),
                    nn.Linear(hidden_unit*2, hidden_unit*2), nn.ReLU(True), nn.Dropout(),
                    nn.Linear(hidden_unit*2, tp.n_nodes)
                    )
        self.softmax = nn.Softmax(dim=0)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, label, req_id, top):
        device = self.args.device

        A_out, A_in = top.make_adj_matrix(req_id)
        A_out = Variable(torch.from_numpy(A_out).type(torch.float)).to(device)
        A_in = Variable(torch.from_numpy(A_in).type(torch.float)).to(device)

        annotation = top.make_annotation_matrix(req_id)
        annotation = Variable(torch.from_numpy(annotation).type(torch.float)).to(device)

        anno_emb = self.anno_emb(annotation) # <N,E>
        anno_emb = anno_emb.view(-1) # <N*E>

        # Does Adj matrix needs normalization?
        A_out = A_out.view(-1) # <N*N>
        A_in = A_in.view(-1) # <N*N>
    
        label = Variable(torch.from_numpy(label).type(torch.long)).to(device)
        gen_nodes = label.clone().detach()

        needVNF, allVNF, _ = top.make_generation_info(req_id, train=True)
        needVNF = Variable(needVNF.type(torch.float)).to(device)
        allVNF = Variable(allVNF.type(torch.float)).to(device)

        needVNF = self.needVNF_emb(needVNF.unsqueeze(0)).squeeze(0) # <L-1,emb_vnf>
        allVNF = self.allVNF_emb(allVNF.unsqueeze(0)).squeeze(0) # <L-1,emb_vnf>

        mask = top.make_mask(req_id)
        mask = Variable(torch.from_numpy(mask).type(torch.float)).to(device)

        total_loss = 0
        for node_idx in range(len(label)-1):
            from_node = label[node_idx]
            label_node = label[node_idx+1]

            node_pos_enc = self.pos_enc[from_node] # <emb_node_dim>
            tmp_needVNF, tmp_allVNF = needVNF[node_idx], allVNF[node_idx]

            concat_input = torch.cat((tmp_needVNF,tmp_allVNF,node_pos_enc,\
                                        anno_emb,A_out,A_in),0).unsqueeze(0)
                                        # <emb_vnf*2 + emb_node + E*N + 2*N*N>
            output = torch.transpose(self.out(concat_input),0,1) # <N,1>

            probs = self.softmax(output) 
            probs = torch.mul(probs, torch.transpose(mask[from_node.item()].unsqueeze(0),0,1))

            gen_node = torch.argmax(probs).item()
            gen_nodes[node_idx+1] = gen_node

            loss = self.criterion(torch.transpose(output,0,1),\
                                label_node.unsqueeze(0).type(torch.long))

            total_loss += loss

        avg_loss = total_loss.item() / float(len(label)-1)

        return gen_nodes.type(torch.long), avg_loss, total_loss


    def generation(self, req_id, top):
        device = self.args.device
        A_out, A_in = top.make_adj_matrix(req_id)
        A_out = Variable(torch.from_numpy(A_out).type(torch.float)).to(device)
        A_in = Variable(torch.from_numpy(A_in).type(torch.float)).to(device)

        annotation = top.make_annotation_matrix(req_id)
        annotation = Variable(torch.from_numpy(annotation).type(torch.float)).to(device)

        anno_emb = self.anno_emb(annotation)
        anno_emb = anno_emb.view(-1)

        A_out = A_out.view(-1)
        A_in = A_in.view(-1)

        needVNF, allVNF, VNF_flag = top.make_generation_info(req_id, train=False)
        if VNF_flag == False:
            return 0, 0, False, idx.FAIL0
        needVNF = Variable(torch.from_numpy(needVNF).type(torch.float)).to(device)
        allVNF = Variable(torch.from_numpy(allVNF).type(torch.float)).to(device)

        from_node = top.reqs[req_id][idx.REQ_SRC]
        from_node = Variable(torch.tensor(from_node).type(torch.long)).to(device)
        dst_node= top.reqs[req_id][idx.REQ_DST]

        gen_nodes = Variable(torch.zeros(self.args.max_gen_len).type(torch.long)).to(device)
        gen_nodes[0] = from_node

        mask = top.make_mask(req_id)
        mask = Variable(torch.from_numpy(mask).type(torch.float)).to(device)
        
        gen_idx = 1
        while True:
            needVNF_emb = self.needVNF_emb(needVNF.unsqueeze(0)).squeeze(0)
            allVNF_emb = self.allVNF_emb(allVNF.unsqueeze(0)).squeeze(0)

            node_pos_enc = self.pos_enc[from_node]
            
            concat_input = torch.cat((needVNF_emb,allVNF_emb,node_pos_enc,\
                                    anno_emb,A_out,A_in),0).unsqueeze(0)

            output = torch.transpose(self.out(concat_input),0,1)

            probs = self.softmax(output)
            probs = torch.mul(probs, torch.transpose(mask[from_node.item()].unsqueeze(0),0,1))

            gen_node = torch.argmax(probs)
            gen_nodes[gen_idx] = gen_node.item()
            gen_idx += 1

            from_node = gen_node

            needVNF, allVNF, VNF_flag = top.make_generation_info(req_id,\
                                                        node_id = gen_node,\
                                                        needVNF = needVNF,\
                                                        allVNF = allVNF)

            if VNF_flag == False:
                return 0, 0, False, idx.FAIL0

            process_flag = True if torch.sum(allVNF) == 0 else False
            if process_flag is True and gen_node.item() == dst_node:
                gen_nodes = gen_nodes[:gen_idx].cpu().numpy()
                maxlat = top.reqs[req_id][idx.REQ_MAXLAT]
                cost = top.compute_cost(gen_nodes, req_id)
                if cost > maxlat:
                    return gen_nodes, cost, True, idx.FAIL2
                else:
                    return gen_nodes, cost, True, 0

            if gen_idx >= self.args.max_gen_len:
                return gen_nodes, 0, False, idx.FAIL1
            
