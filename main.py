from utils import timeSince
from topology import Topology
import index_dict as idx

import os
import time
import numpy as np
import torch
from random import randint



def generation(args, valid_loader, model, use='valid', log=None):
    print(" =========================================\n",\
          "==============Generation=================\n",\
          "=========================================")

    ratio_mean = []
    rand_idx = []
    for i in range(args.n_valid_print):
        rand_idx.append(randint(0, valid_loader.len()*20))

    pack_success = 0
    n_success = 0
    fail_type = np.zeros([3], dtype=np.int)
    gen_iter = 0
    gen_time = 0
    gen_time_start = time.time()
    for i in range(valid_loader.len()):
        model.eval()

        (gen_top, sorted_reqs) = valid_loader.getitem(i)

        n_req = len(sorted_reqs)
        package_fail=0
        for rank_idx in range(n_req):
            gen_iter += 1
            req_id = sorted_reqs[rank_idx]
            if use == 'test':
                label_cost = gen_top.labels[req_id][idx.LABEL_COST]
                label = gen_top.labels[req_id][idx.LABEL_COST+1:]
            else:
                label_cost = 1.0
                label = [0]

            gen_nodes, gen_cost, success, fail = model.generation(req_id, gen_top)
            if success == True:
                n_success += 1
                cost_ratio = (gen_cost / float(label_cost))
                ratio_mean.append(cost_ratio)

                if gen_iter in rand_idx:
                    print("============================")
                    gen_top.print_topology(vnf=True)
                    print("Request : ", gen_top.reqs[req_id])
                    print("Label Sequence : ", label)
                    print("Gen.  Sequence : ", gen_nodes)
                    print("Cost Ratio : ", cost_ratio)
                    print()

                gen_top.update_topology(gen_nodes, req_id)

                if fail == idx.FAIL2:
                    fail_type[fail] += 1
                    package_fail = 1

            else:
                fail_type[fail] += 1
                package_fail = 1

        if package_fail == 0:
            pack_success += 1

        if i % 100 == 0 and i > 0:
            print("{} model {} processed".format(args.model_name,i))
            print("{} pack, {} success, {} fail".format(pack_success, n_success, fail_type))
            log.write("\n{}\t{}\t{}\t{}\t{}\t{}".format(pack_success,n_success, np.sum(fail_type),\
                    fail_type[0],fail_type[1],fail_type[2]))
            
    
    total_try = n_success + np.sum(fail_type)
    fail0_ratio = float(fail_type[idx.FAIL0] / np.sum(fail_type)) * 100
    fail1_ratio = float(fail_type[idx.FAIL1] / np.sum(fail_type)) * 100
    fail2_ratio = float(fail_type[idx.FAIL2] / np.sum(fail_type)) * 100
    fail_ratio = float(np.sum(fail_type) / total_try) * 100

    ratio_var = np.var(np.array(ratio_mean))
    ratio_mean = np.mean(np.array(ratio_mean))
    gen_time = (time.time()-gen_time_start) / valid_loader.len()
    return ratio_mean, ratio_var, gen_time, fail_ratio, fail0_ratio, fail1_ratio, fail2_ratio,\
            n_success, fail_type, pack_success

def open_log(path, dir=False, message=False):
    if dir is True:
        if not os.path.exists(path):
            print("{} directory is made for {}".format(path, message))
            os.makedirs(path)
    else:
        if os.path.exists(path):
            print("Exist {} file is deleted".format(path))
            os.remove(path)
        print("{} file is opended for {}".format(path, mesasge))
        log = open(path, 'a')
        return log

def train_main(args, topo, trainset, validset, model, optimizer):
    print("-----Training Start-----")
    open_log(args.save_dir, dir=True, message="result")
    cost_log = open_log(args.cost_path, message="cost")
    valid_log = open_log(args.valid_path, message="valid")

    cost_log.write("Iter\ttrain_loss\ttrain_acc\ttime\n") 

    start = time.time()
    valid_eval = 0
    best_eval = 99999.0
    LR = args.learning_rate

    # learning decay, early stop
    n_decay = 0
    patience = 0
    early_stop = 0

    # Performances
    total_loss = 0
    total_acc = 0
    best_mean = 0
    best_var = 0
    best_fail_ratio = 0
    best_eval = 99999.0

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epoch):
        print("-----{} Epoch-----".format(epoch))
        if early_stop == 1:
            break
        for i in range(trainset.n_data()):
            train_topo = trainset.getitem(i,topo)
            model.train()

            for reqid in train_topo.reqs.keys():
                label = train_topo.label[reqid]
                gen_nodes, loss = model(label, reqid, train_topo)
                gen_nodes = gen_nodes.cpu().numpy()
                
                model.zero_grad()
                loss.backward()
                optimizer.step()

                train_top.update_topology(label, req_id)

            total_acc += tmp_total_acc / float(n_req)

            if i % args.print_iter == 0 and i > 1:
                avg_acc = (total_acc / args.print_iter) * 100
                avg_loss = total_loss / args.print_iter
                total_loss = 0
                total_acc = 0
                print("- {} epoch, {} gpu, {} model, {} lr_decay, {} patience -".format(\
                        epoch, args.device, args.model_name, n_decay, patience))
                print("{} iters - {:.3f} train_loss - {:.3f} train_acc - {} time".format\
                    (i, avg_loss, avg_acc, timeSince(start)))
                print("Current Best Valid {:.3f}".format(best_eval))
                print("Label Sequence : ", label)
                print("Gen.  Sequence : ", gen_nodes)

            if i % args.valid_iter == 0 and i > 1:
                train_log.write("{}\t{}\t{}\t".format(i,avg_loss,avg_acc))
                print("path : ", args.save_model_path)
                torch.save(model, args.save_model_path)
                print("Save the model : ", args.save_model_path)
                print("-----------Validation Results----------")
                valid_mean, valid_var, valid_time, valid_fail_ratio, fail0, fail1, fail2, _, _, _ = \
                                generation(args, valid_loader, model, use='test')
                print("Valid Ratio Mean      : {:.3f}".format(valid_mean))
                print("Valid Ratio Var.      : {:.3f}".format(valid_var))
                print("Ratio of Fail (%)     : {:.3f}".format(valid_fail_ratio))
                print("Ratio of fail type(%) : \nFAIL0={:.3f} FAIL1={:.3f} FAIL2={:.3f}"\
                                .format(fail0,fail1,fail2))
                print("Spent Avg. Time : {:.3f}".format(valid_time))
                train_log.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(\
                    valid_mean,valid_var, valid_fail_ratio, fail0, fail1, fail2))

                valid_eval = valid_mean + (valid_fail_ratio/25.0)
                if valid_eval < best_eval:
                    print("Find Best Model! Saving : ", args.save_model_path+'.best.pth')
                    torch.save(model, args.save_model_path + '.best.pth')
                    best_mean,best_var,best_fail_ratio,best_fail_type,best_time,best_eval=\
                    valid_mean,valid_var,valid_fail_ratio,(fail0,fail1,fail2),valid_time,\
                    valid_eval

                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        if n_decay >= args.lr_decay:
                            print("Early Stopping...")
                            print("Best Valid Mean : {}".format(best_mean))
                            print("Best Valid Var. : {}".format(best_var))
                            print("Best Fail Ratio : {}".format(best_fail_ratio))
                            print("Best Fail0 : {}, Fail1 : {}, Fail2 : {}".format(\
                                    best_fail_type[0],best_fail_type[1],best_fail_type[2]))
                            print("The time for one packet : {}".format(best_time))
                            print("It too {} packet, {} epoch, {} Time"\
                                .format(i, epoch, timeSince(start)))
                            early_stop = 1
                            break
                        else:
                            n_decay += 1
                            patience = 0
                            print("Learning Rate Decaying...")
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = LR * 0.1
                                LR = LR * 0.1
                            print("Current LR : ", LR)
            if early_stop == 1:
                break
    print("Training End..")
    return train_log


def test_main(args, model, test_loader, use, log=None):
    if log is None:
        if os.path.exists(args.save_log_path):
            os.remove(args.save_log_path)
        log = open(args.save_log_path, 'a')
        log.write("Genearte Test")

    if use == 'test':
        print("\n=======Testing Results========")
        test_mean, test_var, test_time, test_fail_ratio, fail0, fail1, fail2, _, _, _ = generation(\
                    args, test_loader, model, use=use)
        print("Test Ratio Mean       : {:.3f}".format(test_mean))
        print("Test Ratio Var.       : {:.3f}".format(test_var))
        print("Ratio of Fail (%)     : {:.3f}".format(test_fail_ratio))
        print("Ratio of Fail Type (%): \nFAIL0={:.3f} FAIL1={:.3f} FAIL2={:.3f}".format(\
                fail0,fail1,fail2))
        print("Spent Avg. Time : {:.3f}".format(test_time))
        log.write("========Testing=======")
        log.write("\n{}\t{}\t{}\t{}\t{}\t{}\n".format(test_mean,test_var,test_fail_ratio,\
                                                    fail0,fail1,fail2))
        log.close()
    else:
        _, _, _, _, _, _, _, n_success, fail_type, pack_success = generation(args, test_loader, model, use=use, log=log)
        log.write("\n{}\t{}\t{}\t{}\t{}\t{}".format(pack_success,n_success, np.sum(fail_type),\
                    fail_type[0],fail_type[1],fail_type[2]))
        log.close()
        print("Generation results")
        print("{} pack, {} success, {} fail - ( {}, {}, {} )".format(pack_success, n_success, np.sum(fail_type),\
                    fail_type[0], fail_type[1], fail_type[2]))
