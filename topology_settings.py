import numpy as np

n_nodes = 12
n_vnfs = 5
n_sfc = 4
EOI_token = 99

vnf_types = {0:'firewall','firewall':0,\
            1:'ids','ids':1,\
            2:'nat','nat':2,\
            3:'proxy','proxy':3,\
            4:'wano','wano':4}

edges = [
    (0, 1, 1),
    (1, 4, 117),
    (1, 5, 58),
    (1, 11, 84),
    (2, 5, 26),
    (2, 8, 70),
    (3, 6, 63),
    (3, 9, 129),
    (3, 10, 209),
    (4, 6, 90),
    (4, 7, 189),
    (5, 6, 54),
    (7, 9, 36),
    (8, 11, 23),
    (9, 10, 86),
]

edge_capacity = 10000000

vnf_cost = {vnf_types['firewall']:4,\
            vnf_types['ids']:8,\
            vnf_types['nat']:1,\
            vnf_types['proxy']:4,\
            vnf_types['wano']:4}  

vnf_capacity = {vnf_types['firewall']:900000,\
                vnf_types['ids']:600000,\
                vnf_types['nat']:900000,\
                vnf_types['proxy']:900000,\
                vnf_types['wano']:400000}  

sfc_type = {1:np.array([vnf_types['nat'],vnf_types['firewall'],vnf_types['ids']]),\
            2:np.array([vnf_types['nat'],vnf_types['proxy']]),\
            3:np.array([vnf_types['nat'],vnf_types['wano']]),\
            4:np.array([vnf_types['nat'],vnf_types['firewall'],\
                        vnf_types['wano'],vnf_types['ids']]),\
            (vnf_types['nat'],vnf_types['firewall'],vnf_types['ids']):1,\
            (vnf_types['nat'],vnf_types['proxy']):2,\
            (vnf_types['nat'],vnf_types['wano']):3,\
            (vnf_types['nat'],vnf_types['firewall'],\
             vnf_types['wano'],vnf_types['ids']):4
            }
