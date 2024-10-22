import numpy as np
import random
import time
from termcolor import colored
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

from utils.trainer_utils import process_grad, calculate_cosine_dissimilarity
from utils.k_means_cos import K_means_cos
#from flearn.model.mlp import construct_model
from flearn.trainer.groupbase import GroupBase
from collections import Counter

class CLF_selection(GroupBase):
    def __init__(self, train_config):
        super(CLF_selection, self).__init__(train_config)
        self.group_cold_start(random_centers=self.RCC)
        
        # 这里是temptrue xxx，原作者也没有去实现这个东西，可能是其进一步的构想
        # if self.temp_agg == True: 
        #     for g in self.groups: g.aggregation_strategy = 'temp'
        # else:
        #     for g in self.groups: g.aggregation_strategy = 'fedavg'

    """ Cold strat all groups when create the trainer
    """
    def group_cold_start(self, alpha=20, clients=None, random_centers=False):

        # Clustering with all clients by default
        if clients is None: clients = self.clients

        # Strategy #1 (RCC): Randomly pre-train num_group clients as cluster centers
        # It is an optional strategy of FedGroup, named FedGroup-RCC
        if random_centers == True:
            print('Random Cluster Centers.')
            selected_clients = random.sample(clients, k=self.num_group)
            for c, g in zip(selected_clients, self.groups):
                _, _, _, g.latest_params, g.opt_updates = c.pretrain(self.init_params, iterations=50)
                g.latest_updates = g.opt_updates
                c.set_uplink([g])
                g.add_downlink([c])

        # Strategy #2: Pre-train, then clustering the directions of clients' weights
        # <FedGroup> and <FedGrouProx> use this strategy
        if random_centers == False:
            selected_clients = random.sample(clients, k=min(self.num_group*alpha, len(clients)))

            for c in selected_clients: c.clustering = True # Mark these clients as clustering client

            cluster = self.clustering_clients(selected_clients) # {Cluster ID: (cm, [c1, c2, ...])}
            # Init groups accroding to the clustering results
            for g, id in zip(self.groups, cluster.keys()):
                # Init the group latest update
                g.latest_params = cluster[id][0]
                g.opt_updates = cluster[id][1]
                g.latest_updates = g.opt_updates
                # These clients do not need to be cold-started
                # Set the "group" attr of client only, didn't add the client to group
                g.add_downlink(cluster[id][2])
                for c in cluster[id][2]:
                    c.set_uplink([g])

            # We aggregate these clustering results and get the new auxiliary global model
            self.update_auxiliary_global_model(self.groups)
            # Update the discrepancy of clustering client
            '''self.refresh_discrepancy_and_dissmilarity(selected_clients)'''
        return

    """ Clustering clients 
        Return: {Cluster ID: (parameter mean, update mean, client_list ->[c1, c2, ...])}
    """
    def clustering_clients(self, clients, n_clusters=None, max_iter=20):
        if n_clusters is None: n_clusters = self.num_group
        if len(clients) < n_clusters: 
            print("ERROR: Not enough clients for clustering!!")
            return

        # Pre-train these clients first
        csolns, cupdates = {}, {}

        # Record the execution time
        start_time = time.time()
        for c in clients:
            _, _, _, csolns[c], cupdates[c] = c.pretrain(self.init_params, iterations=50)
        print("Pre-training takes {}s seconds".format(time.time()-start_time))

        # 将训练得到的呃clients weights进行向量化
        # 之后将各个被选择的客户端参数合并为一个数组
        update_array = [process_grad(update) for update in cupdates.values()]
        for idx, i in enumerate(update_array):
            update_array[idx] = i / np.linalg.norm(v)
        delta_w = np.vstack(update_array) # shape=(n_clients, n_params)
        
        # 利用k-means算法进行聚类
        # 首先要进行归一化处理
        start_time = time.time()
        labels, center = K_means_cos(delta_w, 10)
        print("K-means takes {}s seconds".format(time.time()-start_time))
 
        '''
        # Normialize cossim to [0,1]
        full_dissim_matrix = (1.0 - full_cossim_matrix) / 2.0
        '''
        
        cluster = {} # {Cluster ID: (cm, [c1, c2, ...])}
        cluster2clients = [[] for _ in range(n_clusters)] # [[c1, c2,...], [c3, c4,...], ...]
        for idx, cluster_id in enumerate(labels):
            #print(idx, cluster_id, len(cluster2clients), n_clusters) # debug
            cluster2clients[cluster_id].append(clients[idx])
        for cluster_id, client_list in enumerate(cluster2clients):
            # calculate the means of cluster
            params_list = [csolns[c] for c in client_list]
            updates_list = [cupdates[c] for c in client_list]
            if params_list:
                 # All client have equal weight
                cluster[cluster_id] = (self.simply_averaging_aggregate(params_list),\
                    self.simply_averaging_aggregate(updates_list), client_list)
            else:
                print("Error, cluster is empty")

        return cluster


    '''Rewrite the schedule client function of GroupBase,
        This function will be call before traning.
    '''
    def schedule_clients(self, round, selected_clients, groups):
        schedule_results = None
        if self.dynamic == True:
            # 1, Redo cold start distribution shift clients
            warm_clients = [wc for wc in self.clients if wc.has_uplink() == True]
            shift_count, migration_count = 0, 0
            for client in warm_clients:
                count = client.check_distribution_shift()
                if count is not None and client.distribution_shift == True:
                    shift_count += 1
                    prev_g = client.uplink[0]
                    prev_g.delete_downlink(client)
                    client.clear_uplink()
                    self.client_cold_start(client)
                    new_g = client.uplink[0]
                    client.train_label_count = count
                    client.distribution_shift = False
                    if prev_g != new_g:
                        migration_count += 1
                        print(colored(f'Client {client.id} migrate from Group {prev_g.id} \
                            to Group {new_g.id}', 'yellow', attrs=['reverse']))
            schedule_results = {'shift': shift_count, 'migration': migration_count}

        # 2, Cold start newcomer: pretrain and assign a group
        for client in selected_clients:
        #for client in self.clients:
            if client.has_uplink() == False:
                self.client_cold_start(client, self.RAC)

        return schedule_results

    ''' Rewrite the schedule group function of GroupBase '''
    def schedule_groups(self, round, clients, groups):
        if self.dynamic == True and self.recluster_epoch is not None:
            # Reculster warm client
            if round in self.recluster_epoch:
                warm_clients = [c for c in clients if c.has_uplink() == True]
                self.recluster(warm_clients, groups)
        return

    """ Reculster () clients and cold start (reassign group) the remain clients
    """
    def recluster(self, clients, groups, alpha=20):
        if len(groups) != len(self.groups):
            print("Warning: Group Number is change!")
            # TODO: dynamic group num
            return 

        print('Reclustering...')
        # Clear the clustering mark
        for c in clients: c.clustering = False

        # Select the clients for clustering first
        selected_clients = random.sample(clients, k=min(len(groups)*alpha, len(clients)))
        remain_clients = [c for c in clients if c not in selected_clients]
        self.group_cold_start(clients=selected_clients)
        for c in remain_clients:
            # Reassign (cold start) the remain clients
            old_group = c.uplink[0]
            old_group.delete_downlink(c)
            c.clear_uplink()
            self.client_cold_start(c, self.RAC, redo=False)

        # Refresh the discrepancy of all clients (clustering clients and reassign clients)
        '''self.refresh_discrepancy_and_dissmilarity(clients)'''
        return 

    def client_cold_start(self, client, random_assign=False, redo=False):
        if client.has_uplink() == True:
            print("Warning: Client already has a group: {:2d}.".format(client.uplink[0].id))
            return

        else:
            _, _, _, csoln, cupdate = client.pretrain(self.init_params, iterations=50)

            # Calculate the cosine dissimilarity between client's update and group's update
            diff_list = []
            for g in self.groups:
                if redo == False:
                    opt_updates = g.opt_updates
                else:
                    opt_updates = g.latest_updates
                diff = calculate_cosine_dissimilarity(cupdate, opt_updates)
                diff_list.append((g, diff))
            if random_assign == True:
                # RAC: Randomly assign client
                assign_group = random.choice(self.groups)
            else:
                # Minimize the differenct
                assign_group = self.groups[np.argmin([tup[1] for tup in diff_list])]
            
            # Set the uplink of client, add the downlink of group
            client.set_uplink([assign_group])
            assign_group.add_downlink([client])

            # Reset the temperature
            client.temperature = client.max_temp
            #print(f'Assign client {client.id} to Group {assign_group.id}!')
        return assign_group
