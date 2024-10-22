from ast import arg
import copy
import importlib
import math
import functools
from pydoc import cli
import numpy as np
import random
import time
import tensorflow as tf
from soupsieve import select
from sympy import utilities
from termcolor import colored
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from utils.read_data import read_federated_data

from utils.trainer_utils import process_grad, calculate_cosine_dissimilarity
from utils.k_means_cos import K_means_cos, k_aggregate
#from flearn.model.mlp import construct_model
from flearn.trainer.groupbase import GroupBase
from collections import Counter
from flearn.server import Server
from flearn.client import Client
from flearn.group import Group


class FedCos(GroupBase):
    def __init__(self, train_config):
        super(FedCos, self).__init__(train_config)
        self.group_cold_start(random_centers=self.RCC)
        if self.cos_agg == True:
            for g in self.groups:
                g.aggregation_strategy = 'fedcos'
        else:
            for g in self.groups:
                g.aggregation_strategy = 'fedavg'

    def construct_actors(self):
        # 1, Read dataset
        clients, train_data, test_data = read_federated_data(self.dataset)

        # print("=============================clients", clients)
        # print("=============================train data",
        #       len(train_data['f_00000']), "\n", train_data['f_00000'])
        # print("=============================test data", len(test_data))

        # 2, Get model loader according to dataset and model name and construct the model
        # Set the model loader according to the dataset and model name
        model_path = 'flearn.model.%s.%s' % (
            self.dataset.split('_')[0], self.model)
        self.model_loader = importlib.import_module(model_path).construct_model
        # Construct the model
        client_model = self.model_loader(
            self.trainer_type, self.client_config['learning_rate'])

        # 3, Construct server
        self.server = Server(model=client_model)

        # 4, Construct clients (don't set their uplink)
        self.clients = [Client(id, self.client_config, train_data[id], test_data[id],
                        model=client_model) for id in clients]

        # 5*, We evaluate the auxiliary global model on server
        # To speed the testing, we need construct a local test dataset for server
        if self.eval_global_model == True:
            server_test_data = {'x': [], 'y': []}
            for c in clients:
                server_test_data['x'].append(test_data[c]['x'])
                server_test_data['y'].append(test_data[c]['y'])
            self.server.test_data['x'] = np.vstack(server_test_data['x'])
            self.server.test_data['y'] = np.hstack(server_test_data['y'])

    """According to the grouping result, select the clients from each group
    """

    def select_clients(self, comm_round, num_clients=20):
        def _sort_by_utility(group, clients):
            # 1. Calculate the data_size utility
            # Hyperparameter, control the penalty level of size_utility
            alpha = self.alpha
            avg_size = group.train_size / len(clients)
            size_utilities = [math.pow(c.train_size/avg_size, alpha)
                              if c.train_size < avg_size else 1 for c in clients]

            # 2. Calculate the convergence utility
            beta = self.beta
            cscale = [0]*len(clients)
            beta_index = [0]*len(clients)
            for i, c in enumerate(clients):
                for v in c.latest_updates:
                    # cscale[i] += np.sum(v.astype(np.float64)**2)
                    cscale[i] += tf.norm(v, ord=2)
                # cscale[i] = cscale[i]**0.5
                beta_index[i] = comm_round - c.newest_round
            convergence_utilities = [i * math.pow(beta, beta_index[idx])
                                     for idx, i in enumerate(cscale)]

            # 3. Sort the clients
            for idx, c in enumerate(clients):
                c.utility = size_utilities[idx] * convergence_utilities[idx]

            def cmp_func(c1, c2):
                if c1.utility >= c2.utility:
                    return -1
                else:
                    return 1
            sorted(clients, key=functools.cmp_to_key(cmp_func))

            return clients
        # cluster_weights = []
        # 获得每个cluster的model weights，并将global weights加入矩阵
        cluster_weights = [process_grad(g.latest_params) for g in self.groups]
        cluster_weights.append(process_grad(self.server.latest_params))
        # for key in self.server.latest_params.keys():
        #     if torch.any(torch.isnan(golbal_weights[key])):
        #         print("========debug========, global weight contains Nan", golbal_weights)
        #         break

        delta_w = np.vstack(cluster_weights)

        full_cossim_matrix = cosine_similarity(
            delta_w)  # shape=(n_clients, n_clients)
        # print("=============debug===========, the cossim matrix is: \n", decomposed_cossim_matrix)

        # 取矩阵里面最后一维数据，代表所有 cluster 和 global 之间的差异
        dis_cossim = [
            (1-i) % 1 if i > 0 else 0 for i in full_cossim_matrix[-1][:-1]]
        sum = np.sum(dis_cossim)
        contri_ws = np.array([i/sum for i in dis_cossim])
        print("cos dissim is", dis_cossim)
        print("contributions of each cluster are", contri_ws)

        # if np.sum(contri_ws) == 1:
        #     print("contribution compution is right")

        select_clients = []
        for idx, g in enumerate(self.groups):
            # Calculate participants number of group g[idx]
            num = contri_ws[idx] * num_clients
            if num < 1 and len(g.downlink) > 0:
                num = 1
                contri_ws[idx] = 1 / num_clients
            num = math.ceil(min(num, len(g.downlink)))

            """ Selection participants by utitlity. (only choose one from each group) """
            # Sort the clients according the utility
            sorted_clients = _sort_by_utility(g, g.downlink)

            # Epsilon-Greedy Algorithm
            props = random.random()
            if props < 0.1:  # Randomly choose one of the low-utility clients with probability 0.1
                unparticipated_clients = []
                for c in sorted_clients[num:]:
                    if c.participated == False:
                        unparticipated_clients.append(c)
                unparticipated_clients = sorted_clients if unparticipated_clients == [
                ] else unparticipated_clients
                tmp_clients = random.sample(unparticipated_clients, k=1)
            else:  # Randomly choose from the high-utility clients
                tmp_clients = random.sample(
                    sorted_clients[:math.ceil(num*1.5)], k=1)
            '''
            tmp_clients = random.sample(
                sorted_clients[:math.ceil(num*1.5)], k=num)
            '''
            # select the top-num clients to participate the training
            for c in tmp_clients:
                select_clients.append(c)

            # tmp_clients = random.sample(g.downlink, k=num)
            # for c in tmp_clients:
            #     select_clients.append(c)

        for c in select_clients:
            c.newest_round = comm_round
            c.participated = True

        # return select_clients
        return select_clients

    """ Cold strat all groups when create the trainer
    """

    def group_cold_start(self, alpha=20, clients=None, random_centers=False):
        # Clustering with all clients by default
        if clients is None:
            clients = self.clients

        # Strategy #1 (RCC): Randomly pre-train num_group clients as cluster centers
        # It is an optional strategy of FedGroup, named FedGroup-RCC
        if random_centers == True:
            print('Random Cluster Centers.')
            selected_clients = random.sample(clients, k=self.num_group)
            for c, g in zip(selected_clients, self.groups):
                _, _, _, g.latest_params, g.opt_updates = c.pretrain(
                    self.init_params, iterations=50)
                g.latest_updates = g.opt_updates
                c.set_uplink([g])
                g.add_downlink([c])

        # Strategy #2: Pre-train, then clustering the directions of clients' weights
        # <FedGroup> and <FedGrouProx> use this strategy
        if random_centers == False:
            selected_clients = random.sample(
                clients, k=min(self.num_group*alpha, len(clients)))

            for c in selected_clients:
                c.clustering = True  # Mark these clients as clustering client

            # for c in clients:
            #     c.clustering = True  # Mark these clients as clustering client

            # {Clusters ID: (cm, [c1, c2, ...])}
            clusters = self.clustering_clients(selected_clients)
            # Init groups accroding to the clustering results
            for g, cluster in zip(self.groups, clusters):
                # Init the group latest update
                g.latest_params = cluster[0]
                g.opt_updates = cluster[1]
                g.latest_updates = g.opt_updates
                # These clients do not need to be cold-started
                # Set the "group" attr of client only, didn't add the client to group
                for i in cluster[2]:
                    g.add_downlink(clients[i])

                for i in cluster[2]:
                    clients[i].set_uplink([g])

            # We aggregate these clustering results and get the new auxiliary global model
            self.update_auxiliary_global_model(self.groups)
            # Update the discrepancy of clustering client
            '''self.refresh_discrepancy_and_dissmilarity(selected_clients)'''
        return

    """ Clustering clients 
        Return: {Cluster ID: (parameter mean, update mean, client_list ->[c1, c2, ...])}
    """

    def clustering_clients(self, clients, n_clusters=None, max_iter=20):
        if n_clusters is None:
            n_clusters = self.num_group
        if len(clients) < n_clusters:
            print("ERROR: Not enough clients for clustering!!")
            return

        # Pre-train these clients first
        csolns, cupdates = {}, {}
        # client_size = [] * len(clients)

        # Record the execution time
        start_time = time.time()
        for idx, c in enumerate(clients):
            _, _, _, csolns[idx], cupdates[idx] = c.pretrain(
                self.init_params, iterations=50)
            # print("------------debuging-------------csolns[c]", csolns[c])
        print("Pre-training takes {}s seconds".format(time.time()-start_time))

        csoln_array = [process_grad(csoln) for csoln in csolns.values()]
        delta_w = np.vstack(csoln_array)  # shape=(n_clients, n_params)

        # print("delta_array's size is, ", delta_w.shape)
        # print("======delta_w======", delta_w)

        # Record the execution time
        start_time = time.time()
        '''
        score, labels = -1, []
        for i in range(n_clusters-2, n_clusters+6):
            k_model = K_means_cos(delta_w, k=i, max_iter=100)
            tmp_labels = k_model.forward()
            tmp_score = silhouette_score(
                delta_w, tmp_labels, metric='euclidean')
            if tmp_score > score:
                labels = tmp_labels
                n_clusters, score = i, tmp_score
            # print("k = ", i, "socre = ", tmp_score)
        print("============ the optimal k value is============", n_clusters)
        '''
        k_model = K_means_cos(delta_w, k=n_clusters, max_iter=100)
        labels = k_model.forward()
        # init Group object according to the adaptive k
        client_model = self.model_loader(
            self.trainer_type, self.client_config['learning_rate'])
        for id in range(n_clusters):
            # We need create the empty datasets for each group
            empty_train_data, empty_test_data = {
                'x': [], 'y': []}, {'x': [], 'y': []}
            self.groups.append(Group(id, self.group_config, empty_train_data, empty_test_data,
                                     [self.server], client_model))
        # Set the server's downlink to groups
        self.server.add_downlink(self.groups)
        print("K-means takes {}s seconds".format(time.time()-start_time))

        # [Cluster ID: [cm, [c1, c2, ...]]]
        cluster = [[] for _ in range(n_clusters)]
        # [[c1, c2,...], [c3, c4,...], ...]
        cluster2clients = [[] for _ in range(n_clusters)]
        for idx, cluster_id in enumerate(labels):
            # print(idx, cluster_id) # debug
            cluster2clients[int(cluster_id)].append(idx)

        # get the aggregate result of each group.
        # Note: In the K_means_cos() function, we use the one-demensional array, so do not get the aggregate result
        for cluster_id, client_list in enumerate(cluster2clients):
            # calculate the centerid of cluster
            params_list = [csolns[c] for c in client_list]
            updates_list = [cupdates[c] for c in client_list]
            if params_list:
                tmp_params = k_aggregate(params_list)
                tmp_updates = [(w1-w0)
                               for w0, w1 in zip(self.groups[cluster_id].latest_params, tmp_params)]
                cluster[cluster_id] = [tmp_params, tmp_updates, client_list]
            else:
                g0 = [np.zeros_like(ws) for ws in self.init_params]
                u0 = [(w1-w0)
                      for w0, w1 in zip(self.init_params, self.init_params)]
                cluster[cluster_id] = [tmp_params, tmp_updates, client_list]
                cluster[cluster_id] = [g0, u0, [1000]]
                print("Error, cluster is empty")

        print('cluster2clients is \n', cluster2clients)

        return cluster

    '''Rewrite the schedule client function of GroupBase,
        This function will be call before traning.
    '''

    def schedule_clients(self, round, selected_clients, groups):
        schedule_results = None
        if self.dynamic == True:
            # 1, Redo cold start distribution shift clients
            warm_clients = [
                wc for wc in selected_clients if wc.has_uplink() == True]
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
            schedule_results = {'shift': shift_count,
                                'migration': migration_count}

        # 2, Cold start newcomer: pretrain and assign a group
        for client in selected_clients:
            # for client in self.clients:
            if client.has_uplink() == False:
                self.client_cold_start(client, self.RAC)

        return schedule_results

    ''' Rewrite the schedule group function of GroupBase 
    '''

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
        for c in clients:
            c.clustering = False

        # Select the clients for clustering first
        selected_clients = random.sample(
            clients, k=min(len(groups)*alpha, len(clients)))
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
            print("Warning: Client already has a group: {:2d}.".format(
                client.uplink[0].id))
            return

        else:
            _, _, _, csoln, cupdate = client.pretrain(
                self.init_params, iterations=50)

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
                assign_group = self.groups[np.argmin(
                    [tup[1] for tup in diff_list])]

            # Set the uplink of client, add the downlink of group
            client.set_uplink([assign_group])
            assign_group.add_downlink([client])

            # Reset the temperature
            # client.temperature = client.max_temp
            #print(f'Assign client {client.id} to Group {assign_group.id}!')
        return assign_group

    # def update_auxiliary_global_model(self, groups, weights):
    #     prev_server_params = self.server.latest_params
    #     new_server_params = self.weighted_aggregate(
    #         [g.latest_params for g in groups], weights)
    #     self.server.latest_updates = [
    #         (new-prev) for prev, new in zip(prev_server_params, new_server_params)]
    #     self.server.latest_params = new_server_params
    #     return
