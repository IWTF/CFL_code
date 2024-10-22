from cProfile import label
import copy
import importlib
import math
import functools
from pydoc import cli
from tokenize import group
from cv2 import threshold
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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
        self.alpha = 0.94
        self.beta = 0.6
        if self.cos_agg == True:
            for g in self.groups:
                g.aggregation_strategy = 'fedcos'
        else:
            for g in self.groups:
                g.aggregation_strategy = 'fedavg'

    def construct_actors(self):
        # 1, Read dataset
        clients, train_data, test_data = read_federated_data(self.dataset)

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

    def train(self):
        for comm_round in range(self.num_rounds):

            print(f'---------- Round {comm_round} ----------')
            # 0, Init time record
            train_time, test_time, agg_time = 0, 0, 0

            # 1, two stage selection for participants
            selected_clients = self.select_clients(comm_round, self.clients_per_round)

            # * Change the clients's data distribution, controled by 'shift_type'
            self.data_distribution_shift(comm_round, self.clients, self.shift_type, self.swap_p)

            # 2, Schedule clients (for example: reassign) or cold start clients, need selected clients only
            schedule_results = self.schedule_clients(comm_round, selected_clients, self.groups)

            # 3, Schedule groups (for example: recluster), need all clients
            self.schedule_groups(comm_round, self.clients, self.groups)

            # 4, Train selected clients
            start_time = time.time()
            train_results = self.server.train(selected_clients)
            train_time = round(time.time() - start_time, 3)
            if train_results == None:
                continue

            # *, Print the grouping information of this round
            gids = [c.uplink[0].id for c in selected_clients]
            count = Counter(gids)
            for id in sorted(count):
                print(
                    f'Round {comm_round}, Group {id} has {count[id]} client.')

            # 5, Inter-group aggregation according to the group learning rate
            if self.group_agg_lr > 0:
                start_time = time.time()
                self.inter_group_aggregation(train_results, self.group_agg_lr)
                agg_time = round(time.time() - start_time, 3)

            # 6, update the discrepancy and dissmilarity between group and client
            diffs = self.refresh_discrepancy_and_dissmilarity(selected_clients)

            # 7, schedule clients after training, in our algorithm, we update utility here
            self.schedule_clients_after_training(
                comm_round, selected_clients, self.groups)

            # 7, Summary this round of training
            train_summary = self.summary_results(
                comm_round, train_results=train_results)

            # 8, Update the auxiliary global model. Simply average group models without weights
            # The empty group will not be aggregated
            # self.update_auxiliary_global_model(
            #     [rest[0] for rest in train_results], weights)
            self.update_auxiliary_global_model(
                [rest[0] for rest in train_results])
            # Set the training model to the new server model, however this step is not important
            self.server.set_params(self.server.latest_params)

            # 9, Test the model (Last round training) every eval_every round and last round
            if comm_round % self.eval_every == 0 or comm_round == self.num_rounds:
                start_time = time.time()

                # Test model on all groups
                test_results = self.server.test(self.server.downlink)
                # Summary this test
                test_summary = self.summary_results(
                    comm_round, test_results=test_results)

                if self.eval_global_model == True:
                    # Test model on the server auxiliary model
                    test_samples, test_acc, test_loss = self.server.test_locally()
                    test_results = [
                        [self.server, test_samples, test_acc, test_loss]]
                    # Summary this test
                    self.summary_results(
                        comm_round, test_results=test_results)
                    
                test_time = round(time.time() - start_time, 3)
                # Write the training result and test result to file
                # Note: Only write the complete test accuracy after all client cold start
                self.writer.write_summary(
                    comm_round, train_summary, test_summary, diffs, schedule_results)

            # 10, Print the train, aggregate, test time
            print(f'Round: {comm_round}, Training time: {train_time}, Test time: {test_time}, \
                Inter-Group Aggregate time: {agg_time}')
    
    """ FedCSS 被选中客户端更新utility
    """
    def schedule_clients_after_training(self, comm_round, selected_clients, groups):
        for c in selected_clients:
            d_avg = c.uplink[0].train_size / len(c.uplink[0].downlink)
            d_term = (c.train_size / d_avg) if c.train_size < d_avg else 1
            c.utility = d_term * pow(self.alpha, c.participate_num) * abs(c.train_loss) 
        p_nums = [c.participate_num for c in self.clients]
        p_utility = [c.utility for c in self.clients]
        # print("++++++++++++++++ participate_num", p_nums)
        # print("++++++++++++++++ participate_num", p_utility)
        return

    """ FedCSS 两阶段客户端选择
    """
    def select_clients(self, comm_round, num_clients=20):
        def _inter_cluster_selection():
            # 1. calculate the cosine dissimilarity between group model and server model
            weights = [g.update_difference()*len(g.downlink)/len(self.clients) for g in self.groups]
            # 2. normalize the dissimilarity, and determine the participants num of each cluster
            sumation = sum(weights)
            num_per_group = [(self.clients_per_round * wi) / sumation for wi in weights]
            # 3. verify, make the num into integer & num < len(group)
            surplus, min_val, min_index = 0, self.clients_per_round, 0
            for index, num  in enumerate(num_per_group):
                num, group_size = round(num), len(self.groups[index].downlink)
                surplus += (num - group_size) if num > group_size else 0
                num_per_group[index] = min(num, group_size)
                if num > 0 and num < min_val:
                    min_index = index
                    min_val = num
            surplus += (self.clients_per_round - np.sum(num_per_group))
            num_per_group[min_index] = min(min_val, len(self.groups[min_index].downlink))
            # print("++++++++++++++++++++++++num_per_group", num_per_group)
            return num_per_group

        def _intra_cluster_selection(g, num):
            def cmp_func(c1, c2):
                if c1.utility >= c2.utility:
                    return -1
                else:
                    return 1
            sorted_arr = sorted(g.downlink, key=functools.cmp_to_key(cmp_func))
            top_user = sorted_arr[:round(num*self.beta)]
            random_user = random.sample(sorted_arr, num-len(top_user))
            for i in range(num-len(top_user)):
                top_user.append(random_user[i])
            if (num <= 2):
                # 给前几层赋值
                num_layers_to_copy = 8  # 要赋值的层数
                for i in range(num_layers_to_copy):
                    g.model.layers[i].set_weights(self.server.model.layers[i].get_weights())
            return top_user

        num_per_group = _inter_cluster_selection()

        participants_num = self.clients_per_round
        selected_clients = []
        for index, g in enumerate(self.groups):
            if (num_per_group[index] <= 0):
                continue
            participants_num -= num_per_group[index]
            tmp_arr = _intra_cluster_selection(g, num_per_group[index])
            selected_clients = [*selected_clients, *tmp_arr]
        
        group_idx = 0
        while (participants_num > 0):
            if (len(self.groups[group_idx].downlink) - num_per_group[group_idx] > 1):
                participants_num -= 1
                tmp_arr = _intra_cluster_selection(g, 1)
                selected_clients = [*selected_clients, *tmp_arr]
                group_idx += 1
        
        return selected_clients

    """ FedCSS k-means++算法，TODO
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
            # selected_clients = random.sample(
            #     clients, k=min(self.num_group*alpha, len(clients)))

            # for c in selected_clients:
            #     c.clustering = True  # Mark these clients as clustering client

            for c in clients:
                c.clustering = True  # Mark these clients as clustering client

            # {Clusters ID: (cm, [c1, c2, ...])}
            clusters = self.clustering_clients(clients)
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
            weights = [1.0] * len(self.init_params)
            # self.update_auxiliary_global_model(self.groups, weights)
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
        print("=======================================")

        # ===================exec HC for each layer==================
        # Get the input matrix of each layer
        n_clusters = 5
        delta_w = []
        for layer in range(len(csolns[0])-3, len(csolns[0])):
            temp = []
            for csoln in csolns.values():
                temp.append(csoln[layer].flatten())
            delta_w.append(temp)

        # Do hierarchical clustering for each layer
        clustering_result = []
        for i in range(len(delta_w)):
            Z = linkage(delta_w[i], 'ward')
            threshold = Z[-n_clusters][2]
            f = fcluster(Z, threshold, 'distance')
            clustering_result.append(f)

        # Combine the multiple clustering results to one result through voting [[], []](nxn)
        voting = [[0]*len(csolns) for i in csolns]
        for result in clustering_result:
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    if result[i] == result[j]:
                        voting[i][j] += 1

        # Change the voting matrix into the group info [[client1, ..., clienti], [...], [...]]
        groups = []
        assigned_nodes = []
        for i in range(len(voting)):
            if i in assigned_nodes:
                continue
            temp = [i]
            assigned_nodes.append(i)
            for j in range(i+1, len(voting)):
                if j not in assigned_nodes and voting[i][j] >= 2:
                    temp.append(j)
                    assigned_nodes.append(j)
            groups.append(temp)

        # Get the group label of each client
        labels = [0 for i in csolns]
        for idx, g in enumerate(groups):
            for i in g:
                labels[i] = idx

        # init Group object according to the adaptive k
        n_clusters = len(groups)
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
        print("Clustering Algorithm takes {}s seconds".format(
            time.time()-start_time))

        # [Cluster ID: [cm, [c1, c2, ...]]]
        cluster = [[] for _ in range(n_clusters)]
        # get the aggregate result of each group.
        for cluster_id, client_list in enumerate(groups):
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

        print('groups is \n', groups)

        return cluster

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
