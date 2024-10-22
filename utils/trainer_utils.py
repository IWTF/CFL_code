from tkinter.tix import Tree
import numpy as np

import tensorflow as tf

'''
    Define the config of trainer,
    The type of trainer contain: fedavg, fedgroup, splitfed*, splitfg*
'''


class TrainConfig(object):
    def __init__(self, dataset, model, trainer):
        self.trainer_type = trainer
        self.results_path = f'results/{dataset}/'

        self.trainer_config = {
            # This is common config of trainer
            'dataset': dataset,
            'model': model,
            'seed': 2077,
            'num_rounds': 200,
            'clients_per_round': 25,
            'eval_every': 1,
            'eval_locally': True,
            'swap_p': 0,  # Randomly swap two warm clients with probability
            'shift_type': None,  # {all, part, increment}
        }

        self.client_config = {
            # This is common config of client
            'local_epochs': 10,
            # However, we compile lr to model
            'learning_rate': 0.01,
            'batch_size': 32,
            # The dynamic reassign clients strategy of FedGroup
            # temperature = None means disable this function
            'temperature': None
        }

        if trainer in ['fedcos', 'fedgroup', 'fesem', 'ifca']:
            if trainer == 'fedcos':
                self.trainer_config.update({
                    'num_group': 10,
                    'group_agg_lr': 0.0,
                    'eval_global_model': False,
                    'pretrain_scale': 20,
                    'measure': 'EDC',  # {EDC, MADC}
                    'RAC': False,
                    'RCC': False,
                    'dynamic': False,
                    'cos_agg': True,
                    'recluster_epoch': [50, 100],  # [100, 200]
                })

            if trainer == 'fedgroup':
                self.trainer_config.update({
                    'num_group': 3,
                    'group_agg_lr': 0.0,
                    'eval_global_model': False,
                    # 'eval_global_model': True,
                    'pretrain_scale': 20,
                    'measure': 'EDC',  # {EDC, MADC}
                    'RAC': False,
                    'RCC': False,
                    'dynamic': False,
                    'temp_metrics': 'l2',  # {l2, consine}
                    # {step, linear, lied, eied} lied-> linear increase&exponential decrease
                    'temp_func': 'step',
                    'temp_agg': False,
                    'recluster_epoch': None  # [50, 100, 150]
                })

            if trainer in ['fesem',  'ifca']:
                self.trainer_config.update({
                    'num_group': 3,
                    # The iter-group aggregation is disabled
                    'group_agg_lr': 0.0,
                    # 'eval_global_model': True
                    'eval_global_model': False,
                })

            self.group_config = {
                # Whether the models of all clients in the group are consistent，
                # which will greatly affect the test results.
                'consensus': False,
                'max_clients': 999,
                'allow_empty': True
            }

        if self.trainer_config['dataset'] == 'femnist':
            self.client_config.update({'learning_rate': 0.003})
            self.trainer_config.update({'num_group': 5})

        if self.trainer_config['dataset'].startswith('cifar'):
            self.client_config.update({'learning_rate': 0.03})
            self.trainer_config.update({'num_group': 10})

        # startwith类似于正则表达式，匹配以mnist开头的字符串
        if self.trainer_config['dataset'].startswith('mnist'):
            self.client_config.update({'learning_rate': 0.03})
            self.trainer_config.update({'num_group': 5})

        if self.trainer_config['dataset'] == 'sent140':
            self.client_config.update({'learning_rate': 0.3})
            self.trainer_config.update({'num_group': 5})
            self.trainer_config.update({'num_rounds': 800})

        if self.trainer_config['dataset'].startswith('synthetic'):
            self.client_config.update({'learning_rate': 0.01})
            self.trainer_config.update({'num_group': 5})

        if self.trainer_config['dataset'] == 'fmnist':
            self.client_config.update({'learning_rate': 0.005})
            self.trainer_config.update({'num_group': 5})

        if trainer == 'splitfed':
            # TODO: plan for split learning
            pass
        if trainer == 'splitfg':
            # TODO:
            pass


def process_grad(grads):
    '''
    Args:
        grads: grad 
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0]  # shape = (784, 10)
    for i in range(1, len(grads)):
        # output a flattened array
        client_grads = np.append(client_grads, grads[i])
        # (784, 10) append (10,)

    return client_grads


def calculate_cosine_dissimilarity(w1, w2):
    flat_w1, flat_w2 = process_grad(w1), process_grad(w2)
    cosine = np.dot(flat_w1, flat_w2) / \
        (np.linalg.norm(flat_w1) * np.linalg.norm(flat_w2))
    dissimilarity = (1.0 - cosine) / 2.0  # scale to [0, 1] then flip
    return dissimilarity


def label2tensor(label, class_num):
    """
    将list类型的label值转化为tensor
    input:label(list):原始数据集
    output:lable(tensor)：样本中的标签的个数
    """
    ret_label = []
    for idx, y in enumerate(label):
        ret_label.append([0] * class_num)
        ret_label[idx][y] = 1
    return tf.reshape(ret_label, [len(label), -1])
