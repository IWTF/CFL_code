from hashlib import new
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import copy


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


def k_aggregate(w):
    # 获得聚类中心的坐标
    if w == []:
        return 0
    w_ret = copy.deepcopy(w[0])
    for laywer in range(len(w_ret)):
        for i in range(0, len(w)):
            if i == 0:
                w_ret[laywer] = w_ret[laywer] / np.linalg.norm(w_ret[laywer])
                continue
            w_ret[laywer] += (w[i][laywer] / np.linalg.norm(w[i][laywer]))
        w_ret[laywer] = w_ret[laywer] / len(w)
        w_ret[laywer] = w_ret[laywer] / np.linalg.norm(w_ret[laywer])
    return w_ret


class K_means_cos():
    def __init__(self, data, k, max_iter):
        self.data = data
        self.k = k
        self.max_iter = max_iter

    def distance(self, p1, p2):
        # 1 - cos(p1, p2)
        return calculate_cosine_dissimilarity(p1, p2)

    def generate_center(self):
        # 随机初始化聚类中心
        n = len(self.data)
        rand_id = random.sample(range(n), self.k)
        center = []
        for id in rand_id:
            center.append(self.data[id])
        return center

    def converge(self, old_center, new_center):
        # 判断是否收敛
        ret = True

        for idx, i in enumerate(old_center):
            if (i == new_center[idx]).all() == False:
                ret = False
                break

        # if ret:
        #     print("return True")
        # else:
        #     print("return False")

        # print('\n\n====old_center========', old_center)
        # print('====new_center========', new_center)
        return ret

    def forward(self):
        center = self.generate_center()
        # print("=======init center========", center)
        n = len(self.data)
        # print(" n value is", n)
        labels = np.zeros(n)
        flag = False
        iter = 0
        while (not flag) and (iter < self.max_iter):
            old_center = copy.deepcopy(center)

            for i in range(n):
                # print("--------------------------------------------------")
                # print("the i-th loop", i)
                cur = self.data[i]
                # print("length is: ", len(cur))
                # print("000000000", cur)
                min_dist = 10*9
                for j in range(self.k):
                    dist = 1 - \
                        cosine_similarity(cur.reshape(1, -1),
                                          center[j].reshape(1, -1))
                    # print("node", i, " node", j, " distance", dis)
                    if dist < min_dist:
                        min_dist = dist
                        labels[i] = j
                # print("this node belongs to, ", labels[i])

            # 更新聚类中心
            for j in range(self.k):
                center[j] = np.mean(self.data[labels == j], axis=0)
                center[j] = center[j] / np.linalg.norm(center[j])

            flag = self.converge(old_center, center)
            iter = iter + 1

        return labels
