import sys

import hdbscan
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
np.set_printoptions(threshold=sys.maxsize)


def hamming_distance(weight_list):
    distance = [[0.0 for j in range(len(weight_list))] for i in range(len(weight_list))]
    weight_list = np.where(weight_list == 1, 0, 1)
    for i in range(len(weight_list)):
        for j in range(len(weight_list)):
            if i != j:
                distance[i][j] = np.sum(weight_list[i] + weight_list[j] - 2 * weight_list[i] * weight_list[j])
                # distance[i][j] = 1-2*(distance[i][j]/d)
    return distance


def model_filtering_layer(weight_list, args):
    '''
        模型过滤层
    '''
    
    weight_list = np.array(weight_list, dtype='float64')
    
    # vote_list = weight_list.copy()
    
    if args["sign"]:
        # if args['vote']:
        #     INFO_LOG.logger.info(f'vote_list:{vote_list}')
        #     vote_list = np.array([np.sign(np.sum(weight_list, axis=0))])
        #     distance_matrix1 = cdist(vote_list, weight_list, metric='hamming')
        #     INFO_LOG.logger.info(f'vote_list.shape:\n{vote_list.shape}')
        #     INFO_LOG.logger.info(f'vote hamming:\n{distance_matrix1}')
        
        
        distance_matrix = cdist(weight_list, weight_list, metric='hamming')

        # INFO_LOG.logger.info('after hamming:\n' + str(distance_matrix))
        
    else:
        distance_matrix = cosine_distances(weight_list)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=len(weight_list) // 2 + 1,
                                gen_min_span_tree=True,
                                metric='precomputed', allow_single_cluster=True, alpha=0.9)
    clusterer.fit(distance_matrix)

    a = {}
    for i in range(max(clusterer.labels_) + 1):
        a[i] = clusterer.labels_.tolist().count(i)  # 非噪声类别的各类所含数目
    admitted_label = 0  # 模型过滤层所接收的类别 (取含模型数量最多的聚类)
    for i in a:
        if a[i] == max(a.values()):
            admitted_label = i
            break
    admitted_index = []  # 模型过滤层所接受的模型（ID in weight_list）

    for i in range(len(clusterer.labels_)):  # len(clusterer.labels_) = len(local_choose)
        if clusterer.labels_[i] == admitted_label:
            admitted_index.append(i)

    return admitted_index



def model_filtering_layer_daguard(weight_list, args):
    '''
        模型过滤层
    '''
    
    weight_list = np.array(weight_list, dtype='float64')

    distance_matrix = cdist(weight_list, weight_list, metric='euclidean')

    size = len(weight_list)
    Eps = np.min(np.sort(distance_matrix, axis=1)[:, (size)// 2])

    clusterer = DBSCAN(eps= Eps, min_samples = (size)// 2, metric='precomputed')
    clusterer.fit_predict(distance_matrix)

    a = {}
    for i in range(max(clusterer.labels_) + 1):
        a[i] = clusterer.labels_.tolist().count(i)  # 非噪声类别的各类所含数目
    admitted_label = 0  # 模型过滤层所接收的类别 (取含模型数量最多的聚类)
    for i in a:
        if a[i] == max(a.values()):
            admitted_label = i
            break
    admitted_index = []  # 模型过滤层所接受的模型（ID in weight_list）

    for i in range(len(clusterer.labels_)):  # len(clusterer.labels_) = len(local_choose)
        if clusterer.labels_[i] == admitted_label:
            admitted_index.append(i)

    return admitted_index


def adaptive_clipping(global_weight, weight_list, admitted_index):
    '''
        自适应裁剪
    '''

    E = euclidean_distances([global_weight], weight_list)[0]
    St = np.median(E)
    for i in admitted_index:
        # weight_list[i] = weight_list[i] * min(1, St / E[i])
        weight_list[i] = global_weight + (weight_list[i] - global_weight) * min(1, St / E[i])
    return weight_list, St


def adaptive_clipping_by_hd(global_weight, weight_list, admitted_index):
    ret = []
    H = [0 for i in range(len(weight_list))]
    global_weight = np.where(global_weight == 1, 0, 1)
    weight_list_tmp = np.where(weight_list == 1, 0, 1)
    for i in range(len(weight_list)):
        H[i] = np.sum(global_weight + weight_list_tmp[i] - 2 * global_weight * weight_list_tmp[i])
    St = np.median(H)
    final_choose_model = []
    for i in range(len(weight_list)):
        w = weight_list[i]
        if i in admitted_index:
            w = w * min(1, St / H[i])
        ret.append(w)
    return ret, St


def adaptive_noising(net, alpha):
    total_length = 0
    for p in net.parameters():
        total_length += p.view(-1).shape[0]
    noise = torch.normal(0.0, alpha, [total_length]).to(next(net.parameters()).device)

    start = 0
    # length = 0
    i = 0
    for p in net.parameters():
        length = p.view(-1).shape[0]
        end = start + length
        # if i < 10:
        #     INFO_LOG.logger.info(f"p.data = \n {p.data} \n")
        #     INFO_LOG.logger.info(f"noise = \n {noise[start:end].view_as(p)} \n")
        p.data += noise[start:end].view_as(p)
        start = end
