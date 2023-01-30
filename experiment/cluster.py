import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

from  sklearn.cluster import DBSCAN



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
    if args['dist'] == "cos":
        distance_matrix = cosine_distances(weight_list)
    else:
        distance_matrix = hamming_distance(weight_list)
        distance_matrix = np.array(distance_matrix, dtype='float64')

    clusterer = hdbscan.HDBSCAN(min_cluster_size=len(weight_list) // 2 + 1, gen_min_span_tree=True,
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
  
    distance_matrix = cosine_distances(weight_list)
    Eps = np.median(distance_matrix)
    clusterer = DBSCAN(eps= Eps, min_sample = len(weight_list) // 2 + 1,  metric='precomputed')
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=len(weight_list) // 2 + 1, gen_min_span_tree=True,
    #                             metric='precomputed', allow_single_cluster=True, alpha=0.9)
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
