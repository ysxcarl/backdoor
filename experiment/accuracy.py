import copy

import torch

from client import poison_batch
from tools import shape_back


def evaluate_accuracy(data_iter, net, dataset_name, device=None, poison_info={}, ):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if len(poison_info):  # 毒化
                # print(poison_info)
                p_info = copy.deepcopy(poison_info)
                p_info['poison_rate'] = 1
                X, y = poison_batch(X, y, p_info, dataset_name=dataset_name)

            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                print("EVALUATER_ERROR")
            n += y.shape[0]
    return acc_sum / n


def predict_accuracy(Y, X, data_size_lst, aim_net, data_iter, poison_info, data_percentage, dataset_name,
                     device='cuda'):
    # Y: 攻击者良性训练得到的更新
    # X：攻击者最终提交的恶意更新
    assert len(Y) == len(X)
    assert len(Y[0]) == len(X[0])
    n = len(Y[0])
    Op = torch.zeros(n).float().to(device)
    Oq = torch.zeros(n).float().to(device)
    attack_data_size = 0
    for i in data_size_lst:
        attack_data_size += i
    for i in range(len(Y)):
        Op += torch.tensor(Y[i] * (data_size_lst[i] / attack_data_size)).float().to(device)
        Oq += torch.tensor(X[i] * (data_size_lst[i] / attack_data_size)).float().to(device)
    W = []
    # W[0]对应全部攻击者都突破的情况; W[-1]对应全部攻击者都未突破的情况,即不发动攻击时的预测权重W_benign
    for percent in data_percentage:
        W.append(shape_back(Op * (1 - percent) + Oq * (percent), aim_net, single=True))

    aim_net = copy.deepcopy(aim_net)
    aim_net = aim_net.to(device)
    ba_prediction = []
    for weight in W:
        with torch.no_grad():
            for p, w_p in zip(aim_net.parameters(), weight):
                w_p = w_p.to(device)
                p.data = w_p.data.clone().detach()
        ba_prediction.append(evaluate_accuracy(data_iter, aim_net, poison_info=poison_info, dataset_name=dataset_name))

    return ba_prediction