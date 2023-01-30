import copy
import torch
import numpy as np


def get_copies(object, nums):
    copies = []
    for i in range(nums):
        copies.append(copy.deepcopy(object))
    return copies


def shape_back(list_of_weights, aim_net, single=False):
    ret = []

    if single:
        list_of_weights = [list_of_weights]

    for i in list_of_weights:
        reshaped = []
        start = 0
        for p in aim_net.parameters():
            # length = len(p.view(-1))
            end = start + len(p.view(-1))
            ################
            if type(i) == torch.Tensor:
                reshaped.append(i.clone().detach().view(-1)[start:end].view_as(p))
            else:
                reshaped.append(torch.tensor(i).view(-1)[start:end].view_as(p))

            start = end
        ret.append(reshaped)

    if single:
        ret = ret[0]

    return ret

def ternGrad(list_of_weights, aim_net, single=False):
    ret = []
    for i in list_of_weights:
        reshaped = np.array([])
        start = 0
        for p in aim_net.parameters():
            end = start + len(p.view(-1))
            tmp = i[start:end]
            maxn = np.linalg.norm(tmp, ord = np.inf)
            tern = np.round(tmp / maxn) * maxn
            reshaped = np.hstack((reshaped, tern))
            start = end
        ret.append(reshaped)
    return ret



def shape_back_to(weights, aim_net):  #
    params = shape_back(weights, aim_net, single=True)
    device = next(aim_net.parameters()).device
    for aim_p, new_p in zip(aim_net.parameters(), params):
        new_p = new_p.to(device)
        aim_p.data = new_p.data.clone().detach()
    return aim_net




def shape_to_1dim(list_of_nets, choose_index=[], single=False, device=None):
    ret = []

    if single:
        list_of_nets = [list_of_nets]

    if len(choose_index) == 0:
        choose_index = list(range(len(list_of_nets)))

    if device is None and isinstance(list_of_nets[choose_index[0]], torch.nn.Module):
        device = list(list_of_nets[choose_index[0]].parameters())[0].device

    for choose in choose_index:
        w = torch.rand(0).view(-1).to(device)
        with torch.no_grad():
            for param in list_of_nets[choose].parameters():
                w = torch.cat((w, param.view(-1)), 0).float()
                w = torch.where(torch.isnan(w), torch.full_like(w, 0), w)
            ret.append(np.array(w.cpu()))
            # ret.append(w)
    if single:
        ret = ret[0]

    return ret


def hand_out(source_net, aim_net):
    assert next(source_net.parameters()).device == next(aim_net.parameters()).device
    with torch.no_grad():
        for g_p, l_p in zip(source_net.parameters(), aim_net.parameters()):
            l_p.data = g_p.data.clone().detach()
    return aim_net


def showVisdom(viz, x, y, win, legend, title='', update='append', xlabel='x', ylabel='y', width=300, height=200):
    if title == '':
        title = win

    opts = {
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "width": width,
        "height": height,
        "legend": legend,
    }

    viz.line(X=[x], Y=[y], win=win, update=update, opts=opts)
    return
