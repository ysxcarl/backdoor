import copy
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torchvision import datasets, transforms


def build_classes_dict(train_dataset):
    cifar_classes = {}
    for ind, x in enumerate(train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
        _, label = x
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    return cifar_classes


def sample_dirichlet_train_data(classes_dict, no_participants, alpha=0.5):
    cifar_classes = copy.deepcopy(classes_dict)
    class_size = len(cifar_classes[0])  # for cifar: 5000
    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())  # for cifar: 10

    image_nums = []
    for n in range(no_classes):
        image_num = []
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            image_num.append(len(sampled_list))
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
        image_nums.append(image_num)
    return per_participant_list


# 获取数据集
def get_dataset(name):
    train_dataset, test_dataset = [], []
    if name == 'MNIST':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]))
    elif name == 'FASHION':
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]))
    elif name == 'CIFAR':
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差

        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    elif name == 'TINY':
        _data_transforms = {
            'train': transforms.Compose([
                # transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(),
            ]),
        }
        _data_dir = './data/tiny-imagenet-200/'
        train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                             _data_transforms['train'])
        test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                            _data_transforms['val'])
        print('reading TINY done')
    else:  # 其他数据集
        pass
    return train_dataset, test_dataset


# 划分数据集
def divide_dataset(train_dataset, test_dataset, num_of_locals, dirichlet=True, dirichlet_alpha=0.5):
    train_dst_lst, test_dst_lst = [], []

    if dirichlet == True:
        cifar_classes = build_classes_dict(train_dataset)
        per_participant_list = sample_dirichlet_train_data(cifar_classes, num_of_locals, alpha=dirichlet_alpha)
        for i in range(num_of_locals):
            train_dst_lst.append(torch.utils.data.Subset(train_dataset, per_participant_list[i]))
            test_dst_lst.append(test_dataset)
    else:
        train_size = len(train_dataset) // num_of_locals
        # test_size = len(test_dataset) // num_of_locals

        all_range = list(range(len(train_dataset)))
        random.shuffle(all_range)

        for i in range(num_of_locals):
            if i == num_of_locals - 1:
                train_dst_lst.append(torch.utils.data.Subset(train_dataset, all_range[i * train_size:]))
            else:
                train_dst_lst.append(
                    torch.utils.data.Subset(train_dataset, all_range[i * train_size: (i + 1) * train_size]))

            test_dst_lst.append(test_dataset)

    return train_dst_lst, test_dst_lst


def load_data(train_set, test_set, batch_size):
    '''
        加载数据集
    '''
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter


def load_data_lst(train_dst_lst, test_dst_lst, batch_size):
    '''
        批量加载数据集
    '''
    train_iter_list, test_iter_list = [], []
    if len(train_dst_lst) != len(test_dst_lst):
        print("LOAD_DATA_LIST_ERROR")
    else:
        for local in range(len(train_dst_lst)):
            train_iter, test_iter = load_data(train_dst_lst[local], test_dst_lst[local], batch_size)
            train_iter_list.append(train_iter)
            test_iter_list.append(test_iter)
    return train_iter_list, test_iter_list


def get_poision_test_iter(test_dataset, batch_size, target):
    """
    删除测试集中标签为target的数据
    Args:
        test_dataset:
        batch_size:
        target:

    Returns:

    """
    test_classes = {}
    for ind, x in enumerate(test_dataset):
        _, label = x
        if label in test_classes:
            test_classes[label].append(ind)
        else:
            test_classes[label] = [ind]

    range_no_id = list(range(0, len(test_dataset)))
    for image_ind in test_classes[target]:
        if image_ind in range_no_id:
            range_no_id.remove(image_ind)
    poison_label_inds = test_classes[target]

    a_iter = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                             range_no_id))
    return a_iter


def model_and_data(args):
    train_dataset, test_dataset = get_dataset(args['dataset'])
    train_dst_lst, test_dst_lst = divide_dataset(train_dataset, test_dataset, args['num_of_clients'], args['IID'],
                                                 args['IID_rate'])
    train_iter_list, test_iter_list = load_data_lst(train_dst_lst, test_dst_lst, args['batchsize'])
    poison_test_iter = get_poision_test_iter(test_dataset, args['batchsize'], args['target'])
    return train_dataset, test_dataset, train_iter_list, test_iter_list, poison_test_iter
