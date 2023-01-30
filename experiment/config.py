import argparse


def agg_des():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")

    """
    未用到的一些超参数，在此对其进行注释

    # parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')

    # parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
    # parser.add_argument('-sf', '--save_freq', type=int, default=100,
                        # help='global model save frequency(of communication)')                       
    # parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    # parser.add_argument('-eta', '--eta', type=float, default=0.02, help='eta')
    
    # parser.add_argument('-WM', '--WEIGHTED_MEAN', type=bool, default=False, help='use weight mean or not')

    # 是否对模型进行预训练
    # parser.add_argument('-pretrain', '--pretrain', type=bool, default=False)
    # # 预训练模型所存储的路径
    # parser.add_argument('-pmp', '--pretrain_model_path', type=str,
    #                     default='tmp/pretrain/pretrain_CIFAR_resnet18_0.745_0.04.pth',
    #                     help='the path of pretrain model')

    """
    parser.add_argument('-name', '--name', type=str, default="", help='target')
    
    parser.add_argument('-defend', '--defend', type=str, default="daguard", help='defend')
    
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

    parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')

    parser.add_argument('-iid', '--IID', type=bool, default=True, help='the way to allocate data to clients')
    parser.add_argument('-iid_rate', '--IID_rate', type=float, default=0.75, help='the rate of iid data')

    # 总共的客户端数量
    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    # 每次参与的客户端数量
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    # 恶意客户端的数量
    parser.add_argument('-NA', '--num_of_attack', type=int, default=4, help='number of attacker')
    # 被植入后门攻击的target
    parser.add_argument('-target', '--target', type=int, default=2, help='target')

    # 数据集
    parser.add_argument('-dataset', '--dataset', type=str, default='MNIST', help='dataset will be used')

    # 服务器与客户端通信次数
    parser.add_argument('-ncomm', '--num_comm', type=int, default=110, help='number of communications')

    # 在第几轮开始进行攻击
    parser.add_argument('-AE', '--attack_epoch_in', type=int, default=10, help='number of under attack epoch')

    # benign 客户端的学习率
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-WD', '--weight_decay', type=float, default=0.0005, help='weight decay')
    # benign 客户端的训练轮数
    parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')

    # 恶意客户端的训练轮数
    parser.add_argument('-LPE', '--local_poison_epoch', type=int, default=10, help='local poison train epoch')
    # 恶意客户端的投毒率 0.46875  0.3125 0.15625
    parser.add_argument('-pr', '--poison_rate', type=float, default=0.3125, help='the poison rate of malicious client')

    # 选择实验的方案，sign=True：使用eFLAME方案， sign=False：使用FLAME方案
    parser.add_argument('-sign', '--sign', type=bool, default=True, help='use signSGD to aggregation')
    # 选用模型
    parser.add_argument('-mn', '--model_name', type=str, default="alexnet", help='choose model')

    # 是否裁剪
    parser.add_argument('-clip', '--clip', type=bool, default=True)
    # 是否加噪
    parser.add_argument('-noise', '--noise', type=bool, default=True)
    parser.add_argument('-λ', '--λ', type=float, default=0.0001, help='λ')


    # Makes aggregation more efficient by vote
    parser.add_argument('-v', '--vote', type=bool, default=False, help="Whether to use voting")
    return parser

def agg_f():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")

    """
    未用到的一些超参数，在此对其进行注释

    # parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')

    # parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
    # parser.add_argument('-sf', '--save_freq', type=int, default=100,
                        # help='global model save frequency(of communication)')                       
    # parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    # parser.add_argument('-eta', '--eta', type=float, default=0.02, help='eta')
    
    # parser.add_argument('-WM', '--WEIGHTED_MEAN', type=bool, default=False, help='use weight mean or not')

    # 是否对模型进行预训练
    # parser.add_argument('-pretrain', '--pretrain', type=bool, default=False)
    # # 预训练模型所存储的路径
    # parser.add_argument('-pmp', '--pretrain_model_path', type=str,
    #                     default='tmp/pretrain/pretrain_CIFAR_resnet18_0.745_0.04.pth',
    #                     help='the path of pretrain model')

    """
    parser.add_argument('-name', '--name', type=str, default="", help='target')
    
    parser.add_argument('-defend', '--defend', type=str, default="ternGrad", help='defend')
    
    parser.add_argument('-g', '--gpu', type=str, default='1', help='gpu id to use(e.g. 0,1,2,3)')

    parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')

    parser.add_argument('-iid', '--IID', type=bool, default=True, help='the way to allocate data to clients')
    parser.add_argument('-iid_rate', '--IID_rate', type=float, default=0.25, help='the rate of iid data')

    # 总共的客户端数量
    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    # 每次参与的客户端数量
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    # 恶意客户端的数量
    parser.add_argument('-NA', '--num_of_attack', type=int, default=4, help='number of attacker')
    # 被植入后门攻击的target
    parser.add_argument('-target', '--target', type=int, default=2, help='target')

    # 数据集
    parser.add_argument('-dataset', '--dataset', type=str, default='FASHION', help='dataset will be used')

    # 服务器与客户端通信次数
    parser.add_argument('-ncomm', '--num_comm', type=int, default=110, help='number of communications')

    # 在第几轮开始进行攻击
    parser.add_argument('-AE', '--attack_epoch_in', type=int, default=10, help='number of under attack epoch')

    # benign 客户端的学习率
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-WD', '--weight_decay', type=float, default=0.0005, help='weight decay')
    # benign 客户端的训练轮数
    parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')

    # 恶意客户端的训练轮数
    parser.add_argument('-LPE', '--local_poison_epoch', type=int, default=10, help='local poison train epoch')
    # 恶意客户端的投毒率 0.46875  0.3125  =0.15625
    parser.add_argument('-pr', '--poison_rate', type=float, default=0.46875, help='the poison rate of malicious client')

    # 选择实验的方案，sign=True：使用eFLAME方案， sign=False：使用FLAME方案
    parser.add_argument('-sign', '--sign', type=bool, default=False, help='use signSGD to aggregation')
    # 选用模型
    parser.add_argument('-mn', '--model_name', type=str, default="alexnet", help='choose model')

    # 是否裁剪
    parser.add_argument('-clip', '--clip', type=bool, default=True)
    # 是否加噪
    parser.add_argument('-noise', '--noise', type=bool, default=True)
    parser.add_argument('-λ', '--λ', type=float, default=0.0001, help='λ')


    # Makes aggregation more efficient by vote
    parser.add_argument('-v', '--vote', type=bool, default=False, help="Whether to use voting")
    return parser


def agg_flame():
    """
    暂时用不到的参数
    # 是否对模型进行预训练
    # parser.add_argument('-pretrain', '--pretrain', type=bool, default=False)
    # 预训练模型所存储的路径
    # parser.add_argument('-pmp', '--pretrain_model_path', type=str,
    #                     default='tmp/pretrain/pretrain_CIFAR_resnet18_0.745_0.04.pth',
    #                     help='the path of pretrain model')
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")

    parser.add_argument('-g', '--gpu', type=str, default='1', help='gpu id to use(e.g. 0,1,2,3)')

    parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')

    parser.add_argument('-iid', '--IID', type=bool, default=False, help='the way to allocate data to clients')
    parser.add_argument('-iid_rate', '--IID_rate', type=float, default=0.5, help='the rate of iid data')

    # 选用模型
    parser.add_argument('-mn', '--model_name', type=str, default="alexnet", help='choose model')

    # 总共的客户端数量
    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    # 每次参与的客户端数量
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    # 恶意客户端的数量
    parser.add_argument('-NA', '--num_of_attack', type=int, default=4, help='number of attacker')

    parser.add_argument('-target', '--target', type=int, default=2, help='target')
    # 服务器与客户端通信次数
    parser.add_argument('-ncomm', '--num_comm', type=int, default=500, help='number of communications')
    # 在第几轮开始进行攻击
    parser.add_argument('-AE', '--attack_epoch_in', type=int, default=50, help='number of under attack epoch')

    # 数据集
    parser.add_argument('-dataset', '--dataset', type=str, default='CIFAR', help='dataset will be used')

    # benign 客户端的学习率
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-WD', '--weight_decay', type=float, default=0.0005, help='weight decay')
    # benign 客户端的训练轮数
    parser.add_argument('-E', '--epoch', type=int, default=2, help='local train epoch')

    # 恶意客户端的训练轮数
    parser.add_argument('-LPE', '--local_poison_epoch', type=int, default=6, help='local poison train epoch')
    # 恶意客户端的投毒率
    parser.add_argument('-pr', '--poison_rate', type=float, default=0.3125, help='the poison rate of malicious client')

    # 选择实验的方案，sign=True：使用eFLAME方案， sign=False：使用FLAME方案
    parser.add_argument('-sign', '--sign', type=bool, default=False, help='use signSGD to aggregation')

    # 是否裁剪
    parser.add_argument('-clip', '--clip', type=bool, default=True)
    # 是否加噪
    parser.add_argument('-noise', '--noise', type=bool, default=True)
    parser.add_argument('-λ', '--λ', type=float, default=0.0001, help='λ')

    return parser


def agg_pretrain():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")

    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

    parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')

    parser.add_argument('-iid', '--IID', type=bool, default=True, help='the way to allocate data to clients')
    parser.add_argument('-iid_rate', '--IID_rate', type=float, default=1, help='the rate of iid data')

    # 总共的客户端数量
    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    # # 每次参与的客户端数量
    # parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
    #                     help='C fraction, 0 means 1 client, 1 means total clients')
    # # 恶意客户端的数量
    # parser.add_argument('-NA', '--num_of_attack', type=int, default=4, help='number of attacker')

    parser.add_argument('-target', '--target', type=int, default=2, help='target')

    # 服务器与客户端通信次数
    parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')

    # 数据集
    parser.add_argument('-dataset', '--dataset', type=str, default='CIFAR', help='dataset will be used')

    # 选用模型
    parser.add_argument('-mn', '--model_name', type=str, default="resnet18", help='choose model')

    # benign 客户端的学习率
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, \
                            use value from origin paper as default")
    parser.add_argument('-WD', '--weight_decay', type=float, default=0.0005, help='weight decay')

    # benign 客户端的训练轮数
    # parser.add_argument('-E', '--epoch', type=int, default=2, help='local train epoch')

    # parser.add_argument('-pretrain', '--pretrain', type=bool, default=False)

    return parser
