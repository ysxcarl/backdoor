# from crypten import mpc
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm

import accuracy
import config
import posion_info
import tools
from Data import model_and_data
from client import create_model
from client import local_train
# from private_filter import fun_test
from filter import *
from util import *


def run(args, dev):
    # 加载并划分数据集
    train_dataset, test_dataset, train_iter_list, test_iter_list, poison_test_iter = model_and_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args['batchsize'], shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batchsize'], shuffle=True, num_workers=4)
    # 实例化模型
    global_net = create_model(args).to(dev)
    # global_net = torch.nn.DataParallel(global_net)  # 多块gpu共同训练

    local_net = copy.deepcopy(global_net)

    global_weight = tools.shape_to_1dim(global_net, single=True)

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))  # 每轮选择聚合的客户端个数

    # 训练
    print("_________________START TRAIN________________________")
    main_acc_local, main_acc_global = [], []
    backdoor_acc = [[] for _ in range(5)]
    true_positive_rate_list = []
    true_negative_rate_list = []
    train_acc_global = []
    all_trigger_lst, trigger_of_acs = posion_info.creat_trigger(args)
    # epoch_under_attack = list(np.random.choice(range(args['num_comm']), int(args['attack_epoch_num']), replace=False))
    epoch_under_attack = list(range(args["attack_epoch_in"], args["num_comm"]))
    acs = list(range(args['num_of_attack']))

    for epoch in tqdm(range(args['num_comm']), desc='Training: ', colour='green'):
        # 挑选恶意参与方以及良性参与方
        if_ATTACK = True if epoch in epoch_under_attack else False
        benigns = list(
            np.random.choice(
                range(args['num_of_attack'], args['num_of_clients']),
                num_in_comm - args['num_of_attack'],
                replace=False
            )
        )
        local_choose = acs + benigns

        local_choose.sort()

        # 开始训练
        local_update_lts = []
        global_weight = tools.shape_to_1dim(global_net, single=True)
        for i in range(num_in_comm):
            local = local_choose[i]
            tools.hand_out(global_net, local_net)
            net, train_iter = local_net, train_iter_list[local]

            if local in acs and if_ATTACK:  # 毒化
                # INFO_LOG.logger.info("Attack")
                poison_info = trigger_of_acs[local]
                opt = torch.optim.SGD(
                    net.parameters(),
                    lr=poison_info['poison_lr'],
                    momentum=0.9,
                    weight_decay=args['weight_decay']
                )
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt,
                    milestones=[0.2 * args['local_poison_epoch'], 0.8 * args['local_poison_epoch']],
                    gamma=0.1
                )
                local_train(
                    net, train_iter, opt, device=dev,
                    num_epochs=args['local_poison_epoch'],
                    poison_info=poison_info,
                    scheduler=scheduler,
                    dataset_name=args['dataset']
                )
            else:
                opt = torch.optim.SGD(
                    net.parameters(),
                    lr=args['learning_rate'], momentum=0.9,
                    weight_decay=args['weight_decay']
                )
                local_train(
                    net, train_iter, opt, device=dev,
                    num_epochs=args['epoch'],
                    dataset_name=args['dataset']
                )
            # local_update_lts.shape = (10, 2797610)
            local_update_lts.append(tools.shape_to_1dim(net, single=True))  # update full model

        if not if_ATTACK:

            admitted_index = local_choose
            # local_net_params.shape = (10, 62)
            local_net_params = tools.shape_back(local_update_lts, local_net)

            sum_params = np.array(local_net_params).sum(axis=0)
            for g_p, l_sum_p in zip(global_net.parameters(), sum_params):
                l_sum_p /= len(admitted_index)  # FedAvg
                l_sum_p = l_sum_p.to(dev)
                g_p.data = l_sum_p.data
        else:
            with torch.no_grad():
                # 模型过滤层
                print("clients index: ", local_choose)
                # print(local_update_lts)
                admitted_index_in_wlst = model_filtering_layer(local_update_lts, args)
                # print("cluster result:", admitted_index_in_wlst, hamming_index_in_wlst)

                admitted_index = []
                for i in admitted_index_in_wlst:
                    admitted_index.append(local_choose[i])
                admitted_index.sort()
                print("choose model: ", admitted_index)

                # 自适应裁剪
                if args['clip']:
                    local_update_lts, St = adaptive_clipping(global_weight, local_update_lts, admitted_index_in_wlst)

                # filter
                benign_w = []
                for i in admitted_index_in_wlst:
                    benign_w.append(local_update_lts[i])
                local_net_params = tools.shape_back(benign_w, local_net)

                # 聚合更新全局模型
                sum_params = np.array(local_net_params).sum(axis=0)
                # sum_params = local_net_params[0]
                # for i in range(1, len(local_net_params)):
                #     for s, l in zip(sum_params, local_net_params[i]):
                #         s.data += l.data

                for g_p, l_sum_p in zip(global_net.parameters(), sum_params):
                    l_sum_p /= len(admitted_index)  # FedAvg
                    l_sum_p = l_sum_p.to(dev)
                    g_p.data = l_sum_p.data

                # 自适应加噪
                if args['noise']:
                    alpha = args['λ'] * St
                    adaptive_noising(global_net, alpha)

        if not if_ATTACK:
            true_negative_rate, true_positive_rate = 1.0, 1.0
        else:
            true_positive_rate = get_true_positive_rate(admitted_index, acs)
            true_negative_rate = get_true_negative_rate(admitted_index, acs)

        true_positive_rate_list.append(true_positive_rate)
        true_negative_rate_list.append(true_negative_rate)

        INFO_LOG.logger.info(f'true_positive_rate: {true_positive_rate_list}')
        INFO_LOG.logger.info(f'true_negative_rate: {true_negative_rate_list}')

        # 全局模型对主任务的精度

        train_acc = accuracy.evaluate_accuracy(
            train_dataloader,
            global_net,
            dataset_name=args['dataset']
        )
        train_acc_global.append(train_acc)

        test_acc = accuracy.evaluate_accuracy(
            test_dataloader,
            global_net,
            dataset_name=args['dataset']
        )
        main_acc_global.append(test_acc)

        # 全局模型对后门任务的精度

        for i in range(5):
            acc = 0.0
            if if_ATTACK:
                acc = accuracy.evaluate_accuracy(
                    poison_test_iter,
                    global_net,
                    poison_info=all_trigger_lst[i],
                    dataset_name=args['dataset']
                )
            backdoor_acc[i].append(acc)

        # 记录每轮全局模型的后门精度
        INFO_LOG.logger.info(f'train_acc: {train_acc_global}')
        INFO_LOG.logger.info(f'main_take_acc: {main_acc_global}')
        INFO_LOG.logger.info(f'global_trigger_acc: {backdoor_acc[0]}')

    # save result

    save_path = f"tmp/flame_result/{args['IID_rate']}-{args['dataset']}-poison_rate-{trigger_of_acs[0]['poison_rate']}-{getNowTime()}-{args['learning_rate']}.pth"
    INFO_LOG.logger.info(f'save result in file:\n {save_path}')

    torch.save(
        [
            train_acc_global,
            main_acc_global,
            backdoor_acc,
            true_positive_rate_list,
            true_negative_rate_list
        ],
        save_path
    )


if __name__ == '__main__':

    parser = config.agg_f()
    args = parser.parse_args()
    args = args.__dict__
    INFO_LOG.logger.info(args)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run(args, dev)
    INFO_LOG.logger.info(args)
