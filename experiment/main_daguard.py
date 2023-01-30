import copy
import os

import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import defend 
import accuracy
import config
import posion_info
import tools
from Data import model_and_data
from client import create_model
from client import local_train
from filter import *
from util import nowTime, INFO_LOG, get_true_negative_rate, get_true_positive_rate

if __name__ == "__main__":

    parser = config.agg_des()
    args = parser.parse_args()
    INFO_LOG.logger.info(args)
    args = args.__dict__
 
    # 设置训练设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 加载并划分数据集
    train_dataset, test_dataset, train_iter_list, test_iter_list, poison_test_iter = model_and_data(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batchsize'], shuffle=True, num_workers=4)
    # 实例化模型
    global_net = create_model(args).to(dev)
    emptry_net = copy.deepcopy(global_net)
    local_net = copy.deepcopy(global_net)

    global_weight = tools.shape_to_1dim(global_net, single=True)
    w_size = len(global_weight)

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))  # 每轮选择聚合的客户端个数

    # 训练
    print("_________________START TRAIN________________________")
    main_acc_local, main_acc_global = [], []

    backdoor_acc = [[] for i in range(5)]
    true_positive_rate_list = []
    true_negative_rate_list = []

    all_trigger_lst, trigger_of_acs = posion_info.creat_trigger(args)
    epoch_under_attack = list(range(args["attack_epoch_in"], args["num_comm"]))
    acs = list(range(args['num_of_attack']))
    print("acs", acs)
    for epoch in tqdm(range(args['num_comm']), desc='Training: '):
        print('\n')
        # 挑选恶意参与方以及良性参与方
        if_ATTACK = True if epoch in epoch_under_attack else False
        benigns = list(
            np.random.choice(range(args['num_of_attack'], args['num_of_clients']), num_in_comm - args['num_of_attack'],
                             replace=False))
        local_choose = acs + benigns

        local_choose.sort()

        # 开始训练
        local_update_lts = []
        # local_norm_lts = []
        alpha_lts = []
        alphas = []
        global_weight = tools.shape_to_1dim(global_net, single=True)
        for i in range(num_in_comm):
            local = local_choose[i]
            tools.hand_out(global_net, local_net)
            net, train_iter, test_iter = local_net, train_iter_list[local], test_iter_list[local]

            if local in acs and if_ATTACK:  # 毒化
                poison_info = trigger_of_acs[local]
                # momentum = 0.5
                opt = torch.optim.SGD(net.parameters(), lr=poison_info['poison_lr'], momentum=0.9,
                                      weight_decay=args['weight_decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[0.2 * args['local_poison_epoch'],
                                                                                  0.8 * args['local_poison_epoch']],
                                                                 gamma=0.1)
                local_train(net, train_iter, opt, device=dev, num_epochs=args['local_poison_epoch'],
                            poison_info=poison_info, scheduler=scheduler, dataset_name=args['dataset'])
            else:
                opt = torch.optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=0.9,
                                      weight_decay=args['weight_decay'])
                local_train(net, train_iter, opt, device=dev, num_epochs=args['epoch'],
                            dataset_name=args['dataset'])

            local_w = tools.shape_to_1dim(net, single=True)  # update full model
            delta = local_w - global_weight
            local_update_lts.append(np.sign(delta))  # update sign grad
            alphas.append(delta)
            norm1 = np.linalg.norm(delta, ord=1)
          
          
            
            # terngrad
            alpha_lts.append(norm1 / w_size)
            
            # norm2 = np.linalg.norm(delta, ord=2)
            # alpha_lts.append(norm2 / math.sqrt(w_size))  # update sign grad

            # norm3 = np.linalg.norm(delta, inf)
        # print("tern", tern[0])
        # print("yess")
        terns = tools.ternGrad(alphas, emptry_net)
   
        with torch.no_grad():
            if not if_ATTACK:
                benign_w = []
                St = np.median(alpha_lts)
                for local_update, alpha_lts in zip(local_update_lts, alpha_lts):
                    benign_w.append(local_update * min(alpha_lts, St))

                global_weight += sum(benign_w)
                global_net = tools.shape_back_to(global_weight, global_net)
            else:
                # print('\n', alpha_lts)
                # 模型过滤层
                if args['defend'] == 'daguard':
                    INFO_LOG.logger.info(f'clients index:  {local_choose}')
                    admitted_index_in_wlst = model_filtering_layer_daguard(terns, args)
                    admitted_index = []
                    for i in admitted_index_in_wlst:
                        admitted_index.append(local_choose[i])
                    admitted_index.sort()
                    INFO_LOG.logger.info(f'choose model:  {admitted_index}')

                    # 自适应裁剪
                    St = np.median(alpha_lts)
                    benign_w = []
                    for i in admitted_index_in_wlst:
                        benign_w.append(local_update_lts[i] * min(alpha_lts[i], St))

                    # 聚合
                    # print(len(benign_w))
                    avg_grad = sum(benign_w)
                    # avg_grad = np.mean(benign_w)
                    global_weight += avg_grad
                    global_net = tools.shape_back_to(global_weight, global_net)

                    # 自适应噪声
                    if args['noise']:
                        alpha = args['λ'] * St * math.sqrt(w_size)
                        adaptive_noising(global_net, alpha)

                elif args['defend'] == 'median':
                    benign_w = defend.median(alphas)
                    global_weight += benign_w
                    global_net = tools.shape_back_to(global_weight, global_net)

                elif args['defend'] == 'avg':
                    benign_w = defend.average(alphas)
                    global_weight += benign_w
                    global_net = tools.shape_back_to(global_weight, global_net)

                elif args['defend'] == 'trimmed_mean':
                    benign_w = defend.trimmed_mean(alphas)
                    global_weight += benign_w
                    global_net = tools.shape_back_to(global_weight, global_net)
             

        if not if_ATTACK or args['defend'] != 'daguard':
            true_negative_rate, true_positive_rate = 1.0, 1.0
        else:
            true_positive_rate = get_true_positive_rate(admitted_index, acs)
            true_negative_rate = get_true_negative_rate(admitted_index, acs)

        true_positive_rate_list.append(true_positive_rate)
        true_negative_rate_list.append(true_negative_rate)

        # 全局模型对主任务的精度
        acc = accuracy.evaluate_accuracy(
            test_dataloader,
            global_net,
            dataset_name=args['dataset']
        )
        main_acc_global.append(acc)

        # 全局模型对后门任务的精度
        for i in range(5):
            acc = accuracy.evaluate_accuracy(poison_test_iter, global_net, poison_info=all_trigger_lst[i],
                                             dataset_name=args['dataset'])
            backdoor_acc[i].append(acc)

        # 记录每轮全局模型的后门精度
        INFO_LOG.logger.info(f'true_positive_rate: {true_positive_rate_list}')
        INFO_LOG.logger.info(f'true_negative_rate: {true_negative_rate_list}')
        INFO_LOG.logger.info(f'main_take_acc: {main_acc_global}')
        INFO_LOG.logger.info(f'global_trigger_acc: {backdoor_acc[0]}')

    # save result, 请提前创建对应文件夹 {args['dataset']}/
    save_path = f"./tmp/daguard/{args['IID_rate']}-{args['defend']}-{args['dataset']}-{nowTime}-{args['num_of_attack']}-posion_rate({trigger_of_acs[0]['poison_rate']}).pth"
    torch.save(
        [
            main_acc_global,
            backdoor_acc,
            true_positive_rate_list,
            true_negative_rate_list
        ],
        save_path
    )
    INFO_LOG.logger.info(f"the result are save in {save_path}")
    INFO_LOG.logger.info(args)
