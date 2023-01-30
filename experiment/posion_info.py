
def creat_trigger(args):
    dataset_name = args['dataset']
    poison_rate = args['poison_rate']
    if dataset_name == 'MNIST' or dataset_name == 'FASHION':
        pattern0 = [[0, 0], [0, 1], [0, 2], [0, 3]]
        pattern1 = [[0, 6], [0, 7], [0, 8], [0, 9]]
        pattern2 = [[3, 0], [3, 1], [3, 2], [3, 3]]
        pattern3 = [[3, 6], [3, 7], [3, 8], [3, 9]]
        target = 2  # DBA原文 为 2

        # poison_rate = 20 / 64  # 20/64
        poison_lr = 0.05

    elif dataset_name == 'CIFAR':
        pattern0 = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
        pattern1 = [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
        pattern2 = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
        pattern3 = [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
        target = 2

        # poison_rate = 10 / 64  # 5/64
        poison_lr = 0.05
    elif dataset_name == 'TINY':
        pattern0 = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [1, 4], [0, 5], [1, 5],
                    [0, 6], [1, 6], [0, 7], [1, 7], [0, 8], [1, 8], [0, 9], [1, 9]]
        pattern1 = [[0, 12], [1, 12], [0, 13], [1, 13], [0, 14], [1, 14], [0, 15], [1, 15], [0, 16], [1, 16], [0, 17],
                    [1, 17], [0, 18], [1, 18], [0, 19], [1, 19], [0, 20], [1, 20], [0, 21], [1, 21]]
        pattern2 = [[4, 0], [5, 0], [4, 1], [5, 1], [4, 2], [5, 2], [4, 3], [5, 3], [4, 4], [5, 4], [4, 5], [5, 5],
                    [4, 6], [5, 6], [4, 7], [5, 7], [4, 8], [5, 8], [4, 9], [5, 9]]
        pattern3 = [[4, 12], [5, 12], [4, 13], [5, 13], [4, 14], [5, 14], [4, 15], [5, 15], [4, 16], [5, 16], [4, 17],
                    [5, 17], [4, 18], [5, 18], [4, 19], [5, 19], [4, 20], [5, 20], [4, 21], [5, 21]]
        target = 2
        # poison_rate = 0.3125
        poison_lr = 0.001
    pattern_global = pattern0 + pattern1 + pattern2 + pattern3

    trigger_0 = {
        'pattern': pattern0,
        'type': '0',
        'poison_rate': poison_rate,
        'target': target,
        'poison_lr': poison_lr
    }
    trigger_1 = {
        'pattern': pattern1,
        'type': '1',
        'poison_rate': poison_rate,
        'target': target,
        'poison_lr': poison_lr
    }
    trigger_2 = {
        'pattern': pattern2,
        'type': '2',
        'poison_rate': poison_rate,
        'target': target,
        'poison_lr': poison_lr
    }
    trigger_3 = {
        'pattern': pattern3,
        'type': '3',
        'poison_rate': poison_rate,
        'target': target,
        'poison_lr': poison_lr
    }
    trigger_global = {
        'pattern': pattern_global,
        'type': 'global',
        'poison_rate': poison_rate,
        'target': target,
        'poison_lr': poison_lr
    }
    all_trigger_lst = [trigger_global, trigger_0, trigger_1, trigger_2, trigger_3]
    trigger_of_acs = {
        0: trigger_0,
        1: trigger_1,
        2: trigger_2,
        3: trigger_3
    }
  
    return all_trigger_lst, trigger_of_acs
