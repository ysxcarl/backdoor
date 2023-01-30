import logging
import time

import colorlog


class DatasetName():
    TINY = 'TINY'
    MNIST = 'MNIST'
    FASHION = 'FASHION'
    CIFAR = 'CIFAR'
    REDDIT = 'REDDIT'
    IOT_TRAFFIC = 'IOT_TRAFFIC'
    

class LogHandler(object):

    def __init__(self, filename, level=logging.INFO):
        self.logger = logging.getLogger(filename)
        self.log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s  %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
            log_colors=self.log_colors_config)

        # 设置日志级别
        self.logger.setLevel(level)
        # 往屏幕上输出
        console_handler = logging.StreamHandler()
        # 输出到文件
        file_handler = logging.FileHandler(filename=filename, mode='a', encoding='utf8')
        file_formatter = logging.Formatter('%(asctime)s  %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s')
        # 设置屏幕上显示的格式
        console_handler.setFormatter(formatter)
        # 设置写入文件的格式
        file_handler.setFormatter(file_formatter)
        # 把对象加到logger里
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)


def getNowTime()->str:
    return time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())


nowTime = getNowTime()


def getINFOLOG():
    return LogHandler(filename=f'log/filter-{nowTime}.log')


INFO_LOG = LogHandler(filename=f'log/filter-{nowTime}.log')


def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        duration_dict = {}
        elapse = (time.perf_counter() - t) * 1000  # 单位为 ms
        if func.__name__ == 'adaptive_clipping':
            duration_dict['clustering'] = elapse
        elif func.__name__ == 'model_filtering_layer':
            duration_dict['clipping'] = elapse
        
        # print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    return fun


@cost_time
def test():
    print('func start')
    time.sleep(2)
    print('func end')


def get_true_positive_rate(ccaf: list, client_poisoned_all: list, total_client: int = 10) -> float:
    choose_poisoned_cnt = 0
    for client in ccaf:
        if client in client_poisoned_all:
            choose_poisoned_cnt += 1

    return (len(client_poisoned_all) - choose_poisoned_cnt) * 1.0 / (total_client - len(ccaf) + 1e-5)


def get_true_negative_rate(ccaf: list, client_poisoned_all: list, total_client: int = 10) -> float:

    rate = 0.0
    choose_poisoned_cnt = 0
    for client in ccaf:
        if client in client_poisoned_all:
            choose_poisoned_cnt += 1
    rate = (len(ccaf) - choose_poisoned_cnt) * 1.0 / (total_client - len(client_poisoned_all) + 1e-5)
    return rate


if __name__ == '__main__':
    logger = LogHandler.getINFOLogger()
