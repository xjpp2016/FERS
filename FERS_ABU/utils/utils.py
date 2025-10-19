import time
import random
import os
import numpy as np
import torch
import re
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from utils.RX import RX_Torch, RX

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()

def write_eval_result(txt, name_list, AU_ROCs, anomaly_ids, write_mode = 'w'):
    f = open(txt, write_mode)
    for i in range(len(anomaly_ids)):
        id = anomaly_ids[i]
        data_path = name_list[id]
        name = data_path.split('/')[-1].split('.')[0]
        auc =AU_ROCs[i]
        f.write('{:}:{:} \n'.format(name,auc))
    f.close()

def write_mean_result(txt, mean_auc, write_mode = 'w'):
    f = open(txt, write_mode)
    f.write('beset_mean_auc:{:} \n'.format(mean_auc))
    f.close()

def write_name(txt, name_list):
    f = open(txt, 'w')
    for i in range(len(name_list)):
        data_path = name_list[i]
        name =re.split(r'[\\/.]', data_path)[-2]
        f.write('scene{:d}:{:}\n'.format(i+1,name))
    f.close()


def seed_torch(seed=50):
    import random, os, numpy as np, torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Force all operations to be deterministic (if possible)
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception as e:
    #     print(f"Warning: Deterministic algorithms not fully enforced: {e}")

def seed_fix(args):
    if args.input_channel == 200 and args.sensor == 'aviris_ng':
        args.seed = 30
    elif args.input_channel == 100 and args.sensor == 'aviris_ng':
        args.seed = 820
    elif args.input_channel == 50 and args.sensor == 'aviris_ng':
        args.seed = 290
    elif args.input_channel == 200 and args.sensor == 'aviris':
        args.seed = 850
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_checkpoint(model_path, model):
    state = {
        'state_dict': model.state_dict(),
    }
    torch.save(state, model_path)


def save_pseudo_rgb(path, data, img_idx):

    # 假设 data.shape = (1, bands, H, W)
    img = data[0].cpu().numpy()  # (bands, H, W)

    # 选取3个波段组成RGB
    rgb = np.stack([
        img[160],  # R
        img[90],  # G
        img[50]   # B
    ], axis=-1)  # (H, W, 3)

    # 归一化到0-1范围
    rgb -= rgb.min()
    rgb /= (rgb.max() + 1e-8)

    # 保存伪彩色图
    plt.imsave(path + str(img_idx) + '.png', rgb)

    from sklearn.decomposition import PCA

    h, w = img.shape[1:]
    img_2d = img.reshape(img.shape[0], -1).T  # (H*W, bands)

    # PCA降到3个通道
    pca = PCA(n_components=3)
    img_pca = pca.fit_transform(img_2d)
    img_pca = img_pca.reshape(h, w, 3)

    # 归一化
    img_pca -= img_pca.min()
    img_pca /= (img_pca.max() + 1e-8)

    plt.imsave(path + str(img_idx) + '.png', img_pca)
    


class LinearDecayLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, initial_lr, min_lr, total_epochs):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.decay_rate = (initial_lr - min_lr) / total_epochs
        super(LinearDecayLR, self).__init__(optimizer)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        lr = max(self.initial_lr - self.decay_rate * current_epoch, self.min_lr)
        return [lr for _ in self.optimizer.param_groups]

def S2One(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    min_max_normalized = (tensor - min_val) / (max_val - min_val)
    return min_max_normalized

def get_admap(data, spa_f, spe_f, tpype=None):

    if tpype=='ham':
        rx_s = RX_Torch(spa_f+data)
        rx_p = RX_Torch(spe_f+data)

        rx_s = S2One(rx_s)
        rx_p = S2One(rx_p)
        rx_r = (rx_s*rx_p)
        rx_r = rx_r.detach().cpu().numpy()

    else:
        rx_r = RX(spa_f+spe_f+data)

    return rx_r