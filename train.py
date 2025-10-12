import os
import random
import argparse
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets.HADDatasets import HADDataset, HADTestDataset, HADDataset_Aug
from models.test.resnet import ConvH, wide_resnet101_2, Pixel_Classifier
from models.spa_branch import HyperSpatialResNet, FeatureRestorationNet
from utils.utils import time_string, convert_secs2time, AverageMeter, print_log, save_checkpoint, write_eval_result, write_mean_result, set_seed, seed_fix, get_admap
from losses.losses import CrossCovarianceLoss, variance_preserve_loss, cos_sim_loss
from kornia.losses import SSIMLoss
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def main():

    parser = argparse.ArgumentParser(description='hyperspectral anomaly detection')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--mask_class', type=str, default='zero', help='spectrum pasted on masks, no, zero, image, sin, other_sensor')
    parser.add_argument('--mask_type', type=str, default='random', help='the type of the spectrum paste, none or random')
    parser.add_argument('--data_path', type=str, default='./data/HAD100Dataset/')
    parser.add_argument('--start_channel_id', type=int, default=0, help='the start id of spectral channel')
    parser.add_argument('--input_channel', type=int, default=200, help='the spectral channel number of input HSI')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60, help='the maximum of training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    parser.add_argument('--seed_fix', choices=['True', 'False'], default='True')
    parser.add_argument('--train_ratio', type=float, default=1, help='data ratio used for training')
    parser.add_argument('--sensor', type=str, default='aviris_ng',help='sensor used in training,  aviris_ng or aviris  test')
    parser.add_argument('--save_txt', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./saved_models/')
    args = parser.parse_args()
    
    # manual seed
    args.seed_fix = (args.seed_fix == 'True') 
    if args.seed_fix:
        args = seed_fix(args)
    set_seed(seed=args.seed)

    # build save path  
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, str(args.input_channel) + 'bands/')    
    args.save_dir = os.path.join('./result/', str(args.input_channel) + 'bands_'
                                 +str(args.sensor))
    epoch_write_dir = os.path.join(args.save_dir, 'epoch')
    if not os.path.exists(epoch_write_dir):
        os.makedirs(epoch_write_dir)
    
    args.saved_models_dir = os.path.join('./result/', str(args.input_channel) + 'bands_'
                                 +str(args.sensor) + '_' + 'seed_' + str(args.seed) + '/saved_models')
    if not os.path.exists(args.saved_models_dir):
        os.makedirs(args.saved_models_dir)

    log = open(os.path.join(args.save_dir, 'training_log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)


    # load fix F model
    convh = ConvH(input_channel=args.input_channel)
    encoder, _ = wide_resnet101_2(pretrained=False)
    classifier = Pixel_Classifier(input_channel=args.input_channel)

    convh_checkpoint = torch.load(args.checkpoint_dir + 'convh.pt', map_location=torch.device('cpu'))
    enc_checkpoint = torch.load(args.checkpoint_dir + 'enc.pt', map_location=torch.device('cpu'))
    pc_checkpoint = torch.load(args.checkpoint_dir + 'pc.pt', map_location=torch.device('cpu'))

    convh.load_state_dict(convh_checkpoint['state_dict'])
    encoder.load_state_dict(enc_checkpoint['state_dict'])
    classifier.load_state_dict(pc_checkpoint['state_dict'])

    convh.cuda(device=args.device_ids[0])
    encoder.cuda(device=args.device_ids[0])
    classifier.cuda(device=args.device_ids[0])

    # load train model
    spa_fen = HyperSpatialResNet(input_channels=args.input_channel)
    spa_fen.cuda(device=args.device_ids[0])

    frn = FeatureRestorationNet(channels=args.input_channel)
    frn.cuda(device=args.device_ids[0])
    
    optimizer = torch.optim.Adam(list(spa_fen.parameters())
                                 +list(frn.parameters()),
                                    lr=args.lr, betas=(0.5,0.999))


    # load dataset
    kwargs = {'num_workers':4, 'pin_memory': True}
    train_dataset = HADDataset(dataset_path=args.data_path,sensor= args.sensor, resize=args.img_size,
                                start_channel=args.start_channel_id, channel=args.input_channel, train_ratio = args.train_ratio)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size*len(args.device_ids), shuffle=True, **kwargs)
    test_dataset = HADTestDataset(dataset_path=args.data_path, resize=args.img_size, channel=args.input_channel)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1*len(args.device_ids), shuffle=False, **kwargs)

    # start training
    start_time = time.time()
    epoch_time = AverageMeter()

    best_auc = 0
    best_auc_ham = 0

    for epoch in range(1, args.epochs + 1):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

        losses = train(args, convh, classifier, encoder, epoch, train_loader, optimizer, log, spa_fen, frn)
        print_log(('Train Epoch: {}  Loss: {:.8f} '.format(epoch,  losses.avg)), log)

        test_imgs, scores, _, gt_imgs = test(args, classifier, convh, encoder, test_loader, spa_fen, frn)

        scores = np.asarray(scores)
        gt_imgs = np.asarray(gt_imgs)

        # calculate ROCAUC
        AU_ROC_per_img = np.zeros(len(test_imgs))
        for i in range(len(test_imgs)):
            AU_ROC_per_img[i] = roc_auc_score(gt_imgs[i, :].flatten() == 1,
                                                              scores[i, :].flatten())
        mean_AU_ROC = np.mean(AU_ROC_per_img)

        if best_auc < mean_AU_ROC:
            best_auc = mean_AU_ROC
            if args.save_txt:
                write_eval_result(os.path.join(args.save_dir, 'best.txt'), test_dataset.test_img,
                                  AU_ROC_per_img,list(range(len(test_dataset.test_img))))
                write_mean_result(os.path.join(args.save_dir, 'seed_' + str(args.seed) + '_best_meanauc.txt'), best_auc)
                
            if args.save_model:
                spp_save_dir = os.path.join(args.saved_models_dir, 'spa_fen.pt')
                save_checkpoint(spp_save_dir, spa_fen)

        print_log('mean pixel ROCAUC: %.5f' % (mean_AU_ROC), log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    log.close()


def train(args, convh, classifier, encoder, epoch, train_loader, optimizer, log, spa_fen, frn):
    
    classifier.eval()
    convh.eval()
    encoder.eval()

    spa_fen.train()
    frn.train()

   # model.train()
    losses = AverageMeter()
    
    for (data) in tqdm(train_loader):

        data = data.cuda(device=args.device_ids[0])
        labels = torch.zeros(data.size(0), data.size(2), data.size(3)).cuda(device=args.device_ids[0])

        _, spe_f = classifier(encoder(convh(data)), data)

        spa_f, false_rgb = spa_fen(data)

        data_r = frn(spa_f+spe_f)

        comp_loss = CrossCovarianceLoss(bands=args.input_channel)
        MseLoss = nn.MSELoss(reduction='mean')
        ssim_loss = SSIMLoss(5)

        loss_sim = comp_loss(spe_f, spa_f) + 0.1*cos_sim_loss(spe_f, spa_f) + 0.1*variance_preserve_loss(spa_f)
        loss_re = MseLoss(data_r, data) + 0.01*ssim_loss(data_r, data)
        loss = loss_sim + loss_re

        losses.update(loss.item(), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

def test(args, classifier, convh, encoder, test_loader, spa_fen, frn):

    classifier.eval()
    convh.eval()
    encoder.eval()

    spa_fen.eval()
    frn.eval()

    scores = []
    test_imgs = []
    gt_imgs = []

    for (data, gt) in tqdm(test_loader):

        test_imgs.extend(data.cpu().numpy())
        gt_imgs.extend(gt.cpu().numpy())

        with torch.no_grad():
            data = data.cuda(device=args.device_ids[0])

            _, spe_f = classifier(encoder(convh(data)), data)
            spa_f, _ = spa_fen(data)

            if args.input_channel > 50:
                admap = get_admap(data, spa_f, spe_f)
            else:
                admap = get_admap(data, spa_f, spe_f, tpype='ham')

            score = np.zeros([data.shape[0],data.shape[-1],data.shape[-1]])
            score_ham = np.zeros([data.shape[0],data.shape[-1],data.shape[-1]])
            for i in range(data.shape[0]):
                score[i, :] = admap
            

        if len(score.shape) == 2:
            score=np.expand_dims(score,axis=0)
        scores.extend(score)

    return test_imgs, scores, _, gt_imgs


if __name__ == '__main__':

    main()
