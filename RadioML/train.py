import torch.optim as optim
import torch.nn as nn
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm
import os

from model import DNCNet, fixed_loss, ResNet18
from dataset import Dataset_IQ
from utils import *

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--bs', default=256, type=int, help='batch_size')
parser.add_argument('--lr', default=2e-3, type=float, help='learning_rate')
parser.add_argument('--epoch', default=200, type=int, help='max_epoch')
parser.add_argument('--lr_rate', default=50, type=int, help='lr_update_freq')
parser.add_argument('--sr', default=20, type=int, help='save_rate')
parser.add_argument('--dir',
                    default='./result/all/train/',
                    type=str,
                    help='result_dir')
parser.add_argument('--signal_dir',
                    default='',
                    required=True,
                    type=str,
                    help='data_dir')
parser.add_argument('--pretrained_dir',
                    default='',
                    required=True,
                    type=str,
                    help='pretrained_classifier_dir')
parser.add_argument('--lambda_1', default=1, type=float, help='loss_ratio')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if torch.cuda.is_available() is True:
    ids = torch.cuda.device_count()
    devices = np.arange(ids).tolist()  # get all GPUs
else:
    devices = []

# train dataset
# signal_images = Dataset_IQ(signal_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5', mode='train')
signal_images = Dataset_IQ(args.signal_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5', mode='train')

signal_dataloader = DataLoader(signal_images, batch_size=args.bs, shuffle=False)

noise_free_images = Dataset_IQ(args.signal_dir, filename='GOLD_XYZ_1024_30dB.hdf5', mode='train')
noise_free_dataloader = DataLoader(noise_free_images, batch_size=args.bs, shuffle=False)

# test data
# signals_test = Dataset_IQ(signal_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5', mode='test')
signals_test = Dataset_IQ(args.signal_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5', mode='test')
dataloader_test = DataLoader(signals_test, batch_size=args.bs, shuffle=False)

# prepare model
model = DNCNet()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = fixed_loss()
criterion.cuda()

classify_model = ResNet18()
state_dict = torch.load(args.pretrained_dir)
classify_model.load_state_dict(state_dict['model'])
classify_model.cuda()
criterion2 = torch.nn.CrossEntropyLoss()
criterion2.cuda()

# begin training
acc_train = []
acc_test = []
best_acc = 0.0
for epoch in range(args.epoch + 1):
    correct_train = 0.0
    total = 0.0
    optimizer = adjust_learning_rate(optimizer, epoch, args.lr_rate)

    st = time.time()
    cnt = 0
    pbar = tqdm(total=len(signal_dataloader))

    for batch1, batch2 in zip(signal_dataloader, noise_free_dataloader):
        # get the inputs
        inputs1, labels1 = batch1
        inputs2, labels2 = batch2

        # plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # plt.plot(inputs1[50][0].squeeze().cpu().detach().numpy())
        # plt.subplot(122)
        # plt.plot(inputs11[50][0].squeeze().cpu().detach().numpy())
        # plt.show()

        inputs_signal = inputs1.cuda()
        inputs_noise_free = inputs2.cuda()
        labels1 = labels1.cuda()

        losses = AverageMeter()
        model.train()

        # forward + backward + optimize
        noise_level_est, output = model(inputs_signal)

        classify_model.eval()  # newly add, fixed the BN layer and Dropout
        classify_result = classify_model(output)
        classify_loss = criterion2(classify_result, labels1.squeeze().long())
        pred = classify_result.detach().max(1)[1]  # classify accuracy
        correct_train += pred.eq(labels1.view_as(pred)).sum()
        total += labels1.size(0)

        loss = args.lambda_1 * criterion(inputs_noise_free, output, noise_level_est) + classify_loss
        # loss = criterion(output, inputs_signal-inputs_noise_free) + classify_loss  # for DnCNN
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cnt += 1

        # save denoised results
        # if epoch % args.sr == 0:
        #     if not os.path.isdir(args.dir + '%03d' % epoch):
        #         os.makedirs(args.dir + '%03d' % epoch)
        #
        #     output_np = output[random.Random(1).randint(1, 16)].squeeze().cpu().detach().numpy()
        #
        #     inputs1_np = inputs1[random.Random(1).randint(1, 16)].squeeze().cpu().detach().numpy()
        #
        #     inputs2_np = inputs2[random.Random(1).randint(1, 16)].squeeze().cpu().detach().numpy()
        #
        #     plt.plot(inputs1_np[0], color='silver', linestyle='dotted', label='noisy')
        #     plt.plot(inputs2_np[0], color='cyan', linestyle='dashed', label='clean')
        #     plt.plot(output_np[0], color='red', linestyle='solid', label='denoised')
        #     plt.legend()
        #     plt.savefig(args.dir + '%03d/%d.png' % (epoch, cnt))
        #     plt.close('all')
        #     plt.clf()

        pbar.update(1)

    pbar.close()

    acc_train.append(correct_train / total)

    # test after each epoch
    with torch.no_grad():
        correct_test = 0.0
        total_test = 0.0
        for data1 in dataloader_test:
            inputs_test = data1[0]
            labels_test = data1[1]

            inputs_test = inputs_test.cuda()
            labels_test = labels_test.cuda()

            model.eval()
            classify_model.eval()

            # denoised
            noise_level_est, output_test = model(inputs_test)

            # classify
            classify_result_test = classify_model(output_test)
            pred_test = classify_result_test.detach().max(1)[1]  # classify accuracy
            correct_test += pred_test.eq(labels_test.view_as(pred_test)).sum()
            total_test += labels_test.size(0)

        acc_test.append(correct_test / total_test)

    # print statistics
    print('[{0}]\t'
          'lr: {lr:.9f}\t'
          'loss_train: {loss:.4f}\t'
          'acc_train: {acc_train:.4f}\t'
          'acc_test: {acc_test:.4f}\t'
          'Time: {time:.3f}'.format(
        epoch,
        lr=optimizer.param_groups[-1]['lr'],
        loss=losses.avg,
        acc_train=correct_train / total,
        acc_test=correct_test / total_test,
        time=time.time() - st))

    if correct_test / total_test > best_acc:
        best_acc = correct_test / total_test
        print('save weight !')
        # save model
        torch.save({'state_dict': model.state_dict()}, args.dir + 'checkpoint_single.pth.tar')

print('Finished Training')

acc_train_list = [acc_train[i].cpu().numpy() for i in range(len(acc_train))]
acc_test_list = [acc_test[i].cpu().numpy() for i in range(len(acc_test))]

# save model here
# torch.save({'state_dict': model.state_dict()}, args.dir + 'checkpoint.pth.tar')

# plot here
# plt.plot(acc_train_list, color='blue', linestyle='solid', marker='o', label='train')
# plt.plot(acc_test_list, color='magenta', linestyle='solid', label='test')
# # plt.ylim((0, 1))
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()

# save results
with open('acc_train_single.txt', 'wb') as file_pi:
    pickle.dump(acc_train_list, file_pi)

with open('acc_test_single.txt', 'wb') as file_pi:
    pickle.dump(acc_test_list, file_pi)

# test whether the weight of classify model is fixed
# state_dict_2 = torch.load(args.pretrained_dir)
# classify_model_origin = ResNet18()
# classify_model_origin.cuda()
# classify_model_origin.load_state_dict(state_dict_2['model'])
# torch.equal(classify_model.layer1[0].conv1.weight.data, classify_model_origin.layer1[0].conv1.weight.data)
# torch.equal(classify_model.layer4[0].conv1.weight.data, classify_model_origin.layer4[0].conv1.weight.data)
