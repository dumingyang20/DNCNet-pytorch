import torch
import argparse
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from matplotlib import pyplot

from dataset import Dataset_complex
from model import DNCNet, LeNet

# matplotlib.use('Agg')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--test_data_dir',
                    default='',
                    required=True,
                    type=str, help='test data')
parser.add_argument('--gt_data_dir',
                    default='',
                    required=True,
                    type=str, help='ground-truth data')
parser.add_argument('--model_dir',
                    default='',
                    required=True,
                    type=str, help='denoised model')
parser.add_argument('--classify_model_dir',
                    default='',
                    required=True,
                    type=str, help='pretrained model')
parser.add_argument('--save_dir',
                    default='./result/all/train_at_noisefree_LeNet/test_at_5dB/',
                    type=str, help='test result')
parser.add_argument('--bs', default=64,
                    type=int, help='batch size')
args = parser.parse_args()

"""load model to test another data set"""
model_CKPT = torch.load(args.model_dir)
model = DNCNet()
model.cuda()
print('1. loading denoised model ...')
model.load_state_dict(model_CKPT['state_dict'])
model.eval()
print('1. Finish loading !')

model_CKPT2 = torch.load(args.classify_model_dir)
classify_model = LeNet()
classify_model.cuda()
print('2. loading classifying model ...')
classify_model.load_state_dict(model_CKPT2['model'])
classify_model.eval()
print('2. Finish loading !')

"""prepare the test dataset"""
print('3. loading data ...')
test_signal = Dataset_complex(args.test_data_dir, mode='test')  # SIGNAL-8 dataset
test_signal_dataloader = DataLoader(test_signal, batch_size=args.bs, shuffle=False)

"""load ground-truth data"""
gt_signal = Dataset_complex(args.gt_data_dir, mode='test')  # SIGNAL-8 dataset
gt_signal_dataloader = DataLoader(gt_signal, batch_size=args.bs, shuffle=False)
print('3. Finish loading !')

"""denoised and classify"""
idx = 0
correct = 0.0
total_test = 0.0
test_acc_list = []
snr_before_list = []
snr_after_list = []
for data1, data2 in zip(test_signal_dataloader, gt_signal_dataloader):
    inputs = data1[0]
    labels = data1[1]
    inputs = inputs.cuda()
    labels = labels.cuda()

    # denoised
    noise_level_est, output = model(inputs)

    # classify
    classify_result = classify_model(output)
    pred = classify_result.detach().max(1)[1]  # classify accuracy
    correct += pred.eq(labels.view_as(pred)).sum()
    total_test += labels.size(0)

    idx += 1

    acc = correct / total_test
    # print('test accuracy in %03d batch：%.2f%%' % (idx, (100 * acc)))
    test_acc_list.append(acc)

    # calculate SNR gains
    gt_np = data2[0].squeeze().cpu().detach().numpy()
    inputs_np = inputs.squeeze().cpu().detach().numpy()
    output_np = output.squeeze().cpu().detach().numpy()

    inputs_complex = 1j * inputs_np[:, 1, :]
    inputs_complex += inputs_np[:, 0, :]

    output_complex = 1j * output_np[:, 1, :]
    output_complex += output_np[:, 0, :]

    gt_complex = 1j * gt_np[:, 1, :]
    gt_complex += gt_np[:, 0, :]
    gt_power = [np.sum(pow(abs(gt_complex[i]), 2)) / np.size(gt_complex[i]) for i in range(len(gt_complex))]

    noise_before = inputs_complex - gt_complex
    noise_before_power = [np.sum(pow(abs(noise_before[i]), 2)) / np.size(noise_before[i]) for i in
                          range(len(noise_before))]

    noise_after = output_complex - gt_complex
    noise_after_power = [np.sum(pow(abs(noise_after[i]), 2)) / np.size(noise_after[i]) for i in range(len(noise_after))]

    snr_before = [10 * np.log10(gt_power[i] / noise_before_power[i]) for i in range(len(noise_before_power))]
    snr_after = [10 * np.log10(gt_power[i] / noise_after_power[i]) for i in range(len(noise_after_power))]

    snr_before_list.extend(snr_before)
    snr_after_list.extend(snr_after)

    # for plot
    # gt_np = data2[0].cuda()[random.Random(1).randint(1, 16)].squeeze().cpu().detach().numpy()
    # inputs_np = inputs[random.Random(1).randint(1, 16)].squeeze().cpu().detach().numpy()
    # output_np = output[random.Random(1).randint(1, 16)].squeeze().cpu().detach().numpy()

    # plt results
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42

    # if not os.path.isdir(args.save_dir):
    #     os.makedirs(args.save_dir)

    # style 1
    # plt.plot(inputs_np[0], color='silver', linestyle='dotted', label='noisy')
    # plt.plot(gt_np[0], color='cyan', linestyle='dashed', label='clean')
    # plt.plot(output_np[0], color='red', linestyle='solid', label='denoised')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(args.save_dir + '%d.png' % idx)
    # plt.close('all')
    # plt.clf()

    # style 2
    # plot setting
    # palette = pyplot.get_cmap('Paired')
    # color1 = palette(4)
    # color2 = palette(1)
    # color3 = palette(9)
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 18,
    #          }
    #
    # plt.figure(1)
    # plt.subplot(311)
    # plt.plot(inputs_np[0], color=color1, linestyle='solid', label='有噪声')
    # plt.xticks([])
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.legend(loc='upper left', fontsize=16)
    # #
    # plt.subplot(312)
    # plt.plot(gt_np[0], color=color2, linestyle='solid', label='干净')
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.ylim([-1.5, 1.5])
    # plt.legend(loc='upper left', fontsize=16)
    # plt.xticks([])
    # #
    # plt.subplot(313)
    # plt.plot(output_np[0], color=color3, linestyle='solid', label='去噪后')
    # plt.ylim([-1.5, 1.5])
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.xticks(fontproperties='Times New Roman', size=18)
    # plt.legend(loc='upper left', fontsize=16)
    # plt.tight_layout()
    # #
    # plt.savefig(args.save_dir + '%d.pdf' % idx)
    # plt.close('all')
    # plt.clf()

    # style 3 -- final
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    palette = pyplot.get_cmap('Paired')
    color1 = palette(4)
    color2 = palette(1)
    color3 = palette(9)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }

    # plt.figure(figsize=(6, 3))
    # plt.plot(inputs_np[0], color=color1, linestyle='solid', label='有噪声')
    # plt.xticks(fontproperties='Times New Roman', size=18)
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.legend(loc='upper left', fontsize=16)
    # plt.tight_layout()
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(1) + '.png')
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(1) + '.pdf')
    #
    # plt.figure(figsize=(6, 3))
    # plt.plot(gt_np[0], color=color2, linestyle='solid', label='干净')
    # plt.ylim([-1.5, 1.5])
    # plt.legend(loc='upper left', fontsize=16)
    # plt.xticks(fontproperties='Times New Roman', size=18)
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.tight_layout()
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(2) + '.png')
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(2) + '.pdf')
    #
    # plt.figure(figsize=(6, 3))
    # plt.plot(output_np[0], color=color3, linestyle='solid', label='去噪后')
    # # plt.ylim([-1.5, 1.5])
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.xticks(fontproperties='Times New Roman', size=18)
    # plt.legend(loc='upper left', fontsize=16)
    # plt.tight_layout()
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(3) + '.png')
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(3) + '.pdf')

    # plt.close('all')
    # plt.clf()

print('average value of test accuracy：%.2f%%' % (100 * sum(test_acc_list) / len(test_acc_list)))
print('SNR Gains: %.2f -> %.2f ' % (
sum(snr_before_list) / len(snr_before_list), sum(snr_after_list) / len(snr_after_list)))

# SNR gains for each class
snr_each_before = []
snr_each_after = []
labels = gt_signal.label_info.cpu().detach().numpy()
n_class = int(max(labels)) + 1
for i in range(n_class):
    idx = np.where(labels == i)[0]
    snr_each_1 = [snr_before_list[i] for i in idx]
    snr_each_2 = [snr_after_list[i] for i in idx]

    snr_each_before.append(sum(snr_each_1) / len(snr_each_1))
    snr_each_after.append(sum(snr_each_2) / len(snr_each_2))

