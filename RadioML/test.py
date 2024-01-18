import torch
import argparse
from torch.utils.data import DataLoader
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pickle
import os

from model import DNCNet, VGG, ResNet18, num_class
from dataset import Dataset_complex, Dataset_IQ

matplotlib.use('Agg')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data_dir',
                    default='',
                    required=True,
                    type=str, help='test data')
parser.add_argument('--model_dir',
                    default='',
                    required=True,
                    type=str, help='denoised model')
parser.add_argument('--classify_model_dir',
                    default='',
                    required=True,
                    type=str, help='pretrained model')

parser.add_argument('--save_dir',
                    default='',
                    required=True,
                    type=str, help='test result')
parser.add_argument('--bs', default=64,
                    type=int, help='batch size')
parser.add_argument('--label_idx', default=4,
                    type=int, help='specific_label')
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
classify_model = ResNet18()
classify_model.cuda()
print('2. loading classifying model ...')
classify_model.load_state_dict(model_CKPT2['model'])
# classify_model.load_state_dict(model_CKPT2['state_dict'])
classify_model.eval()
print('2. Finish loading !')

"""prepare the test dataset"""
print('3. loading data ...')
# test_signal = Dataset_IQ(args.data_dir, filename='GOLD_XYZ_1024_30dB_5dB_impulse.hdf5', mode='test')
test_signal = Dataset_IQ(args.data_dir, filename='GOLD_XYZ_1024_30dB_-1dB.hdf5', mode='test')

# test_signal = Dataset_IQ_specific(args.data_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5',
#                                   idx=args.label_idx, mode='test')
test_signal_dataloader = DataLoader(test_signal, batch_size=args.bs, shuffle=False)

"""load ground-truth data"""
gt_signal = Dataset_IQ(args.data_dir, filename='GOLD_XYZ_1024_30dB.hdf5', mode='test')
# gt_signal = Dataset_IQ_specific(args.data_dir, filename='GOLD_XYZ_1024_30dB.hdf5',
#                                 idx=args.label_idx, mode='test')
gt_signal_dataloader = DataLoader(gt_signal, batch_size=args.bs, shuffle=False)
print('3. Finish loading !')

"""denoised and classify"""
idx = 0
correct = 0.0
total_test = 0.0
test_acc_list = []
snr_before_list = []
snr_after_list = []
confusion_matrix = torch.zeros(num_class, num_class)
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
    # label_np = data2[1][random.Random(1).randint(1, 16)].cpu().numpy()
    # label_np = int(label_np)

    idx += 1

    test_acc_list.append(correct / total_test)

    # plt results
    # if not os.path.isdir(args.save_dir):
    #     os.makedirs(args.save_dir)

    # plt.plot(inputs_np[0], color='silver', linestyle='dotted', label='noisy' + '_' + str(label_np))
    # plt.plot(gt_np[0], color='cyan', linestyle='dashed', label='clean' + '_' + str(label_np))
    # plt.plot(output_np[0], color='red', linestyle='solid', label='denoised' + '_' + str(label_np))
    # tx0 = 0
    # tx1 = 620
    # ty0 = 1.4
    # ty1 = 0.4

    # tx0 = 580
    # tx1 = 800
    # ty0 = 1.6
    # ty1 = 0.3
    # sx = [tx0, tx1, tx1, tx0, tx0]
    # sy = [ty0, ty0, ty1, ty1, ty0]
    # # plt.plot(sx, sy, "purple")
    # # plt.legend(loc='lower right', fontsize=16)
    # plt.tick_params(labelsize=12)
    # plt.show()

    # plt.savefig(args.save_dir + '%d.png' % idx)
    # plt.close('all')
    # plt.clf()

    for t, p in zip(labels.view(-1), pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    #
    # # break
    #
    # if idx == 22:
    #     break
    # else:
    #     plt.close('all')

    # style 3
    # plt.rcParams['font.sans-serif'] = ['SimSun']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['pdf.fonttype'] = 42
    # palette = pyplot.get_cmap('Paired')
    # color1 = palette(4)
    # color2 = palette(1)
    # color3 = palette(9)
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 18,
    #          }
    #
    # plt.figure(figsize=(6, 3))
    # plt.plot(inputs_np[0], color=color1, linestyle='solid', label='有噪声')
    # plt.xticks(fontproperties='Times New Roman', size=18)
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.legend(loc='upper left', fontsize=16)
    # plt.tight_layout()
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(1) + '.png')
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(1) + '.pdf')
    # #
    # plt.figure(figsize=(6, 3))
    # plt.plot(gt_np[0], color=color2, linestyle='solid', label='干净')
    # plt.ylim([-2, 2])
    # plt.legend(loc='upper left', fontsize=16)
    # plt.xticks(fontproperties='Times New Roman', size=18)
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.tight_layout()
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(2) + '.png')
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(2) + '.pdf')
    # #
    # plt.figure(figsize=(6, 3))
    # plt.plot(output_np[0], color=color3, linestyle='solid', label='去噪后')
    # # plt.ylim([-1.5, 1.5])
    # plt.yticks(fontproperties='Times New Roman', size=18)
    # plt.xticks(fontproperties='Times New Roman', size=18)
    # plt.legend(loc='upper left', fontsize=16)
    # plt.tight_layout()
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(3) + '.png')
    # plt.savefig(args.save_dir + '%d' % idx + '_' + str(3) + '.pdf')
    #
    # plt.close('all')
    # plt.clf()

print('average value of test accuracy：%.2f%%' % (100 * sum(test_acc_list) / len(test_acc_list)))
print('SNR Gains: %.2f -> %.2f ' % (
sum(snr_before_list) / len(snr_before_list), sum(snr_after_list) / len(snr_after_list)))

# To get the per-class accuracy:
# print(confusion_matrix.diag()/confusion_matrix.sum(1))

# save numerical logs
# with open('confusion_matrix.txt', 'wb') as file_pi:
#     pickle.dump(confusion_matrix, file_pi)

# plot confusion matrix
# label = ['0', '1', '2', '3', '4', '5', '6', '7',
#          '8', '9', '10', '11', '12', '13', '14', '15',
#          '16', '17', '18', '19', '20', '21', '22', '23', ]
# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
# plt.colorbar()
# tick_marks = np.arange(num_class)
# plt.xticks(tick_marks, label)
# plt.yticks(tick_marks, label)
# plt.ylabel('Real label')
# plt.xlabel('Prediction')
# plt.tight_layout()
# plt.show()
