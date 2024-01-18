import torch
import os
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle

from model import DNCNet, VGG, ResNet18
from dataset import Dataset_complex, Dataset_IQ
from utils import adjust_learning_rate

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data_dir',
                    default='',
                    required=True,
                    type=str,
                    help='data ')
parser.add_argument('--model_dir',
                    default='',
                    required=True,
                    type=str, help='denoised model')
parser.add_argument('--classify_model_dir',
                    default='',
                    required=True,
                    type=str, help='pretrained classifier')
parser.add_argument('--dir',
                    default='./result/all/train/',
                    type=str, help='result_dir')
parser.add_argument('--bs',
                    # default=64,
                    default=256,
                    type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='max_epoch')
parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--lr_rate', default=20, type=int, help='lr_update_freq')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

"""prepare the model"""
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
# optimizer = optim.Adam(classify_model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
criterion.cuda()
print('2. Finish loading !')

# fix the first 5 layers and only update the weight of linear layer
count = 0
para_optim = []
for k in classify_model.children():
    count += 1
    # 6 should be changed properly
    if count > 5:
        for param in k.parameters():
            para_optim.append(param)
    else:
        for param in k.parameters():
            param.requires_grad = False

optimizer = optim.Adam(para_optim, lr=args.lr)

"""prepare the retrain dataset"""
print('3. loading data ...')
# retrain_signal = Dataset_IQ(args.data_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5', mode='train')  # IQ data
retrain_signal = Dataset_IQ(args.data_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5', mode='train')  # IQ data
retrain_signal_dataloader = DataLoader(retrain_signal, batch_size=args.bs, shuffle=False)

# signals_test = Dataset_IQ(args.data_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5', mode='test')
signals_test = Dataset_IQ(args.data_dir, filename='GOLD_XYZ_1024_30dB_5dB.hdf5', mode='test')
dataloader_test = DataLoader(signals_test, batch_size=args.bs, shuffle=False)
print('3. Finish loading !')

"""denoised and classify"""
# acc_test_list = []
acc_test = []
acc_train = []
best_acc = 0.0
for epoch in range(args.epoch + 1):
    correct = 0.0
    running_loss = 0.0
    total = 0.0
    optimizer = adjust_learning_rate(optimizer, epoch, args.lr_rate)
    for batch_idx, data in enumerate(retrain_signal_dataloader):
        inputs, labels = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        classify_model.train()
        # denoised
        noise_level_est, output = model(inputs)

        optimizer.zero_grad()

        # classify
        classify_result = classify_model(output)
        loss = criterion(classify_result, labels.squeeze().long())
        pred = classify_result.detach().max(1)[1]  # classify accuracy
        correct += pred.eq(labels.view_as(pred)).sum()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)

        # print loss and accuracy per 100 batches
        if batch_idx % 100 == 99:
            print('epoch:{} batch:{} loss:{} lr:{} accuracy:{}'
                  .format(epoch + 1,
                          batch_idx + 1,
                          running_loss / 100,
                          optimizer.param_groups[-1]['lr'],
                          correct / total))

            running_loss = 0.0  # zero loss

    acc_train.append(correct / total)

    # print test accuracy after each epoch
    with torch.no_grad():
        correct = 0.0
        total_test = 0.0
        for data in dataloader_test:
            images, labels = data

            images, labels = images.cuda(), labels.cuda()
            noise_level_est_test, output_test = model(images)  # denoise
            classify_model.eval()
            classify_result_test = classify_model(output_test)  # classify
            _, predicted = torch.max(classify_result_test.data, 1)
            total_test += labels.size(0)
            correct += (predicted == labels.squeeze()).sum()

        acc_test.append(correct / total_test)

        if correct / total_test > best_acc:
            best_acc = correct / total_test
            print('save weight')
            # save model
            # torch.save({'model': classify_model.state_dict()}, 'ResNet2_30dB_impulse.pth.tar')
            state = {'model': classify_model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, args.dir + 'ResNet2_30dB.pth.tar')

        print('epoch %d,  test accuracy：%.2f%%' % (epoch + 1, 100 * correct / total_test))

# save retrain model
# state = {'model': classify_model.state_dict(), 'optimizer': optimizer.state_dict()}
# torch.save(state, 'ResNet_trial_30dB_impulse.pth.tar')

acc_train_list = [acc_train[i].cpu().numpy() for i in range(len(acc_train))]
acc_test_list = [acc_test[i].cpu().numpy() for i in range(len(acc_test))]

# save results
with open('acc_retrain_single.txt', 'wb') as file_pi:
    pickle.dump(acc_train_list, file_pi)

with open('acc_retest_single.txt', 'wb') as file_pi:
    pickle.dump(acc_test_list, file_pi)


