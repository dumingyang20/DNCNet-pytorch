import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import pickle

from model import DNCNet, LeNet
from dataset import Dataset_complex
from utils import adjust_learning_rate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data_dir',
                    default='',
                    required=True,
                    type=str, help='retrain data')
parser.add_argument('--model_dir',
                    default='',
                    required=True,
                    type=str, help='denoised model')
parser.add_argument('--classify_model_dir',
                    default='',
                    required=True,
                    type=str, help='pretrained classifier')
parser.add_argument('--bs', default=64,
                    type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='max_epoch')
parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--lr_rate', default=20, type=int, help='lr_update_freq')
args = parser.parse_args()


"""prepare the model"""
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
classify_model.load_state_dict(model_CKPT2)
optimizer = optim.Adam(classify_model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
criterion.cuda()
print('2. Finish loading !')

"""prepare the retrain dataset"""
print('3. loading data ...')
retrain_signal = Dataset_complex(args.data_dir, mode='train')  # SIGNAL-8 dataset
retrain_signal_dataloader = DataLoader(retrain_signal, batch_size=args.bs, shuffle=False)

signals_test = Dataset_complex(args.data_dir, mode='test')
dataloader_test = DataLoader(signals_test, batch_size=args.bs, shuffle=False)
print('3. Finish loading !')


"""denoised and classify"""
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
        print('epoch %d, test accuracy：%.2f%%' % (epoch + 1, 100 * correct / total_test))

    if correct / total_test > best_acc:
        best_acc = correct / total_test
        print('save weight !')
        # save model
        state = {'model': classify_model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, './result/all/train/LeNet2_noisefree.pth.tar')


# save retrain model
# state = {'model': classify_model.state_dict(), 'optimizer': optimizer.state_dict()}
# torch.save(state, './result/all/train/LeNet2_noisefree.pth.tar')

acc_train_list = [acc_train[i].cpu().numpy() for i in range(len(acc_train))]
acc_test_list = [acc_test[i].cpu().numpy() for i in range(len(acc_test))]

# save results
with open('acc_retrain_single.txt', 'wb') as file_pi:
    pickle.dump(acc_train_list, file_pi)

with open('acc_retest_single.txt', 'wb') as file_pi:
    pickle.dump(acc_test_list, file_pi)

