from __future__ import print_function
import argparse
import time
import os
import numpy as np
import tqdm
from sklearn.metrics import f1_score

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from numpy import *
from torch.autograd import Variable
from data_preprocessing.data_from_prepared_dataset import train_dataset, colorize_mask, fast_hist, validation_dataset
from networks.seg_net import Segnet

parser = argparse.ArgumentParser(description='Training a Segnet model')
parser.add_argument('--batch_size', type=int, default=4, help='equivalent to instance normalization with batch_size=1')
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=2, help='equivalent to numclass')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--num_workers', type=int, default=4, help='how many threads of cpu to use while loading data')
parser.add_argument('--size_w', type=int, default=320, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=320, help='scale image to this size')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--net', type=str, default='', help='path to pre-trained network')
parser.add_argument('--data_path', default='data_segnet/training', help='path to training images')
parser.add_argument('--val_path', default='data_segnet/validation', help='path to validation images')
parser.add_argument('--outf', default='checkpoint/Segnet', help='folder to output images and model checkpoints')
parser.add_argument('--save_epoch', default=1, help='how many epochs to save once')
parser.add_argument('--test_step', default=300, help='how many epochs to visualize')
parser.add_argument('--log_step', default=4, help='how many epochs to write log')
parser.add_argument('--num_GPU', default=1, help='number of GPU')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
    os.makedirs(opt.outf + '/model/')
except OSError:
    pass

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
cudnn.benchmark = True

#training set loader & validation set loader
train_datatset_ = train_dataset(opt.data_path, opt.size_w, opt.size_h, opt.flip)
train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=opt.num_workers)
validation_datatset_ = validation_dataset(opt.val_path, opt.size_w, opt.size_h, opt.flip)
validation_loader = torch.utils.data.DataLoader(dataset=validation_datatset_, batch_size=opt.batch_size, shuffle=False,
                                           num_workers=opt.num_workers)

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#choose SegNet
net = Segnet(opt.input_nc, opt.output_nc)

if opt.net != '':
    net.load_state_dict(torch.load(opt.netG))
else:
    net.apply(weights_init)
if opt.cuda:
    net.cuda()
if opt.num_GPU > 1:
    net = nn.DataParallel(net)

###########   LOSS & OPTIMIZER   ##########
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
###########   GLOBAL VARIABLES   ###########
initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
semantic_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
initial_image = Variable(initial_image)
semantic_image = Variable(semantic_image)

if opt.cuda:
    initial_image = initial_image.cuda()
    semantic_image = semantic_image.cuda()
    criterion = criterion.cuda()

if __name__ == '__main__':

    log = open('./checkpoint/Segnet/train_Segnet_log.txt', 'w')
    start = time.time()

    hist = np.zeros((opt.output_nc, opt.output_nc))
    for epoch in range(1, opt.niter + 1):
        net.train()
        loader = iter(train_loader)
        for i in range(0, train_datatset_.__len__(), opt.batch_size):
            initial_image_, semantic_image_, name = loader.next()
            initial_image.resize_(initial_image_.size()).copy_(initial_image_)
            semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)
            semantic_image_pred = net(initial_image)

            ### loss ###
            # from IPython import embed;embed()
            assert semantic_image_pred.size()[2:] == semantic_image.size()[1:]
            loss = criterion(semantic_image_pred, semantic_image.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### evaluate ###
            predictions = semantic_image_pred.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            gts = semantic_image.data[:].squeeze_(0).cpu().numpy()
            hist += fast_hist(label_pred=predictions.flatten(), label_true=gts.flatten(),
                              num_classes=opt.output_nc)
            train_acc = np.diag(hist).sum() / hist.sum()

            ########### Logging ##########
            if i % opt.log_step == 0:
                print('[%d/%d][%d/%d] Loss: %.4f TrainAcc: %.4f' %
                      (epoch, opt.niter, i, len(train_loader) * opt.batch_size, loss.item(), train_acc))
                log.write('[%d/%d][%d/%d] Loss: %.4f TrainAcc: %.4f' %
                          (epoch, opt.niter, i, len(train_loader) * opt.batch_size, loss.item(), train_acc))
            if i % opt.test_step == 0:
                gt = semantic_image[0].cpu().numpy().astype(np.uint8)
                gt_color = colorize_mask(gt)
                predictions = semantic_image_pred.data.max(1)[1].squeeze_(1).cpu().numpy()
                prediction = predictions[0]
                predictions_color = colorize_mask(prediction)
                width, height = opt.size_w, opt.size_h
                save_image = Image.new('RGB', (width * 2, height))
                save_image.paste(gt_color, box=(0 * width, 0 * height))
                save_image.paste(predictions_color, box=(1 * width, 0 * height))
                save_image.save(opt.outf + '/epoch_%03d_%03d_gt_pred.png' % (epoch, i))

        ### validation ###
        net.eval()
        valid_loss = 0
        correct=0
        total=0
        with torch.no_grad():
            for images, labels, _ in validation_loader:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                ouputs = net(images)
                loss = criterion(ouputs, labels)
                valid_loss += loss.item()
                _,predicted = ouputs.max(1)
                total += labels.size(0)*labels.size(1)*labels.size(2)
                correct += predicted.eq(labels).sum().item()
        test_loss=valid_loss/len(validation_loader)
        acc=correct/total
        #write validation acc and loss into log file
        log.write('[%d/%d] Validation Loss: %.4f Validation Acc: %.4f' %
                  (epoch, opt.niter, test_loss, acc))
        print(f'Epoch {epoch}  \t\t Validation Loss: {test_loss}   \t\t Validation Acc: {acc}')

        if epoch % opt.save_epoch == 0:
            torch.save(net.state_dict(), '%s/model/netG_%s.pth' % (opt.outf, str(epoch)))

    end = time.time()
    torch.save(net.state_dict(), '%s/model/netG_final.pth' % opt.outf)
    print('Program processed ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600, 'h')
    log.close()
