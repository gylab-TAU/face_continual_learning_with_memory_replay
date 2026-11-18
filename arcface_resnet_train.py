from __future__ import print_function

import glob
import os
from dataset import *
from metrics import *
from focal_loss import *
import torch
from torch.utils import data
import torch.nn.functional as F
from resnet import *
import torchvision
# from visualizer import *
# from view_model import *
import torch
import numpy as np
import random
import time
import config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from config import *
import cv2

def save_model(model, save_path, name, iter_cnt, is_best = False):
    if is_best:
        best_checkpoint = glob.glob(os.path.join(save_path,'*best.pth'))
        if len(best_checkpoint) > 0:
            if os.path.isfile(best_checkpoint[0]):
                os.remove(best_checkpoint[0])
        save_name = os.path.join(save_path, name + '_best.pth')
    else:
        save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')

    torch.save(model.state_dict(), save_name)
    return save_name


def write_image_names(opt):
    root_dir = opt.train_root
    dirs = os.listdir(root_dir)
    with open(os.path.join(root_dir,opt.train_list), 'w') as train_list:
        label = 0
        for dir in dirs:
            if os.path.isdir(os.path.join(root_dir,dir)):
                imlist = glob.glob('*.png',root_dir=os.path.join(root_dir,dir))
                for im in imlist:
                    train_list.write(f'{dir}/{im} {label}')
                    train_list.write('\n')
                label += 1



def train_arcface_resnet(opt = None, trainloader = None, get_from_checkpoint = False):
    if opt is None:
        opt = Config()
    device = torch.device("cuda")

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    embedding_size = 512
    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se, embedding_size=embedding_size)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    # view_model(model, opt.input_shape)
    # print(model)

    if get_from_checkpoint:
        best_checkpoint = glob.glob(os.path.join(opt.checkpoints_path,'*best.pth'))
        if len(best_checkpoint) > 0:
            if os.path.isfile(best_checkpoint[0]):
                model.load_state_dict(torch.load(best_checkpoint[0]))
                print(f'get_from_checkpoint found model {best_checkpoint[0]}')
                return model
        # else
        print('could not find model in checkpoint')
        return

#    model = DataParallel(model)
    if opt.resume != '':
        if os.path.exists(opt.resume):
            print(f'resume from {opt.resume}')
            model.load_state_dict(torch.load(opt.resume))
        else:
            print(f'path: {opt.resume} doesnot exist, not resuming')

    model.to(device)

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(embedding_size, opt.num_classes, s=20, m=0.35, easy_margin=opt.easy_margin)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(embedding_size, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(embedding_size, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(embedding_size, opt.num_classes)


    metric_fc.to(device)
#    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    if trainloader is None:
        write_image_names(opt)
        train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                      batch_size=opt.train_batch_size,
                                      shuffle=True,
                                      num_workers=opt.num_workers)

    #identity_list = get_lfw_list(opt.lfw_test_list)
    #img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))


#    best_lfw = 0.936
#    best_lfw = 0
    best_acc = 0

    # lambda_var = 1e-3

    start = time.time()
    for i in range(1, opt.max_epoch):
        scheduler.step()
        batch_acc = []
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                batch_acc.append(acc)

                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                # if opt.display:
                #     visualizer.display_current_results(iters, loss.item(), name='train_loss')
                #     visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

            # if ii > 1000:
            #     break
        max_acc = np.max(batch_acc)
        if max_acc > best_acc:
            save_model(model, opt.checkpoints_path, opt.backbone, i, is_best=True)
            best_acc = max_acc
            model.eval()
            # lfw_acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
            # print(f'saving model acc: {acc}, lfw_acc: {lfw_acc}')
            print(f'saving model max_acc: {max_acc} epoch: {i}')
            if max_acc == 1.0:
                print('got mean_acc=1.0, stopping')
                return model
            model.train()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)


    # model.eval()
    # acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    # if acc > best_lfw:
    #     save_model(model, opt.checkpoints_path, opt.backbone, i, is_best = True)
    #     best_lfw = acc

        # if opt.display:
        #     visualizer.display_current_results(iters, acc, name='test_acc')

    return model

def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


def arcface_resnet_test(opt, model = None):

    if model is None:
        opt = Config()
        if opt.backbone == 'resnet18':
            model = resnet_face18(opt.use_se)
        elif opt.backbone == 'resnet34':
            model = resnet34()
        elif opt.backbone == 'resnet50':
            model = resnet50()

    #    model = DataParallel(model)
        # load_model(model, opt.test_model_path)
        model.load_state_dict(torch.load(opt.test_model_path))

    model.to(torch.device("cuda"))

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    model.eval()
    lfw_acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    return lfw_acc




