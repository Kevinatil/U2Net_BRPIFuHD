import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import RescaleT, RandomCrop, ToTensorLab, SalObjDataset, ToTensor, Rescale
from U2net import U2NET, U2NETP

import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction='mean')

loss_recorder=[]
loss_epoch_recorder=[]

tra_lbl_name_list = []########

def get_path():
    global tra_lbl_name_list
    tra_lbl_name_list = []
    data_dir='D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1'
    tra_image_dir = 'JPEGImages'
    tra_label_dir = 'SegmentationClass'
    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*'))
    for img_path in tra_img_name_list:
        img_name = img_path.split("\\")[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]

        imidx = bbb[0]
        for i in range(1, len(bbb)):

            imidx = imidx + "." + bbb[i]


        tra_lbl_name_list.append(data_dir + "\\" + tra_label_dir + "\\" + imidx + '_matte.png')

def get_path_test():
    global tra_lbl_name_list
    tra_lbl_name_list = []
    data_dir='D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1'
    tra_image_dir = 'test_img'
    tra_label_dir = 'test_mask'
    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*'))
    for img_path in tra_img_name_list:
        img_name = img_path.split("\\")[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]

        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + "\\" + tra_label_dir + "\\" + imidx + '_matte.png')


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    loss_recorder.append([loss0.item()+ loss1.item()+ loss2.item()+ loss3.item()+ loss4.item()+ loss5.item()+ loss6.item()])

    return loss0, loss


def main():
    # ------- 2. set the directory of training dataset --------
    model_name = 'u2net'  # 'u2netp'


    data_dir = 'D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1'
    tra_image_dir = 'JPEGImages'
    tra_label_dir = 'SegmentationClass'

    image_ext = '.jpg'
    label_ext = '_matte.png'  # 标签的后缀名

    model_dir = 'D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1/saved_models/' + model_name + '/'  # 保存的参数模型所在的文件夹名
    params_path = model_dir + model_name + ".pth"
    epoch_num = 10
    batch_size_train = 5
    batch_size_val = 2

    train_num = 0
    val_num = 0


    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*'))
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split("\\")[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]  #
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + "\\" + tra_label_dir + "\\" + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))  # 422
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([

            RescaleT(100),

            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    # ------- 3. define model --------
    # define the net
    if (model_name == 'u2net'):
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.cuda()

    if os.path.exists(params_path):
        net.load_state_dict(torch.load(params_path))
    else:
        print("No parameters!")

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000  # save the model every 2000 iterations########################################??

    for epoch in range(0, epoch_num):
        net.train()  # 训练模式

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():  # 转移到cuda
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)  # 6个sup作损失

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # delete temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))


        torch.save(net.state_dict(), params_path)
        loss_epoch_recorder.append(running_loss)
        net.train()




def predict(percent=1):
    data_dir='D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1'
    tra_image_dir = 'JPEGImages'
    tra_label_dir = 'SegmentationClass'

    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*'))
    get_path()
    salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(100),
        #RandomCrop(288),
        ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    counter=0
    predictnum=int(percent*len(tra_img_name_list))

    for i, data in enumerate(salobj_dataloader):
        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)


        net=U2NET(3, 1)
        if torch.cuda.is_available():
            net.cuda()  # 网络转移至GPU
        params_path='D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1/saved_models/u2net/u2net.pth'
        net.load_state_dict(torch.load(params_path))
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)

        plt.imshow(np.array((d0[0][0].cpu().detach())*255).astype('uint8'))
        plt.savefig('D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1/predicts/{0}predict.png'.format(counter))
        plt.imshow(np.array((labels[0][0])*255).astype('uint8'))
        plt.savefig('D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1/predicts/{0}label.png'.format(counter))
        counter+=1
        if counter>predictnum:
            break

def evaluate_trainingset(percent=1):

    assert percent<=1

    data_dir='D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1'
    tra_image_dir = 'JPEGImages'
    tra_label_dir = 'SegmentationClass'

    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*'))
    get_path()
    salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(100),
        ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=1, shuffle=True, num_workers=1)

    score_recorder=[]
    counter=0
    evalnumber=int(percent*len(tra_lbl_name_list))

    for i, data in tqdm(enumerate(salobj_dataloader)):
        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        net=U2NET(3, 1)
        if torch.cuda.is_available():
            net.cuda()  # GPU
        params_path=data_dir+'/saved_models/u2net/u2net.pth'
        net.load_state_dict(torch.load(params_path))
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        #print(labels[0][0])
        #print(labels[0][0].shape)
        #out1 = net(img)
        predict=np.array((d0[0][0].cpu().detach()))
        predict[predict>=0.5]=1
        predict[predict<0.5]=0
        predict=predict.astype(bool)
        label=np.array((labels[0][0])).astype(bool)
        score=(predict*label).sum()/(predict+label).sum()
        #print(score)
        counter+=1
        score_recorder.append(score)
        if counter>evalnumber:
            break

    print(score_recorder)
    scores=np.array(score_recorder)*100
    print('mean:{0},var:{1},max:{2},min:{3}'.format(scores.mean(),scores.var(),max(scores),min(scores)))
    plt.figure()
    plt.plot(scores)
    plt.plot([0,len(scores)],[scores.mean(),scores.mean()])
    plt.savefig(data_dir+'/experiment/scores_train.png')

    plt.figure()
    plt.hist(scores)
    plt.savefig(data_dir+'/experiment/scores_hist_train.png')
    #plt.text(0,80,'500',c='r')
    #plt.show()

def evaluate_testset():
    data_dir='D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1'
    tra_image_dir = 'test_img'
    tra_label_dir = 'test_mask'

    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*'))
    get_path_test()
    salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(100),
        ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=1, shuffle=True, num_workers=1)

    score_recorder=[]
    counter=0

    for i, data in tqdm(enumerate(salobj_dataloader)):
        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        net=U2NET(3, 1)
        if torch.cuda.is_available():
            net.cuda()
        params_path=data_dir+'/saved_models/u2net/u2net.pth'
        net.load_state_dict(torch.load(params_path))
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        predict=np.array((d0[0][0].cpu().detach()))
        predict[predict>=0.5]=1
        predict[predict<0.5]=0
        predict=predict.astype(bool)
        label=np.array((labels[0][0])).astype(bool)
        score=(predict*label).sum()/(predict+label).sum()
        #print(score)
        counter+=1
        score_recorder.append(score)

    print(score_recorder)
    scores=np.array(score_recorder)*100
    print('mean:{0},var:{1},max:{2},min:{3}'.format(scores.mean(),scores.var(),max(scores),min(scores)))
    plt.figure()
    plt.plot(scores)
    plt.plot([0,len(scores)],[scores.mean(),scores.mean()])
    plt.savefig(data_dir+'/experiment/scores_test.png')

    plt.figure()
    plt.hist(scores)
    plt.savefig(data_dir+'/experiment/scores_hist_test.png')


if __name__ == '__main__':

    trainb=1
    predictb=0
    trainingevalb=0
    testevalb=1

    if trainb:
        start=time.time()
        main()
        end=time.time()
        print('training time: ',end-start)
        plt.figure()
        plt.plot(loss_recorder)
        plt.savefig('D:/dance/new_dance_scoring/3Dreconstruction/trainingtest1/experiment/loss.png')
    if predictb:
        start=time.time()
        predict(percent=0.006)
        end=time.time()
        print('predict time: ',end-start)
    if trainingevalb:
        start=time.time()
        evaluate_trainingset(percent=0.002)
        end=time.time()
        print('training eval time: ',end-start)
    if testevalb:
        start=time.time()
        evaluate_testset()
        end=time.time()
        print('test eval time ',end-start)