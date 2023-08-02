import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset  # 加载数据用（返回图片的索引，归一化后的图片，标签）

from U2netp import U2NETp, U2NETPp

# from my_U2_Net.model import U2NET  # full size version 173.6 MB         #导入两个网络
# from my_U2_Net.model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):  # 归一化
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred.squeeze()  # 删除单维度
    predict_np = predict.cpu().data.numpy()  # 转移到CPU上

    im = Image.fromarray(predict_np * 255).convert('RGB')  # 转为PIL，从归一化的图片恢复到正常0到255之间
    img_name = image_name.split("\\")[-1]  # 取出后缀类型
    # print(image_name)
    # print(img_name)
    image = io.imread(
        image_name)  # io.imread读出图片格式是uint8(unsigned int)；value是numpy array；图像数据是以RGB的格式进行存储的，通道值默认范围0-255
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    # pb_np = np.array(imo)                                  #多余的

    aaa = img_name.split(".")  # 图片名字被切分为一个列表
    bbb = aaa[0:-1]  # 取出图片名称的前缀
    # print(aaa)                                             #['5', 'jpg']
    # print(bbb)                                             #['5']
    # print("---------------------------------------------")
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')  # 保存图片到指定路径


def main():
    rootpath='D:/dance/new_dance_scoring/3Dreconstruction/facedataset_u2netplus_3perceptionfields/'
    # --------- 1. get image path and name ---------
    model_name = 'u2net-plus'  # u2netp                              #保存的模型的名称

    image_dir = rootpath+'test_img/'  # 将要预测的图片所在的文件夹路径
    prediction_dir = rootpath+'predicts/'  # 预测结果的保存的文件夹路径
    # model_dir = '../saved_models/'+ model_name + '/' + model_name + '.pth'
    # model_dir = r"../saved_models/u2net/u2net_bce_itr_422_train_3.743319_tar_0.546805.pth"
    model_dir = rootpath+"saved_models/u2net-plus/u2net-plus.pth"  # 模型参数的路径

    img_name_list = glob.glob(image_dir + '*')  # 图片文件夹下的所有数据（携带路径）
    #print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(100),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)  # 加载数据

    # --------- 3. model define ---------
    if (model_name == 'u2net-plus'):  # 分辨使用的是哪一个模型参数
        print("...load U2NET-plus---173.6 MB")
        net = U2NETp(3, 1)
    else:
        print("...load U2NEP-plus---4.7 MB")
        net = U2NETPp(3, 1)
    net.load_state_dict(torch.load(model_dir))  # 加载训练好的模型
    if torch.cuda.is_available():
        net.cuda()  # 网络转移至GPU
    net.eval()  # 测评模式

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split("/")[-1])  # test_images\5.jpg
        # print(data_test)                                   #'imidx': tensor([[0]], dtype=torch.int32), 'image': tensor([[[[ 1.4051,  ...'label': tensor([[[[0., 0., 0.,  ...,
        inputs_test = data_test['image']  # 测试的是图片
        inputs_test = inputs_test.type(torch.FloatTensor)  # 转为浮点型

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
            # Variable是对Tensor的一个封装，操作和Tensor是一样的，但是每个Variable都有三个属性，
            # tensor不能反向传播，variable可以反向传播。它会逐渐地生成计算图。
            # 这个图就是将所有的计算节点都连接起来，最后进行误差反向传递的时候，
            # 一次性将所有Variable里面的梯度都计算出来，而tensor就没有这个能力
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)  # 将图片传入网络

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)  # 对预测的结果做归一化

        # save results to test_results folder
        save_output(img_name_list[i_test], pred, prediction_dir)  # 原始图片名、预测结果，预测图片的保存目录   #save_output保存预测的输出值

        del d1, d2, d3, d4, d5, d6, d7  # del 用于删除对象。在 Python，一切都是对象，因此 del 关键字可用于删除变量、列表或列表片段等。


if __name__ == "__main__":
    main()