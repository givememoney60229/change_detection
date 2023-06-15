# pytorch code of TGRS paper
# "HyperNet: Self-Supervised Hyperspectral SpatialSpectral Feature Understanding Network for Hyperspectral Change Detection"
import collections
import os
import torch.nn as nn
import random
import math
import argparse
from sklearn import metrics
from sklearn.cluster import KMeans
from operator import truediv
import numpy as np
import torch.backends.cudnn as cudnn
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
#from evalution import two_cls_access
from HyperNet_model import HyperNet, BasicBlock
from PIL import Image

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(2)

def two_cls_access(reference,result):
    # for Hermiston dataset
    # reference:change_value=1;unchange_value=0
    # result: predicted map:change_value=1;unchange_value=0
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # False Positive
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp)+0.00000001)
    recall = len(tp) / (len(tp) + len(fn)+0.00000001)

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall++0.00000001)
    F1 = round(F1, 4)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))

    oa = (len(tp)+len(tn))/m/n      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/m/n/m/n
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    return oa_kappa

def kmean(ctemp):
    x=ctemp.shape[0]
    y=ctemp.shape[1]
    ctemp=np.reshape(ctemp,(-1,1))
    cfinal = np.ones((x*y, 1))
    kmeans = KMeans(n_clusters=2)
    idx = kmeans.fit_predict(ctemp)
    idx=np.array(idx)
    print(idx)
    print('kmean result size:',np.shape(idx))
    cfinal[idx == 1] = 0
    cfinal[idx == 2] = 1
    temp = cfinal.reshape((x, y))
    mean1 = np.mean(ctemp[cfinal == 0])
    mean2 = np.mean(ctemp[cfinal == 1])
    if mean2 >mean1:
        map = np.zeros((x, y))
        map[temp == 0] = 1
    else:
        map = temp
    return  map


def initNetParams_v2(net):
    # Init net parameters
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def get_args_viareggio_EX_1(seed):
    print('---------------------------func: get_args_viareggio_EX_1---------------------------')
    print('---------------------------EX need be a number to set seed ------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Viareggio_data.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='EX_1', type=str, help='China dataset use img_1 and img_2 as input;')
    parser.add_argument('--idx_file', default='./data/num_idx_ex1.mat',type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--save_model_path',default=path + '/EX1_Viareggio_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',default=path + '/EX1_Viareggio_result' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--GT_ex1', default=r'/data/meiqi.hu/PycharmProjects/ACD/data/ref_EX1.bmp',
                        type=str, help='ground_truth map for EX-1')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)

    return args


def get_args_viareggio_EX_2(seed):
    print('---------------------------func: get_args_viareggio_EX_2---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Viareggio_data.mat', type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='EX_2', type=str, help='China dataset use img_1 and img_2 as input;')
    parser.add_argument('--idx_file',default='./data/num_idx_ex2.mat',type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--save_model_path',default=path + '/EX2_Viareggio_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',default=path + '/EX2_Viareggio_result' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--GT_ex2', default=r'/data/meiqi.hu/PycharmProjects/ACD/data/ref_EX2.bmp',
                        type=str, help='ground_truth map for EX-2')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)

    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)

    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args


def get_args_simulation_EX_3(seed):
    print('---------------------------func: get_args_simulation_EX_3---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Hymap.mat', type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='EX_3', type=str, help='EX_1 use img_1 and img_2_RE as input; EX_2 use img_1 and img_3_RE as input')
    parser.add_argument('--idx_file', default='./data/num_idx_ex3.mat', type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--save_model_path', default=path+'/EX3_Viareggio_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path+'/EX3_Viareggio_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)

    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)

    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args



def train_HyperNet_HACD(args):
    print('---------------------------func: train_HyperNet_HACD---------------------------')
    model, idx, img_2, groundTruth = [], [], [], []
    print('\n')
    data = sio.loadmat(args.data_name)
    print('input data for test:', args.data_name, sep='\n')
    img_1 = data['img_1']
    if args.EX_num == 'EX_1':
        model = HyperNet(BasicBlock, layernum=[127, 64, 128, 64], gamma=args.gamma)  # for EX-1 & EX-2
        img_2 = data['img_2_RE']  # H,W,# C
        print('img_1 and img_2_RE is input for test')
        idx = sio.loadmat(args.idx_file)['idx_EX1']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        groundTruth = Image.open(args.GT_ex1)

    elif args.EX_num == 'EX_2':
        model = HyperNet(BasicBlock, layernum=[127, 64, 128, 64], gamma=args.gamma)  # for EX-1 & EX-2
        img_2 = data['img_3_RE']  # H,W,C
        print('img_1 and img_3_RE is input for test')
        idx = sio.loadmat(args.idx_file)['idx_EX2']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        groundTruth = Image.open(args.GT_ex2)

    elif args.EX_num == 'EX_3':
        model = HyperNet(BasicBlock, layernum=[126, 64, 128, 64], gamma=args.gamma)  # for EX-3
        img_2 = data['img_2']  # H,W,C
        print('img_1 and img_2 is input for test')
        idx = sio.loadmat(args.idx_file)['idx_EX3']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        groundTruth = data['GT']

    # H, W, C = img_1.shape
    X1 = torch.tensor(np.transpose(img_1, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    X2 = torch.tensor(np.transpose(img_2, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    print('input.shape:', X1.shape)
    del img_1, img_2, data

    model.apply(initNetParams_v2)
    init_lr = args.lr
    model.cuda()
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    Tra_ls = []
    print('trainging begins----------------------------')
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        loss = model(X1, X2, idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Tra_ls.append(loss.item())
        if epoch % 10 == 0:
            print('epoch [{}/{}],train:{:.4f}'.format(epoch, args.epochs, loss.item()))
    print('--------------SSL: training is sucessfully down--------------')
    print('--------------model save_path:', args.save_model_path, sep='\n')
    # torch.save(model.state_dict(), args.save_model_path)

    plt.figure(args.EX_num)
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(args.epochs), np.asarray(Tra_ls), 'r-o', label="SSL")
    plt.legend()

    model.eval()
    print('---------------------- fuse feature---------------------')
    f1, f2 = model(X1, X2, 0)  # [1, 32, 450, 375]
    f1, f2 = f1.squeeze(), f2.squeeze()
    f1, f2 = np.array(f1.permute(1, 2, 0)), np.array(f2.permute(1, 2, 0))  # 450, 375,32]

    r = diff_RX(f1, f2)
    x, y, auc = plot_roc(r, groundTruth)
    plt.figure(args.EX_num + ' SSL fuse feature')
    plt.imshow(r, cmap='hot')

    sio.savemat(args.save_result_path, {'r': r, 'x': x, 'y': y, 'auc': auc})
    print('--------------result save_path:', args.save_result_path, sep='\n')

    return r, x, y, auc



def get_args_USA(seed):
    print('---------------------------func: get_args_USA---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/USA.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='USA',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file',
                        default='/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Hermiston/HermistonBinary.mat',
                        type=str, help='path filename of the trained model')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    parser.add_argument('--save_model_path',default=path + '/USA_HyperNet_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path + '/USA_HyperNet_result_' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--save_result_path2', default=path + '/USA_HyperNet_result_KMEAN_' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=260, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)

    return args


def get_args_Bay(seed):
    print('---------------------------func: get_args_Bay---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Bay.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='Bay',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file', default='/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/bayArea/bayArea_gtChanges2.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    parser.add_argument('--save_model_path',
                        default= path + '/Bay_HyperNet_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path2',
                        default= path + '/Bay_HyperNet_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/Bay_HyperNet_resultkmean_'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.08, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args


def get_args_Wetland(seed):
    print('---------------------------func: get_args_Wetland---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Wetland.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='Wetland',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file', default='/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Wetland/Reference_Map_Binary.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    parser.add_argument('--save_model_path',
                        default= path + '/Wetland_HyperNet_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/Wetland_HyperNet_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=260, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.07, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args
    
    
def get_args_River(seed):
    print('---------------------------func: get_args_Bay---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/River.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='River',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file', default='/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/zuixin/groundtruth.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    parser.add_argument('--save_model_path',
                        default= path + '/River_HyperNet_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/River_HyperNet_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path2',
                        default= path + '/River_HyperNet_result_kmean'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=260, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.08, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args



def get_args_Yancheng(seed):
    print('---------------------------func: get_args_Bay---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Yancheng.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='Yancheng',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file', default='/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Farm/label.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    parser.add_argument('--save_model_path',
                        default= path + '/Yancheng_HyperNet_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/Yancheng_HyperNet_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path2',
                        default= path + '/Yancheng_HyperNet_result_kman'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=260, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.07, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args

def get_args_Barbara(seed, mode):
    print('---------------------------func: get_args_Barbara---------------------------')
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--EX_num', default='Barbara',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    if mode == 'Barbara_half1':
        parser.add_argument('--data_name',default='./data/Barbara_half1.mat',
                            type=str, help='path filename of training data')
        parser.add_argument('--idx_file',default='./data/num_idx_Barbara_half1.mat',
                            type=str, help='path filename of the trained model')
        parser.add_argument('--save_model_path', default=path +'/Barbara_half1_HyperNet_' + str(seed) + '.pkl',
                            type=str, help='path filename of the trained model')
        parser.add_argument('--save_result_path', default=path +'/Barbara_half1_HyperNet_result' + str(seed) + '.mat',
                            type=str, help='path filename of the trained model')
    elif mode == 'Barbara_half2':
        parser.add_argument('--data_name', default='./data/Barbara_half2.mat',
                            type=str, help='path filename of training data')
        parser.add_argument('--idx_file', default='./data/num_idx_Barbara_half2.mat',
                            type=str, help='path filename of the trained model')

        parser.add_argument('--save_model_path', default=path + '/Barbara_half2_HyperNet_' + str(seed) + '.pkl',
                            type=str, help='path filename of the trained model')
        parser.add_argument('--save_result_path', default=path + '/Barbara_half2_HyperNet_result' + str(seed) + '.mat',
                            type=str, help='path filename of the trained model')
    else:
        print('-----mode should be Barbara_half1 or Barbara_half2-----------')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args

def normalize_result(subimg):
    subimg_min = np.min(subimg)
    subimg_max = np.max(subimg)
    subimg_n = (subimg - subimg_min) / (subimg_max - subimg_min)
    return subimg_n
def get_loss_idx(idx,cidx_len,ucidx_len):
    idx=np.reshape(idx,-1)
    idx_change=(np.where(idx==0))[0]
    np.random.shuffle(idx_change)
    print(idx_change)
    idx_unchange=(np.where(idx==0))[0]
    np.random.shuffle(idx_unchange)
    idx1=idx_change[0:cidx_len]
    idx2=idx_unchange[0:ucidx_len]
    select_loss_idx=np.append(idx1,idx2)
    return select_loss_idx


def train_HyperNet_HBCD(args):
    print('---------------------------func: train_HyperNet_HBCD---------------------------')
    print('\n')
    model, idx = [], []
    if args.EX_num == 'USA':
        
        #idx = sio.loadmat(args.idx_file)['idx_usa']
        idx = sio.loadmat(args.idx_file)['HermistonBinary']
        #idx=np.arange(780000)
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        img_1 = sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Hermiston/hermiston2004.mat')['HypeRvieW']
        listing = list(range(242))
        img_1 = np.delete(img_1,listing[0:7]+ listing[57:81]+listing[119:130]+listing[164:186]+listing[218:241], axis=2)
        select_loss_idx=sio.loadmat('/home/ihclserver/Desktop/change_dec/HyperNet-main/Hermistom_cvx.mat')['background_cvx']
        select_loss_idx=select_loss_idx[0]-1
        img_2 = sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Hermiston/hermiston2007.mat')['HypeRvieW']
        listing = list(range(242))
        img_2 = np.delete(img_2, listing[0:7]+ listing[57:81]+listing[119:130]+listing[164:186]+listing[218:241], axis=2)
        R,C,B=img_1.shape
        #model = HyperNet(BasicBlock, layernum=[155, 72, 144+155, 112], gamma=args.gamma,size=(1,B,R,C))  # for USA
        model = HyperNet(BasicBlock, layernum=[155, 72, 144, 112], gamma=args.gamma,size=(1,B,R,C))  # for USA
    elif args.EX_num == 'Bay':
        
        idx = sio.loadmat(args.idx_file)['HypeRvieW']
        idx[idx!=0]=1
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        img_1 =sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/bayArea/Bay_Area_2013.mat')['HypeRvieW']
        listing = list(range(224))
        img_1 = np.delete(img_1,listing[107:114]+listing[153:169], axis=2)
        
        img_2 = sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/bayArea/Bay_Area_2015.mat')['HypeRvieW']
        listing = list(range(224))
        img_2 = np.delete(img_2,listing[107:114]+listing[153:169], axis=2)
        R,C,B=img_1.shape
        model =HyperNet(BasicBlock,  layernum=[201, 101, 202, 112], gamma=1,size=(1,B,R,C))  # for Bay area dataset
    elif args.EX_num == 'Yancheng':
        
        idx = sio.loadmat(args.idx_file)['label']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        img_1 =sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Farm/Farm1.mat')['imgh']  
        listing = list(range(224))
        img_1 = np.delete(img_1,listing[107:108], axis=2)
        img_2 = sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Farm/Farm2.mat')['imghl']
        listing = list(range(224))
        img_2 = np.delete(img_2,listing[107:108], axis=2)
        R,C,B=img_1.shape
        #model =HyperNet(BasicBlock,  layernum=[154, 77, 154+154, 77], gamma=1,size=(1,B,R,C))  
        model =HyperNet(BasicBlock,  layernum=[154, 77, 154, 77], gamma=1,size=(1,B,R,C)) 
        
    elif args.EX_num == 'Wetland':
       #model =HyperNet(BasicBlock,  layernum=[132, 66, 132, 112], gamma=1)  # for Bay area dataset
       idx = sio.loadmat(args.idx_file)['Ref_map_binary']
       idx = torch.from_numpy(idx.squeeze()).cuda()
       print('unchanged idx path:', args.idx_file)
       img_1 =sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Wetland/PreImg_2006.mat')["img_2006"]
       img_2 = sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/Wetland/PostImg_2007.mat')['img_2007']
       R,C,B=img_1.shape
       model =HyperNet(BasicBlock,  layernum=[132, 66, 132, 112], gamma=1,size=(1,B,R,C))
    elif args.EX_num == 'River':
       
       idx = sio.loadmat(args.idx_file)['lakelabel_v1']
       idx[idx==255]=1
       idx = torch.from_numpy(idx.squeeze()).cuda()
       print('unchanged idx path:', args.idx_file)
       img_1 =sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/zuixin/river_before.mat')["river_before"]
       listing = list(range(198))
       img_1 = np.delete(img_1,listing[0:4]+listing[48:58]+listing[67:74]+listing[89:107]+listing[138:165]+[listing[176]]+listing[195:198], axis=2)
       listing = list(range(198))
       img_2 = sio.loadmat('/home/ihclserver/Desktop/change_dec/hyperspectral change detection methods/Dataset/zuixin/river_after.mat')['river_after']
        #listing = list(range(224))
       R,C,B=img_1.shape
       print('River shape',np.shape(img_2))
       select_loss_idx=sio.loadmat('/home/ihclserver/Desktop/change_dec/HyperNet-main/river_cvx.mat')['background_cvx']
       select_loss_idx=select_loss_idx[0]-1
       img_2 = np.delete(img_2,listing[0:4]+listing[48:58]+listing[67:74]+listing[89:107]+listing[138:165]+[listing[176]]+listing[195:198], axis=2)       
       #model =HyperNet(BasicBlock,  layernum=[128, 72, 144+128, 112], gamma=1,size=(1,B,R,C)) 
       model =HyperNet(BasicBlock,  layernum=[128, 72, 144, 112], gamma=1,size=(1,B,R,C))  # for Bay area dataset
    elif args.EX_num == 'Barbara':
        print('------------training for Barbara dataset ------------------')
        model = HyperNet(BasicBlock, layernum=[224, 112, 224, 112], gamma=1)  # for Santa Barbara dataset
        idx = sio.loadmat(args.idx_file)['idx_barbara']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
    select_loss_idx=get_loss_idx(idx.cpu().numpy(),0,8912)
    select_loss_idx = torch.from_numpy(select_loss_idx).cuda()
    model.apply(initNetParams_v2)
    print('----------------model.apply(initNetParams_v2)-----------------')

    #data = sio.loadmat(args.data_name)
    #print('input data for test:', args.data_name, sep='\n')
    
    #img_1 = data['img_1']
    #img_2 = data['img_2']  # H,W,C
    print('img_1 and img_2 is input for test')
    H, W, C = img_1.shape
    X1 = torch.tensor(np.transpose(img_1, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    X2 = torch.tensor(np.transpose(img_2, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    print('input.shape:', X1.shape)
    #del img_1, img_2, data
    del img_1, img_2

    init_lr = args.lr
    model.cuda()
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    Tra_ls = []
    print('trainging begins----------------------------')
    print(np.shape(idx))
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        #print(np.shape(X1),np.shape(X2),np.shape(idx))
        loss = model(X1, X2,select_loss_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Tra_ls.append(loss.item())
        if epoch % 10 == 0:
            print('epoch [{}/{}],train:{:.4f}'.format(epoch, args.epochs, loss.item()))
    print('--------------SSL: training is sucessfully down--------------')
    print('--------------model save_path:', args.save_model_path, sep='\n')
    torch.save(model.state_dict(), args.save_model_path)
    plt.figure(args.EX_num)
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(args.epochs), np.asarray(Tra_ls), 'r-o', label="SSL")
    plt.legend()

    model.eval()
    f1, f2 = model(X1, X2, 0)
    #r=diff_RX(f1, f2)
    
   
    f1, f2 = f1.squeeze(), f2.squeeze()
    f1 = f1.permute(1, 2, 0)
    f2 = f2.permute(1, 2, 0)
    f1 = f1.reshape([-1, f1.shape[2]])
    f2 = f2.reshape([-1, f2.shape[2]])
    print('fi size is ',np.shape(f1))
    print('f2 size is ',np.shape(f2))
    
    
   
    mse_criterion =nn.CosineSimilarity(dim=1, eps=1e-6)
    MSE_result= mse_criterion(f1, f2)
   # MSE_result_n=normalize_result(MSE_result.reshape([H, W]))
    MSE_result=normalize_result(MSE_result.reshape([H, W]).numpy())
    #print(MSE_result)
    MSE_result_kmean=kmean(MSE_result)
    #sio.savemat(args.save_result_path, { 'MSE_result': MSE_result})
    #sio.savemat(args.save_result_path2, { 'MSE_result_kmean': MSE_result_kmean})
    save_png=str(args.save_result_path+'MSE_result.png')
    print(idx)
  
    print("-------- HyperNet performance-----------")
    OA, kappa, precision, recall, F1=calculate_metrics(idx.cpu().numpy(), MSE_result_kmean)
    print('OA',OA)
    print('precetion',precision)
    print('recall',recall)
    print('kappa',kappa)
    print('F1',F1)
   

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(MSE_result,cmap='gray')
    plt.title('MSE')
    plt.subplot(1,3,2)
    plt.imshow(MSE_result_kmean,cmap='gray')
    plt.title('MSE_resultkmean(normalize 1)')
    plt.subplot(1,3,3)
    plt.imshow(idx.cpu().numpy(),cmap='gray')
    plt.title('gt')
    plt.show()
    #plt.savefig(save_png)

    
    
    print('--------------save_result_path:', args.save_result_path, sep='\n')
    return OA,precision,recall,kappa,F1

def diff_RX(img1, img2):
    # img[H, W, C]
    H, W, C = img1.shape
    img1_2d = np.reshape(img1, [H * W, C])
    img2_2d = np.reshape(img2, [H * W, C])
    diff = np.absolute(img1_2d - img2_2d)
    diff_mean = np.mean(diff, axis=0)
    print('diff_mean shape:', diff_mean.shape)  # [1,C]

    diff_cov = np.cov(diff, rowvar=False)
    diff_mean0 = diff - diff_mean  # [H*W,C]
    del img1, img2, C, img1_2d, img2_2d, diff
    T1 = np.matmul(diff_mean0, np.linalg.inv(diff_cov))  # [H*W,C]
    T2 = np.sum(T1 * diff_mean0, axis=1)
    print('shape:', T2.shape)

    T2 = np.reshape(T2, [H, W])
    plt.figure('Diff_RX')
    plt.imshow(T2, cmap='hot')
    return T2


# for evaluating the performance of the anomaly change detection result
def plot_roc(predict, ground_truth):
    """
    INPUTS:
     predict - anomalous change intensity map
     ground_truth - 0or1
    OUTPUTS:
     X, Y for ROC plotting
     auc
    """
    max_value = np.max(ground_truth)
    if max_value != 1:
        ground_truth = ground_truth / max_value

    # initial point（1.0, 1.0）
    x = 1.0
    y = 1.0
    hight_g, width_g = ground_truth.shape
    hight_p, width_p = predict.shape
    if hight_p != hight_g:
        predict = np.transpose(predict)

    ground_truth = ground_truth.reshape(-1)
    predict = predict.reshape(-1)
    # compuate the number of positive and negagtive pixels of the ground_truth
    pos_num = np.sum(ground_truth == 1)
    neg_num = np.sum(ground_truth == 0)
    # step in axis of  X and Y
    x_step = 1.0 / neg_num
    y_step = 1.0 / pos_num
    # ranking the result map
    index = np.argsort(list(predict))
    ground_truth = ground_truth[index]
    """ 
    for i in ground_truth:
     when ground_truth[i] = 1, TP minus 1，one y_step in the y axis, go down
     when ground_truth[i] = 0, FP minus 1，one x_step in the x axis, go left
    """
    X = np.zeros(ground_truth.shape)
    Y = np.zeros(ground_truth.shape)
    for idx in range(0, hight_g * width_g):
        if ground_truth[idx] == 1:
            y = y - y_step
        else:
            x = x - x_step
        X[idx] = x
        Y[idx] = y

    auc = -np.trapz(Y, X)
    if auc < 0.5:
        auc = -np.trapz(X, Y)
        t = X
        X = Y
        Y = t
    print('auc: ', auc)
    return X, Y, auc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def zz(seed):
    print('---------------everything will be ok-------------')
    print('current seed:', seed)

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def compute_acc(gt_test,pred_test_fdssc):
    gt_test=np.reshape(gt_test,(-1,1))
    pred_test_fdssc=np.reshape(pred_test_fdssc,(-1,1))
    precision=metrics.precision_score(pred_test_fdssc.astype(np.int16),
                                          gt_test.astype(np.int16), average=None)
    test_pre = precision[1]
    recall=metrics.recall_score(pred_test_fdssc.astype(np.int16),
                                          gt_test.astype(np.int16), average=None)
    test_recall = recall[1]
    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test)
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test)
    print(confusion_matrix_fdssc)
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test)
    F1=2*(test_pre*test_recall)/(test_recall+test_pre)
    F1.astype(np.int16)
    return test_pre,test_recall,overall_acc_fdssc,average_acc_fdssc,kappa,F1

def calculate_metrics(gt, output):
    gt=np.reshape(gt,(-1,1))
    output=np.reshape(output,(-1,1))
    #TN, FP, FN, TP = metrics.confusion_matrix(gt, output).ravel()
    con = metrics.confusion_matrix(gt, output)
    print(con)
    TP = con[1, 1]
    FP = con[0, 1]
    FN = con[1, 0]
    TN = con[0, 0]
    print('TP',TP)
    print('FP',FP)
    print('FN',FN)
    print('TN',TN)
    OA = (TP + TN) / (TP + TN + FP + FN)
    PRE = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / (TP + TN + FP + FN) ** 2
    kappa = (OA - PRE) / (1 - PRE)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (recall + precision)

    return OA, kappa, precision, recall, F1

if __name__ == "__main__":
    import time
    OA=0
    K=0
    PRE=0
    RECALL=0
    TIME=[]
    F1=0
    iter=1
    data='Yancheng'
    for i in range(iter):
        train_start= time.time()
        if data == 'River':
            dataset: 'River'
            SEED = np.arange(1, 6)
            for i in np.arange(0, 1):
                seed = SEED[i]
                print('\n')
                args = get_args_River(seed)
                setup_seed(args.seed)
                zz(seed)
                overall_acc_fdssc,test_pre,test_recall,kappa,f1=train_HyperNet_HBCD(args) 
            
            
        if data == 'Hermiston':
            dataset: 'USA(Hermiston)'
            SEED = np.arange(1, 6)
            for i in np.arange(0, 1):
                seed = SEED[i]
                print('\n')
                args = get_args_USA(seed)
                setup_seed(args.seed)
                zz(seed)
                overall_acc_fdssc,test_pre,test_recall,kappa,f1=train_HyperNet_HBCD(args)
            
        if data == 'bayArea':
            dataset: 'Bay'
            SEED = np.arange(1, 6)
            for i in np.arange(0, 1):
                seed = SEED[i]
                print('\n')
                args = get_args_Bay(seed)
                setup_seed(args.seed)
                zz(seed)
                overall_acc_fdssc,test_pre,test_recall,kappa,f1=train_HyperNet_HBCD(args)
            
            
        if data == 'Wetland':
            dataset: 'Wetland'
            SEED = np.arange(1, 6)
            for i in np.arange(0, 1):
                seed = SEED[i]
                print('\n')
                args = get_args_Wetland(seed)
                setup_seed(args.seed)
                zz(seed)
                overall_acc_fdssc,test_pre,test_recall,kappa,f1=train_HyperNet_HBCD(args)  

        if data == 'Yancheng':
            dataset: 'Yancheng'
            SEED = np.arange(1, 6)
            for i in np.arange(0, 1):
                seed = SEED[i]
                print('\n')
                args = get_args_Yancheng(seed)
                setup_seed(args.seed)
                zz(seed)
                overall_acc_fdssc,test_pre,test_recall,kappa,f1=train_HyperNet_HBCD(args) 
        time_finish= time.time() 

        time_one_roop=time_finish-train_start
        TIME.append(time_one_roop)
        OA=OA+overall_acc_fdssc
        #OA.append(overall_acc_fdssc)
        K=K+kappa
        #K.append(kappa)
        PRE=PRE+test_pre
       # PRE.append(test_pre)
        RECALL=RECALL+test_recall
        #RECALL.append(test_recall)
        F1=F1+f1
        #with torch.no_grad():
            #model.load_state_dict(torch.load(args.save_model_path))
            #model.eval()
            #output = model(net_input_before, net_input_after, net_input_concat)
            
    print('=============Finish 10 times training=============')
    print('OA',OA/(iter))
    print('kappa',K/(iter))
    print('recall',RECALL/(iter))
    print('precetion',PRE/(iter))
    print('F1',F1/(iter))    
    print('time',np.mean(TIME))
    """ For (ACD)anomalous change detection of the HyperNet """
        # dataset:Viareggio 2013 EX1
        # SEED = np.arange(1, 11)
        # EX1_time = []
        # for i in np.arange(0, 5):
        #     time1 = time.clock()
        #     seed = SEED[i]
        #     print('\n')
        #     args = get_args_viareggio_EX_1(seed)
        #     setup_seed(args.seed)
        #     zz(seed)
        #     train_HyperNet_HACD(args)
        #     time2 = time.clock()
        #     EX1_time.append(time2 - time1)

        # dataset: Viareggio 2013 EX2
        # SEED = np.arange(1, 11)
        # EX1_time = []
        # for i in np.arange(0, 5):
        #     time1 = time.clock()
        #     seed = SEED[i]
        #     print('\n')
        #     args = get_args_viareggio_EX_2(seed)
        #     setup_seed(args.seed)
        #     zz(seed)
        #     train_HyperNet_HACD(args)
        #     time2 = time.clock()
        #     EX1_time.append(time2 -time1)

        # dataset: simulated_Hymap dataset
        # SEED = np.arange(1, 11)
        # EX3_time = []
        # for i in np.arange(0, 5):
        #     time1 = time.clock()
        #     seed = SEED[i]
        #     print('\n')
        #     args = get_args_simulation_EX_3(seed)
        #     setup_seed(args.seed)
        #     zz(seed)
        #     train_HyperNet_HACD(args)
        #     time2 = time.clock()
        #     EX3_time.append(time2 -time1)

    """ For (BCD)binary change detection of the HyperNet """
    

        # dataset: Barbara dataset
        # SEED = np.arange(1, 11)
        # mode = 'Barbara_half1'
        # Barbara_time = []
        # for i in np.arange(0, 5):
        #     time1 = time.clock()
        #     seed = SEED[i]
        #     print('\n')
        #     args = get_args_Barbara(seed, mode)
        #     setup_seed(args.seed)
        #     zz(seed)
        #     train_HyperNet_HBCD(args)
        #     time2 = time.clock()
        #     Barbara_time.append(time2 - time1)










