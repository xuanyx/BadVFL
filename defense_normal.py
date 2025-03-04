import os
import sys
import argparse
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
import numpy as np
from badnets import *
from PIL import Image
import torch.nn.functional as F
import random
from logger import Logger
from data_loader import *
import math

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Backdoor Attack in Vertical Federated Learning!')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128, metavar='LR', help='learning rate')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="vgg16",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--depth', type=int, default=28, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--dataset', type=str, default="imagenet", help="choose from cifar10,svhn")
parser.add_argument('--num_classes',type=int, default=10,help='the num of class')

parser.add_argument('--resume', type=bool, default=False,help = "whether to resume from a file")
parser.add_argument('--resume1', type=str, default='./backdoor_model/s15_6_500cifar10_resnet1814_badnets__checkpoint.pth.tar', help='whether to resume training, default: None')
parser.add_argument('--resume2', type=str, default='./backdoor_model/s15_6_500cifar10_resnet1824_badnets__checkpoint.pth.tar', help='whether to resume training, default: None')
parser.add_argument('--resume3', type=str, default='./backdoor_model/s15_6_500cifar10_resnet1834_badnets__checkpoint.pth.tar', help='whether to resume training, default: None')

parser.add_argument('--target_idx', type=int, default=5, help='target sample index')
parser.add_argument('--num', type=int, default=5, help='the number of sample to change img in each batch')
parser.add_argument('--poison_num', type=int, default=100, help='the min number of total poison sample ')
parser.add_argument('--out_dir', type=str, default='./defense_normal_imagenet', help='dir of output')
parser.add_argument('--laplace_param', type=float, default=1e-2, help='laplace noise param')
parser.add_argument('--thre', type=float, default=0.6, help='thre for similarity')
parser.add_argument('--an', type=str, default="11_normal", help='type name')
parser.add_argument('--gpu', type=str, default='1', help='gpu')

args = parser.parse_args()

DEVICE = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# setup data loader
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=2)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='../../data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True)
if args.dataset == "GTSRB":
    train_imgs,train_y,test_imgs,test_y,class_num = load_data_from_GTSRB("../../data/GTSRB/GTSRB_JPG",args.seed)
    trainset = Data_Set(train_imgs,train_y,is_train=True,dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    testset = Data_Set(test_imgs,test_y,is_train=False,dataset=args.dataset)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print("GTSRB!!!")
if args.dataset == "imagenet":
    train_imgs,train_y,test_imgs,test_y,class_num = load_data_from_imagenet("../../data/ILSVRC2012",args.seed)
    trainset = Data_Set(train_imgs,train_y,is_train=True,dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,num_workers=2)
    testset = Data_Set(test_imgs,test_y,is_train=False,dataset=args.dataset)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=2)
    print("imagenet!!!")

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN()
    net = "smallcnn"
if args.net == "vgg11":
    model = VGG11(num_classes=args.num_classes)
    net = "vgg11"
if args.net == "vgg16":
    modelA = VGG("VGG16",num_classes=args.num_classes).to(DEVICE)
    modelB = VGG("VGG16",num_classes=args.num_classes).to(DEVICE)
    modelC = FCNN5(num_classes=args.num_classes).to(DEVICE)
    net = "VGG16"
if args.net == "resnet18":
    modelA = ResNet18(num_classes=args.num_classes).to(DEVICE)
    modelB = ResNet18(num_classes=args.num_classes).to(DEVICE)
    modelC = FCNN4(num_classes=args.num_classes).to(DEVICE)
    net = "resnet18"
if args.net == "WRN":
  # e.g., WRN-34-10
    modelA = Wide_ResNet(depth=args.depth, num_classes=args.num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate)
    modelB = Wide_ResNet(depth=args.depth, num_classes=args.num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate)
    modelC = FCNN4()
    net = "WRN{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
if args.net == 'WRN_madry':
  # e.g., WRN-32-10
    model = Wide_ResNet_Madry(depth=args.depth, num_classes=args.num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate)
    net = "WRN_madry{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
print(net)


optimizer1 = optim.SGD(modelA.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  #momentum=args.momentum,
optimizer2 = optim.SGD(modelB.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer3 = optim.SGD(modelC.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # momentum=args.momentum,


img_save=transforms.ToPILImage()

def save_img(img1,img2,target):  # img1,img2,target  img1,target
    filename1=args.dataset+str(target)+'_111.png'    #target.item()
    filename2=args.dataset+str(target)+'_112.png'
    # filename1 = str(target)+"_all.png"
    # filename2 = str(target)+"_2.png"
    # img1 = resize(img1)
    img1 = img_save(img1)
    img2=img_save(img2)
    img1.save(os.path.join("img_poison",filename1), quality=100, subsampling=0)
    img2.save(os.path.join("img_poison",filename2), quality=100, subsampling=0)
    return


if args.resume==False:
    # logger = Logger(os.path.join(args.out_dir,"test.txt"),title="Clean")
    logger = Logger(os.path.join(args.out_dir, args.an+str(args.laplace_param)+'_'+str(args.thre)+'_s'+str(args.seed)+'_log_'+str(args.num)+"_"+str(args.poison_num)+"_"+str(args.target_idx)+'_'+args.dataset+"_"+args.net+'_badnets.txt'), title="Clean")
    # logger.set_names(['GradA', 'GradB'])
    logger.set_names(['Epoch', 'Natural Test Acc','backdoor ASR'])


def train(modelA, modelB, modelC,train_loader,test_loader,start_epoch):
    if args.resume:
        print("resume!")
        checkpoint1 = torch.load(args.resume1)
        checkpoint2 = torch.load(args.resume2)
        checkpoint3 = torch.load(args.resume3)
        start_epoch = checkpoint1['epoch']
        modelA.load_state_dict(checkpoint1['state_dict'])
        modelB.load_state_dict(checkpoint2['state_dict'])
        modelC.load_state_dict(checkpoint3['state_dict'])
        clean_acc = test_clean(modelA, modelB, modelC, start_epoch,test_loader)
        badnets_acc = test_badnets(modelA, modelB, modelC, start_epoch, test_loader,trainset[args.target_idx][1])
        sys.exit(0)



    # epoch = train_clean(modelA, modelB, modelC,train_loader,test_loader,start_epoch)
    epoch=-1
    train_badnets(modelA, modelB, modelC,train_loader,test_loader, epoch+1)

    return

def train_badnets(modelA, modelB, modelC,train_loader,test_loader,start_epoch):
    best_acc = 0
    next_epoch, cluster, target_label = get_cluster(modelA, modelB, modelC,train_loader,test_loader,start_epoch)
    for epoch in range(next_epoch+1, args.epochs):
        adjust_learning_rate(optimizer1, epoch + 1)
        adjust_learning_rate(optimizer2, epoch + 1)
        adjust_learning_rate(optimizer3, epoch + 1)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)  #[128,3,32,32]
            data1,data2 = torch.chunk(data, 2, 3)
            if batch_idx in cluster.keys():
                other_class_idx = np.delete(np.arange(args.batch_size),np.array(cluster[batch_idx]))
                for i in cluster[batch_idx]:
                    j = np.random.choice(other_class_idx,replace = False)
                    data2[i] = infect_X(data2[j].clone().detach(),args.dataset,DEVICE,position='mid')  #同一个batch里的替换效果会不会较差？

            modelA.train()
            modelB.train()
            modelC.train()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            act1, output1 = modelA(data1)
            act2, output2 = modelB(data2)
            size = list(act1.size())
            act1 = torch.reshape(act1, (size[0], -1))
            act2 = torch.reshape(act2, (size[0], -1))

            act1_copy = act1.clone().detach()
            act2_copy = act2.clone().detach()

            # for i in range(act1.shape[0]):
            #     normal_noise12 =  np.random.normal(0,args.laplace_param,act1[0].shape)
            #     normal_noise12 = torch.tensor(normal_noise12).float().to(DEVICE)
            #     normal_noise13 =  np.random.normal(0,args.laplace_param,act2[0].shape)
            #     normal_noise13 = torch.tensor(normal_noise13).float().to(DEVICE)
            #     act1_copy[i] = act1_copy[i].clone().detach() + normal_noise12.clone().detach()
            #     act2_copy[i] = act2_copy[i].clone().detach() + normal_noise13.clone().detach()

            act1_copy.requires_grad=True
            act2_copy.requires_grad=True
            data3 = torch.cat((act1_copy,act2_copy), axis=1 )
            output3 = modelC(data3)

            loss = nn.CrossEntropyLoss(reduction='mean')(output3, target)

            loss.backward(retain_graph=True)
            optimizer3.step()

            act1_grad = act1_copy.grad.clone().detach()
            act2_grad = act2_copy.grad.clone().detach()

            act1_grad_noise = act1_grad
            act2_grad_noise = act2_grad
            for i in range(act1_grad.shape[0]):

                normal_noise2 =  np.random.normal(0,args.laplace_param,act1_grad[0].shape)
                normal_noise2 = torch.tensor(normal_noise2).float().to(DEVICE)
                normal_noise3 =  np.random.normal(0,args.laplace_param,act2_grad[0].shape)
                normal_noise3 = torch.tensor(normal_noise3).float().to(DEVICE)
                act1_grad_noise[i] = act1_grad[i].clone().detach() + normal_noise2.clone().detach()
                act2_grad_noise[i] = act2_grad[i].clone().detach() + normal_noise3.clone().detach()
                # nn2,pos2 = torch.topk(torch.abs(act1_grad_noise.clone().detach()),k=100,largest=False,sorted=False)
                # nn3,pos3 = torch.topk(torch.abs(act2_grad_noise.clone().detach()),k=100,largest=False,sorted=False)
                # act1_grad_noise[i][pos2]=0
                # act2_grad_noise[i][pos3]=0


            optimizer2.zero_grad()
            act2.backward(act2_grad_noise.clone().detach())
            optimizer2.step()
            act2_copy.grad.zero_()

            optimizer1.zero_grad()
            act1.backward(act1_grad_noise.clone().detach())
            optimizer1.step()
            act1_copy.grad.zero_()

        clean_acc = test_clean(modelA, modelB, modelC, epoch, test_loader)
        badnets_acc = test_badnets(modelA, modelB, modelC, epoch, test_loader,target_label)
        logger.append([epoch + 1, clean_acc, badnets_acc])

        if clean_acc > best_acc:
            best_acc = clean_acc
            rootA = args.an+str(args.laplace_param)+"_"+str(args.thre)+"_s"+str(args.seed)+'_'+str(args.num)+"_"+str(args.poison_num)+args.dataset+"_"+args.net+'1'+str(args.target_idx)+'_badnets_'+'_checkpoint.pth.tar'
            rootB = args.an+str(args.laplace_param)+"_"+str(args.thre)+"_s"+str(args.seed)+'_'+str(args.num)+"_"+str(args.poison_num)+args.dataset+"_"+args.net+'2'+str(args.target_idx)+'_badnets_'+'_checkpoint.pth.tar'
            rootC = args.an+str(args.laplace_param)+"_"+str(args.thre)+"_s"+str(args.seed)+'_'+str(args.num)+"_"+str(args.poison_num)+args.dataset+"_"+args.net+'3'+str(args.target_idx)+'_badnets_'+'_checkpoint.pth.tar'
            # torch.save(state, adv_test_root)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': modelA.state_dict(),
            #     'clean_acc': best_acc,
            #     'badnets_acc':badnets_acc,
            # },filename = rootA)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': modelB.state_dict(),
            #     'clean_acc': best_acc,
            #     'badnets_acc':badnets_acc,
            # },filename = rootB)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': modelC.state_dict(),
            #     'clean_acc': best_acc,
            #     'badnets_acc':badnets_acc,
            # },filename = rootC)
    return

import torch.nn.functional as F

from sklearn.decomposition import FastICA, PCA
# projector = FastICA(n_components=50, max_iter=500, tol=0.0001)
projector = PCA(n_components=100)

def get_cluster(modelA, modelB, modelC,train_loader,test_loader,start_epoch):
    target_img,target_label = trainset[args.target_idx]  #target attack img
    print("target_label:",target_label)
    target_img = torch.chunk(target_img, 2, 2)[1]
    # target_img = torch.chunk(target_img, 2, 2)
    # save_img(target_img[0],target_img[1],target_label)
    # sys.exit(0)
    if_changed = np.zeros(51000)
    if args.dataset == "cifar10":
        other_class = np.zeros(50000)
    else:
        other_class = np.zeros(10531)
    cluster = {}
    count = 0
    right_idx = 0
    target_grad = None

    id1=[]
    id2=[]
    simi_all=[]
    true_label = []
    thre=0

    for epoch in range(0, 2*args.batch_size//args.num): #2*args.batch_size//args.num
        adjust_learning_rate(optimizer1, epoch + 1)
        adjust_learning_rate(optimizer2, epoch + 1)
        adjust_learning_rate(optimizer3, epoch + 1)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)  #[128,3,32,32]
            data1,data2 = torch.chunk(data, 2, 3)

            if batch_idx in cluster.keys():
                other_class_idx = np.delete(np.arange(args.batch_size),np.array(cluster[batch_idx]))
                for i in cluster[batch_idx]:
                    j = np.random.choice(other_class_idx,replace = False)
                    data2[i] = infect_X(data2[j].clone().detach(),args.dataset,DEVICE,position="mid")  #同一个batch里的替换效果会不会较差？

            if count < args.poison_num:
                choice_idx = np.where(if_changed[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]==0)[0]
                if choice_idx.size < args.num:
                    change_idx = choice_idx
                else:
                    change_idx = np.random.choice(choice_idx,args.num,replace=False)

                # target = target.cpu().numpy()
                # # change_idx = np.random.choice(np.where(target == target_label)[0],4,replace=False)
                # change_idx = np.random.choice(np.delete(np.arange(0,data2.shape[0]),np.where(target == target_label)[0]),4,replace=False)
                # target = torch.from_numpy(target).to(DEVICE)

                data2[change_idx] = target_img.to(DEVICE)
                if_changed[batch_idx*args.batch_size+change_idx] = 1


            modelA.train()
            modelB.train()
            modelC.train()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            act1, output1 = modelA(data1)
            act2, output2 = modelB(data2)
            # print(act2)
            # sys.exit(0)
            size = list(act1.size())
            act1 = torch.reshape(act1, (size[0], -1))
            act2 = torch.reshape(act2, (size[0], -1))

            act1_copy = act1.clone().detach()
            act2_copy = act2.clone().detach()
            # for i in range(act1.shape[0]):
            #     normal_noise12 =  np.random.normal(0,3,act1[0].shape)
            #     normal_noise12 = torch.tensor(normal_noise12).float().to(DEVICE)
            #     normal_noise13 =  np.random.normal(0,3,act2[0].shape)
            #     normal_noise13 = torch.tensor(normal_noise13).float().to(DEVICE)
            #     act1_copy[i] = act1_copy[i].clone().detach() + normal_noise12.clone().detach()
            #     act2_copy[i] = act2_copy[i].clone().detach() + normal_noise13.clone().detach()
                # print(normal_noise12)
                # print(normal_noise13)
            act1_copy.requires_grad=True
            act2_copy.requires_grad=True
            data3 = torch.cat((act1_copy,act2_copy), axis=1 )
            output3 = modelC(data3)

            loss1 = nn.CrossEntropyLoss(reduction='mean')(output3, target)

            loss1.backward(retain_graph=True)
            optimizer3.step()

            act1_grad = act1_copy.grad.clone().detach()
            act2_grad = act2_copy.grad.clone().detach()
            # print("act2:",act2_grad[0])
            # print("act2_sign:",torch.sign(act2_grad[0]))
            act1_grad_noise = act1_grad.clone().detach()
            act2_grad_noise = act2_grad.clone().detach()
            for i in range(act1_grad.shape[0]):

                normal_noise2 =  np.random.normal(0,args.laplace_param,act1_grad[0].shape)
                normal_noise2 = torch.tensor(normal_noise2).float().to(DEVICE)
                normal_noise3 =  np.random.normal(0,args.laplace_param,act2_grad[0].shape)
                normal_noise3 = torch.tensor(normal_noise3).float().to(DEVICE)
                act1_grad_noise[i] =  act1_grad[i].clone().detach() + normal_noise2.clone().detach()
                act2_grad_noise[i] =  act2_grad[i].clone().detach() + normal_noise3.clone().detach()
            # print("act2_noise:",act2_grad_noise[0])
            # print("act2_noise_sign:",torch.sign(act2_grad_noise[0]))
            # aa = torch.sign(act2_grad[0]).eq(torch.sign(act2_grad_noise[0])).nonzero()
            # # print(aa)
            # print("size:",aa.shape[0])

            optimizer2.zero_grad()
            act2.backward(act2_grad_noise.clone().detach())
            optimizer2.step()

            optimizer1.zero_grad()
            act1.backward(act1_grad_noise.clone().detach())
            optimizer1.step()

            data2_grad = act2_grad_noise.clone().detach().view(args.batch_size,-1)
            act1_copy.grad.zero_()
            act2_copy.grad.zero_()

            # reduced_data2_grad = torch.from_numpy(projector.fit_transform(data2_grad.cpu().numpy()))

            if batch_idx == math.floor(args.target_idx / args.batch_size):
                target_grad = data2_grad[args.target_idx % args.batch_size]


            if target_grad is None:
                continue
            # if count < args.poison_num:
            #     for i in change_idx:
            #         simi = angular_distance(target_grad,data2_grad[i])
            #         # print(simi)
            #         simi_all.append(simi.cpu())
            #         id1.append(batch_idx)
            #         id2.append(i)
            #         true_label.append(target[i])

            if count < args.poison_num:
                same_class_idx = []
                for i in range(data2_grad.shape[0]):  #  range(data2_grad.shape[0])
                    simi = angular_distance(target_grad,data2_grad[i])
                    # print(simi)
                    # simi = reg_angular_distance(target_grad,data2_grad[i])
                    if simi > args.thre:  #different label  #0.25
                        same_class_idx.append(i)
                        # print(simi)
                        other_class[batch_idx*args.batch_size+change_idx] = 1
                        if target[i] != target_label:
                            print("not equal!! epoch:",epoch)
                            print("idx:",batch_idx,i,simi)
                            print("target:",target[i])

                # if len(same_class_idx) != 0:
                #     count = count + len(same_class_idx)
                #     if count > args.poison_num:
                #         excess_num = count-args.poison_num
                #         del same_class_idx[-excess_num:]
                #         count = count-excess_num
                #     for i in same_class_idx:
                #         if target[i] == target_label:
                #             right_idx = right_idx + 1
                #     if batch_idx in cluster.keys():
                #         cluster[batch_idx] = cluster[batch_idx] + same_class_idx
                #     else:
                #         cluster[batch_idx] = same_class_idx

                # repeat choose for same class
                if len(same_class_idx) !=0:
                    for i in same_class_idx:
                        if batch_idx not in cluster.keys():
                            count=count+1
                            cluster[batch_idx] = [i]
                            if target[i] == target_label:
                                right_idx = right_idx + 1
                        else:
                            if i in cluster[batch_idx]:
                                continue
                            else:
                                count=count+1
                                if target[i] == target_label:
                                    right_idx = right_idx + 1
                                cluster[batch_idx].append(i)
        print("right_idx: %d , count: %d " % (right_idx,count))

        # if count <  args.poison_num and simi_all!=[]:
        #     print("hhhhhhh")
        #     thre,change_dict,count,right_idx,other_class = determine_thre(id1,id2,simi_all,true_label,count,right_idx,target_label,epoch,other_class)
        #     cluster = merge_dict(cluster,change_dict)
        #     id1=[]
        #     id2=[]
        #     simi_all=[]
        #     true_label = []
        #     if count == 0 and epoch % (args.batch_size//args.num)==0:
        #         if_changed = np.zeros(51000)
        # print("right_idx: %d,count: %d" % (right_idx,count))

        clean_acc = test_clean(modelA, modelB, modelC, epoch, test_loader)
        badnets_acc = test_badnets(modelA, modelB, modelC, epoch, test_loader,target_label)
        logger.append([epoch + 1, clean_acc, badnets_acc])

        if count >= args.poison_num:
            break
        if (epoch+1) %(args.batch_size//args.num) == 0 :
            print(epoch+1,args.batch_size//args.num)
            if_changed = other_class

    print("right_idx:",right_idx)
    print("count:",count)
    print(cluster)
    logger.append([0,right_idx,count])
    # sys.exit(0)
    return epoch,cluster,target_label

def determine_thre(id1,id2,simi_all,true_label,count,right_count,target_label,epoch,other_class):
    simi_all = np.array(simi_all)
    # print(simi_all)
    thre = np.percentile(simi_all, 99)
    print(thre)
    print(simi_all.size)
    print(len(id1))
    print(len(id2))
    change_dict = {}
    if thre < 0.1:
        return thre,change_dict,count,right_count,other_class
    for i in range(simi_all.size):
        if simi_all[i] > thre:
            count = count+1
            if id1[i] in change_dict.keys():
                change_dict[id1[i]] = change_dict[id1[i]] + [id2[i]]
                other_class[id1[i]*args.batch_size+id2[i]] = 1
            else:
                change_dict[id1[i]] = [id2[i]]
                other_class[id1[i]*args.batch_size+id2[i]] = 1
            if true_label[i] == target_label:
                right_count = right_count + 1
            else:
                print("not equal!")
                print(true_label[i],id1[i],id2[i])
            if count >= args.poison_num:
                return thre,change_dict,count,right_count,other_class

    return thre,change_dict,count,right_count,other_class

def merge_dict(dict1,dict2):  #dict2 is small one
    for k in dict2.keys():
        if k in dict1.keys():
            dict1[k] = dict1[k] + dict2[k]
        else:
            dict1[k] = dict2[k]
    return dict1


def test_clean(modelA, modelB, modelC, epoch,loader):
    modelA.eval()
    modelB.eval()
    modelC.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # iterator = tqdm(loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            data1,data2 = torch.chunk(inputs, 2, 3)
            with torch.no_grad():
                act1, output1 = modelA(data1)
                act2, output2 = modelB(data2)
                size = list(act1.size())
                act1 = torch.reshape(act1, (size[0], -1))
                act2 = torch.reshape(act2, (size[0], -1))
                data3 = torch.cat((act1,act2), axis=1 )
                output3 = modelC(data3)
                loss = nn.CrossEntropyLoss(reduction='mean')(output3, targets)
            test_loss += loss.item()
            _, predicted = output3.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    acc = 100.*correct/total
    print("epoch: %d, acc: %f "  % (epoch+1, acc))
    return acc

def test_badnets(modelA, modelB, modelC, epoch, loader,target_label):
    modelA.eval()
    modelB.eval()
    modelC.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # iterator = tqdm(loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            poison_targets = torch.ones_like(targets).to(DEVICE)*target_label
            data1 = torch.chunk(inputs, 2, 3)[0]
            data2 = torch.chunk(inputs, 2, 3)[1]  #data[:, :, :, 16:]
            # save_img(data1[0].cpu(),data2[0].cpu(),targets[0].cpu())
            for i in range(data2.shape[0]):
                 data2[i] = infect_X(data2[i].clone().detach(),args.dataset,DEVICE,position='mid')

            with torch.no_grad():
                act1, output1 = modelA(data1)
                act2, output2 = modelB(data2)
                size = list(act1.size())
                act1 = torch.reshape(act1, (size[0], -1))
                act2 = torch.reshape(act2, (size[0], -1))
                data3 = torch.cat((act1,act2), axis=1 )
                output3 = modelC(data3)
            _, predicted = output3.max(1)
            total += targets.size(0)
            correct += predicted.eq(poison_targets).sum().item()


    acc = 100.*correct/total
    print("BadNets!!! epoch: %d, acc: %f "  % (epoch+1, acc))
    return acc

def save_checkpoint(state, checkpoint=args.out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    if epoch >= 150:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def angular_distance(logits1, logits2):
    # _,pos1 = torch.topk(torch.abs(logits1.clone().detach()),k=50,largest=False,sorted=False)
    # logits1 = torch.sign(logits1[pos1])
    # # print(logits1)
    # _,pos2 = torch.topk(torch.abs(logits2.clone().detach()),k=50,largest=False,sorted=False)
    # logits2 = torch.sign(logits2[pos2])
    # print(logits2)
    # _,pos1 = torch.topk(torch.abs(logits1.clone().detach()),k=100,largest=False,sorted=False)
    # _,pos2 = torch.topk(torch.abs(logits2.clone().detach()),k=100,largest=False,sorted=False)
    # logits1 = torch.sign(logits1[pos1])
    # logits2 = torch.sign(logits2[pos1])
    # logits1 = logits1[pos1]
    # logits2 = logits2[pos1]
    numerator = logits1.mul(logits2).sum()
    # print(numerator)
    logits1_l2norm = logits1.mul(logits1).sum().sqrt()
    logits2_l2norm = logits2.mul(logits2).sum().sqrt()
    denominator = logits1_l2norm.mul(logits2_l2norm)

    return torch.div(numerator, denominator)

def angular_distance1(logits1, logits2):
    numerator = logits1.mul(logits2).sum(1)
    logits1_l2norm = logits1.mul(logits1).sum(1).sqrt()
    logits2_l2norm = logits2.mul(logits2).sum(1).sqrt()
    denominator = logits1_l2norm.mul(logits2_l2norm)
    for i, _ in enumerate(numerator):
        if numerator[i] > denominator[i]:
            numerator[i]=denominator[i]
    D = torch.sub(1.0, torch.abs(torch.div(numerator, denominator)))
    return D

def reg_angular_distance(logits1, logits2):
    zero_set = torch.zeros(logits1.shape).to(DEVICE)
    u1 = torch.mean(logits1)
    s1 = torch.var(logits1)
    u2 = torch.mean(logits2)
    s2 = torch.var(logits2)
    # logits1 = (logits1 - u1)/s1
    # logits2 = (logits2 - u2)/s2
    logits1 = (logits1 - u1)
    logits2 = (logits2 - u2)
    # pos1 = torch.where(logits1>-s1,logits1,zero_set)
    # pos1 = torch.where(logits1[pos1]<s1,logits1,zero_set).nonzero()
    # pos2 = torch.where(logits2>-s2 and logits2<s2,logits2,0)
    # pos3 = pos1 + pos2
    # print(pos1)
    # print(pos3)
    # print("logits1:",logits1)
    # print("logits2:",logits2)
    numerator = logits1.mul(logits2).sum()
    logits1_l2norm = logits1.mul(logits1).sum().sqrt()
    logits2_l2norm = logits2.mul(logits2).sum().sqrt()
    denominator = logits1_l2norm.mul(logits2_l2norm)
    # print("numerator:",numerator)
    # print("denominator:",denominator)
    # sys.exit(0)
    return torch.div(numerator, denominator)


def l2_norm(logits1,logits2):
    numerator = logits1 - logits2
    dis = torch.norm(numerator, p=2)
    return dis


if __name__=="__main__":
    train(modelA, modelB, modelC,train_loader,test_loader,0)
    # train_clean(modelA, modelB, modelC, train_loader,test_loader,0)
