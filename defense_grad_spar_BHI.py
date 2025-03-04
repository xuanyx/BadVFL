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

parser = argparse.ArgumentParser(description='Backdoor Attack in Vertical Federated Learning!')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128, metavar='LR', help='learning rate')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet18",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--depth', type=int, default=28, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--dataset', type=str, default="BHI", help="choose from cifar10,svhn")
parser.add_argument('--num_classes',type=int, default=2,help='the num of class')

# parser.add_argument('--resume1', type=str, default='./trained_models/GTSRB_resnet18_1_clean_normliz_checkpoint.pth.tar', help='whether to resume training, default: None')
# parser.add_argument('--resume2', type=str, default='./trained_models/GTSRB_resnet18_2_clean_normliz_checkpoint.pth.tar', help='whether to resume training, default: None')
# parser.add_argument('--resume3', type=str, default='./trained_models/GTSRB_resnet18_fcnn4_normliz_checkpoint.pth.tar', help='whether to resume training, default: None')

parser.add_argument('--resume', type=bool, default=False,help = "whether to resume from a file")
parser.add_argument('--resume1', type=str, default='./backdoor_model/s15_6_500cifar10_resnet1814_badnets__checkpoint.pth.tar', help='whether to resume training, default: None')
parser.add_argument('--resume2', type=str, default='./backdoor_model/s15_6_500cifar10_resnet1824_badnets__checkpoint.pth.tar', help='whether to resume training, default: None')
parser.add_argument('--resume3', type=str, default='./backdoor_model/s15_6_500cifar10_resnet1834_badnets__checkpoint.pth.tar', help='whether to resume training, default: None')

parser.add_argument('--target_idx', type=int, default=23, help='target sample index')
parser.add_argument('--num', type=int, default=5, help='the number of sample to change img in each batch')
parser.add_argument('--poison_num', type=int, default=1300, help='the min number of total poison sample ')
parser.add_argument('--out_dir', type=str, default='./defense_spar_BHI', help='dir of output')
parser.add_argument('--thre', type=float, default=0.6, help='thre for similarity')
parser.add_argument('--compress_rate', type=float, default=0.1, help='thre for similarity')
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
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "BHI":
    train_imgs,train_y,test_imgs,test_y,class_num = load_data_from_BHI("../../data/BHI",args.seed)
    trainset = Data_Set(train_imgs,train_y,is_train=True,dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,num_workers=2)
    testset = Data_Set(test_imgs,test_y,is_train=False,dataset=args.dataset)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=2)
    print("BHI!!!")
else:
    print("This file is just for training BHI dataset!")
    sys.exit(0)

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
    modelC = FCNN_BHI(num_classes=args.num_classes).to(DEVICE)
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
# optimizer2 = optim.RMSprop(modelB.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)


img_save=transforms.ToPILImage()
# resize = transforms.Resize([200,200])

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
    logger = Logger(os.path.join(args.out_dir, str(args.compress_rate)+'_'+str(args.thre)+'_s'+str(args.seed)+'_log_'+str(args.num)+"_"+str(args.poison_num)+"_"+str(args.target_idx)+"_"+args.dataset+"_"+args.net+'_badnets.txt'), title="Clean")
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
            target = target.to(DEVICE)  #[128,3,32,32]
            data1 = data[0].to(DEVICE)
            data2 = data[1].to(DEVICE)
            if batch_idx in cluster.keys():
                other_class_idx = np.delete(np.arange(args.batch_size),np.array(cluster[batch_idx]))
                for i in cluster[batch_idx]:
                    j = np.random.choice(other_class_idx,replace = False)
                    data2[i] = infect_X(data2[j].clone().detach(),args.dataset,DEVICE,position="mid")  #同一个batch里的替换效果会不会较差？

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
            act1.retain_grad()
            act2.retain_grad()
            data3 = torch.cat((act1,act2), axis=1 )
            output3 = modelC(data3)

            loss = nn.CrossEntropyLoss(reduction='mean')(output3, target)

            loss.backward(retain_graph=True)
            optimizer3.step()

            act1_grad = act1.grad.clone().detach()
            act2_grad = act2.grad.clone().detach()

            clip_num = math.floor((1-args.compress_rate)*act1_grad.shape[1])
            for i in range(act1_grad.shape[0]):
                _, pos1 = torch.topk(torch.abs(act1_grad[i].clone().detach()),k=clip_num,largest=False,sorted=False)
                act1_grad[i][pos1] = 0
                _, pos2 = torch.topk(torch.abs(act2_grad[i].clone().detach()),k=clip_num,largest=False,sorted=False)
                act2_grad[i][pos2] = 0
            # if batch_idx in cluster.keys():
            #     for i in cluster[batch_idx]:
            #         act2_grad[i] = act2_grad[i]

            optimizer2.zero_grad()
            act2.backward(act2_grad)
            optimizer2.step()
            act2.grad.zero_()

            optimizer1.zero_grad()
            act1.backward(act1_grad)
            optimizer1.step()
            act1.grad.zero_()

        clean_acc = test_clean(modelA, modelB, modelC, epoch, test_loader)
        badnets_acc = test_badnets(modelA, modelB, modelC, epoch, test_loader,target_label)
        logger.append([epoch + 1, clean_acc, badnets_acc])

        if clean_acc > best_acc:
            best_acc = clean_acc
            rootA = str(args.compress_rate)+"_"+str(args.thre)+"_s"+str(args.seed)+'_'+str(args.num)+"_"+str(args.poison_num)+args.dataset+"_"+args.net+'1'+str(args.target_idx)+'_badnets_'+'_checkpoint.pth.tar'
            rootB = str(args.compress_rate)+"_"+str(args.thre)+"_s"+str(args.seed)+'_'+str(args.num)+"_"+str(args.poison_num)+args.dataset+"_"+args.net+'2'+str(args.target_idx)+'_badnets_'+'_checkpoint.pth.tar'
            rootC = str(args.compress_rate)+"_"+str(args.thre)+"_s"+str(args.seed)+'_'+str(args.num)+"_"+str(args.poison_num)+args.dataset+"_"+args.net+'3'+str(args.target_idx)+'_badnets_'+'_checkpoint.pth.tar'
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



def get_cluster(modelA, modelB, modelC,train_loader,test_loader,start_epoch):
    target_img,target_label = trainset[args.target_idx]  #target attack img
    print("target_label:",target_label)
    # target_img = torch.chunk(target_img, 2, 2)
    # save_img(target_img[0],target_img[1],target_label)
    # sys.exit(0)
    if_changed = np.zeros(110893)
    cluster = {}
    count = 0
    right_idx = 0
    target_grad = None
    for epoch in range(start_epoch, args.batch_size//args.num):
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.to(DEVICE)  #[128,3,32,32]
            data1 = data[0].to(DEVICE)
            data2 = data[1].to(DEVICE)

            if count < args.poison_num:
                choice_idx = np.where(if_changed[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]==0)[0]
                change_idx = np.random.choice(choice_idx,args.num,replace=False)
                # change_idx = np.random.choice(np.where(target == target_label)[0],4,replace=False)
                # change_idx = np.random.choice(np.delete(np.arange(0,data2.shape[0]),np.where(target == target_label)[0]),4,replace=False)

                # target = target.cpu().numpy()
                # change_idx = np.random.choice(np.where(target == target_label)[0],4,replace=False)
                # change_idx = np.random.choice(np.delete(np.arange(0,data2.shape[0]),np.where(target == target_label)[0]),4,replace=False)

                # target = torch.from_numpy(target).to(DEVICE)

                data2[change_idx] = target_img[1].to(DEVICE)
                if_changed[batch_idx*args.batch_size+change_idx] = 1

            modelA.train()
            modelB.train()
            modelC.train()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            # data1 = Variable(data1.to(DEVICE), requires_grad=True)
            # data2 = Variable(data2.to(DEVICE), requires_grad=True)
            act1, output1 = modelA(data1)
            act2, output2 = modelB(data2)
            size = list(act1.size())
            act1 = torch.reshape(act1, (size[0], -1))
            act2 = torch.reshape(act2, (size[0], -1))

            act1.retain_grad()
            act2.retain_grad()
            data3 = torch.cat((act1,act2), axis=1 )
            output3 = modelC(data3)

            loss1 = nn.CrossEntropyLoss(reduction='mean')(output3, target)

            loss1.backward(retain_graph=True)
            # optimizer1.step()
            # optimizer2.step()
            optimizer3.step()

            act1_grad = act1.grad.clone().detach()
            act2_grad = act2.grad.clone().detach()
            clip_num = math.floor((1-args.compress_rate)*act1_grad.shape[1])

            for i in range(act1_grad.shape[0]):
                _, pos1 = torch.topk(torch.abs(act1_grad[i].clone().detach()),k=clip_num,largest=False,sorted=False)
                act1_grad[i][pos1] = 0
                _, pos2 = torch.topk(torch.abs(act2_grad[i].clone().detach()),k=clip_num,largest=False,sorted=False)
                act2_grad[i][pos2] = 0


            optimizer2.zero_grad()
            act2.backward(act2_grad)
            optimizer2.step()

            optimizer1.zero_grad()
            act1.backward(act1_grad)
            optimizer1.step()
            # data1_grad = data1.grad.clone().detach().view(args.batch_size,-1)
            # data2_grad = data2.grad.clone().detach().view(args.batch_size,-1)
            data1_grad = act1_grad.clone().detach().view(args.batch_size,-1)
            data2_grad = act2_grad.clone().detach().view(args.batch_size,-1)
            act1.grad.zero_()
            act2.grad.zero_()
            # data1.grad.zero_()
            # data2.grad.zero_()
            if batch_idx == math.floor(args.target_idx / args.batch_size):
                target_grad = data2_grad[args.target_idx % args.batch_size]
                # print(pos)
                # print(target_grad_norm)
                # sys.exit(0)
                # print(math.floor(args.target_idx / args.batch_size))
                # print(args.target_idx % args.batch_size)
                # print(data1_grad)
            if target_grad is None:
                continue
            if count < args.poison_num:
                same_class_idx = []
                for i in change_idx:
                    simi = angular_distance(target_grad,data2_grad[i])
                    # print(simi)
                    if simi > args.thre:  #different label  #0.25
                        same_class_idx.append(i)
                        if target[i] != target_label:
                            print("not equal!! epoch:",epoch)
                            print("idx:",batch_idx,i)
                            print("target:",target[i])

                if len(same_class_idx) != 0:
                    count = count + len(same_class_idx)
                    if count > args.poison_num:
                        excess_num = count-args.poison_num
                        del same_class_idx[-excess_num:]
                        count = count-excess_num
                    for i in same_class_idx:
                        if target[i] == target_label:
                            right_idx = right_idx + 1
                    if batch_idx in cluster.keys():
                        cluster[batch_idx] = cluster[batch_idx] + same_class_idx
                    else:
                        cluster[batch_idx] = same_class_idx
        print(count)



        clean_acc = test_clean(modelA, modelB, modelC, epoch, test_loader)
        badnets_acc = test_badnets(modelA, modelB, modelC, epoch, test_loader,target_label)
        logger.append([epoch + 1, clean_acc, badnets_acc])

        if count >= args.poison_num:
            break
    # sys.exit(0)
    print("right_idx:",right_idx)
    print("count:",count)
    print(cluster)
    logger.append([0,right_idx,count])
    # sys.exit(0)
    return epoch,cluster,target_label


def test_clean(modelA, modelB, modelC, epoch,loader):
    modelA.eval()
    modelB.eval()
    modelC.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # iterator = tqdm(loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets = targets.to(DEVICE)
            data1 = inputs[0].to(DEVICE)
            data2 = inputs[1].to(DEVICE)
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
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets = targets.to(DEVICE)
            data1 = inputs[0].to(DEVICE)
            data2 = inputs[1].to(DEVICE)
            poison_targets = torch.ones_like(targets).to(DEVICE)*target_label
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
            # iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

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
    numerator = logits1.mul(logits2).sum()
    # print(numerator)
    logits1_l2norm = logits1.mul(logits1).sum().sqrt()
    logits2_l2norm = logits2.mul(logits2).sum().sqrt()
    denominator = logits1_l2norm.mul(logits2_l2norm)
    return torch.div(numerator, denominator)



if __name__=="__main__":
    train(modelA, modelB, modelC,train_loader,test_loader,0)
    # train_clean(modelA, modelB, modelC, train_loader,test_loader,0)
