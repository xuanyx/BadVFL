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
from functools import partial

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
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--num_classes',type=int, default=10,help='the num of class')

parser.add_argument('--resume', type=bool, default=False,help = "whether to resume from a file")
parser.add_argument('--resume1', type=str, default='./replace_model_cifar10/s1_g10_500_5_cifar10_resnet1813_badnets_checkpoint.pth.tar', help='whether to resume training, default: None')
parser.add_argument('--resume2', type=str, default='./replace_model_cifar10/s1_g10_500_5_cifar10_resnet1823_badnets_checkpoint.pth.tar', help='whether to resume training, default: None')
parser.add_argument('--resume3', type=str, default='./replace_model_cifar10/s1_g10_500_5_cifar10_resnet1833_badnets_checkpoint.pth.tar', help='whether to resume training, default: None')

parser.add_argument('--target_idx', type=int, default=1, help='poison target img idx')
parser.add_argument('--out_dir', type=str, default='./replace_model_cifar10', help='dir of output')
parser.add_argument('--poison_num', type=int, default=500, help='poison num of clean img')
parser.add_argument('--num', type=int, default=5, help='poison num of clean img in each batch')
parser.add_argument('--position', type=str, default="mid", help='poison num of clean img in each batch')
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
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, drop_last=True)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='../../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,drop_last=True)
if args.dataset == "GTSRB":
    train_imgs,train_y,test_imgs,test_y,class_num = load_data_from_GTSRB("../../data/GTSRB/GTSRB_JPG",args.seed)
    trainset = Data_Set(train_imgs,train_y,is_train=True,dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    testset = Data_Set(test_imgs,test_y,is_train=False,dataset=args.dataset)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    print("GTSRB!!!")
if args.dataset == "imagenet":
    train_imgs,train_y,test_imgs,test_y,class_num = load_data_from_imagenet("../../data/ILSVRC2012",args.seed)
    trainset = Data_Set(train_imgs,train_y,is_train=True,dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    testset = Data_Set(test_imgs,test_y,is_train=False,dataset=args.dataset)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,drop_last=True)
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
# optimizer2 = optim.RMSprop(modelB.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

img_save=transforms.ToPILImage()

def save_img(img1,img2,target):
    filename1=args.dataset+str(target.item())+'_1.png'
    filename2=args.dataset+str(target.item())+'_2.png'
    img1=img_save(img1)
    img2=img_save(img2)
    img1.save(os.path.join("img_poison",filename1), quality=100, subsampling=0)
    img2.save(os.path.join("img_poison",filename2), quality=100, subsampling=0)
    return
def save_img1(img,target):
    filename1=args.dataset+str(target)+'.png'
    img=img_save(img)
    img.save(os.path.join("img_poison",filename1), quality=100, subsampling=0)
    return

if args.resume==False:
    logger = Logger(os.path.join(args.out_dir, 'new1_g10_log_replace_'+args.position+"_"+str(args.poison_num)+"_"+str(args.num)+"_"+str(args.target_idx)+"_"+args.dataset+"_"+args.net+'_badnets.txt'), title="Clean")
    # logger.set_names(['GradA', 'GradB'])
    logger.set_names(['Epoch', 'Natural Test Acc','backdoor ASR','SIMI'])


def train(modelA, modelB, modelC,train_loader,test_loader,start_epoch):
    if args.resume:
        print("resume")
        checkpoint1 = torch.load(args.resume1)
        checkpoint2 = torch.load(args.resume2)
        checkpoint3 = torch.load(args.resume3)
        start_epoch = checkpoint1['epoch']
        modelA.load_state_dict(checkpoint1['state_dict'])
        modelB.load_state_dict(checkpoint2['state_dict'])
        modelC.load_state_dict(checkpoint3['state_dict'])
        clean_acc = test_clean(modelA, modelB, modelC, start_epoch, test_loader)
        badnets_acc = test_badnets(modelA, modelB, modelC, start_epoch, train_loader,trainset[args.target_idx][1])
        clean_ori_acc,poison_ori_acc = test_ori(modelA, modelB, modelC, start_epoch, test_loader)
        return

    train_badnets(modelA, modelB, modelC,train_loader,test_loader, start_epoch)

    return



def train_badnets(modelA, modelB, modelC,train_loader,test_loader,start_epoch):
    best_acc = 0
    target_img,target_label = trainset[args.target_idx]  #target attack img
    print("target_label:",target_label)
    poison_num = 0
    # save_img1(target_img,target_label)
    # sys.exit(0)

    change_dict = get_poison_idx(train_loader,target_label)
    print(change_dict)


    change_idx = []

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer1, epoch + 1)
        adjust_learning_rate(optimizer2, epoch + 1)
        adjust_learning_rate(optimizer3, epoch + 1)
        for batch_idx, (data, target) in enumerate(train_loader):
            # if batch_idx == 0:
            #     print(target[args.target_idx])
            #     sys.exit(0)
                # continue
            data, target = data.to(DEVICE), target.to(DEVICE)  #[128,3,32,32]
            data1,data2 = torch.chunk(data, 2, 3)

            if batch_idx in change_dict.keys():
                change_idx = change_dict[batch_idx]
                for i in change_idx:
                    data2[i] = infect_X(data2[i].clone().detach(),args.dataset,DEVICE,position=args.position)

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
            # act2.retain_grad()

            # upload random act2 to prevent the devil party to establish the mapping between the poioned one with its true label
            act2_devil = act2.clone().detach()

            if len(change_idx) != 0:
                for i in change_idx:
                    if args.dataset == 'imagenet':
                        normal_noise = np.random.normal(0,1e-6,4096)
                        normal_noise = torch.tensor(normal_noise).float().to(DEVICE)
                        act2_devil[i] = normal_noise
                        # act2_devil[i] = torch.rand(4096)
                    else:
                        # act2_devil[i] = torch.rand(1024)
                        normal_noise = np.random.normal(0,1e-6,1024)
                        normal_noise = torch.tensor(normal_noise).float().to(DEVICE)
                        act2_devil[i] = normal_noise
            act2_devil.requires_grad = True

            data3 = torch.cat((act1,act2_devil), axis=1 )
            output3 = modelC(data3)

            loss1 = nn.CrossEntropyLoss(reduction='mean')(output3, target)

            loss1.backward(retain_graph=True)

            act1_grad = act1.grad.detach().clone()
            act2_grad = act2_devil.grad.detach().clone()
            # print(act_grad)
            # optimizer1.step()
            optimizer3.step()

            # gradient replacement
            if batch_idx == 0:
                # act2_ori=act2[args.target_idx].clone().detach()
                target_grad = act2_devil.grad[args.target_idx].clone().detach()
            if len(change_idx) != 0:
                act2_grad[change_idx] = target_grad*10
                # for ci in change_idx:
                #     simi = simi + similar(act2_ori,act2[ci])
                change_idx = []


            optimizer2.zero_grad()
            act2.backward(act2_grad)
            optimizer2.step()
            act2_devil.grad.zero_()

            optimizer1.zero_grad()
            act1.backward(act1_grad)
            optimizer1.step()
            act1.grad.zero_()

        clean_acc = test_clean(modelA, modelB, modelC, epoch, test_loader)
        badnets_acc = test_badnets(modelA, modelB, modelC, epoch, test_loader,target_label)
        logger.append([epoch + 1, clean_acc, badnets_acc])

        if clean_acc > best_acc:
            best_acc = clean_acc
            rootA = "new1_g10_"+args.position+"_"+str(args.poison_num)+"_"+str(args.num)+"_"+args.dataset+"_"+args.net+'1'+str(args.target_idx)+'_badnets'+'_checkpoint.pth.tar'
            rootB = "new1_g10_"+args.position+"_"+str(args.poison_num)+"_"+str(args.num)+"_"+args.dataset+"_"+args.net+'2'+str(args.target_idx)+'_badnets'+'_checkpoint.pth.tar'
            rootC = "new1_g10_"+args.position+"_"+str(args.poison_num)+"_"+str(args.num)+"_"+args.dataset+"_"+args.net+'3'+str(args.target_idx)+'_badnets'+'_checkpoint.pth.tar'
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

def similar(logits1, logits2):
    numerator = logits1.mul(logits2).sum()
    # print(numerator)
    logits1_l2norm = logits1.mul(logits1).sum().sqrt()
    logits2_l2norm = logits2.mul(logits2).sum().sqrt()
    denominator = logits1_l2norm.mul(logits2_l2norm)
    # print(denominator)
    # if numerator > denominator:
    #     numerator[i]=denominator[i]
    # D = torch.sub(1.0, torch.abs(torch.div(numerator, denominator)))
    return torch.div(numerator, denominator)

def test_clean(modelA, modelB, modelC, epoch,test_loader):
    modelA.eval()
    modelB.eval()
    modelC.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # iterator = tqdm(loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            data1 = torch.chunk(inputs, 2, 3)[0]
            data2 = torch.chunk(inputs, 2, 3)[1]  #data[:, :, :, 16:]
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

    # Save checkpoint.
    acc = 100.*correct/total
    print("epoch: %d, acc: %f "  % (epoch, acc))
    return acc

def test_badnets(modelA, modelB, modelC, epoch, test_loader,target_label):
    modelA.eval()
    modelB.eval()
    modelC.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # iterator = tqdm(loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            poison_targets = torch.ones_like(targets).to(DEVICE)*target_label
            data1 = torch.chunk(inputs, 2, 3)[0]
            data2 = torch.chunk(inputs, 2, 3)[1]  #data[:, :, :, 16:]
            # save_img(data1[0].cpu(),data2[0].cpu(),targets[0].cpu())
            for i in range(data2.shape[0]):
                 data2[i] = infect_X(data2[i].clone().detach(),args.dataset,DEVICE,position=args.position)

            with torch.no_grad():
                act1, output1 = modelA(data1)
                act2, output2 = modelB(data2)
                size = list(act1.size())
                act1 = torch.reshape(act1, (size[0], -1))
                act2 = torch.reshape(act2, (size[0], -1))
                data3 = torch.cat((act1,act2), axis=1 )
                output3 = modelC(data3)
            _, predicted = output3.max(1)
            # print(output3[args.target_idx])
            # print(output3[0])
            # print(predicted)
            # sys.exit(0)
            total += targets.size(0)
            correct += predicted.eq(poison_targets).sum().item()

    acc = 100.*correct/total
    print("BadNets!!! epoch: %d, acc: %f "  % (epoch, acc))
    return acc

def test_ori(modelA, modelB, modelC, epoch,test_loader):
    target_img,target_label = trainset[args.target_idx]  #target attack img
    target_img = torch.chunk(target_img, 2, 2)[1]
    modelA.eval()
    modelB.eval()
    modelC.eval()
    test_loss = 0
    clean_correct = 0
    poison_correct = 0
    total = 0
    with torch.no_grad():
        # iterator = tqdm(loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            poison_targets = torch.ones_like(targets).to(DEVICE)*target_label
            data1 = torch.chunk(inputs, 2, 3)[0]
            data2 = torch.chunk(inputs, 2, 3)[1]  #data[:, :, :, 16:]
            for i in range(data2.shape[0]):
                data2[i] = target_img.to(DEVICE)
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

            clean_correct += predicted.eq(targets).sum().item()
            poison_correct += predicted.eq(poison_targets).sum().item()

    # Save checkpoint.
    clean_acc = 100.*clean_correct/total
    poison_acc = 100.*poison_correct/total
    print("epoch: %d,clean acc: %f ,poison_acc: %f"  % (epoch, clean_acc,poison_acc))
    return clean_acc,poison_acc



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

def get_poison_idx(train_loader,target_label):
    change_dict = {}
    poison_num = 0
    invalid_num = 0
    data_size=len(trainset)
    data_size_left=data_size - (data_size % args.batch_size)
    print(data_size_left)
    choice_idx = np.delete(np.arange(data_size_left),[args.target_idx])
    change_idx = np.random.choice(choice_idx,args.poison_num,replace=False)
    change_idx = change_idx.tolist()
    for i in change_idx:
        batch_idx= i // args.batch_size
        if batch_idx in change_dict.keys():
            change_dict[batch_idx].append(i%args.batch_size)
        else:
            change_dict[batch_idx] = [i%args.batch_size]
    # print(change_dict)
    return change_dict

    '''
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print(targets)
        # sys.exit(0)
        if poison_num < args.poison_num:
            num = np.random.randint(0, args.num)  #each batch max num to change grad
            choice_idx = np.delete(np.arange(args.batch_size),[args.target_idx])
            change_idx = np.random.choice(choice_idx,num,replace=False)
            change_idx = change_idx.tolist()
            if len(change_idx) != 0:
                change_dict[batch_idx] = change_idx
                poison_num = poison_num + len(change_idx)
                if poison_num > args.poison_num:
                    excess_num = poison_num-args.poison_num
                    del change_idx[-excess_num:]
                    poison_num = poison_num - excess_num
                for i in change_idx:
                    if targets[i] == target_label:
                        invalid_num = invalid_num + 1
        else:
            print("invalid_num:",invalid_num)
            print("poison_num:",poison_num)
            return change_dict
        '''





if __name__=="__main__":

    train(modelA, modelB, modelC,train_loader,test_loader,0)
