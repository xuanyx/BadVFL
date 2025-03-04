import numpy as np
import sys
from torch.utils.data import Dataset
from six.moves import cPickle
import os
import json
from collections import Counter
from torchvision import transforms
from PIL import Image
import random
import math

def default_loader_from_file(path,is_noise=False,is_trian=None,dataset='cifar10'): #for GTSRB
    if dataset == 'imagenet':
        resize = 224
    elif dataset == "GTSRB":
        resize = 40
    elif dataset == "BHI":
        resize = 50
    else:
        print("No dataset!!!")
        sys.exit(0)

    preprocess_norm_img = transforms.Compose([
        transforms.Resize([resize,resize]),    #224,224
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5582, 0.4957, 0.5228], std=[0.3251, 0.2837, 0.2890]),
    ])
    if dataset=="BHI":
        input_image0 = Image.open(path[0])
        input_image1 = Image.open(path[1])
        # input_image2 = Image.open(path[2])
        # input_image3 = Image.open(path[3])
        # input_image4 = Image.open(path[4])
        # input_image5 = Image.open(path[5])
        # input_image6 = Image.open(path[6])
        # input_image7 = Image.open(path[7])
        # input_image8 = Image.open(path[8])
        # input_image9 = Image.open(path[9])
        input_tensor0 = preprocess_norm_img(input_image0)
        input_tensor1 = preprocess_norm_img(input_image1)
        # input_tensor2 = preprocess_norm_img(input_image2)
        # input_tensor3 = preprocess_norm_img(input_image3)
        # input_tensor4 = preprocess_norm_img(input_image4)
        # input_tensor5 = preprocess_norm_img(input_image5)
        # input_tensor6 = preprocess_norm_img(input_image6)
        # input_tensor7 = preprocess_norm_img(input_image7)
        # input_tensor8 = preprocess_norm_img(input_image8)
        # input_tensor9 = preprocess_norm_img(input_image9)
        input_tensor=[input_tensor0,input_tensor1]
        # input_tensor=[input_tensor0,input_tensor1,input_tensor2,input_tensor3,input_tensor4,input_tensor5,input_tensor6,input_tensor7,input_tensor8,input_tensor9]  #,input_tensor4,input_tensor5,input_tensor6,input_tensor7
    else:
        input_image = Image.open(path)
        input_tensor = preprocess_norm_img(input_image)

    return input_tensor

class Data_Set(Dataset):
    def __init__(self,images,targets,is_train=False,dataset='cifar10',loader=default_loader_from_file):
        self.images = images
        self.targets = targets
        self.loader = loader
        self.is_train=is_train
        self.dataset=dataset
    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn,is_trian=self.is_train,dataset=self.dataset)
        target = self.targets[index]
        return img,target
    def __len__(self):
        return len(self.images)


def load_data_from_GTSRB(data_dir,seed):
    # load data from GTSRB
    train_imgs=[]
    train_y=[]
    test_imgs=[]
    test_y=[]
    # load train data
    train_dir=os.path.join(data_dir,"train")
    label_dirs=os.listdir(train_dir)  #List all directories under this file
    for label_dir in label_dirs:
        label=int(label_dir)
        imgs=os.listdir(os.path.join(train_dir,label_dir))
        for img in imgs:
            train_imgs.append(os.path.join(train_dir,label_dir,img))
            train_y.append(label)

    # load test data
    test_dir=os.path.join(data_dir,"test")
    img_label_map={}
    with open(os.path.join(data_dir,"GT-final_test.csv")) as f:
        next(f)
        for line in f:
            line=line.strip().split(";")
            img,label=line[0].split(".")[0],line[-1]
            img_label_map[img]=int(label)
    imgs=os.listdir(os.path.join(data_dir,"test"))
    for img in imgs:
        label=img_label_map[img.split(".")[0]]
        test_imgs.append(os.path.join(test_dir,img))
        test_y.append(label)

    class_num=len(set(train_y))
    size1 = len(train_y)
    idx1 = np.random.choice(size1, size1, replace=False)
    train_imgs = train_imgs[idx1]
    train_y = train_y[idx1]
    size2 = len(test_y)
    idx2 = np.random.choice(size2, size2, replace=False)
    test_imgs = test_imgs[idx2]
    test_y = test_y[idx2]
    return train_imgs,train_y,test_imgs,test_y,class_num

def load_data_from_imagenet(data_dir,seed,class_num_choice=10):
    X_train=[]  #store path of each pic
    y_train=[]
    X_test=[]
    y_test=[]
    # dirs is the label ids
    img_dirs=os.listdir(os.path.join(data_dir,"train"))

    class_map= json.load(open(os.path.join(data_dir,"imagenet_class_map.json")))

    for img_dir in img_dirs:
        label=class_map[img_dir]
        for img_file in os.listdir(os.path.join(data_dir,"train",img_dir)):
            X_train.append(os.path.join(data_dir,"train",img_dir,img_file))
            y_train.append(label)

    for img_dir in img_dirs:
        label=class_map[img_dir]
        for img_file in os.listdir(os.path.join(data_dir,"test",img_dir)):
            X_test.append(os.path.join(data_dir,"test",img_dir,img_file))
            y_test.append(label)

    a = Counter(y_train)
    #print("# every class:",a)
    print("load imagenet data done","train size:",len(y_train),"test size:",len(y_test))

    size1 = len(y_train)
    idx1 = np.random.choice(size1, size1, replace=False)

    X_train = np.array(X_train)[idx1]
    y_train = np.array(y_train)[idx1]
    size2 = len(y_test)
    idx2 = np.random.choice(size2, size2, replace=False)
    X_test = np.array(X_test)[idx2]
    y_test = np.array(y_test)[idx2]


    return X_train,y_train,X_test,y_test,class_num_choice

def load_data_from_BHI(data_dir,seed,client_num=2,class_num_choice=2):
    X_train=[]  #store path of each pic
    y_train=[]
    X_test=[]
    y_test=[]
    X_data=[]
    y_data=[]

    img_dirs=os.listdir(data_dir)

    for img_dir in img_dirs:
        for img_class in os.listdir(os.path.join(data_dir, img_dir)):
            if img_class not in ['0','1']:
                continue
            label = 0 if img_class=='0' else 1
            img_files = os.listdir(os.path.join(data_dir, img_dir, img_class))
            img_num = math.floor(len(img_files)/client_num)
            for i in range(img_num):
                x1=[]
                for j in range(client_num):
                    x1.append(os.path.join(data_dir,img_dir,img_class,img_files[i*client_num+j]))
                X_data.append(x1)
                y_data.append(label)

    print(len(X_data))
    print(len(y_data))

    X_train=X_data[:math.floor(len(X_data)*0.8)]
    y_train=y_data[:math.floor(len(y_data)*0.8)]
    X_test=X_data[math.floor(len(X_data)*0.8):]
    y_test=y_data[math.floor(len(y_data)*0.8):]

    # print(len(X_train))
    # print(len(y_train))
    # print(len(X_test))
    # print(len(y_test))
    # a = Counter(y_train)
    # print("# every class:",a)
    size1 = len(y_train)
    idx1 = np.random.choice(size1, size1, replace=False)
    X_train = np.array(X_train)[idx1]
    y_train = np.array(y_train)[idx1]
    size2 = len(y_test)
    idx2 = np.random.choice(size2, size2, replace=False)
    X_test = np.array(X_test)[idx2]
    y_test = np.array(y_test)[idx2]

    return X_train,y_train,X_test,y_test,class_num_choice
