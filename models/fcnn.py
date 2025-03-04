import torch.nn as nn
import torch
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self,num_classes=10):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,num_classes)
        # self.fc5 = nn.Linear(2048,num_classes)
        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        # out = self.fc5(x)
        # out = self.soft(out)
        return out

class FCNN4_V1(nn.Module):
    def __init__(self,num_classes=10):
        super(FCNN4_V1, self).__init__()
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,num_classes)
        # self.fc5 = nn.Linear(2048,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        # out = self.fc5(x)
        return out

class FCNN2(nn.Module):
    def __init__(self,num_classes=10):
        super(FCNN2, self).__init__()
        self.fc1 = nn.Linear(2048,512)
        self.fc2 = nn.Linear(512,num_classes)
        # self.fc5 = nn.Linear(2048,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        # out = self.fc5(x)
        return out

class FCNN5(nn.Module):
    def __init__(self,num_classes=10):
        super(FCNN5, self).__init__()
        self.fc1 = nn.Linear(8192,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

class FCNN5_NL(nn.Module):
    def __init__(self,num_classes=10):
        super(FCNN5_NL, self).__init__()
        self.fc1 = nn.Linear(8192,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return out


class FCNN_BHI(nn.Module):
    def __init__(self,num_classes=10):
        super(FCNN_BHI, self).__init__()
        self.fc1 = nn.Linear(9216,1024)   #2:9216  4/8:4096  6:4080
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


class FCNN4_NL(nn.Module):
    def __init__(self,num_classes=10):
        super(FCNN4_NL, self).__init__()
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return out


def FCNN4(num_classes=10):
    return FCNN(num_classes)
