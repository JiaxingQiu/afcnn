import torch
from torchvision import models
import torch.nn as nn



# googlenet transferred learning
class cnn_conv2d_ggl(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        self.ggl = models.googlenet(pretrained=True)
        self.flat = nn.Flatten()
        self.l = nn.Linear(1000, dim_out)

    def forward(self, x):
        x = self.ggl(x)
        x = self.flat(x)
        x = self.l(x)

        return x
    
# DIY self designed model structure
class cnn_conv2d_diy(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.relu = nn.ReLU()
#         self.conv2d_1 = nn.Conv2d(3, 64, (55, 33), stride=(1, 1))
#         self.conv2d_2 = nn.Conv2d(64, 128, (9, 9), stride=(1, 1))
#         self.conv2d_3 = nn.Conv2d(128, 128, (7, 7), stride=(1, 1))
#         self.conv2d_4 = nn.Conv2d(128, 256, (5, 5), stride=(1, 1))
#         self.conv2d_5 = nn.Conv2d(256, 256, (3, 3), stride=(1, 1))
#         self.conv2d_6 = nn.Conv2d(256, 512, (3, 3), stride=(1, 1))
#         self.dropout = nn.Dropout(p=0.2) 
#         self.flat = nn.Flatten()
#         self.linear = nn.Linear(180224,4) #nn.LazyLinear(dim_out)
        
        self.conv2d_1 = nn.Conv2d(3, 30, (55, 33), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(30, 30, (9, 9), stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(30, 30, (7, 7), stride=(1, 1))
        self.conv2d_4 = nn.Conv2d(30, 30, (5, 5), stride=(1, 1))
        self.conv2d_5 = nn.Conv2d(30, 30, (3, 3), stride=(1, 1))
        self.conv2d_6 = nn.Conv2d(30, 30, (3, 3), stride=(1, 1))
        self.dropout = nn.Dropout(p=0.2) 
        self.flat = nn.Flatten()
        self.linear = nn.Linear(17760,4) #nn.LazyLinear(dim_out)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

#         x = self.conv2d_3(x)
#         x = self.relu(x)
#         x = self.conv2d_3(x)
#         x = self.relu(x)
        x = self.conv2d_4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

#         x = self.conv2d_5(x)
#         x = self.relu(x)
#         x = self.conv2d_5(x)
#         x = self.relu(x)
        x = self.conv2d_6(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.flat(x)
        #print(x.shape)
        x = self.linear(x)

        return x
    
    
# VGG16
class cnn_conv2d_vgg16(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2) 
        self.conv2d_1 = nn.Conv2d(3, 3, (55, 33), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(3, 3, (5, 5), stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(3, 3, (3, 3), stride=(1, 1))
        self.vgg16 = models.vgg16(pretrained=True)
        self.flat = nn.Flatten()
#         self.l1 = nn.Linear(1000, 500)
#         self.l2 = nn.Linear(500, 300)
#         self.l3 = nn.Linear(300, 100)
        self.l4 = nn.Linear(1000, dim_out)

    def forward(self, x):
#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.conv2d_2(x)
#         x = self.relu(x)
#         x = self.conv2d_3(x)
#         x = self.relu(x)
        x = self.vgg16(x)
        x = self.flat(x)
#         x = self.l1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.l2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.l3(x)
#         x = self.relu(x)
#         x = self.dropout(x)
        x = self.l4(x)

        return x