import torch
from torchvision import models
import torch.nn as nn


# VGG16
class cnn_conv2d_vgg16(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        self.relu = nn.ReLU()
        self.vgg16 = models.vgg16(pretrained=True)
        self.flat = nn.Flatten()
        self.l4 = nn.Linear(1000, dim_out)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.flat(x)
        x = self.l4(x)

        return x


# ResNet whole 
class cnn_resnet(nn.Module):
  # number of trainable parameters = 25,677,152

    def __init__(self, dim_out):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.flat = nn.Flatten()
        self.l = nn.Linear(1000, dim_out)

    def forward(self, x):
        x = self.resnet(x)
        x = self.flat(x)
        x = self.l(x)

        return x

# ResNet modified keep 2 layers
class cnn_res2(nn.Module):
# number of trainable parameters = 2,316,612
    def __init__(self, dim_out):
        super().__init__()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.res_regulate = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(512, 256, (1, 1), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(256) )
        self.layer_final = nn.Sequential(
                    nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.identical = nn.Sequential()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(256, 4)

    def forward(self, x):
        x = self.res_regulate(x)
        x = self.reduce_dim(x)
        x_snap = self.identical(x)
        x = self.layer_final(x)
        x = x + x_snap
        x = self.flat(x)
        x = self.linear(x)

        return x
    
    
# ResNet modified keep 3 layers
class cnn_res3(nn.Module):
# number of trainable parameters = 11,643,204
    def __init__(self, dim_out):
        super().__init__()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.res_regulate = nn.Sequential(
                    resnet50.conv1,
                    resnet50.bn1,
                    resnet50.relu,
                    resnet50.maxpool,
                    resnet50.layer1,
                    resnet50.layer2,
                    resnet50.layer3
                    )
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(1024, 256, (3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(256) )
        self.layer_final = nn.Sequential(
                    nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(256)
                )
        self.identical = nn.Sequential()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flat = nn.Flatten()
        self.linear = nn.Linear(256, 4)

    def forward(self, x):
        x = self.res_regulate(x)
        x = self.reduce_dim(x)
        x_snap = self.identical(x)
        x = self.layer_final(x)
        x = x + x_snap
        x = self.pool(x)
        x = self.flat(x)
        x = self.linear(x)

        return x
    
# ResNet modified keep 4 layers
class cnn_res4(nn.Module):
# number of trainable parameters = 24,772,932
    def __init__(self, dim_out):
        super().__init__()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.res_regulate = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3, 
            resnet50.layer4,
            nn.Dropout(p=0.2), 
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(2048, 256, (1, 1), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(256) )
        self.layer_final = nn.Sequential(
                    nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(128), 
                    nn.ReLU(),
                    nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(128), 
                    nn.ReLU(),
                    nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding="same"),
                    nn.BatchNorm2d(256), nn.Dropout(p=0.2), 
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.identical = nn.Sequential()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(256, 4)

    def forward(self, x):
        x = self.res_regulate(x)
        x = self.reduce_dim(x)
        x_snap = self.identical(x)
        x = self.layer_final(x)
        x = x + x_snap
        x = self.flat(x)
        x = self.linear(x)

        return x
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
class cnn_conv2d_diy_lite(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.relu = nn.ReLU()
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
        x = self.conv2d_4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2d_6(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.flat(x)
        #print(x.shape)
        x = self.linear(x)

        return x
    
# DIY self designed model structure
class cnn_conv2d_diy(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.relu = nn.ReLU()
        self.conv2d_1 = nn.Conv2d(3, 64, (55, 33), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(64, 128, (9, 9), stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(128, 128, (7, 7), stride=(1, 1))
        self.conv2d_4 = nn.Conv2d(128, 256, (5, 5), stride=(1, 1))
        self.conv2d_5 = nn.Conv2d(256, 256, (3, 3), stride=(1, 1))
        self.conv2d_6 = nn.Conv2d(256, 512, (3, 3), stride=(1, 1))
        self.dropout = nn.Dropout(p=0.2) 
        self.flat = nn.Flatten()
        self.linear = nn.Linear(180224,4) #nn.LazyLinear(dim_out)
        

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv2d_3(x)
        x = self.relu(x)
        x = self.conv2d_3(x)
        x = self.relu(x)
        x = self.conv2d_4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv2d_5(x)
        x = self.relu(x)
        x = self.conv2d_5(x)
        x = self.relu(x)
        x = self.conv2d_6(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.flat(x)
        #print(x.shape)
        x = self.linear(x)

        return x


# slightly surged VGG16
class cnn_conv2d_vgg16_custom(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        # print vgg16 layers, we can use receptive field size calculater
        self.vgg16_custom = models.vgg16(pretrained=True).features[0:16] # keep the first 9 layers
        self.conv2d_1 = nn.Conv2d(256, 64, (1, 1), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(64, 64, (1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.dropout = nn.Dropout(p=0.2)
        self.flat = nn.Flatten()
        self.l = nn.Linear(3200, dim_out)

    def forward(self, x):
        x = self.vgg16_custom(x)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.l(x)

        return x