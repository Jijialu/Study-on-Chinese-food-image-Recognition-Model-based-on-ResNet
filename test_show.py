# 使用vgg预训练模型做迁移学习,vgg19最后一层是1000，所以冻结其他层，只训练新分类层，进行cifar分类，
from collections import namedtuple
import torch
from torch.autograd import Variable
import os
import torchvision.transforms as trans
from torchvision.transforms import ToPILImage
import torchvision.models as models
show = ToPILImage()
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import torch.nn as nn
import utils.logger as log
import random
from data.foodDataset import FoodDataset
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import basicblock as B
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
setup_seed(100)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# myResNet
class myResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        self.sa = B.SpatialAttention()
        self.eca = B.eca_layer(512)
        self.pool = nn.Sequential(
            nn.Conv2d(2048 * 3, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 172),
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # x1 = self.deformconv(x)
        sa = self.sa(x)
        eca = self.eca(x)
        att = torch.cat([sa,eca,x],1)
        x = self.pool(att)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y
# myResNet without sa (没有空间注意力)
class myResNet_noSA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        # self.sa = B.SpatialAttention()
        self.eca = B.eca_layer(512)
        self.pool = nn.Sequential(
            nn.Conv2d(2048 * 2, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 172),
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # x1 = self.deformconv(x)
        # sa = self.sa(x)
        eca = self.eca(x)
        att = torch.cat([eca,x],1)
        x = self.pool(att)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y
# myResNet without eca (没有通道注意力)
class myResNet_noECA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        self.sa = B.SpatialAttention()
        # self.eca = B.eca_layer(512)
        self.pool = nn.Sequential(
            nn.Conv2d(2048 * 2, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 172),
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # x1 = self.deformconv(x)
        sa = self.sa(x)
        # eca = self.eca(x)
        att = torch.cat([sa,x],1)
        x = self.pool(att)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y
# myResNet without eca and as (没有通道，空间注意力)
class myResNet_noECA_noSA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        # self.sa = B.SpatialAttention()
        # self.eca = B.eca_layer(512)
        self.pool = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 172),
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # x1 = self.deformconv(x)
        # sa = self.sa(x)
        # eca = self.eca(x)
        # att = torch.cat([sa,x],1)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y

def test(model):
    model.eval()
    count = 0
    correct = 0
    for index, data in enumerate(tqdm.tqdm(test_loader)):
        img, lab = data
        b, c, h, w = img.shape
        img = img
        lab = lab
        with torch.no_grad():
            pre = model(img)
        _, lab_pre = torch.max(pre.data, 1)
        currect = torch.sum(lab_pre == lab.data)
        count += b
        correct += currect

        # 展示图像
        # import matplotlib.pyplot as plt
        # from torchvision.transforms import ToPILImage
        # img = img[0]
        # img[0] = img[0] * 0.229 + 0.485
        # img[1] = img[1] * 0.224 + 0.456
        # img[2] = img[2] * 0.225 + 0.406
        # img = ToPILImage()(img)
        # plt.imshow(img)
        # plt.show()

        print('prediction:',lab_pre)
        print('real',lab)
    return torch.true_divide(correct, count)


if __name__ == '__main__':
    '''
        Init parameters
    '''
    batch_size = 1
    netname = 'vgg16'
    netvision = 'v1'
    class_num = 172
    path = os.getcwd()
    path = os.path.join(path, 'model_zoo', str(netname), str(netvision))

    mynet = myResNet()
    mynet = nn.DataParallel(mynet)
    mynet.load_state_dict(torch.load(path +'/resnet101_87.06.pth',map_location=torch.device('cpu')),strict=True)
    mynet = mynet.module
    # 没加数据增强前是68
    transform = trans.Compose([
        trans.Scale((320,320)),
        trans.CenterCrop((224,224)),
        # trans.RandomRotation(0.5),
        trans.ToTensor(),
        # trans.RandomVerticalFlip(0.5),
        # trans.RandomHorizontalFlip(0.5),
        trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_data = FoodDataset('test',transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    acc = test(mynet)