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
def setup_seed(seed): #设置一个随机数种子，这样就可以复现
    torch.manual_seed(seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(seed) # gpu 为当前GPU设置随机种子
    np.random.seed(seed) #仅一次有效
setup_seed(100)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #调用的GPU编号

# myResNet
class myResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #拿到一个预训练的resnet101
        self.model = models.resnet101(pretrained=True)   #基本块是resnet101，只到卷积层，平均池化层及之后就不要了
        #空间注意力
        self.sa = B.SpatialAttention() #定义函数是空间注意力，前面假设了basicblock.py为B
        #通道注意力
        self.eca = B.eca_layer(512) #512个通道的通道注意力
        #池化层
        self.pool = nn.Sequential(
            nn.Conv2d(2048 * 3, 512, 3, 1, 1), #2048*3通道变成512通道降低计算量
            nn.ReLU(True), #激活函数RELU
            nn.AdaptiveAvgPool2d((1, 1))  #将feature map变成1*1 自适应平均池化
        )
        #分类     （64，512，1，1）参数
        self.classifier = nn.Sequential( #用于连接模型的函数
            nn.Linear(512, 172), #把512个通道分成172类
        )
                 #输入
    def forward(self, x):
        #resnet101的全部卷积层 x(64,3,224,224)
        x = self.model.conv1(x)
        x = self.model.bn1(x) #数据归一化方法，在深度神经网络中激活层之前。其作用可以加快模型训练时的收敛速度，使得模型训练过程更加稳定，避免梯度爆炸或者梯度消失。并且起到一定的正则化作用
        x = self.model.relu(x) #激活函数
        x = self.model.maxpool(x) #最大池化
        x = self.model.layer1(x) # 输出64通道
        x = self.model.layer2(x) #3
        x = self.model.layer3(x) #224
        x = self.model.layer4(x) #224

        #后面是原创的东西
        sa = self.sa(x)     #空间注意力     (64,2048,14,14)
        eca = self.eca(x)     #通道注意力      (64,2048,14,14)
        att = torch.cat([sa,eca,x],1)  #cat把三个拼接起来，成为2048*3通道  (64,2048,14,14) 
        x = self.pool(att)           #(64,512,1,1)
        x = torch.flatten(x, 1)     #(64,512) 按列横向拼接
        y = self.classifier(x)      #(64,172)
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
    #习惯性的用法：测试的开始
    model.eval() #把实例化model指定为eval，以免固定住BN和Dropout不会取平均而是用固定好的值
    count = 0
    correct = 0
    #加载测试数据
    for index, data in enumerate(tqdm.tqdm(test_loader)): #返回一个枚举对象的函数 从测试集里列举对象
        img, lab = data
        b, c, h, w = img.shape
        img = img.cuda()
        lab = lab.cuda()
        #no_grad指不计算梯度
        with torch.no_grad(): #不会被计算梯度的部分
            pre = model(img)
        _, lab_pre = torch.max(pre.data, 1) #每行的最大值
        currect = torch.sum(lab_pre == lab.data) #当测试和实际值对上的时候就是正确了
        count += b #计数用的参数
        correct += currect  #计算预测对了几个
    model.train()
    return torch.true_divide(correct, count)     #返回测试集的准确率 两个数相除且返回浮点数

def train():  #开始训练了
    mynet = myResNet().cuda()
    #多GPU并行
    # mynet = MyViT('B_16_imagenet1k',num_layers=6,image_size=224,num_classes=12,dropout_rate=0, pretrained=True)
    mynet = nn.DataParallel(mynet)       #这个函数是让多个GPU共同训练加快速度

    # mynet.load_state_dict(torch.load(path +'/best.pth'),strict=True)
    print('-----start train model------')        #定义优化器，是Adam 初始学习率是0.0001，权重衰减值0.0001
    # test(model)
    optimiter = torch.optim.Adam(mynet.parameters(),lr=0.0001,betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0001)
    ######### Scheduler ########### parameters用于迭代优化的参数或者定义参数组的dicts lr是学习率 betas用于计算梯度的平均和平方的系数 eps为了提高数值稳定性而添加到分母的一个项 weight_decay：权重衰减
    #余弦退火 调整学习率
    warmup = True #这是优化学习率的方法，在预热期间lr从0上升到lr再降回0的过程
    if warmup:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiter, 10, eta_min=1e-7) #optimiter是要进行学习率衰减的优化器，10为余弦周期的一半，eta_min是学习率衰减的最小值。学习率衰减是因为一开始步长可以长一点，但是之后要确定范围那就要精确一些才能找到最小/大值


    loss_f1 = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

    #开始训练了
    best = 0

    for epo in range(200):         #训练轮次设置为200，实际是100左右。一轮就是所有数据都用一次
        currect = 0
        count = 0
        # for k, v in mynet.named_parameters():
        #     if 'block' in k:
        #         v.requires_grad = False

        #获取一个batch数据
        for i, data in enumerate(tqdm.tqdm(train_loader)): #循环读取训练集中的数据，每次betch都是随机抽取64个
            # print(i)
            img, lab = data  #图像 标签    image大小是（64，3，224，224） 64个，3个通道，224*224 因为网络模型就是合适于224*224


            #--- 数据增强 ---#
            h = img.shape[2]   #高 是224 取shape的第三个数据
            w = img.shape[3]     #宽 是224 取shape的第四个数据

            for n in range(2):
                y = np.random.randint(h)    #y是（0，h ）随机数
                x = np.random.randint(w)          #x是（0，w ）随机数
                #在（x,y）为中心找一个40*40的矩形区域
                y1 = np.clip(y - 20 // 2, 0, h) #输入的数组是y-20//2，这个数值必须在（0，h）范围内，超出的部分会变成0/h
                y2 = np.clip(y + 20 // 2, 0, h)
                x1 = np.clip(x - 20 // 2, 0, w)
                x2 = np.clip(x + 20 // 2, 0, w)

             #一半的概率将这个矩形区域变为0 ，一半的概率变为噪音  ，就是为了防止过拟合。不用cutmix的原因是中餐有很多地方食材是重复重叠的，很可能会造成混淆，中餐是细颗粒度
                type_rand = np.random.rand()#生成一个服从（0，1）的均匀分布的样本值
                if type_rand < 0.5:
                    img[:,:, y1: y2, x1: x2] = 0. #y1:y2表示（y1,y2）这个范围
                else:
                    img[:,:, y1: y2, x1: x2] = torch.randn(img.shape[0], img.shape[1] , y2 - y1, x2 - x1) #img.shape[0]是垂直尺寸（高度），img.shape[1]是水平尺寸（宽度），img.shape[2]图像的通道数
            # ----------- #

            b, c, h, w = img.shape
            #把tensor变成变量
            img = Variable(img).cuda()    #variable是个可以反向传播不断变化的变量，tensor里面的变量都是这个格式，cuda是将数据移动到GPU显存中去
            lab = Variable(lab).cuda()

                  #给模型输入图像得到预测值
            pre = mynet(img)    #这个预测值是一个172位的数组，是指该图像属于每个类的概率

            #算误差
            loss = loss_f1(pre, lab) #用交叉熵损失函数算误差

       #BP，Adam优化器，让loss接近0,误差减小
            optimiter.zero_grad()    #梯度置零，adam，不让上次的梯度影响这次的
            loss.backward()        #误差反向传播
            optimiter.step()         #更新参数

              #把概率最大对应的索引拿到，拿到分类结果
            _, lab_pre = torch.max(pre.data, 1) #按行寻找最大值
            # 计算该batch对了几个
            currect += torch.sum(lab_pre == lab.data)
            count += b

                  #打印日志文件
        logger.info('train correct rate:[{}] epoch:[{}] current learning rate:[{}]'.format((int(currect)/int(count)),epo,optimiter.param_groups[0]['lr']))

        #每5轮更新一次学习率
        if epo % 5 == 0 and epo > 0:
            scheduler.step()
            print('epoch', epo, ' current learning rate', optimiter.param_groups[0]['lr'])
        #训练完一轮就测试一次
        test_val_ = test(mynet)    #测试函数 ，得到准确率
        #保存参数
        if test_val_ > best:        #如果这一轮准确率比之前最好的好就保存这个，否则不保存
            torch.save(mynet.state_dict(), path +'/best.pth')
            best = test_val_
            print('best',best)
            
            #打印日志
        logger.info('test correct rate:[{}] epoch:[{}]'.format(test_val_,epo))

if __name__ == '__main__':
    '''
        初始化参数
    '''
    batch_size = 64   #每批次64个图像，一次训练所抓取的数据样本数量
    netname = 'vgg16'
    netvision = 'v1'
    class_num = 172
     #创建路径
    path = os.getcwd() #得到当前路径
    path = os.path.join(path, 'model_zoo', str(netname), str(netvision)) #这是传递路径的意思
    # model path
    if not os.path.exists(path):
        os.makedirs(path) #创建目录
        print('path create')
    # log path日志文件的路径r
    log_path = os.path.join(path, 'train.log') #创建log路径
    logger = log.get_logger(log_path)

    # 数据预处理 pytorch自带的transform，都是为了防止过拟合（训练集表现好但是测试集表现不好）
    transform = trans.Compose([ #compose函数是一个将transforms组合起来的函数，循序执行
        trans.Scale((320,320)),  #把图像统一变为320*320大小
        trans.CenterCrop((224,224)), #取图像中心的224*224大小的区域
        trans.RandomRotation(0.5),  #
        trans.ToTensor(),    #PIL格式的图像数据转为tensor格式
        trans.RandomVerticalFlip(0.5),   #以0.5的概率垂直翻转
        trans.RandomHorizontalFlip(0.5),    #水平翻转
        trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   #归一化 mean是三个通道的平均灰度值，std是三个通道的方差灰度值，Normalsize是归一化：（每张图片的三个通道（灰度值-平均值）/方差）
    ])
    #加载数据
    train_data = FoodDataset('train',transform)   #fooddataset是自己建立的类
    test_data = FoodDataset('test',transform)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)     #shuffle是是否打乱顺序 加载器
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    train()