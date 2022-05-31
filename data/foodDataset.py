import torch
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
img_to_tensor = transforms.ToTensor()


def loaddatavireo(textfile):#获取图像路径
    path = r"D:\Vireo food-172\VireoFood172\ready_chinese_food"
    textfile = open(textfile, "r", encoding="utf-8").readlines()
    data = list()
    for line in textfile:
        line = line[:-1].split("\t")
        imgpath = path + line[0]
        label = int((line[0].split("/")[-2])) - 1 # (注意在数据集里，是从1 - 172， 但是为了pytorch适用，所以这里变为 0-171)
        data.append([imgpath, label])
    return data


class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, type='train', transform=None):
        if type == 'train':
            #train用的是训练集的数据集和验证集 test是测试集 要改一下自己的路径
            filepath = r'D:\Vireo food-172\VireoFood172\TR.txt'
            self.data = loaddatavireo(filepath) #把读取的数据放到self.data里
            # adding
            filepath = r'D:\Vireo food-172\VireoFood172\VAL.txt'
            data2 = loaddatavireo(filepath)
            self.data.extend(data2)
        else:
            filepath = r'D:\Vireo food-172\VireoFood172\TE.txt'
            self.data = loaddatavireo(filepath)

        self.transforms = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        image = Image.open(data[0])
        image = image.convert('RGB')
        label = data[1]
        if self.transforms != None:
            image = self.transforms(image)
            # label = torch.Tensor(label) # 这句话不能有，有的话会将int变成了一个数组。
        return image, label

if __name__ == '__main__':
    dataset = FoodDataset()