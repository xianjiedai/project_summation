from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch

def read_cifar10(batchsize,data_dir):
    transform_train = transforms.Compose([
                                    # transforms.RandomRotation(),  # 随机旋转
                                    transforms.RandomCrop(32, padding=4),  # 填充后裁剪
                                    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    # transforms.ColorJitter(brightness=1),  # 颜色变化。亮度
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

    data_train = datasets.CIFAR10(root=data_dir,
                                  train=True,
                                  transform=transform_train,
                                  download=True)

    data_test = datasets.CIFAR10(root=data_dir,
                                 train=False,
                                 transform=transform_test,
                                 download=True
                                 )

    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=False)
                                   #drop_last=True)
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=False)
    return data_loader_train,data_loader_test

def add_Gaussian_noise(inputs, mean=0, std=1):
    inputs = inputs.double()
    std = torch.rand(1)*std
    noise = torch.randn_like(inputs)*std+torch.tensor(mean)
    output = torch.clamp(inputs + noise, min=0, max=255)
    return output.type(torch.uint8)

