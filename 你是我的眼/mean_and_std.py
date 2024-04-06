# 此程序计算数据集的均值和标准差，以用于数据预处理时的标准化
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

# 加载数据
train_dataset = datasets.ImageFolder(root='D:\\code\\eye\\dataset\\train',transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset,batch_size=50,shuffle=True)

def get_mean_std_value(loader):
    data_sum,data_squared_sum,num_batches = 0,0,0
    for data in loader[0]: # 只需要数据的张量，而不需要其标签
        # 计算dim=0,2,3维度的均值和以及平方均值和，dim=1为通道数量，不用参与计算。 [batch_size,channels,height,width]
        data_sum += torch.mean(data,dim=[0,2,3])    
        data_squared_sum += torch.mean(data**2,dim=[0,2,3])  
        # 统计batch的数量
        num_batches += 1
    # 计算均值与标准差
    mean = data_sum/num_batches
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std

mean,std = get_mean_std_value(train_loader)
print(f'mean = {mean},std = {std}')