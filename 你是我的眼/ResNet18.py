# 此程序实现用ResNet-18模型进行图片分类，并将效果可视化
import torch  
import torchvision.models as models  
import torchvision.transforms as transforms  
import torchvision.datasets as datasets  
from torch.utils.data import DataLoader  
import torch.nn as nn  
import matplotlib.pyplot as plt  
import warnings  
  
warnings.filterwarnings("ignore")  # 忽略不必要的警告，使结果美观

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])  # 此处经由程序mean_and_std计算，并保留三位小数
])  

# 数据加载
train_dataset = datasets.ImageFolder(root='D:\\code\\eye\\dataset\\train', transform=data_transform)  
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)  
test_dataset = datasets.ImageFolder(root='D:\\code\\eye\\dataset\\test', transform=data_transform)  
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)  # 测试集没有必要打乱顺序

# 模型定义
model = models.resnet18(pretrained=True)
change = model.fc.in_features
model.fc = nn.Linear(change, 10) # 上述在调用resnet18模型时，选择默认分为1000类，需要更改参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 如果没有可用的gpu则会采用cpu，保证程序的正常运行
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)# 设置动量为0.9以加速收敛

epochs = 10
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    # 模型训练
    model.train()
    for inputs, labels in train_loader:
        # 前向传播
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        optimizer.step()
    # 模型评估
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度，节省计算资源
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算测试集上的损失和准确率
    test_loss /= total
    test_acc = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # 打印每个epoch的测试结果
    print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# 画图可视化
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(test_losses, label='Test Loss')
plt.title('Test Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')  
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout() # 自动调整布局
plt.show()