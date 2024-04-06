# 此程序研究三个参数（迭代次数，学习率，样本容量）对损失函数的影响。
import numpy as np
import matplotlib.pyplot as plt
import random
# 部分1-此部分研究损失函数loss与迭代次数n的关系，学习率rate = 0.01，样本容量m = 100。
# 创建数据
np.random.seed(0)
m = 100
x = np.random.rand(m,1)
y = 5 + 3 * x + np.random.rand(m,1)
# 拟合模型
K = 0
rate = 0.01
loss = []
for n in range(3000):
    y_hat = K * x  
    small_loss = np.sum((y_hat-y)**2) / (2*m)
    loss.append(small_loss)
    K += rate / m * np.dot(x.T,(y-y_hat)) # 通过矩阵的乘法来实现多个乘积的累加。
    K = K.item() # 经过line 19计算出的K是一个1x1的矩阵，并不是float类型，这里将K变成float类型，便于输出观察，虽然此程序没有输出K。
# 画图可视化
plt.plot(loss,label='loss and n')
plt.xlabel('n')
plt.ylabel('loss')
plt.legend()
plt.show()
# 部分2-此部分研究损失函数loss与学习率rate的关系，迭代次数n = 1000，样本容量m = 100。
K = 0
numbers = [random.uniform(0, 1) for _ in range(100)]
loss = []
for rate in numbers:
    for n in range(1000):
        y_hat = K * x
        small_loss += np.sum((y_hat-y)**2)/(2*m)
        K += rate/m*np.dot(x.T,(y-y_hat))
        K = K.item()
    loss.append(small_loss)
plt.plot(loss,label='loss and rate')
plt.xlabel('rate')
plt.ylabel('loss')
plt.legend()
plt.show()
# 部分3-此部分研究损失函数loss与样本容量m的关系，迭代次数n = 200，学习率rate = 0.40。
# 关于此部分：考虑到若样本容量过多程序会进行缓慢，减少了迭代次数，增大了学习率。
# 结果反思：从图像上观察到loss与m成正比例关系，但是loss越大表明模型拟合越不准确，一般而言，样本容量越大，模型拟合应该更精确，对此部分代码存疑。
K = 0
rate = 0.40
numbers = 50 * [random.randint(10,81) for _ in range(25)]
loss = []
for m in numbers:
    x = np.random.rand(m,1)
    y = 5 + 3 * x + np.random.rand(m,1)
    for n in range(200):
        y_hat = K * x
        small_loss += np.sum((y_hat-y)**2)/(2*m)
        K += rate/m*np.dot(x.T,(y-y_hat))
        K = K.item()
    loss.append(small_loss)
plt.plot(loss,label='loss and storage')
plt.xlabel('storage')
plt.ylabel('loss')
plt.legend()
plt.show()