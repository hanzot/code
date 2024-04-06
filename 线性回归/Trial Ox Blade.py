# 此程序利用已有模型实现二元线性回归并将结果可视化表现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 创建数据集
np.random.seed(0)
x = np.random.rand(300,1)
y = 5 + 4 * x + np.random.rand(300,1)
# 分为训练集与测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
# 应用模型
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
# 结果可视化
plt.scatter(x_test,y_test,color='b',label='actual')
plt.plot(x_test,y_pred,color='r',label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()