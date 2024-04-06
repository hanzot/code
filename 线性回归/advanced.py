# 此程序利用已有模型实现多元线性回归并将结果可视化。
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 此处进行加载波士顿房价数据，在新版sklearn中已删除该数据，下载了1.1.1版本的sklearn库，仍然失败，于是查询后直接使用下列代码实现数据调用。
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# 划分为训练组和测试组
x = data
y = target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# 应用模型
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# 结果可视化
plt.scatter(y_test, y_pred,color='b')
plt.plot(y_test,y_test,color='r') # 在整个坐标系中，纵坐标是y_pred，横坐标是y_test，此处采用y=x来描述回归线，通过观察回归线与点的重合程度同样能够判断模拟效果。
plt.xlabel('actual_price')
plt.ylabel('predict_price')
plt.show()