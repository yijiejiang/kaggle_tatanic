# #
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
# from pylab import mpl

# mpl.rcParams['font.sans-serif'] = ['FangSong_GB2312'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# data_train = pd.read_csv("/Users/jiangyijie/Downloads/kaggle_tatanic/Kaggle competetion Tatanic/train.csv")
data_train = pd.read_csv("/home/yijie/PycharmProjects/kaggle_tatanic/Kaggle competetion Tatanic/train.csv")
# print(data_train)   # 打印数据
print(data_train.info())    # 打印概览

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind = 'bar')
plt.title(u"dead or survival(1=survival)")
plt.ylabel(u'number')

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.title(u"distribution of Pclass")
plt.ylabel(u'number')

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u'age')
plt.grid(b=True, which="major", axis='y')
plt.title(u"distribution of age")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u'age')
plt.ylabel(u'density')
plt.title(u'distribution of Pclass')
plt.legend((u'1st', u'2nd', u'3rd'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u'people from different entrance')
plt.ylabel(u'number')

plt.show()

