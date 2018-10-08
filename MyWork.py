import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv("/Users/jiangyijie/Downloads/kaggle_tatanic/Kaggle competetion Tatanic/train.csv")
print(data_train)
print(data_train.info())