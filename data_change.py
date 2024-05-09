import pandas as pd
import torch

def process_data(data):
    data = pd.get_dummies(data, columns=['log2FoldChange'])    # 变成one-hot数据
    return data


data1 = pd.read_csv(r'D:\PyCharmProjects\pythonProject\TAB_DEGNET\KIRC\train_targets_scored.csv')
data_one_hot1 = process_data(data1)
data2 = pd.read_csv(r'D:\PyCharmProjects\pythonProject\TAB_DEGNET\KIRC\test_targets_scored.csv')
data_one_hot2 = process_data(data2)

data_one_hot1.to_csv(r'D:\PyCharmProjects\pythonProject\TAB_DEGNET\KIRC\train_targets_one_hot_scored.csv',index=False,header=True)
data_one_hot2.to_csv(r'D:\PyCharmProjects\pythonProject\TAB_DEGNET\KIRC\test_targets_one_hot_scored.csv',index=False,header=True)

#
# import pickle
# F=open(r'D:\PyCharmProjects\pythonProject\tabnet_degnext\my_interim\2_TCGA-BLCA_train_nonscored_pred.pkl','rb')
#
# content=pickle.load(F)
#
# print(content)

# input_dim = 534
# a = torch.ones(input_dim, dtype=torch.bool)
# print(a)