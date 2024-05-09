import pandas as pd
import torch

import argparse

model_artifact_name = "2-stage-nn-tabnet"
parser = argparse.ArgumentParser(description='Training 2-Stage NN+TabNet')
parser.add_argument('-input', metavar='INPUT',
                    help='Input folder', default=r"D:\PyCharmProjects\pythonProject\TAB_DEGNET\COAD")
parser.add_argument('-output', metavar='OUTPUT',
                    help='Output folder', default=r"D:\PyCharmProjects\pythonProject\TAB_DEGNET\COAD")
parser.add_argument('-batch-size', type=int, default=256,
                    help='Batch size')
args = parser.parse_args()
input_folder = args.input
output_folder = args.output

import os

os.makedirs(f'{output_folder}/model', exist_ok=True)
os.makedirs(f'{output_folder}/interim', exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# file name prefix
NB = 'TCGA-COAD'

IS_TRAIN = True

MODEL_DIR = f"{output_folder}/model"  # "../model"
INT_DIR = f"{output_folder}/interim"  # "../interim"

# DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# import data
from sklearn.model_selection import train_test_split
all_data = pd.read_csv(f'{input_folder}/TCGA-COAD_all_gene_array.csv')
n_total = len(all_data)


train_indices, test_indices = train_test_split(
    range(n_total), test_size=0.2, random_state=10)

train_features = all_data.iloc[train_indices,:-1]
train_targets_scored = all_data.iloc[train_indices,[0,-1]]
train_features.to_csv(f'{input_folder}/train_features.csv',index=False,header=True)
train_targets_scored.to_csv(f'{input_folder}/train_targets_scored.csv',index=False,header=True)

test_data = all_data.drop(train_indices,axis=0)
test_features = test_data.iloc[:,:-1]
test_targets_scored = test_data.iloc[:,[0,-1]]
test_features.to_csv(f'{input_folder}/test_features.csv',index=False,header=True)
test_targets_scored.to_csv(f'{input_folder}/test_targets_scored.csv',index=False,header=True)


def process_data(data):
    data = pd.get_dummies(data, columns=['log2FoldChange'])    # 变成one-hot数据
    return data


data_one_hot1 = process_data(train_targets_scored)
data_one_hot2 = process_data(test_targets_scored)

data_one_hot1.to_csv(f'{input_folder}/train_targets_one_hot_scored.csv',index=False,header=True)
data_one_hot2.to_csv(f'{input_folder}/test_targets_one_hot_scored.csv',index=False,header=True)
