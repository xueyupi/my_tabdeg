import sys

import argparse

model_artifact_name = "2-stage-nn-tabnet"
parser = argparse.ArgumentParser(description='Training 2-Stage NN+TabNet')
parser.add_argument('-input', metavar='INPUT',
                    help='Input folder', default="/home/fsf/pycharm/COAD/deg_tab")
parser.add_argument('-output', metavar='OUTPUT',
                    help='Input folder', default="/home/fsf/pycharm/COAD/deg_tab")
parser.add_argument('-batch-size', type=int, default=512,   # 256 512
                    help='Batch size')
args = parser.parse_args()
input_folder = args.input
output_folder = args.output

import os

os.makedirs(f'{output_folder}/model', exist_ok=True)
os.makedirs(f'{output_folder}/interim', exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


from scipy.sparse.csgraph import connected_components
from umap import UMAP
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, RepeatedMultilabelStratifiedKFold

import numpy as np
import scipy as sp
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
import time
# import joblib

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(f"is cuda available: {torch.cuda.is_available()}")

import warnings


# warnings.filterwarnings('ignore')






# file name prefix
NB = 'TCGA-COAD'

IS_TRAIN = True

MODEL_DIR = f"{output_folder}/model"  # "../model"
INT_DIR = f"{output_folder}/interim"  # "../interim"

# DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






############################################################################################
#

## 503-203-tabnet-with-nonscored-features-10fold3seed



from pytorch_tabnet.tab_model import TabNetRegressor
#
#
# def seed_everything(seed_value):
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     os.environ['PYTHONHASHSEED'] = str(seed_value)
#
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
# # seed_everything(42)
# DEFAULT_SEED=42   #512
# seed_everything(seed_value=DEFAULT_SEED)







print('part2================================================================================')




## 503-203-tabnet-with-nonscored-features-10fold3seed



from pytorch_tabnet.tab_model import TabNetRegressor


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# seed_everything(42)
DEFAULT_SEED = 256    #256 42
seed_everything(seed_value=DEFAULT_SEED)


# file name prefix
NB = 'TCGA-COAD'
NB_PREV = '2_TCGA-COAD'



# label smoothing
PMIN = 0.0
PMAX = 1.0

# submission smoothing
SMIN = 0.0
SMAX = 1.0

# model hyper params

# training hyper params
# EPOCHS = 25
# BATCH_SIZE = 256
NFOLDS = 10 # 10
NREPEATS = 1
NSEEDS = 5 # 3 # 5

# Adam hyper params
# LEARNING_RATE = 5e-4
LEARNING_RATE = 0.001  # 0.001   # 之前是这个
# WEIGHT_DECAY = 1e-5
WEIGHT_DECAY = 1e-5

# scheduler hyper params
PCT_START = 0.2
DIV_FACS = 1e3
MAX_LR = 1e-2


# ========================================================================

import torch
import torch.nn as nn
from pytorch_tabnet.metrics import Metric

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, n_cls=2):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing + smoothing / n_cls
        self.smoothing = smoothing / n_cls

    def forward(self, x, target):
        probs = torch.nn.functional.sigmoid(x,)
        target1 = self.confidence * target + (1-target) * self.smoothing
        loss = -(torch.log(probs+1e-15) * target1 + (1-target1) * torch.log(1-probs+1e-15))
        return loss.mean()

class SmoothedLogLossMetric(Metric):
    """
    BCE with logit loss
    """
    def __init__(self, smoothing=0.001):
        self._name = f"{smoothing:.3f}" # write an understandable name here
        self._maximize = False
        self._lossfn = LabelSmoothing(smoothing)

    def __call__(self, y_true, y_score):
        """
        """
        y_true = torch.from_numpy(y_true.astype(np.float32)).clone()
        y_score = torch.from_numpy(y_score.astype(np.float32)).clone()
        return self._lossfn(y_score, y_true).to('cpu').detach().numpy().copy().take(0)

class LogLossMetric(Metric):
    """
    BCE with logit loss
    """
    def __init__(self, smoothing=0.0):
        self._name = f"{smoothing:.3f}" # write an understandable name here
        self._maximize = False
        self._lossfn = LabelSmoothing(smoothing)

    def __call__(self, y_true, y_score):
        """
        """
        y_true = torch.from_numpy(y_true.astype(np.float32)).clone()
        y_score = torch.from_numpy(y_score.astype(np.float32)).clone()
#         print("log loss metric: ", self._lossfn(y_score, y_true).to('cpu').detach().numpy().copy())
        return self._lossfn(y_score, y_true).to('cpu').detach().numpy().copy().take(0)




# def process_data(data):
# #     data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
#     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2, 0: 0, 1: 1, 2: 2})
#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1, 0: 0, 1: 1})
#     return data

def run_training_tabnet(train, test, trn_idx, val_idx, feature_cols, target_cols, fold, seed, filename="tabnet"):

    seed_everything(seed)

    # train_ = process_data(train)
    # test_ = process_data(test)
    train_ = train
    test_ = test

    train_df = train_.loc[trn_idx,:].reset_index(drop=True)
    valid_df = train_.loc[val_idx,:].reset_index(drop=True)

    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values


    # model = TabNetRegressor(n_d=32, n_a=32, n_steps=1, lambda_sparse=0,
    #                         cat_dims=[3, 2], cat_emb_dim=[1, 1], cat_idxs=[0, 1],
    #                         optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    #                         mask_type='entmax',  # device_name=DEVICE,
    #                         scheduler_params=dict(milestones=[100, 150], gamma=0.9),#)
    #                         scheduler_fn=torch.optim.lr_scheduler.MultiStepLR,
    #                         verbose=10,
    #                         seed = seed)
    model = TabNetRegressor(n_d=8, n_a=8, n_steps=1, lambda_sparse=0,    # 8 , 8
                            cat_dims=[], cat_emb_dim=[], cat_idxs=[],
                            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2, weight_decay=1e-5),# optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
                            mask_type='entmax',  # device_name=DEVICE,
                            scheduler_params=dict(milestones=[50, 100, 150], gamma=0.95),#  scheduler_params=dict(milestones=[50, 100], gamma=0.95)   scheduler_params=dict(milestones=[50, 100], gamma=0.9) scheduler_params=dict(milestones=[100, 150], gamma=0.9),#)
                            scheduler_fn=torch.optim.lr_scheduler.MultiStepLR,
                            verbose=10,
                            seed = seed)

    loss_fn = LabelSmoothing(0.001)

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))


    if IS_TRAIN:
        model.fit(X_train=x_train, y_train=y_train,
                  eval_set=[(x_valid, y_valid)], eval_metric=[LogLossMetric, SmoothedLogLossMetric],
                  max_epochs=200, patience=50, batch_size=512, virtual_batch_size=64,  # patience=60, batch_size=512,virtual_batch_size=64
                    num_workers=0, drop_last=False, loss_fn=loss_fn
                  )
        model.save_model(f"{MODEL_DIR}/{NB}_{filename}_SEED{seed}_FOLD{fold}")

    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values

    model = TabNetRegressor(n_d=8, n_a=8, n_steps=1, lambda_sparse=0,       # 8,8
                            # cat_dims=[3, 2], cat_emb_dim=[1, 1], cat_idxs=[0, 1],
                            cat_dims=[], cat_emb_dim=[], cat_idxs=[],
                            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=1e-2, weight_decay=1e-5),# optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
                            mask_type='entmax',  # device_name=DEVICE,
                            scheduler_params=dict(milestones=[25,50,75,100], gamma=0.95),  #  scheduler_params=dict(milestones=[50, 75, 100], gamma=0.95),  # scheduler_params=dict(milestones=[100, 150], gamma=0.9),#)
                            scheduler_fn=torch.optim.lr_scheduler.MultiStepLR,
                            verbose=10,
                            seed = seed)

    model.load_model(f"{MODEL_DIR}/{NB}_{filename}_SEED{seed}_FOLD{fold}.zip")

    valid_preds = model.predict(x_valid)

    valid_preds = torch.sigmoid(torch.as_tensor(valid_preds)).detach().cpu().numpy()
    oof[val_idx] = valid_preds

    predictions = model.predict(x_test)
    predictions = torch.sigmoid(torch.as_tensor(predictions)).detach().cpu().numpy()

    return oof, predictions





def run_k_fold(train, test, feature_cols, target_cols, NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    mskf = MultilabelStratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = seed)
    # print('mskf',mskf)    # mskf MultilabelStratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        oof_, pred_ = run_training_tabnet(train, test, t_idx, v_idx, feature_cols, target_cols, f, seed)

        predictions += pred_ / NFOLDS / NREPEATS
        oof += oof_ / NREPEATS

    return oof, predictions

def run_seeds(train, test, feature_cols, target_cols, nfolds=NFOLDS, nseed=NSEEDS):
    seed_list = range(nseed)
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    time_start = time.time()

    for seed in seed_list:

        oof_, predictions_ = run_k_fold(train, test, feature_cols, target_cols, nfolds, seed)
        oof += oof_ / nseed
        predictions += predictions_ / nseed
        print(f"seed {seed}, elapsed time: {time.time() - time_start}")
    print('target_cols:',target_cols)
    print('test.columns',test.columns)
    print('len(test)',len(test))
    train[target_cols] = oof
    test[target_cols] = predictions



# 7>6=3>5>4>2
# train.to_pickle(f"{INT_DIR}/{NB}_pre_train.pkl")
# test.to_pickle(f"{INT_DIR}/{NB}_pre_test.pkl")
# train = pd.read_pickle(f"{INT_DIR}/{NB}_pre_train.pkl")
# test = pd.read_pickle(f"{INT_DIR}/{NB}_pre_test.pkl")
train = pd.read_pickle(f"{INT_DIR}/{NB}_pre_train7.pkl")
test = pd.read_pickle(f"{INT_DIR}/{NB}_pre_test7.pkl")



train_features = pd.read_csv(f'{input_folder}/train_features.csv')
train_targets_one_hot_scored = pd.read_csv(f'{input_folder}/train_targets_one_hot_scored.csv')
test_targets_one_hot_scored = pd.read_csv(f'{input_folder}/test_targets_one_hot_scored.csv')


test_features = pd.read_csv(f'{input_folder}/test_features.csv')
train_targets_one_hot_scored2 = pd.read_csv(f'{input_folder}/train_targets_one_hot_scored2.csv')
test_targets_one_hot_scored2 = pd.read_csv(f'{input_folder}/test_targets_one_hot_scored2.csv')


target = train[train_targets_one_hot_scored2.columns]
target_cols = target.drop('genes', axis=1).columns.values.tolist()  # ['logFC_0', 'logFC_1', 'logFC_2']

feature_cols = [c for c in train.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['genes']]


# In[ ]:


run_seeds(train, test, feature_cols, target_cols, NFOLDS, NSEEDS)




# train.to_pickle(f"{INT_DIR}/{NB}_train.pkl")
# test.to_pickle(f"{INT_DIR}/{NB}_test.pkl")


# train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))
# valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
valid_results = train_targets_one_hot_scored2.drop(columns=target_cols).merge(train[['genes']+target_cols], on='genes', how='left').fillna(0)


# y_true = train_targets_scored[target_cols].values
y_true = train_targets_one_hot_scored2[target_cols].values
# y_true = y_true > 0.5
y_pred = valid_results[target_cols].values


score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]

print("CV log_loss: ", score)


# for root, dirs, files in os.walk(f"{MODEL_DIR}/"):
#     for f in files:
#         if f[-3:] == "zip":
#             print(f)
#             os.rename(f"{MODEL_DIR}/" + f, f"{MODEL_DIR}/" + f[:-3]+"model")





#
# train = pd.read_pickle(f"{INT_DIR}/{NB}_train.pkl")
# test = pd.read_pickle(f"{INT_DIR}/{NB}_test.pkl")

# test[target_cols] = np.maximum(PMIN, np.minimum(PMAX, test[target_cols]))
valid_results = test_targets_one_hot_scored2.drop(columns=target_cols).merge(test[['genes']+target_cols], on='genes', how='left').fillna(0)
y_true = test_targets_one_hot_scored2[target_cols].values
y_pred = valid_results[target_cols].values
index_true = np.argmax(y_true, axis=1)
index_pred = np.argmax(y_pred, axis=1)
print(index_true)
print(index_pred)
match_value=0
for i in range(len(index_pred)):
    if index_true[i]==index_pred[i]:
        match_value+=1
print(match_value/len(index_pred))





# 输出准确率 召回率 F值
from sklearn import metrics

print('1',metrics.classification_report(index_true, index_pred))
print('2',metrics.confusion_matrix(index_true, index_pred))

# from sklearn.metrics import accuracy_score
# test_acc = accuracy_score(y_pred=y_pred, y_true=y_true)
#
# print(f"FINAL TEST SCORE FOR {NB} : {test_acc}")

# index_pred = pd.DataFrame({'RISK':index_pred})
# index_pred.to_csv(f"{output_folder}/{NB}_pred.csv",index=False,header=False)

index_pred = pd.DataFrame(index_pred)
index_pred.to_csv(f"/home/fsf/pycharm/COAD/deg_tab/pred2.csv",header='class')