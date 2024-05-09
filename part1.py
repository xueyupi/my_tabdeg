import sys

import argparse

model_artifact_name = "2-stage-nn-tabnet"
parser = argparse.ArgumentParser(description='Training 2-Stage NN+TabNet')
parser.add_argument('-input', metavar='INPUT',
                    help='Input folder', default=r"D:\PyCharmProjects\pythonProject\TAB_DEGNET\KIRC")
parser.add_argument('-output', metavar='OUTPUT',
                    help='Output folder', default=r"D:\PyCharmProjects\pythonProject\TAB_DEGNET\KIRC")
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


DEFAULT_SEED = 512
seed_everything(seed_value=DEFAULT_SEED)




# file name prefix
NB = 'TCGA-KIRC'

IS_TRAIN = True

MODEL_DIR = f"{output_folder}/model"  # "../model"
INT_DIR = f"{output_folder}/interim"  # "../interim"

# DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# import data
from sklearn.model_selection import train_test_split
all_data = pd.read_csv(f'{input_folder}/TCGA-KIRC_non_bio.csv')
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






from sklearn.preprocessing import QuantileTransformer

train_features = pd.read_csv(f'{input_folder}/train_features.csv')
test_features = pd.read_csv(f'{input_folder}/test_features.csv')
# print(train_features.shape)     # (11702, 599)
# print(test_features.shape)      # (2926, 599)

NORMAL = [col for col in train_features.columns if col.startswith('11',13,15)]
TUMOR = [col for col in train_features.columns if col.startswith('01',13,15)]

# print(len(NORMAL))  # 72
# print(len(TUMOR))   # 526

for col in (NORMAL + TUMOR):
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = pd.concat([train_features, test_features])[col].values.reshape(vec_len+vec_len_test, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')

    train_features[col] = transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]
    test_features = test_features.copy()
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

# print('test_features.shape',test_features.shape)  # (2926, 599)

#

# NORMAL
n_comp = 15
n_dim = 5

data = pd.concat([pd.DataFrame(train_features[NORMAL]), pd.DataFrame(test_features[NORMAL])])
# print('train_features[NORMAL].shape',train_features[NORMAL].shape)      # train_features[NORMAL].shape (7572, 19)
# print('test_features[NORMAL].shape',test_features[NORMAL].shape)      # test_features[NORMAL].shape (1893, 19)
# print('data.shape',data.shape)               # data.shape (9465, 19)


if IS_TRAIN:
    pca = PCA(n_components=n_comp, random_state=DEFAULT_SEED).fit(train_features[NORMAL])
    umap = UMAP(n_components=n_dim, random_state=DEFAULT_SEED).fit(train_features[NORMAL])
    pd.to_pickle(pca, f"{MODEL_DIR}/{NB}_pca_normal.pkl")
    pd.to_pickle(umap, f"{MODEL_DIR}/{NB}_umap_normal.pkl")
else:
    pca = pd.read_pickle(f"{MODEL_DIR}/{NB}_pca_normal.pkl")
    umap = pd.read_pickle(f"{MODEL_DIR}/{NB}_umap_normal.pkl")

data2 = pca.transform(data[NORMAL])           # (9465, 10)
data3 = umap.transform(data[NORMAL])          # (9465, 5)

train2 = data2[:train_features.shape[0]]      # (7572, 10)
test2 = data2[-test_features.shape[0]:]       # (1893, 10)
train3 = data3[:train_features.shape[0]]      # (7572, 5)
test3 = data3[-test_features.shape[0]:]       # (1893, 5)


train2 = pd.DataFrame(train2, columns=[f'pca_N-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_N-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'pca_N-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_N-{i}' for i in range(n_dim)])

# print(train_features.shape)   # (11702, 599)
# print(test_features.shape)    # (2926, 599)


train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

# print(train_features.shape)   # (11702, 619)
# print(test_features.shape)    # (2926, 619)



print('==================================================')






#TUMOR
n_comp = 50
n_dim = 15

data = pd.concat([pd.DataFrame(train_features[TUMOR]), pd.DataFrame(test_features[TUMOR])])

# print('train_features[TUMOR].shape',train_features[TUMOR].shape)    # train_features[TUMOR].shape (7572, 405)
# print('data.shape',data.shape)       # data.shape (9465, 405)


if IS_TRAIN:
    pca = PCA(n_components=n_comp, random_state=DEFAULT_SEED).fit(train_features[TUMOR])
    umap = UMAP(n_components=n_dim, random_state=DEFAULT_SEED).fit(train_features[TUMOR])
    pd.to_pickle(pca, f"{MODEL_DIR}/{NB}_pca_tumor.pkl")
    pd.to_pickle(umap, f"{MODEL_DIR}/{NB}_umap_tumor.pkl")
else:
    pca = pd.read_pickle(f"{MODEL_DIR}/{NB}_pca_tumor.pkl")
    umap = pd.read_pickle(f"{MODEL_DIR}/{NB}_umap_tumor.pkl")

data2 = pca.transform(data[TUMOR])
data3 = umap.transform(data[TUMOR])

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_T-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_T-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'pca_T-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_T-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

print('train_features.shape',train_features.shape)   # (7572, 505)
print('test_features.shape',test_features.shape)     # (1893, 505)

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]




print('==================================================')


from sklearn.feature_selection import VarianceThreshold

if IS_TRAIN:
    var_thresh = VarianceThreshold(threshold=0.5).fit(train_features.iloc[:, 1:])
    pd.to_pickle(var_thresh, f"{MODEL_DIR}/{NB}_variance_thresh0_5.pkl")
else:
    var_thresh = pd.read_pickle(f"{MODEL_DIR}/{NB}_variance_thresh0_5.pkl")

data = train_features.append(test_features)
data_transformed = var_thresh.transform(data.iloc[:, 1:])

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0] : ]


# print('train_features_transformed.shape',train_features_transformed.shape)  # train_features_transformed.shape (7572, 487)
# print('test_features_transformed.shape',test_features_transformed.shape)    # test_features_transformed.shape (1893, 487)


train_features = pd.DataFrame(train_features['genes'].values.reshape(-1, 1),columns=['genes'])
train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)
test_features = pd.DataFrame(test_features['genes'].values.reshape(-1, 1),columns=['genes'])
test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)


# print('train_features.shape',train_features.shape)       # train_features.shape (7572, 488)
# print(train_features)
# print('test_features.shape',test_features.shape)         # test_features.shape (1893, 488)
# print(test_features)

train = train_features
test = test_features
train.to_pickle(f"{INT_DIR}/{NB}_train_preprocessed.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_test_preprocessed.pkl")



# 已完成pca和umap的分组降维 #
############################################################################################




# file name prefix
NB = '2_TCGA-KIRC'

IS_TRAIN = True

MODEL_DIR = f"{output_folder}/model"  # "../model"
INT_DIR = f"{output_folder}/interim"  # "../interim"

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# label smoothing
PMIN = 0.0
PMAX = 1.0

# submission smoothing
SMIN = 0.0
SMAX = 1.0

# model hyper params
HIDDEN_SIZE = 2048

# training hyper params
EPOCHS = 15
BATCH_SIZE = args.batch_size
NFOLDS = 10 # 10
NREPEATS = 1
NSEEDS = 5 # 5

# Adam hyper params
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

# scheduler hyper params
PCT_START = 0.2
DIV_FACS = 1e3
MAX_LR = 1e-2



# def process_data(data):
#     data = pd.get_dummies(data, columns=['cp_time','cp_dose'])    # 变成one-hot数据
#     return data


class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct


class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        # print('inputs.shape',inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


def calc_valid_log_loss(train, target, target_cols):
    y_pred = train[target_cols].values
    y_true = target[target_cols].values

    y_true_t = torch.from_numpy(y_true.astype(np.float64)).clone()
    y_pred_t = torch.from_numpy(y_pred.astype(np.float64)).clone()

    return torch.nn.BCELoss()(y_pred_t, y_true_t).to('cpu').detach().numpy().copy()




class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size=HIDDEN_SIZE):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x





def run_training(train, test, trn_idx, val_idx, feature_cols, target_cols, fold, seed):

    seed_everything(seed)

    # train_ = process_data(train)
    # test_ = process_data(test)

    train_ = train
    test_ = test

    train_df = train_.loc[trn_idx,:].reset_index(drop=True)
    valid_df = train_.loc[val_idx,:].reset_index(drop=True)

    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values

    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=len(feature_cols),
        num_targets=len(target_cols),
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=PCT_START, div_factor=DIV_FACS,
                                              max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    loss_fn = nn.BCEWithLogitsLoss()

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    best_loss_epoch = -1

    if IS_TRAIN:
        for epoch in range(EPOCHS):

            train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_loss_epoch = epoch
                oof[val_idx] = valid_preds
                model.to('cpu')
                torch.save(model.state_dict(), f"{MODEL_DIR}/{NB}_nonscored_SEED{seed}_FOLD{fold}_.pth")
                model.to(DEVICE)

            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(f"seed: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, best_loss: {best_loss:.6f}, best_loss_epoch: {best_loss_epoch}")

    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=len(feature_cols),
        num_targets=len(target_cols),
    )

    model.load_state_dict(torch.load(f"{MODEL_DIR}/{NB}_nonscored_SEED{seed}_FOLD{fold}_.pth"))
    model.to(DEVICE)

    if not IS_TRAIN:
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        oof[val_idx] = valid_preds

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)

    return oof, predictions


def run_k_fold(train, test, feature_cols, target_cols, NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    mskf = RepeatedMultilabelStratifiedKFold(n_splits=NFOLDS, n_repeats=NREPEATS, random_state=None)

    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        oof_, pred_ = run_training(train, test, t_idx, v_idx, feature_cols, target_cols, f, seed)

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

    train[target_cols] = oof
    test[target_cols] = predictions


train_features = pd.read_csv(f'{input_folder}/train_features.csv')
train_targets_scored = pd.read_csv(f'{input_folder}/train_targets_one_hot_scored.csv')

test_features = pd.read_csv(f'{input_folder}/test_features.csv')
# sample_submission = pd.read_csv(f'{input_folder}/sample_submission.csv')
# test_targets_scored = pd.read_csv(f'{input_folder}/test_targets_one_hot_scored.csv')



# non-scored labels prediction

# remove nonscored labels if all values == 0
# train_targets_scored = train_targets_scored.loc[:, train_targets_nonscored.sum() != 0]
# print('train_targets_scored.shape',train_targets_scored.shape)     # (7572, 4)

train = train.merge(train_targets_scored,on='genes')
print('train',train)
print('train.shape:',train.shape)         # (7572, 491)

target = train[train_targets_scored.columns]
print('target',target)
# print('target.shape',target.shape)        # (7572, 4)
target_cols = target.drop('genes', axis=1).columns.values.tolist()
print('target_cols',target_cols)        # ['log2FoldChange_0', 'log2FoldChange_1', 'log2FoldChange_2']
feature_cols = [c for c in train.columns if c not in target_cols and c not in ['kfold','genes']]
# feature_cols = [c for c in process_data(train).columns if c not in target_cols and c not in ['kfold','sig_id']]
print('feature_cols',feature_cols)   # feature_cols [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486]




run_seeds(train, test, feature_cols, target_cols)



print(f"train shape: {train.shape}")
print(f"test  shape: {test.shape}")
print(f"features : {len(feature_cols)}")
print(f"targets  : {len(target_cols)}")




valid_loss_total = calc_valid_log_loss(train, target, target_cols)
print(f"CV loss: {valid_loss_total}")


# In[ ]:


train.to_pickle(f"{INT_DIR}/{NB}_train_nonscored_pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_test_nonscored_pred.pkl")


# In[ ]:


valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['genes']+target_cols], on='genes', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]

print("CV log_loss: ", score)

