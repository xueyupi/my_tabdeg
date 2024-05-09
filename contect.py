import sys

import argparse

model_artifact_name = "2-stage-nn-tabnet"
parser = argparse.ArgumentParser(description='Training 2-Stage NN+TabNet')
parser.add_argument('-input', metavar='INPUT',
                    help='Input folder', default="D:\PyCharmProjects\pythonProject\TAB_DEGNET\BLCA")
parser.add_argument('-output', metavar='OUTPUT',
                    help='Output folder', default="D:\PyCharmProjects\pythonProject\TAB_DEGNET\BLCA")
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
NB = 'TCGA-BLCA'

IS_TRAIN = True

MODEL_DIR = f"{output_folder}/my_model"  # "../model"
INT_DIR = f"{output_folder}/my_interim"  # "../interim"

# DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


# import data
from sklearn.model_selection import train_test_split

all_data = pd.read_csv(f'{input_folder}/TCGA-BLCA_non_bio.csv')
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
test_targets_scored.to_csv(f'{input_folder}/est_targets_scored.csv',index=False,header=True)

# print(test_features)
# print('test_features.shape',test_features.shape)   # (1893, 424)




from sklearn.preprocessing import QuantileTransformer

train_features = pd.read_csv(r'D:\PyCharmProjects\pythonProject\tabnet_degnext\train_features.csv')
test_features = pd.read_csv(r'D:\PyCharmProjects\pythonProject\tabnet_degnext\test_features.csv')
# print(train_features.shape)     # (7572, 424)
# print(test_features.shape)      # (1893, 424)

NORMAL = [col for col in train_features.columns if col.startswith('11',13,15)]
TUMOR = [col for col in train_features.columns if col.startswith('01',13,15)]

# print(len(NORMAL))  # 19
# print(len(TUMOR))   # 405

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

# print('test_features.shape',test_features.shape)  # (1893, 424)



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


train2 = pd.DataFrame(train2, columns=[f'pca_N-{i}' for i in range(n_comp)])    # (7572, 10)
train3 = pd.DataFrame(train3, columns=[f'umap_N-{i}' for i in range(n_dim)])    # (7572, 5)
test2 = pd.DataFrame(test2, columns=[f'pca_N-{i}' for i in range(n_comp)])      # (1893, 10)
test3 = pd.DataFrame(test3, columns=[f'umap_N-{i}' for i in range(n_dim)])      # (1893, 5)

# print(train_features.shape)   (7572, 424)
# print(test_features.shape)   (1893, 424)


train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

# print(train_features.shape)   # (7572, 439)
# print(test_features.shape)    # (1893, 439)


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



############################################################################################




# file name prefix
NB = '2_TCGA-BLCA'

IS_TRAIN = True

MODEL_DIR = f"{output_folder}/my_model"  # "../model"
INT_DIR = f"{output_folder}/my_interim"  # "../interim"

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
test_targets_scored = pd.read_csv(f'{input_folder}/test_targets_one_hot_scored.csv')



## non-scored labels prediction

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
#
#
# # In[ ]:
#

train.to_pickle(f"{INT_DIR}/{NB}_train_nonscored_pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_test_nonscored_pred.pkl")

#
# # In[ ]:
#

valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['genes']+target_cols], on='genes', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]

print("CV log_loss: ", score)







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

seed_everything(42)



# file name prefix
NB = 'TCGA-BLCA'
NB_PREV = '2_TCGA-BLCA'

# IS_TRAIN = False

# MODEL_DIR = "../input/moa503/503-tabnet" # "../model"
# INT_DIR = "../input/moa503/203-nonscored-pred" # "../interim"

# DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

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
NSEEDS = 3 # 5

# Adam hyper params
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

# scheduler hyper params
PCT_START = 0.2
DIV_FACS = 1e3
MAX_LR = 1e-2




train_features = pd.read_csv(f'{input_folder}/train_features.csv')
train_targets_one_hot_scored = pd.read_csv(r'D:\PyCharmProjects\pythonProject\tabnet_degnext\train_targets_one_hot_scored.csv')
test_targets_one_hot_scored = pd.read_csv(r'D:\PyCharmProjects\pythonProject\tabnet_degnext\test_targets_one_hot_scored.csv')

train_targets_scored2 = pd.read_csv(f'{input_folder}/train_targets_scored2.csv')
# train_targets_nonscored = pd.read_csv(f'{input_folder}/train_targets_nonscored.csv')


test_features = pd.read_csv(f'{input_folder}/test_features.csv')
test_targets_scored2 = pd.read_csv(f'{input_folder}/test_targets_scored2.csv')
# sample_submission = pd.read_csv(f'{input_folder}/sample_submission.csv')
train_targets_one_hot_scored2 = pd.read_csv(r'D:\PyCharmProjects\pythonProject\tabnet_degnext\train_targets_one_hot_scored2.csv')
test_targets_one_hot_scored2 = pd.read_csv(r'D:\PyCharmProjects\pythonProject\tabnet_degnext\test_targets_one_hot_scored2.csv')




print("(nsamples, nfeatures)")
print(train_features.shape)                      # (7572, 425)
print(train_targets_scored.shape)                # (7572, 2)
print(train_targets_one_hot_scored2.shape)       # (7572, 4)
print(test_features.shape)                       # (1893, 425)
print(test_targets_scored.shape)                 # (1893, 2)
print(test_targets_one_hot_scored2.shape)        # (1893, 4)




NORMAL = [col for col in train_features.columns if col.startswith('11',13,15)]
TUMOR = [col for col in train_features.columns if col.startswith('01',13,15)]


# In[ ]:


from sklearn.preprocessing import QuantileTransformer

use_test_for_preprocessing = False

for col in (NORMAL + TUMOR):

    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)

    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        if use_test_for_preprocessing:
            raw_vec = pd.concat([train_features, test_features])[col].values.reshape(vec_len+vec_len_test, 1)
            transformer.fit(raw_vec)
        else:
            raw_vec = train_features[col].values.reshape(vec_len, 1)
            transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')


    train_features[col] = transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]



# GENES

n_comp = 5

data = pd.concat([pd.DataFrame(train_features[NORMAL]), pd.DataFrame(test_features[NORMAL])])
if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp, random_state=42).fit(data[NORMAL])
    pd.to_pickle(fa, f'{MODEL_DIR}/{NB}_factor_analysis_normal.pkl')
else:
    fa = pd.read_pickle(f'{MODEL_DIR}/{NB}_factor_analysis_normal.pkl')

data2 = (fa.transform(data[NORMAL]))
train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_N-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_N-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


#CELLS

n_comp = 50

data = pd.concat([pd.DataFrame(train_features[TUMOR]), pd.DataFrame(test_features[TUMOR])])

if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp, random_state=42).fit(data[TUMOR])
    pd.to_pickle(fa, f'{MODEL_DIR}/{NB}_factor_analysis_tumor.pkl')
else:
    fa = pd.read_pickle(f'{MODEL_DIR}/{NB}_factor_analysis_tumor.pkl')

data2 = (fa.transform(data[TUMOR]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_T-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_T-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)




from sklearn.cluster import KMeans
def fe_cluster(train, test, n_clusters_normal = 35, n_clusters_tumor = 5, SEED = 123):

    features_N = list(train.columns[1:20])
    features_T = list(train.columns[20:425])

    def create_cluster(train, test, features, kind = 'N', n_clusters = n_clusters_normal):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)

        if IS_TRAIN:
            kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
            pd.to_pickle(kmeans, f"{MODEL_DIR}/{NB}_kmeans_{kind}.pkl")
        else:
            kmeans = pd.read_pickle(f"{MODEL_DIR}/{NB}_kmeans_{kind}.pkl")

        train[f'clusters_{kind}'] = kmeans.predict(train_)
        test[f'clusters_{kind}'] = kmeans.predict(test_)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test

    train, test = create_cluster(train, test, features_N, kind = 'N', n_clusters = n_clusters_normal)
    train, test = create_cluster(train, test, features_T, kind = 'T', n_clusters = n_clusters_tumor)
    return train, test

train_features ,test_features=fe_cluster(train_features,test_features)


print('train_features.shape',train_features.shape)    # (7572, 520)
print('test_features.shape',test_features.shape)      # (1893, 520)




def fe_stats(train, test):

    features_N = list(train.columns[1:20])
    features_T = list(train.columns[20:425])

    for df in train, test:
        df['N_mean'] = df[features_N].mean(axis = 1)
        df['N_std'] = df[features_N].std(axis = 1)
        df['N_kurt'] = df[features_N].kurtosis(axis = 1)
        df['N_skew'] = df[features_N].skew(axis = 1)
        df['T_mean'] = df[features_T].mean(axis = 1)
        df['T_std'] = df[features_T].std(axis = 1)
        df['T_kurt'] = df[features_T].kurtosis(axis = 1)
        df['T_skew'] = df[features_T].skew(axis = 1)
        df['NT_mean'] = df[features_N + features_T].mean(axis = 1)
        df['NT_std'] = df[features_N + features_T].std(axis = 1)
        df['NT_kurt'] = df[features_N + features_T].kurtosis(axis = 1)
        df['NT_skew'] = df[features_N + features_T].skew(axis = 1)

    return train, test

train_features,test_features=fe_stats(train_features,test_features)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('train_features.shape',train_features.shape)    # (7572, 532)
print('test_features.shape',test_features.shape)      # (1893, 532)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



remove_vehicle = True

if remove_vehicle:
    train_features = train_features.reset_index(drop=True)
    train_targets_one_hot_scored2 = train_targets_one_hot_scored2.reset_index(drop=True)
    # train_targets_scored2=train_targets_scored2.reset_index(drop=True)
else:
    pass




train = train_features.merge(train_targets_one_hot_scored2,on='genes').reset_index(drop=True)
# train = train_features.merge(train_targets_scored2,on='genes').reset_index(drop=True)
test = test_features.reset_index(drop=True)




target = train[train_targets_one_hot_scored2.columns]
# target = train[train_targets_scored2.columns]
target_cols = target.drop('genes', axis=1).columns.values.tolist()  # ['logFC_0', 'logFC_1', 'logFC_2']



print('********************************')
print(target.shape)             # (7572, 4)
print(train_features.shape)     # (7572, 532)
print(test_features.shape)      # (1893, 532)
print(train.shape)              # (7572, 535)
print(test.shape)               # (1893, 532)
print('********************************')





train_nonscored_pred = pd.read_pickle(f'{INT_DIR}/{NB_PREV}_train_nonscored_pred.pkl')
test_nonscored_pred = pd.read_pickle(f'{INT_DIR}/{NB_PREV}_test_nonscored_pred.pkl')


# train_targets_nonscored = train_targets_nonscored.loc[:, train_targets_nonscored.sum() != 0]



# train = train.merge(train_nonscored_pred[train_targets_scored.columns], on='genes')
# test = test.merge(test_nonscored_pred[train_targets_scored.columns], on='genes')
train = train.merge(train_nonscored_pred[train_targets_one_hot_scored.columns], on='genes')
test = test.merge(test_nonscored_pred[test_targets_one_hot_scored.columns], on='genes')


print('train',train)    # [7572 rows x 538 columns]
print('test',test)      # [1893 rows x 535 columns]
print('train.columns',train.columns)
# (['genes', 'TCGA.BT.A20Q.11A', 'TCGA.GC.A3BM.11A', 'TCGA.K4.A3WV.11A',
#        'TCGA.BT.A20U.11A', 'TCGA.K4.A54R.11A', 'TCGA.BT.A20N.11A',
#        'TCGA.CU.A0YN.11A', 'TCGA.BT.A2LB.11A', 'TCGA.BT.A2LA.11A',
#        ...
#        'NT_mean', 'NT_std', 'NT_kurt', 'NT_skew', 'logFC_0', 'logFC_1',
#        'logFC_2', 'log2FoldChange_0', 'log2FoldChange_1', 'log2FoldChange_2'],
#       dtype='object', length=538)
print('test.columns',test.columns)
# (['genes', 'TCGA.BT.A20Q.11A', 'TCGA.GC.A3BM.11A', 'TCGA.K4.A3WV.11A',
#        'TCGA.BT.A20U.11A', 'TCGA.K4.A54R.11A', 'TCGA.BT.A20N.11A',
#        'TCGA.CU.A0YN.11A', 'TCGA.BT.A2LB.11A', 'TCGA.BT.A2LA.11A',
#        ...
#        'T_std', 'T_kurt', 'T_skew', 'NT_mean', 'NT_std', 'NT_kurt', 'NT_skew',
#        'log2FoldChange_0', 'log2FoldChange_1', 'log2FoldChange_2'],
#       dtype='object', length=535)




from sklearn.preprocessing import QuantileTransformer

nonscored_target = [c for c in train_targets_one_hot_scored.columns if c != "genes"]
# print('nonscored_target',nonscored_target)  # ['log2FoldChange_0', 'log2FoldChange_1', 'log2FoldChange_2']


for col in (nonscored_target):
    vec_len = len(train[col].values)
    vec_len_test = len(test[col].values)
    raw_vec = train[col].values.reshape(vec_len, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')

    train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test[col] = transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]




feature_cols = [c for c in train.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['genes']]
len(feature_cols)
# print('feature_cols',feature_cols)  # ['TCGA.BT.A20Q.11A', 'TCGA.GC.A3BM.11A', 'TCGA.K4.A3WV.11A', 'TCGA.BT.A20U.11A', 'TCGA.K4.A54R.11A', 'TCGA.BT.A20N.11A', 'TCGA.CU.A0YN.11A', 'TCGA.BT.A2LB.11A', 'TCGA.BT.A2LA.11A', 'TCGA.GD.A2C5.11A', 'TCGA.GC.A6I3.11A', 'TCGA.BT.A20W.11A', 'TCGA.BL.A13J.11A', 'TCGA.GD.A3OP.11A', 'TCGA.BT.A20R.11A', 'TCGA.CU.A0YR.11A', 'TCGA.GC.A3WC.11A', 'TCGA.GD.A3OQ.11A', 'TCGA.K4.A5RI.11A', 'TCGA.FD.A3SS.01A', 'TCGA.E7.A97P.01A', 'TCGA.GC.A3WC.01A', 'TCGA.DK.A1A3.01A', 'TCGA.GC.A6I3.01A', 'TCGA.XF.AAMJ.01A', 'TCGA.GC.A6I1.01A', 'TCGA.ZF.A9RL.01A', 'TCGA.DK.A3IL.01A', 'TCGA.E7.A6MF.01A', 'TCGA.BT.A20R.01A', 'TCGA.H4.A2HQ.01A', 'TCGA.GV.A3QH.01A', 'TCGA.E7.A4XJ.01A', 'TCGA.XF.A8HE.01A', 'TCGA.FD.A5BT.01A', 'TCGA.G2.AA3C.01A', 'TCGA.BL.A3JM.01A', 'TCGA.XF.AAN3.01A', 'TCGA.FD.A3SR.01A', 'TCGA.4Z.AA89.01A', 'TCGA.G2.A2EO.01A', 'TCGA.KQ.A41R.01A', 'TCGA.BT.A20V.01A', 'TCGA.DK.A2HX.01A', 'TCGA.YF.AA3M.01A', 'TCGA.GV.A3QF.01A', 'TCGA.XF.A9T3.01A', 'TCGA.BT.A20T.01A', 'TCGA.XF.AAN0.01A', 'TCGA.E7.A7DV.01A', 'TCGA.GV.A40E.01A', 'TCGA.GC.A3OO.01A', 'TCGA.XF.A9ST.01A', 'TCGA.GV.A3JX.01A', 'TCGA.FJ.A3ZF.01A', 'TCGA.YC.A89H.01A', 'TCGA.CF.A9FM.01A', 'TCGA.DK.A1AD.01A', 'TCGA.G2.A2EK.01A', 'TCGA.E5.A2PC.01A', 'TCGA.C4.A0F1.01A', 'TCGA.FD.A43N.01A', 'TCGA.BL.A13J.01A', 'TCGA.DK.A1AE.01A', 'TCGA.FD.A3SO.01A', 'TCGA.DK.A3IK.01A', 'TCGA.BT.A20X.01A', 'TCGA.ZF.AA52.01A', 'TCGA.SY.A9G0.01A', 'TCGA.GV.A3JW.01A', 'TCGA.FD.A3N6.01A', 'TCGA.CU.A5W6.01A', 'TCGA.DK.A2I4.01A', 'TCGA.DK.AA6M.01A', 'TCGA.4Z.AA7Y.01A', 'TCGA.DK.A3WY.01A', 'TCGA.E7.A3X6.01A', 'TCGA.2F.A9KP.01A', 'TCGA.DK.A3IQ.01A', 'TCGA.CF.A5U8.01A', 'TCGA.CU.A3QU.01A', 'TCGA.FD.A6TI.01A', 'TCGA.GU.A42P.01A', 'TCGA.FD.A3B3.01A', 'TCGA.YF.AA3L.01A', 'TCGA.DK.AA76.01A', 'TCGA.ZF.A9R2.01A', 'TCGA.GU.AATQ.01A', 'TCGA.FD.A5C1.01A', 'TCGA.DK.A2I1.01A', 'TCGA.UY.A9PB.01A', 'TCGA.DK.A3X1.01A', 'TCGA.DK.A1AC.01A', 'TCGA.FD.A62N.01A', 'TCGA.DK.AA75.01A', 'TCGA.BL.A5ZZ.01A', 'TCGA.4Z.AA86.01A', 'TCGA.XF.AAN2.01A', 'TCGA.BT.A2LD.01A', 'TCGA.ZF.A9R1.01A', 'TCGA.ZF.A9R9.01A', 'TCGA.DK.A6AV.01A', 'TCGA.XF.AAMY.01A', 'TCGA.GU.A763.01A', 'TCGA.FD.A3NA.01A', 'TCGA.CF.A47T.01A', 'TCGA.GC.A3BM.01A', 'TCGA.GD.A3OP.01A', 'TCGA.SY.A9G5.01A', 'TCGA.KQ.A41S.01A', 'TCGA.4Z.AA7M.01A', 'TCGA.FD.A3SJ.01A', 'TCGA.S5.A6DX.01A', 'TCGA.UY.A8OD.01A', 'TCGA.DK.A3IV.01A', 'TCGA.CF.A5UA.01A', 'TCGA.GD.A2C5.01A', 'TCGA.ZF.AA56.01A', 'TCGA.XF.AAMT.01A', 'TCGA.FD.A6TD.01A', 'TCGA.FD.A62O.01A', 'TCGA.BT.A2LB.01A', 'TCGA.FD.A3N5.01A', 'TCGA.G2.A3VY.01A', 'TCGA.DK.A1AA.01A', 'TCGA.XF.A9SY.01A', 'TCGA.DK.AA6Q.01A', 'TCGA.G2.A2EL.01A', 'TCGA.E7.A6ME.01A', 'TCGA.HQ.A2OF.01A', 'TCGA.BT.A20N.01A', 'TCGA.FD.A3SP.01A', 'TCGA.BT.A20P.01A', 'TCGA.GV.A40G.01A', 'TCGA.CF.A47S.01A', 'TCGA.XF.A9T2.01A', 'TCGA.CF.A9FH.01A', 'TCGA.ZF.AA4X.01A', 'TCGA.XF.AAMQ.01A', 'TCGA.DK.AA6W.01A', 'TCGA.KQ.A41O.01A', 'TCGA.S5.AA26.01A', 'TCGA.DK.A1A5.01A', 'TCGA.XF.A9SP.01A', 'TCGA.FD.A3SL.01A', 'TCGA.FD.A6TB.01A', 'TCGA.FD.A6TA.01A', 'TCGA.XF.A8HB.01A', 'TCGA.UY.A8OB.01A', 'TCGA.GU.A767.01A', 'TCGA.FD.A43U.01A', 'TCGA.ZF.AA5H.01A', 'TCGA.5N.A9KI.01A', 'TCGA.XF.A8HF.01A', 'TCGA.UY.A78L.01A', 'TCGA.FD.A5C0.01A', 'TCGA.UY.A9PE.01A', 'TCGA.K4.A6FZ.01A', 'TCGA.XF.A8HD.01A', 'TCGA.LC.A66R.01A', 'TCGA.GU.A762.01A', 'TCGA.XF.A9T5.01A', 'TCGA.4Z.AA7O.01A', 'TCGA.DK.A6B5.01A', 'TCGA.C4.A0F0.01A', 'TCGA.G2.A3IB.01A', 'TCGA.G2.AA3F.01A', 'TCGA.ZF.AA53.01A', 'TCGA.LT.A8JT.01A', 'TCGA.XF.A8HI.01A', 'TCGA.FJ.A3ZE.01A', 'TCGA.4Z.AA82.01A', 'TCGA.DK.AA6X.01A', 'TCGA.XF.A9SJ.01A', 'TCGA.CF.A7I0.01A', 'TCGA.GD.A3OQ.01A', 'TCGA.CU.A0YO.01A', 'TCGA.DK.AA71.01A', 'TCGA.CF.A9FL.01A', 'TCGA.ZF.AA5N.01A', 'TCGA.UY.A9PF.01A', 'TCGA.UY.A78P.01A', 'TCGA.CF.A1HR.01A', 'TCGA.DK.A3IN.01A', 'TCGA.XF.AAMG.01A', 'TCGA.GD.A76B.01A', 'TCGA.ZF.AA4U.01A', 'TCGA.ZF.AA4V.01A', 'TCGA.DK.AA6R.01A', 'TCGA.FT.A61P.01A', 'TCGA.2F.A9KQ.01A', 'TCGA.K4.A54R.01A', 'TCGA.BT.A20W.01A', 'TCGA.ZF.A9R3.01A', 'TCGA.BT.A42F.01A', 'TCGA.K4.A83P.01A', 'TCGA.4Z.AA80.01A', 'TCGA.XF.A9SX.01A', 'TCGA.KQ.A41N.01A', 'TCGA.FD.A3SM.01A', 'TCGA.DK.A3WX.01A', 'TCGA.DK.A3WW.01A', 'TCGA.XF.A9SK.01A', 'TCGA.FD.A3B7.01A', 'TCGA.CF.A3MG.01A', 'TCGA.CU.A0YR.01A', 'TCGA.UY.A78M.01A', 'TCGA.FD.A3SN.01A', 'TCGA.ZF.A9RN.01A', 'TCGA.4Z.AA81.01A', 'TCGA.E7.A8O8.01A', 'TCGA.GC.A3RB.01A', 'TCGA.BT.A20O.01A', 'TCGA.G2.A2ES.01A', 'TCGA.DK.AA6P.01A', 'TCGA.FD.A43P.01A', 'TCGA.E7.A678.01A', 'TCGA.2F.A9KT.01A', 'TCGA.CF.A9FF.01A', 'TCGA.UY.A9PA.01A', 'TCGA.ZF.AA58.01A', 'TCGA.2F.A9KW.01A', 'TCGA.4Z.AA87.01A', 'TCGA.GU.A42Q.01A', 'TCGA.HQ.A5ND.01A', 'TCGA.UY.A78K.01A', 'TCGA.UY.A9PH.01A', 'TCGA.BT.A3PH.01A', 'TCGA.ZF.AA4R.01A', 'TCGA.ZF.AA5P.01A', 'TCGA.DK.A3X2.01A', 'TCGA.YC.A8S6.01A', 'TCGA.FD.A3B5.01A', 'TCGA.GC.A3RC.01A', 'TCGA.BT.A0S7.01A', 'TCGA.GU.AATP.01A', 'TCGA.CF.A1HS.01A', 'TCGA.CF.A3MH.01A', 'TCGA.GC.A4ZW.01A', 'TCGA.ZF.A9RD.01A', 'TCGA.GU.A766.01A', 'TCGA.G2.A2EF.01A', 'TCGA.FD.A5BY.01A', 'TCGA.CF.A8HX.01A', 'TCGA.GV.A6ZA.01A', 'TCGA.FD.A3B4.01A', 'TCGA.4Z.AA7N.01A', 'TCGA.GC.A3RD.01A', 'TCGA.HQ.A2OE.01A', 'TCGA.FD.A5BZ.01A', 'TCGA.BT.A3PJ.01A', 'TCGA.DK.AA6L.01A', 'TCGA.CU.A72E.01A', 'TCGA.FD.A6TH.01A', 'TCGA.GD.A3OS.01A', 'TCGA.DK.A3IS.01A', 'TCGA.K4.A3WS.01A', 'TCGA.FD.A6TF.01A', 'TCGA.4Z.AA7R.01A', 'TCGA.DK.A6B1.01A', 'TCGA.E5.A4TZ.01A', 'TCGA.DK.A6AW.01A', 'TCGA.FD.A62S.01A', 'TCGA.DK.A3IU.01A', 'TCGA.ZF.AA4W.01A', 'TCGA.GC.A3YS.01A', 'TCGA.4Z.AA7Q.01A', 'TCGA.LT.A5Z6.01A', 'TCGA.GC.A3I6.01A', 'TCGA.E7.A5KF.01A', 'TCGA.DK.A3IM.01A', 'TCGA.FJ.A3Z9.01A', 'TCGA.CF.A47X.01A', 'TCGA.BT.A2LA.01A', 'TCGA.K4.A4AC.01A', 'TCGA.XF.AAMX.01A', 'TCGA.ZF.A9R5.01A', 'TCGA.XF.A8HC.01A', 'TCGA.ZF.A9RF.01A', 'TCGA.H4.A2HO.01A', 'TCGA.BT.A42E.01A', 'TCGA.FD.A5BU.01A', 'TCGA.XF.A9T6.01A', 'TCGA.ZF.AA4T.01A', 'TCGA.CF.A47W.01A', 'TCGA.K4.A5RJ.01A', 'TCGA.XF.A9SW.01A', 'TCGA.XF.AAMR.01A', 'TCGA.DK.AA6S.01A', 'TCGA.FD.A62P.01A', 'TCGA.FD.A6TC.01A', 'TCGA.XF.AAN1.01A', 'TCGA.GV.A3JV.01A', 'TCGA.FD.A43S.01A', 'TCGA.PQ.A6FN.01A', 'TCGA.K4.A5RI.01A', 'TCGA.ZF.A9R7.01A', 'TCGA.XF.A9SM.01A', 'TCGA.FD.A6TE.01A', 'TCGA.XF.AAME.01A', 'TCGA.BT.A42C.01A', 'TCGA.CF.A47Y.01A', 'TCGA.E7.A519.01A', 'TCGA.FD.A43Y.01A', 'TCGA.XF.A9SL.01A', 'TCGA.CF.A3MF.01A', 'TCGA.FD.A3B8.01A', 'TCGA.DK.A1A7.01A', 'TCGA.DK.A1A6.01A', 'TCGA.DK.A6B2.01A', 'TCGA.DK.AA74.01A', 'TCGA.ZF.AA4N.01A', 'TCGA.BT.A0YX.01A', 'TCGA.R3.A69X.01A', 'TCGA.XF.AAMW.01A', 'TCGA.C4.A0EZ.01A', 'TCGA.XF.A9T0.01A', 'TCGA.UY.A78N.01A', 'TCGA.E7.A7DU.01A', 'TCGA.DK.A3IT.01A', 'TCGA.MV.A51V.01A', 'TCGA.UY.A78O.01A', 'TCGA.FD.A6TK.01A', 'TCGA.4Z.AA83.01A', 'TCGA.BL.A0C8.01A', 'TCGA.DK.A2I6.01A', 'TCGA.XF.AAN4.01A', 'TCGA.CF.A27C.01A', 'TCGA.5N.A9KM.01A', 'TCGA.BT.A20U.01A', 'TCGA.YC.A9TC.01A', 'TCGA.E7.A6MD.01A', 'TCGA.K4.A5RH.01A', 'TCGA.XF.AAMH.01A', 'TCGA.DK.A1AB.01A', 'TCGA.GV.A3QI.01A', 'TCGA.FD.A43X.01A', 'TCGA.XF.A9SU.01A', 'TCGA.BT.A20J.01A', 'TCGA.E7.A4IJ.01A', 'TCGA.FD.A3B6.01A', 'TCGA.XF.A9SV.01A', 'TCGA.K4.AAQO.01A', 'TCGA.CU.A3YL.01A', 'TCGA.DK.A2I2.01A', 'TCGA.4Z.AA84.01A', 'TCGA.DK.A1AF.01A', 'TCGA.DK.AA6T.01A', 'TCGA.E5.A4U1.01A', 'TCGA.BT.A20Q.01A', 'TCGA.ZF.A9R4.01A', 'TCGA.KQ.A41Q.01A', 'TCGA.E7.A8O7.01A', 'TCGA.XF.AAN7.01A', 'TCGA.XF.A8HH.01A', 'TCGA.FD.A5BX.01A', 'TCGA.ZF.A9RE.01A', 'TCGA.FT.A3EE.01A', 'TCGA.XF.AAN8.01A', 'TCGA.XF.AAML.01A', 'TCGA.BT.A3PK.01A', 'TCGA.CF.A3MI.01A', 'TCGA.E7.A541.01A', 'TCGA.ZF.AA54.01A', 'TCGA.XF.AAMZ.01A', 'TCGA.CF.A8HY.01A', 'TCGA.GV.A3QG.01A', 'TCGA.DK.A6B6.01A', 'TCGA.DK.A6B0.01A', 'TCGA.E7.A97Q.01A', 'TCGA.K4.A3WV.01A', 'TCGA.E7.A85H.01A', 'TCGA.XF.A8HG.01A', 'TCGA.XF.A9T4.01A', 'TCGA.GD.A6C6.01A', 'TCGA.G2.A2EJ.01A', 'TCGA.UY.A9PD.01A', 'TCGA.E7.A677.01A', 'TCGA.FD.A3SQ.01A', 'TCGA.DK.AA77.01A', 'TCGA.K4.A6MB.01A', 'TCGA.ZF.AA51.01A', 'TCGA.E7.A7PW.01A', 'TCGA.KQ.A41P.01A', 'TCGA.G2.A3IE.01A', 'TCGA.E7.A5KE.01A', 'TCGA.FD.A6TG.01A', 'TCGA.2F.A9KO.01A', 'TCGA.CF.A47V.01A', 'TCGA.G2.AA3B.01A', 'TCGA.PQ.A6FI.01A', 'TCGA.C4.A0F7.01A', 'TCGA.FD.A5BS.01A', 'TCGA.ZF.A9RM.01A', 'TCGA.CU.A3KJ.01A', 'TCGA.XF.A9T8.01A', 'TCGA.FD.A5BR.01A', 'TCGA.4Z.AA7S.01A', 'TCGA.XF.A9SH.01A', 'TCGA.DK.A1AG.01A', 'TCGA.2F.A9KR.01A', 'TCGA.DK.AA6U.01A', 'TCGA.C4.A0F6.01A', 'TCGA.CU.A0YN.01A', 'TCGA.FJ.A871.01A', 'TCGA.G2.A2EC.01A', 'TCGA.FD.A5BV.01A', 'TCGA.E7.A7XN.01A', 'TCGA.GU.A764.01A', 'TCGA.FJ.A3Z7.01A', 'TCGA.XF.A9SZ.01A', 'TCGA.GU.AATO.01A', 'TCGA.ZF.A9RC.01A', 'TCGA.ZF.A9R0.01A', 'TCGA.BL.A13I.01A', 'TCGA.4Z.AA7W.01A', 'TCGA.HQ.A5NE.01A', 'TCGA.UY.A8OC.01A', 'TCGA.E7.A3Y1.01A', 'TCGA.XF.AAN5.01A', 'TCGA.G2.AA3D.01A', 'TCGA.GU.A42R.01A', 'TCGA.XF.A9SI.01A', 'TCGA.GV.A3JZ.01A', 'pca_N-0', 'pca_N-1', 'pca_N-2', 'pca_N-3', 'pca_N-4', 'pca_T-0', 'pca_T-1', 'pca_T-2', 'pca_T-3', 'pca_T-4', 'pca_T-5', 'pca_T-6', 'pca_T-7', 'pca_T-8', 'pca_T-9', 'pca_T-10', 'pca_T-11', 'pca_T-12', 'pca_T-13', 'pca_T-14', 'pca_T-15', 'pca_T-16', 'pca_T-17', 'pca_T-18', 'pca_T-19', 'pca_T-20', 'pca_T-21', 'pca_T-22', 'pca_T-23', 'pca_T-24', 'pca_T-25', 'pca_T-26', 'pca_T-27', 'pca_T-28', 'pca_T-29', 'pca_T-30', 'pca_T-31', 'pca_T-32', 'pca_T-33', 'pca_T-34', 'pca_T-35', 'pca_T-36', 'pca_T-37', 'pca_T-38', 'pca_T-39', 'pca_T-40', 'pca_T-41', 'pca_T-42', 'pca_T-43', 'pca_T-44', 'pca_T-45', 'pca_T-46', 'pca_T-47', 'pca_T-48', 'pca_T-49', 'clusters_N_0', 'clusters_N_1', 'clusters_N_2', 'clusters_N_3', 'clusters_N_4', 'clusters_N_5', 'clusters_N_6', 'clusters_N_7', 'clusters_N_8', 'clusters_N_9', 'clusters_N_10', 'clusters_N_11', 'clusters_N_12', 'clusters_N_13', 'clusters_N_14', 'clusters_N_15', 'clusters_N_16', 'clusters_N_17', 'clusters_N_18', 'clusters_N_19', 'clusters_N_20', 'clusters_N_21', 'clusters_N_22', 'clusters_N_23', 'clusters_N_24', 'clusters_N_25', 'clusters_N_26', 'clusters_N_27', 'clusters_N_28', 'clusters_N_29', 'clusters_N_30', 'clusters_N_31', 'clusters_N_32', 'clusters_N_33', 'clusters_N_34', 'clusters_T_0', 'clusters_T_1', 'clusters_T_2', 'clusters_T_3', 'clusters_T_4', 'N_mean', 'N_std', 'N_kurt', 'N_skew', 'T_mean', 'T_std', 'T_kurt', 'T_skew', 'NT_mean', 'NT_std', 'NT_kurt', 'NT_skew', 'log2FoldChange_0', 'log2FoldChange_1', 'log2FoldChange_2']



num_features=len(feature_cols)  # 534
num_targets=len(target_cols)    # 3





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
    model = TabNetRegressor(n_d=32, n_a=32, n_steps=1, lambda_sparse=0,
                            cat_dims=[], cat_emb_dim=[], cat_idxs=[],
                            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                            mask_type='entmax',  # device_name=DEVICE,
                            scheduler_params=dict(milestones=[100, 150], gamma=0.9),#)
                            scheduler_fn=torch.optim.lr_scheduler.MultiStepLR,
                            verbose=10,
                            seed = seed)

    loss_fn = LabelSmoothing(0.001)

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))


    if IS_TRAIN:
        model.fit(X_train=x_train, y_train=y_train,
                  eval_set=[(x_valid, y_valid)], eval_metric=[LogLossMetric, SmoothedLogLossMetric],
                  max_epochs=200, patience=50, batch_size=512, virtual_batch_size=64,
                    num_workers=0, drop_last=False, loss_fn=loss_fn
                  )
        model.save_model(f"{MODEL_DIR}/{NB}_{filename}_SEED{seed}_FOLD{fold}")

    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values

    model = TabNetRegressor(n_d=32, n_a=32, n_steps=1, lambda_sparse=0,
                            cat_dims=[3, 2], cat_emb_dim=[1, 1], cat_idxs=[0, 1],
                            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                            mask_type='entmax',  # device_name=DEVICE,
                            scheduler_params=dict(milestones=[100, 150], gamma=0.9),#)
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

    train[target_cols] = oof
    test[target_cols] = predictions





train.to_pickle(f"{INT_DIR}/{NB}_pre_train.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_pre_test.pkl")





# In[ ]:


run_seeds(train, test, feature_cols, target_cols, NFOLDS, NSEEDS)




train.to_pickle(f"{INT_DIR}/{NB}_train.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_test.pkl")




train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))
# valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
valid_results = train_targets_one_hot_scored2.drop(columns=target_cols).merge(train[['genes']+target_cols], on='genes', how='left').fillna(0)


# y_true = train_targets_scored[target_cols].values
y_true = train_targets_one_hot_scored2[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values


score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]

print("CV log_loss: ", score)


for root, dirs, files in os.walk(f"{MODEL_DIR}/"):
    for f in files:
        if f[-3:] == "zip":
            print(f)
            os.rename(f"{MODEL_DIR}/" + f, f"{MODEL_DIR}/" + f[:-3]+"model")



# train = pd.read_pickle(f"{INT_DIR}/{NB}_train.pkl")
# test = pd.read_pickle(f"{INT_DIR}/{NB}_test.pkl")
#
# test[target_cols] = np.maximum(PMIN, np.minimum(PMAX, test[target_cols]))
# valid_results = test_targets_one_hot_scored2.drop(columns=target_cols).merge(test[['genes']+target_cols], on='genes', how='left').fillna(0)
# y_true = test_targets_one_hot_scored2[target_cols].values
# y_pred = valid_results[target_cols].values
# index_true = np.argmax(y_true, axis=1)
# index_pred = np.argmax(y_pred, axis=1)
# print(index_true)
# print(index_pred)
# match_value=0
# for i in range(len(index_pred)):
#     if index_true[i]==index_pred[i]:
#         match_value+=1
# print(match_value/len(index_pred))




