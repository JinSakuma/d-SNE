import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import cv2
import glob
from PIL import Image 
from tqdm import tqdm
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

from utils.utils import set_random_seeds, log_dict
from utils.data import set_paths, make_abcd_dataset, SingleDataLoader, PairDataLoader
from utils.data import PairDataset, SingleDataset
from utils.visualizer import get_feat, show_UMAP_2D
from utils.trainer import train, val
from models.models import build_CNN, build_Classifier
from models.loss import CombinedLoss


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type=str, default='AlexNet', help='AlexNet or ResNet')
parser.add_argument('-o', '--output', type=str, default='logs/test0311_Adam_1e-5/')
# parser.add_argument('-o', '--output', type=str, default='logs/step2_2_lambda30_gamma_2_sgd1-5/')
parser.add_argument('-e', '--epoch', type=int, default=50)
parser.add_argument('-g', '--gpuid', type=int, default=0)

args = parser.parse_args()

##########################################################################
# Config
##########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
seed = 0
set_random_seeds(seed)

num_epochs = args.epoch
output_dir = os.path.join(os.getcwd(), args.output)
weight_dir = output_dir.replace('logs', 'weights')
method=args.method
batch_size=16


if method=='AlexNet':
    classifier = build_Classifier(method, cls_num=10)
    print('useing AlexNet')
elif method=='ResNet':
    classifier = build_Classifier(method, cls_num=10)
    print('using ResNet')
else:
    assert False, 'method Error: check -m(-method) option'
    
extractor = build_CNN(method)

train_criterion = CombinedLoss()
val_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([{'params': extractor.parameters()},
                        {'params': classifier.parameters()}], lr=1e-5)

# optimizer = optim.SGD([{'params': extractor.parameters()},
#                        {'params': classifier.parameters()}], lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor.to(device)
classifier.to(device)
train_criterion.to(device)
val_criterion.to(device)

##########################################################################
# Data
##########################################################################

root = '/mnt/aoni04/jsakuma/data'
mnist_train = set_paths(root, 'mnist', 'train')
mnist_test = set_paths(root, 'mnist', 'test')
mnist_m_train = set_paths(root, 'mnist-m', 'train')
mnist_m_test = set_paths(root, 'mnist-m', 'test')

(X_a_train, y_a_train), (X_b_train, y_b_train), (X_c_train, y_c_train),(X_d_train, y_d_train) = make_abcd_dataset(mnist_train, mnist_m_train, max_num=5000, cls_flg=False)
(X_a_test, y_a_test), (X_b_test, y_b_test), (X_c_test, y_c_test), (X_d_test, y_d_test) = make_abcd_dataset(mnist_test, mnist_m_test, max_num=800, cls_flg=False)


# transform
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#データセットの作成
batch_size=16
X_ab_train = np.concatenate([X_a_train, X_b_train])
y_ab_train = np.concatenate([y_a_train, y_b_train])

ds_train = PairDataset(X_ab_train, y_ab_train, X_c_train, y_c_train, src_num=-1, tgt_num=300,
                       sample_ratio=3, transform=transform_train)

ds_a_test = SingleDataset(X_a_test, y_a_test, transform_test)
ds_b_test = SingleDataset(X_b_test, y_b_test, transform_test)
ds_c_test = SingleDataset(X_c_test, y_c_test, transform_test)
ds_d_test = SingleDataset(X_d_test, y_d_test, transform_test)

#loaderの作成
loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
loader_a_val = DataLoader(ds_a_test, batch_size=batch_size, shuffle=False)
loader_b_val = DataLoader(ds_b_test, batch_size=batch_size, shuffle=False)
loader_c_val = DataLoader(ds_c_test, batch_size=batch_size, shuffle=False)
loader_d_val = DataLoader(ds_d_test, batch_size=batch_size, shuffle=False)
loader_pair_ac_test = PairDataLoader(ds_a_test, ds_c_test, batch_size=1, shuffle=False)
loader_pair_bd_test = PairDataLoader(ds_b_test, ds_d_test, batch_size=1, shuffle=False)

##########################################################################
# Train/Val Model
##########################################################################

# make output dir
os.makedirs(output_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(output_dir+'/figures', exist_ok=True)
os.makedirs(output_dir+'/history', exist_ok=True)

# num of steps
train_steps = 1000
val_steps = len(loader_a_val)

# output log
results_log = {'train_acc': [], 'train_loss': [], 'train_Lcls': [], 'train_Ldsne': [],
               'val_acc_A': [], 'val_acc_B': [], 'val_acc_C': [], 'val_acc_D': [],
               'val_loss_A': [], 'val_loss_B': [], 'val_loss_C': [], 'val_loss_D': [],
#                'val_Lcls_A': [], 'val_Lcls_B': [], 'val_Lcls_C': [], 'val_Lcls_D': [],
#                'val_Ldsne_A': [], 'val_Ldsne_B': [], 'val_Ldsne_C': [], 'val_Ldsne_D': []
               }

# train/val loop
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')

    for phase in ['train', 'val']:
        # train
        if phase == 'train':
            results_train = train(train_steps, extractor, classifier, loader_train, optimizer, train_criterion, device)
            results_log = log_dict(results_train, results_log, phase, None)
        # val
        else:
            results_A_val = val(val_steps, extractor, classifier, loader_a_val, val_criterion, device)
            results_log = log_dict(results_A_val, results_log, phase, 'A')
            
            results_B_val = val(val_steps, extractor, classifier, loader_b_val, val_criterion, device)
            results_log = log_dict(results_B_val, results_log, phase, 'B')           
            
            results_C_val = val(val_steps, extractor, classifier, loader_c_val, val_criterion, device)
            results_log = log_dict(results_C_val, results_log, phase, 'C')
            
            results_D_val = val(val_steps, extractor, classifier, loader_d_val, val_criterion, device)
            results_log = log_dict(results_D_val, results_log, phase, 'D')
            
            # save model
            accD = results_D_val[-1]
            torch.save(extractor.state_dict(), os.path.join(weight_dir, 'extractor_epoch{}_acc{:.3f}.pth'.format(epoch, accD)))
            torch.save(classifier.state_dict(), os.path.join(weight_dir, 'classifier_epoch{}_acc{:.3f}.pth'.format(epoch, accD)))
            
            
    # plot feature by UMAP
    target_dictAC = {'0':0, '1': 0, '2': 0, '3': 0, '4': 0} 
    target_dictBD = {'5':0, '6': 0, '7': 0, '8': 0, '9': 0} 
    featA, featC = get_feat(extractor, loader_pair_ac_test, device, target_dictAC, N=10)
    featB, featD = get_feat(extractor, loader_pair_bd_test, device, target_dictBD, N=10)

    featAB = featA+featB
    featABCD = featA+featB+featC+featD
    show_UMAP_2D(featAB, featABCD, os.path.join(output_dir, 'figures/umap_2d_plot_epoch{}.png'.format(epoch)))

        
##########################################################################
# Save Log
##########################################################################
df = pd.DataFrame(results_log)
df.to_csv(os.path.join(output_dir, 'result.csv'), encoding='utf_8_sig')

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
plt.plot(results_log['train_acc'], label='trainABC')
plt.plot(results_log['val_acc_A'], label='valA')
plt.plot(results_log['val_acc_B'], label='valB')
plt.plot(results_log['val_acc_C'], label='valC')
plt.plot(results_log['val_acc_D'], label='valD')
plt.legend()
plt.savefig(os.path.join(output_dir, 'history/history_acc.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
plt.plot(results_log['train_loss'], label='trainABC')
plt.plot(results_log['val_loss_A'], label='valA')
plt.plot(results_log['val_loss_B'], label='valB')
plt.plot(results_log['val_loss_C'], label='valC')
plt.plot(results_log['val_loss_D'], label='valD')
plt.legend()
plt.savefig(os.path.join(output_dir, 'history/history_loss.png'))
plt.close()

# plt.figure(figsize=(8, 6))
# plt.rcParams["font.size"] = 15
# plt.plot(results_log['train_Lcls'], label='trainABC')
# plt.plot(results_log['val_Lcls_A'], label='valA')
# plt.plot(results_log['val_Lcls_B'], label='valB')
# plt.plot(results_log['val_Lcls_C'], label='valC')
# plt.plot(results_log['val_Lcls_D'], label='valD')
# plt.legend()
# plt.savefig(os.path.join(output_dir, 'history/history_Lcls.png'))
# plt.close()

# plt.figure(figsize=(8, 6))
# plt.rcParams["font.size"] = 15
# plt.plot(results_log['train_Ldsne'], label='trainABC')
# plt.plot(results_log['val_Ldsne_A'], label='valA')
# plt.plot(results_log['val_Ldsne_B'], label='valB')
# plt.plot(results_log['val_Ldsne_C'], label='valC')
# plt.plot(results_log['val_Ldsne_D'], label='valD')
# plt.legend()
# plt.savefig(os.path.join(output_dir, 'history/history_Ldsne.png'))
# plt.close()

