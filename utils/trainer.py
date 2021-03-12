import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def train(num_steps, extractor, classifier, train_dataloader, optimizer, criterion, device):
    extractor.train()
    classifier.train()

    correct = 0.
    epoch_loss = 0.
    epoch_cls_loss = 0.
    epoch_dsne_loss = 0.
    train_cnt = 0.
    for batch_idx, (X, y) in zip(tqdm(range(num_steps)), train_dataloader):
        X = {k: v.to(device) for k, v in X.items()}  # Send X to GPU
        y = {k: v.to(device) for k, v in y.items()}  # Send y to GPU

        # Repeat train step for both target and source datasets
        for train_name in X.keys():
            optimizer.zero_grad()

            ft, y_pred = {}, {}
            for name in X.keys():
                ft[name] = extractor(X[name])
                y_pred[name] = classifier(ft[name])
                
                cls_pred = y_pred[name].argmax(dim=1, keepdim=True)
                cls_pred = cls_pred.view(-1)
#                 print(cls_pred)
#                 print(y[name])
                correct += cls_pred.eq(y[name].view_as(cls_pred)).sum().item()
                train_cnt += X[name].shape[0]

            loss, loss_cls, loss_dsne = criterion(ft, y_pred, y, train_name)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_cls_loss += loss_cls.item()
            epoch_dsne_loss += loss_dsne.item()
    
    
    cls_acc = correct / float(train_cnt)
    epoch_loss = epoch_loss / float(train_cnt)
    epoch_cls_loss = epoch_cls_loss / float(train_cnt)
    epoch_dsne_loss = epoch_dsne_loss / float(train_cnt)
    print('train')
    print('loss: {:.3f}, loss cls: {:.3f}, loss dsne: {:.3f}, cls_acc: {:.3f}'.format(epoch_loss, epoch_cls_loss, epoch_dsne_loss, cls_acc))
    return epoch_loss, epoch_cls_loss, epoch_dsne_loss, cls_acc
            
            
def val(num_steps, extractor, classifier, val_dataloader, criterion, device):
    extractor.eval()
    classifier.eval()

    correct = 0.
    epoch_loss = 0.
    epoch_cls_loss = 0.
    epoch_dsne_loss = 0.
    train_cnt = 0.
    with torch.no_grad():
        for batch_idx, (X, y) in zip(tqdm(range(num_steps)), val_dataloader):
            X = X.to(device)
            y = y.to(device)

            features = extractor(X)           
            output = classifier(features)

            cls_pred = output.argmax(dim=1, keepdim=True)
            correct += cls_pred.eq(y.view_as(cls_pred)).sum().item()
            train_cnt += X.shape[0]
            
            loss = criterion(output, y)
            epoch_loss += loss.item()

    cls_acc = correct / float(train_cnt)
    epoch_loss = epoch_loss / float(train_cnt)
#     epoch_cls_loss = epoch_cls_loss / float(train_cnt)
#     epoch_dsne_loss = epoch_dsne_loss / float(train_cnt)
    print('val')
    print('loss: {:.3f}, cls_acc: {:.3f}'.format(epoch_loss, cls_acc))
#     print('loss: {:.3f}, loss cls: {:.3f}, loss dsne: {:.3f}, cls_acc: {:.3f}'.format(epoch_loss, epoch_cls_loss, epoch_dsne_loss, cls_acc))
    return epoch_loss, cls_acc
# return epoch_loss, epoch_cls_loss, epoch_dsne_loss, cls_acc


def test(num_steps, extractor, classifier, val_dataloader, criterion, device):
    extractor.eval()
    classifier.eval()

    correct = 0.
    epoch_loss = 0.
    train_cnt = 0.
    label_list, pred_list = np.array([]), np.array([])
    with torch.no_grad():
        for batch_idx, (X, y) in zip(tqdm(range(num_steps)), val_dataloader):
            X = X.to(device)
            y = y.to(device)

            features = extractor(X)
            output = classifier(features)

            cls_pred = output.argmax(dim=1, keepdim=True)
            correct += cls_pred.eq(y.view_as(cls_pred)).sum().item()
            train_cnt += X.shape[0]
            
            loss = criterion(output, y)
            epoch_loss += loss.item()
            
            label_list = np.append(label_list, y.detach().cpu().numpy())
            pred_list = np.append(pred_list, cls_pred.detach().cpu().numpy())

    cls_acc = correct / float(train_cnt)
    epoch_loss = epoch_loss / float(train_cnt)
    print('val')
    print('loss: {:.3f}, cls_acc: {:.3f}'.format(epoch_loss, cls_acc))
    return epoch_loss, cls_acc, label_list, pred_list