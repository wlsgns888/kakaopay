from pickletools import optimize
from tkinter import N
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from config.config import *

def _calc_accuracy(prediction, label):
    if type(prediction).__name__ =='Tensor':
        _, max_indices = torch.max(prediction, 1)
        accuracy = (max_indices == label).sum().data.cpu().numpy() / max_indices.size()[0]

        prediction = torch.argmax(prediction, dim=-1).data.cpu().numpy()
        label = label.data.cpu().numpy()
    else:
        accuracy = (np.array(prediction) == np.array(label)).sum()/ len(prediction)
    f1 = np.mean(
        f1_score(
            prediction,
            label,
            average=None,
        )
    )
    return accuracy, f1, prediction, label

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_func, device, epochs, lr, lr_decay, gamma, nc) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.lr_decay=lr_decay
        self.gamma = gamma
        self.label = list(label_decode_dict(nc).values())


    def train(self, train_loader, epoch):
        total_loss = 0.0
        total_acc = 0.0
        total_f1_score = 0.0
        total_batch = 0
        self.model.train()

        train_loader_len = len(train_loader)
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=train_loader_len)

        n = 0
        for i, item in pbar:
            adjust_learning_rate(self.optimizer, self.epochs, epoch, i, train_loader_len, self.lr, self.gamma, lr_decay=self.lr_decay)
            input = item['img'].to(self.device)
            target = item['label'].to(self.device)
            output = self.model(input)
            loss = self.loss_func(output, target)

            acc, f1, prediction, label = _calc_accuracy(output, target)
            
            total_loss += loss.item()
            total_acc += acc
            total_f1_score += f1
            total_batch += 1
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            n += input.size(0)
            pbar.desc = '(Train | {batch}/{size} | Loss: {loss:.4f} | acc: {acc:.4f} | f1_score: {f1:.4f})'.format(
                        batch=total_batch, 
                        size=train_loader_len,
                        loss = total_loss / total_batch,
                        acc = total_acc / total_batch,
                        f1 = total_f1_score /total_batch,
                    )
        pbar.close()

        return (
            round((total_loss / total_batch), 4),
            round((total_acc / total_batch), 4),
            round((total_f1_score / total_batch), 4),
        )

    def validate(self, val_loader, epoch):
        total_loss = 0.0
        total_acc = 0.0
        total_f1_score = 0.0
        total_batch = 0
        self.val_predict = []
        self.val_label = []

        self.model.eval()

        val_loader_len = len(val_loader)
        pbar = enumerate(val_loader)
        pbar = tqdm(pbar, total=val_loader_len)
        n=0
        with torch.no_grad():
            for i, item in pbar:
                input = item['img'].to(self.device)
                target = item['label'].to(self.device)
                output = self.model(input)
                loss = self.loss_func(output, target)

                acc, f1, prediction, label = _calc_accuracy(output, target)
                self.val_predict.extend(prediction)
                self.val_label.extend(label)
                total_loss += loss.item()
                total_acc += acc
                total_f1_score += f1
                total_batch += 1

                n += input.size(0)
                pbar.desc = '(Val | {batch}/{size} | Loss: {loss:.4f} | acc: {acc:.4f} | f1_score: {f1:.4f})'.format(
                            batch=total_batch, 
                            size=val_loader_len,
                            loss = total_loss / total_batch,
                            acc = total_acc / total_batch,
                            f1 = total_f1_score /total_batch,
                        )
        pbar.close()

        total_acc, total_f1_score, prediction, label = _calc_accuracy(self.val_predict, self.val_label)
        return (
            round(total_loss/total_batch, 4),
            round(total_acc, 4),
            round((total_f1_score), 4),
        )
    def save_model(self, save_path):
        torch.save(
        {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            #"scheduler": self.scheduler.state_dict(),
        },
        save_path,
    )

    def save_confusion_matrix(self, save_folder_path: str):
        matrix = confusion_matrix(self.val_label, self.val_predict)
        recalls = [matrix[i][i]/matrix[i].sum() for i in range(len(matrix))]
        precisions = [matrix[i][i]/matrix[:,i].sum() for i in range(len(matrix))]

        num_predictions = [int(matrix[:,i].sum()) for i in range(len(matrix))]
        num_gts = [int(matrix[i].sum()) for i in range(len(matrix))]

        avg_recall = np.array(recalls).sum()/len(recalls)
        avg_precision = np.array(precisions).sum()/len(precisions)
        f1_score = 2.0 / (1/avg_precision + 1/avg_recall)
        accuracy = np.diag(matrix).sum()/matrix.sum()

        xticklabels= ['{}({})\n:precision {:0.2f}'.format(l,cnt,prec) for l, cnt, prec in zip(self.label, num_predictions, precisions)]
        yticklabels= ['{}({})\n:recall {:0.2f}'.format(l, cnt, recall) for l, cnt, recall in zip(self.label, num_gts, recalls)]

        data_frame = pd.DataFrame(matrix, columns=self.label, index=self.label)
        plt.figure(figsize=(12, 11))
        cm = sns.heatmap(
            data_frame,
            cmap="Blues",
            annot_kws={"size": 20},
            annot=True,
            fmt="d",
            linecolor="grey",
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Prediction (Avg. Precision: {:0.2f})\n F1-score:{:0.2f}, Accuracy:{:0.2f}".format(avg_precision,f1_score, accuracy),fontsize=20)
        plt.ylabel("GT (Avg. Recall: {:0.2f})".format(avg_recall),fontsize=20)

        save_path = os.path.join(save_folder_path, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()

from math import cos, pi
def adjust_learning_rate(optimizer, epochs, epoch, iteration, num_iter, lr, gamma, lr_decay='cos'):
    warmup_epoch = 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter

    if lr_decay == 'step':
        lr = lr * (gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif lr_decay == 'cos':
        lr = lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif lr_decay == 'linear':
        lr = lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    else:
        raise ValueError('Unknown lr mode {}'.format(lr_decay))

    if epoch < warmup_epoch:
        lr = lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
