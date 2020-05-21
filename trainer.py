from utils import AverageMeter
import numpy as np
import wandb
from tqdm import tqdm
import torch


class MILTrainer:
    def __init__(self):
        pass


class ClassificationTrainer:

    def __init__(self, model, optimizer, train_loader, val_loader, batch_size, epochs, patience=10):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.num_train = len(train_loader)
        self.num_val = len(val_loader)
        self.epochs = epochs
        self.curr_epoch = 0
        self.use_gpu = next(self.model.parameters()).is_cuda
        # TODO: this assumes a global learning rate
        self.lr = self.optimizer.param_groups[0]['lr']
        self.patience = patience
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        print(f"\n[*] Train on {self.num_train} samples, validate on {self.num_val} samples")
        best_val_loss = np.inf
        epochs_since_best = 0
        for epoch in range(self.epochs):
            self.curr_epoch = epoch
            print(f'\nEpoch {epoch}/{self.epochs} -- lr = {self.lr}')
            train_loss, train_acc = self.run_one_epoch(training=True)
            val_loss, val_acc = self.run_one_epoch(training=False)
            msg = f'train loss {train_loss:.3f} train acc {train_acc:.3f} -- val loss {val_loss:.3f} val acc {val_acc:.3f}'
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                msg += '[*]'
                # TODO: implement model saving
                # save_model
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch)
            if epochs_since_best > self.patience:
                epochs_since_best = 0
                self.lr = self.lr / np.sqrt(10)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def run_one_epoch(self, training):
        losses = AverageMeter()
        accs = AverageMeter()
        if training:
            amnt = self.num_train
            loader = self.train_loader
        else:
            amnt = self.num_val
            loader = self.val_loader
        with tqdm(total=amnt) as pbar:
            for i, data in enumerate(loader):
                x, y, = data
                if self.use_gpu:
                    x, y, = x.cuda(), y.cuda()
                output = self.model(x)
                if training:
                    self.optimizer.zero_grad()
                loss = self.criterion(output, y)
                _, preds = torch.max(output, 1)
                acc = torch.sum(preds == y.data) / len(y)
                if training:
                    loss.backward()
                    self.optimizer.step()
                losses.update(loss.data)
                accs.update(acc.data)
                pbar.set_description(f" - loss: {losses.avg:.3f} acc {accs.avg:.3f}")
                pbar.update(self.batch_size)
        return losses.avg, accs.avg
