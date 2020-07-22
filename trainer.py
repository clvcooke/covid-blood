from utils import AverageMeter, save_model
from sklearn.metrics import roc_auc_score
import numpy as np
import wandb
from tqdm import tqdm
import torch
import os


class ClassificationTrainer:

    def __init__(self, model, optimizer, train_loader, val_loader, test_loader=None, test_interval=5, batch_size=8,
                 epochs=50, patience=10, negative_control=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.negative_control = negative_control
        self.test_interval = test_interval
        self.batch_size = batch_size
        self.num_train = len(train_loader)
        self.num_val = len(val_loader)
        self.num_test = len(test_loader) if test_loader is not None else 0
        self.epochs = epochs
        self.curr_epoch = 0
        self.use_gpu = next(self.model.parameters()).is_cuda
        # TODO: this assumes a global learning rate
        self.lr = self.optimizer.param_groups[0]['lr']
        self.patience = patience
        self.criterion = torch.nn.CrossEntropyLoss()
        # hack to get the wandb unique ID
        self.run_name = os.path.basename(wandb.run.path)

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
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }

            if self.test_loader is not None and ((epoch % self.test_interval) == 0 or epoch == (self.epochs - 1)):
                test_loss, test_acc = self.run_one_epoch(training=False, testing=True)
                test_auc, control_auc = self.get_auc()
                metrics.update({
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'test_auc': test_auc,
                    'control_auc': control_auc
                })
                msg += f' -- test loss {test_loss:.3f} test acc {test_acc:.3f} test auc {test_auc:.3f} control auc {control_auc:.3f}'
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                msg += '[*]'
                # TODO: implement model saving
                save_model(self.model, self.run_name)
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            print(msg)

            wandb.log(metrics, step=epoch)
            if epochs_since_best > self.patience:
                epochs_since_best = 0
                self.lr = self.lr / np.sqrt(10)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def run_one_epoch(self, training, testing=False):
        losses = AverageMeter()
        accs = AverageMeter()
        if training:
            # using train set, doing updates
            if testing:
                raise RuntimeError()
            amnt = self.num_train
            loader = self.train_loader
            self.model.train()
        elif testing:
            # evaling test set
            amnt = self.num_test
            loader = self.test_loader
            self.model.eval()
        else:
            # evaling val set
            amnt = self.num_val
            loader = self.val_loader
            self.model.eval()
        with tqdm(total=amnt * self.batch_size) as pbar:
            for i, data in enumerate(loader):
                (x, _), y, = data
                if self.use_gpu:
                    x, y, = x.cuda(), y.cuda()
                if training:
                    output = self.model(x)
                    self.optimizer.zero_grad()
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        output = self.model(x)
                        loss = self.criterion(output, y)
                _, preds = torch.max(output, 1)
                acc = torch.sum(preds == y.data).float() / len(y)
                losses.update(loss.data)
                accs.update(acc.data)
                pbar.set_description(f" - loss: {losses.avg:.3f} acc {accs.avg:.3f}")
                pbar.update(self.batch_size)
        return losses.avg, accs.avg

    def get_auc(self):
        # for every image
        inference_results = {}
        with torch.no_grad():
            for (images, filenames), labels in tqdm(self.test_loader):
                images = images.cuda()
                results = self.model(images)
                preds = torch.nn.functional.softmax(results, dim=-1)[:, 1]
                preds = preds.tolist()
                for filename, pred, label in zip(filenames, preds, labels):
                    order = os.path.basename(filename).split('_')[0]
                    try:
                        int(order)
                    except:
                        order = os.path.basename(os.path.dirname(filename))
                        int(order)
                    if order not in inference_results:
                        inference_results[order] = {}
                        inference_results[order]['predictions'] = []
                        inference_results[order]['label'] = int(label)
                    inference_results[order]['predictions'].append(pred)
        control_results = {key: value for key, value in inference_results.items() if value['label'] == 1}
        with torch.no_grad():
            for (images, filenames), labels in tqdm(self.negative_control):
                images = images.cuda()
                results = self.model(images)
                preds = torch.nn.functional.softmax(results, dim=-1)[:, 1]
                preds = preds.tolist()
                for filename, pred, label in zip(filenames, preds, labels):
                    order = os.path.basename(filename).split('_')[0]
                    try:
                        int(order)
                    except:
                        order = os.path.basename(os.path.dirname(filename))
                        int(order)
                    if order not in control_results:
                        control_results[order] = {}
                        control_results[order]['predictions'] = []
                        control_results[order]['label'] = int(label)
                    control_results[order]['predictions'].append(pred)
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions']) for values in inference_results.values()]
        control_preds = [np.median(values['predictions']) for values in control_results.values()]
        control_labels = [values['label'] for values in control_results.values()]
        test_auc = roc_auc_score(labels, predictions)
        control_auc = roc_auc_score(control_labels, control_preds)
        return test_auc, control_auc
