from utils import AverageMeter, save_model
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
import torch
import os


class ClassificationTrainer:

    def __init__(self, model, optimizer, train_loader, val_loader, test_loader=None, test_interval=5, batch_size=8,
                 epochs=50, patience=10, negative_control=None, lq_loss=None, scheduler=None, schedule_type=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_type = schedule_type
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
        self.step_num = 0
        self.use_gpu = next(self.model.parameters()).is_cuda
        # TODO: this assumes a global learning rate
        self.lr = self.optimizer.param_groups[0]['lr']
        self.patience = patience
        if lq_loss is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = lambda y_pred, y_true: self.lq_loss(lq_loss, y_pred, y_true)
        # hack to get the wandb unique ID
        self.run_name = os.path.basename(wandb.run.path)

    @staticmethod
    def lq_loss(q, y_pred, y_true):
        y_one_hot = torch.eye(y_pred.shape[-1], dtype=y_pred.dtype, device=y_pred.device)[y_true]
        return torch.mean(torch.sum((1 - F.softmax(y_pred, 1).pow(q)) * y_one_hot / q, axis=1))

    def train(self):
        print(f"\n[*] Train on {self.num_train} samples, validate on {self.num_val} samples")
        best_val_auc = 0
        epochs_since_best = 0
        for epoch in range(self.epochs):
            self.curr_epoch = epoch
            print(f'\nEpoch {epoch}/{self.epochs} -- lr = {self.lr}')
            train_loss, train_acc = self.run_one_epoch(training=True)
            val_loss, val_acc = self.run_one_epoch(training=False)
            val_auc = self.get_auc(self.get_inference_results(self.val_loader))
            msg = f'train loss {train_loss:.3f} train acc {train_acc:.3f} -- val loss {val_loss:.3f} val acc {val_acc:.3f} val auc {val_auc:.3f}'
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc
            }
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_model(self.model, self.run_name)
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            if self.test_loader is not None and (
                    (epoch % self.test_interval) == 0 or epoch == (self.epochs - 1) or epochs_since_best == 0):
                test_loss, test_acc = self.run_one_epoch(training=False, testing=True)
                test_auc, control_auc = self.get_test_control_auc()
                metrics.update({
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'test_auc': test_auc,
                    'control_auc': control_auc
                })
                msg += f' -- test loss {test_loss:.3f} test acc {test_acc:.3f} test auc {test_auc:.3f} control auc {control_auc:.3f}'
            if epochs_since_best == 0:
                msg += '[*]'

            print(msg)
            for param_group in self.optimizer.param_groups:
                curr_lr = param_group['lr']
                break

            metrics['curr_lr'] = curr_lr
            wandb.log(metrics, step=epoch)
            if self.schedule_type == 'plateau':
                self.scheduler.step(val_auc)
            # if epochs_since_best > self.patience:
            #     epochs_since_best = 0
            #     self.lr = self.lr / np.sqrt(10)
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = self.lr

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
                    if self.schedule_type == 'cyclic':
                        self.scheduler.step()

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

    def get_inference_results(self, loader):
        inference_results = {}
        self.model.eval()
        with torch.no_grad():
            for (images, filenames), labels in tqdm(loader):
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
        return inference_results

    @staticmethod
    def get_auc(inference_results):
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions']) for values in inference_results.values()]
        auc = roc_auc_score(labels, predictions)
        return auc

    def get_test_control_auc(self):
        # for every image
        inference_results_test = self.get_inference_results(self.test_loader)
        inference_results_control = self.get_inference_results(self.negative_control)
        positive_control_results = {key: value for key, value in inference_results_test.items() if value['label'] == 1}
        inference_results_control.update(positive_control_results)
        test_auc = self.get_auc(inference_results_test)
        control_auc = self.get_auc(inference_results_control)
        return test_auc, control_auc
