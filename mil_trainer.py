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
                 epochs=50, patience=10, negative_control=None, lq_loss=None):
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
        self.beta = 0.95
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

            wandb.log(metrics, step=epoch)
            if epochs_since_best > self.patience:
                epochs_since_best = 0
                self.lr = self.lr / np.sqrt(10)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    @staticmethod
    def avg_soft(tensor):
        mil = torch.mean(tensor, dim=0, keepdim=True)
        mil_soft = torch.nn.functional.softmax(mil, dim=-1)
        return mil_soft


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
        beta = self.beta ** self.curr_epoch
        simple_loss = torch.nn.NLLLoss()
        with tqdm(total=amnt * self.batch_size) as pbar:
            for i, data in enumerate(loader):
                x, y, = data
                if self.use_gpu:
                    x, y, = x.cuda(), y.cuda()
                if training:
                    # output is going to be a MIL output and a bunch of SIL outputs
                    mil_out, sil_out = self.model(x)
                    # sil_out is non_softmaxed probs
                    mil_log_probs = torch.log(self.avg_soft(sil_out))
                    # sil_soft = torch.nn.functional.softmax(sil_out, dim=-1)
                    # mil_soft = torch.mean(sil_soft, dim=0, keepdim=True)
                    # mil_log_probs = torch.log(mil_soft)
                    self.optimizer.zero_grad()
                    ratio = 1.0
                    simple_mil_loss = simple_loss(mil_log_probs, y)
                    simple_sil_loss = self.criterion(sil_out, torch.cat([y]*len(sil_out)))
                    total_loss = ratio*simple_mil_loss + (1-ratio) * simple_sil_loss
                    total_loss.backward()
                    # mil_loss = (1-beta)*self.criterion(mil_out, y)
                    # sil_loss = beta*self.criterion(sil_out, torch.cat([y]*len(sil_out)))
                    # loss = mil_loss + sil_loss
                    # simple_mil_loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        mil_out, sil_out = self.model(x)
                        sil_soft = torch.nn.functional.softmax(sil_out, dim=-1)
                        mil_soft = torch.mean(sil_soft, dim=0, keepdim=True)
                        mil_log_probs = torch.log(mil_soft)
                        simple_mil_loss = simple_loss(mil_log_probs, y)
                        # mil_loss = (1 - beta) * self.criterion(mil_out, y)
                        # sil_loss = beta * self.criterion(sil_out, torch.cat([y] * len(sil_out)))
                        # loss = mil_loss + sil_loss
                _, preds = torch.max(mil_out, 1)
                acc = torch.sum(preds == y.data).float() / len(y)
                losses.update(float(simple_mil_loss.data))
                accs.update(float(acc.data))
                pbar.set_description(f" - loss: {losses.avg:.3f} acc {accs.avg:.3f}")
                pbar.update(self.batch_size)
        return losses.avg, accs.avg

    def get_inference_results(self, loader):
        inference_results = {}
        self.model.eval()
        rand = 0
        beta = 1.0
        with torch.no_grad():
            for images, labels in tqdm(loader):
                images = images.cuda()
                results_mil, results_sil = self.model(images)
                rand += 1
                sil_preds = self.avg_soft(results_sil)[:,1].tolist()
                assert len(sil_preds) == 1
                # sil_preds = torch.nn.functional.softmax(results_sil, dim=-1)[:, 1].tolist()
                # sil_preds = np.median(sil_preds)
                # mil_preds = torch.nn.functional.softmax(results_mil, dim=-1)[:, 1].tolist()
                # assert len(mil_preds) == 1
                # mil_preds = mil_preds[0]
                preds = sil_preds[0]
                inference_results[str(rand)] = {
                    'predictions': preds,
                    'label': int(labels[0])
                }
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
        if self.negative_control is not None:
            inference_results_control = self.get_inference_results(self.negative_control)
            positive_control_results = {key: value for key, value in inference_results_test.items() if value['label'] == 1}
            inference_results_control.update(positive_control_results)
            control_auc = self.get_auc(inference_results_control)
        else:
            control_auc = 0

        test_auc = self.get_auc(inference_results_test)
        return test_auc, control_auc
