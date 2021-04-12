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

    def train(self):
        print(f"\n[*] Train on {self.num_train} samples, validate on {self.num_val} samples")
        best_val_auc = 0
        epochs_since_best = 0

        wandb.watch(self.model)

        for epoch in range(self.epochs):
            self.curr_epoch = epoch
            print(f'\nEpoch {epoch}/{self.epochs} -- lr = {self.lr}')
            train_loss, train_acc = self.run_one_epoch(training=True, rounds=1)
            val_inference_results = self.get_inference_results(self.val_loader)
            val_acc = self.get_acc(val_inference_results)
            val_auc = self.get_auc(val_inference_results)
            msg = f'train loss {train_loss:.3f} train acc {train_acc:.3f} -- val acc {val_acc:.3f} val auc {val_auc:.3f}'
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
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

            for param_group in self.optimizer.param_groups:
                curr_lr = param_group['lr']
                break
            metrics['curr_lr'] = curr_lr
            print(msg)

            wandb.log(metrics, step=epoch)
            if self.schedule_type == 'plateau':
                self.scheduler.step(val_auc)
            # if epochs_since_best > self.patience:
            #     epochs_since_best = 0
            #     self.lr = self.lr / np.sqrt(10)
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = self.lr

    @staticmethod
    def avg_soft(tensor):
        mil = torch.mean(tensor, dim=0, keepdim=True)
        mil_soft = torch.nn.functional.softmax(mil, dim=-1)
        return mil_soft


    def run_one_epoch(self, training, testing=False, rounds=1):
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
        simple_loss = torch.nn.CrossEntropyLoss()
        if testing:
            accum_amnt = 1
            max_size = 1
        else:
            accum_amnt = 8
            max_size = 4
        accum_counter = 0
        curr_shape = None
        batch_data = []
        with tqdm(total=amnt * self.batch_size * rounds) as pbar:
            for _ in range(rounds):
                for i, data in enumerate(loader):
                    x, y, = data
                    batch_data.append(data)
                    accum_counter += 1
                    if accum_counter < accum_amnt:
                        continue
                    accum_counter = 0
                    # now stack if possible
                    curr_shape = None
                    x_batched = []
                    y_batched = []
                    for x, y in batch_data:
                        if x.shape != curr_shape or len(x_batched) == 0 or len(x_batched[-1]) >= max_size:
                            curr_shape = x.shape
                            x_batched.append([])
                            y_batched.append([])
                        x_batched[-1].append(x)
                        y_batched[-1].append(y)
                    x_batched = [torch.stack(xb).squeeze(1) for xb in x_batched]
                    y_batched = [torch.stack(yb).squeeze(1) for yb in y_batched]
                    total_loss = 0
                    total_acc = 0
                    if training:
                        self.optimizer.zero_grad()
                    for x, y in zip(x_batched, y_batched):
                        if self.use_gpu:
                            x, y, = x.cuda(), y.cuda()
                        if training:
                            # output is going to be a MIL output and a bunch of SIL outputs
                            mil_out, sil_out = self.model(x)
                            simple_mil_loss = (simple_loss(mil_out, y) / accum_amnt)*x.shape[0]
                            simple_mil_loss.backward()
                            total_loss += float(simple_mil_loss.detach().cpu())
                        else:
                            with torch.no_grad():
                                mil_out, sil_out = self.model(x)
                                simple_mil_loss = (simple_loss(mil_out, y) / accum_amnt) * x.shape[0]
                                total_loss += float(simple_mil_loss.detach().cpu())
                        _, preds = torch.max(mil_out, 1)
                        with torch.no_grad():
                            total_acc += float(torch.sum(preds == y.data).float().detach())
                    if training:
                        self.optimizer.step()
                        if self.schedule_type == 'cyclic':
                            self.scheduler.step()
                    batch_data = []
                    acc = total_acc / accum_amnt

                    losses.update(total_loss)
                    accs.update(acc)

                    pbar.set_description(f" - loss: {losses.avg:.3f} acc {accs.avg:.3f}")
                    pbar.update(accum_amnt)
        return losses.avg, accs.avg

    def get_inference_results(self, loader):
        inference_results = {}
        self.model.eval()
        rand = 0
        with torch.no_grad():
            for images, labels in tqdm(loader):
                images = images.cuda()
                results_mil, results_sil = self.model(images)
                rand += 1
                preds = F.softmax(results_mil, dim=1)[0, 1]
                inference_results[str(rand)] = {
                    'predictions': preds,
                    'label': int(labels[0])
                }
        return inference_results

    @staticmethod
    def get_auc(inference_results):
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions'].cpu().detach().numpy()) for values in inference_results.values()]
        auc = roc_auc_score(labels, predictions)
        return auc

    @staticmethod
    def get_acc(inference_results):
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions'].cpu().detach().numpy()) for values in inference_results.values()]
        acc = np.sum(np.round(predictions) == labels) / len(labels)
        return acc

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
