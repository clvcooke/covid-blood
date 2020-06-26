from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pwd


class AverageMeter:
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_torch(random_seed, use_gpu, gpu_number=0):
    torch.manual_seed(random_seed)
    torch.set_num_threads(1)
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
        torch.cuda.manual_seed(random_seed)


def aggregate_scores(score_list, method='median'):
    """
    :param score_list: confidence scores for a patient, in list or as a np array. Should be for a single patient
    :param method: aggregation method to use (recommend mean or median)
    :return: aggregated score (single float)
    """
    scores_np = np.float_(score_list)
    if method == 'median':
        return np.median(scores_np)
    elif method == 'mean':
        return np.mean(scores_np)
    elif method == 'max':
        return np.max(scores_np)
    elif method == 'min':
        return np.min(scores_np)
    elif method == 'range':
        return np.max(scores_np) - np.min(scores_np)
    elif method == 'chance':
        return 1


def plot_roc_curve(ground_truth, scores, aggregation_method='mean'):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    predicted_scores = aggregate_scores(scores, aggregation_method)
    fpr, tpr, thresholds = roc_curve(ground_truth, predicted_scores)
    auc = roc_auc_score(ground_truth, predicted_scores)
    axs.set_title(f"ROC Curve \n AUC = {auc:.3f}")
    axs.plot(fpr, tpr)
    axs.set_xlabel("False Positive Rate")
    axs.set_ylabel("True Positive Rate")
    fig.show()


def save_model(model, model_name, model_dir=None):
    if model_dir is None:
        model_dir = f"/hddraid5/data/{pwd.getpwuid(os.getuid()).pw_name}/models"
    model_path = os.path.join(model_dir, model_name + '.pth')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


def load_model(model, model_path=None, model_id=None, strict=True):
    if model_path is not None:
        model_state_dict = torch.load(model_path)
    else:
        assert model_id is not None
        if os.path.exists('/hddraid5/data'):
            model_path = f"/hddraid5/data/colin/models/{model_id}.pth"
        else:
            model_path = f"/home/col/models/{model_id}.pth"
        return load_model(model, model_path, strict=strict)
    try:
        model.load_state_dict(model_state_dict, strict=strict)
    except RuntimeError:
        pass
    return model


def get_covid_transforms(image_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.10, contrast=0.20, saturation=0.20, hue=0.20),
            transforms.RandomAffine(degrees=10, scale=(1.05, 0.95), shear=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms
