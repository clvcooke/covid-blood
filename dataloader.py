from torchvision import datasets, transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn import model_selection
from pandas import read_excel
import os
import glob

# used to filter out large images (not of single cells)
IMAGE_SIZE_CUTOFF_UPPER = 800000
# size in bytes
IMAGE_SIZE_CUTOFF_LOWER = 100


def get_fold(data, random_state=0, fold_index=0, folds=6):
    """
    :param data: data to split
    :param random_state: for re-producability, default 0 for most use-cases
    :param split_index: which of the splits to use
    :param folds: how many splits you want to use
    :return:
    """
    assert fold_index < folds
    folder = model_selection.KFold(n_splits=folds, shuffle=True, random_state=random_state)
    split = folder.split(X=data)[fold_index]
    data = np.array(data)
    return data[split[0]], data[split[1]]


def get_patient_orders(exclude_orders=None):
    base_path = '/hddraid5/data/colin/covid-data/'
    label_files = glob.glob(os.path.join(base_path, '*.xlsx'))
    orders = []
    test_results = []
    for label_file in label_files:
        table = read_excel(label_file)
        table_orders = list(table['Order #'])
        table_test_results = list(table['Covid Test result'])
        orders = orders + table_orders
        test_results = test_results + table_test_results
    positive_images = {}
    negative_images = {}
    for order, test_result in zip(orders, test_results):
        try:
            label = 'positive' in test_result.lower()
            np.int(order)
        except (TypeError, AttributeError):
            continue
        all_image_paths = glob.glob(os.path.join(base_path, 'COVID Research Images', '**', str(order), '**', '*.jpg'),
                                    recursive=True)
        image_paths = [image_path for image_path in all_image_paths if
                       (os.path.getsize(image_path) < IMAGE_SIZE_CUTOFF_UPPER and os.path.getsize(
                           image_path) > IMAGE_SIZE_CUTOFF_LOWER)]
        if label:
            positive_images[str(order)] = image_paths
        else:
            negative_images[str(order)] = image_paths
    # sort by order number, python 3.7 has dictionaries ordered by default
    negative_images = dict(sorted(negative_images))
    positive_images = dict(sorted(positive_images))
    return negative_images, positive_images


def load_all_patients(group_by_patient=False, batch_size=8, fold_number=0, fold_seed=0, fold_count=6):
    """
    Loads all the data from the Duke COVID +/- dataset
    :param group_by_patient: return one entry per patient (bag style), default false
    :param batch_size: images to load at a time
    :param fold_number: which fold to use for validation, default 0
    :param fold_seed: seed for fold generation, default 0
    :param fold_count: number of folds to use, validation split = 1/fold_count
    :return: iterable dataset objects for training and validation
    """
    negative_images, positive_images = get_patient_orders()
    negative_orders = list(negative_images.keys())
    positive_orders = list(positive_images.keys())
    train_positive_orders, val_positive_orders = get_fold(positive_orders, fold_index=fold_number)
    train_negative_orders, val_negative_orders = get_fold(negative_orders, fold_index=fold_number)