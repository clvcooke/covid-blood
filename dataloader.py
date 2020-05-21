import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn import model_selection
from pandas import read_excel
import os
import glob
from PIL import Image
from tqdm import tqdm

# used to filter out large images (not of single cells)
IMAGE_SIZE_CUTOFF_UPPER = 800000
# size in bytes
IMAGE_SIZE_CUTOFF_LOWER = 100


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, data_transforms=None, metadata=None):
        """

        :param data_x: input data to network
        :param data_y: labels
        :param data_transforms: image transforms to be applied to images
        :param metadata: any extra data that should be stored with the images for convenience
        """
        assert len(data_x) == len(data_y), 'length mismatch between x and y'
        self.data_x = data_x
        self.data_y = data_y
        self.metadata = metadata
        self.data_transforms = data_transforms

    def __getitem__(self, index):
        image = self.data_x[index]
        label = self.data_y[index]
        if self.data_transforms is not None:
            image = self.data_transforms(image)
        return image, label


def get_fold(data, fold_seed=0, fold_index=0, fold_count=6):
    """
    :param data: data to split
    :param fold_seed: for re-producability, default 0 for most use-cases
    :param fold_index: which of the splits to use
    :param fold_count: how many splits you want to use
    :return:
    """
    assert fold_index < fold_count
    folder = model_selection.KFold(n_splits=fold_count, shuffle=True, random_state=fold_seed)
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


def load_orders(orders, image_paths, label):
    all_images = []
    all_labels = []
    all_orders = []
    all_files = []
    for order in tqdm(orders):
        images = []
        labels = []
        orders = []
        files = []
        for image_path in image_paths[order]:
            image = Image.open(image_path)
            images.append(image)
            labels.append(label)
            orders.append(order)
            files.append(os.path.basename(image_path))
        all_images += images
        all_labels += labels
        all_orders += orders
        all_files += files
    return all_images, all_labels, all_orders, all_files


def load_pbc_data(train_transforms=None, val_transforms=None, batch_size=8):
    """

    :param train_transforms:
    :param val_transforms:
    :param batch_size:
    :param fold_number:
    :param fold_seed:
    :param fold_count:
    :return:
    """
    data_dir = '/hddraid5/data/colin/cell_classification/data'
    data_transforms = {
        'train': train_transforms,
        'val': val_transforms
    }
    # luckily torchvision has a nice class for this scenario
    # Create training and validation datasets
    image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    train_loader, val_loader = [torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']]
    return train_loader, val_loader

def load_all_patients(train_transforms=None, val_transforms=None, group_by_patient=False, batch_size=8, fold_number=0,
                      fold_seed=0,
                      fold_count=6):
    """
    Loads all the data from the Duke COVID +/- dataset
    :param group_by_patient: return one entry per patient (bag style), default false
    :param batch_size: images to load at a time
    :param fold_number: which fold to use for validation, default 0
    :param fold_seed: seed for fold generation, default 0
    :param fold_count: number of folds to use, validation split = 1/fold_count
    :return: iterable dataset objects for training and validation
    """
    negative_image_paths, positive_image_paths = get_patient_orders()
    negative_orders = list(negative_image_paths.keys())
    positive_orders = list(positive_image_paths.keys())
    # split into train/val
    # TODO: add test data splitting
    train_positive_orders, val_positive_orders = get_fold(positive_orders, fold_index=fold_number, fold_seed=fold_seed,
                                                          fold_count=fold_count)
    train_negative_orders, val_negative_orders = get_fold(negative_orders, fold_index=fold_number, fold_seed=fold_seed,
                                                          fold_count=fold_count)
    if group_by_patient:
        raise RuntimeError("Needs to be implemented still")
    else:
        # first we load the data into memory from disk
        train_pos_images, train_pos_labels, train_pos_orders, train_pos_files = load_orders(train_positive_orders,
                                                                                            positive_image_paths, 1)
        train_neg_images, train_neg_labels, train_neg_orders, train_neg_files = load_orders(train_negative_orders,
                                                                                            negative_image_paths, 0)
        train_images = train_pos_images + train_neg_images
        train_labels = train_pos_labels + train_neg_labels
        train_orders = train_pos_orders + train_neg_orders
        train_files = train_pos_files + train_neg_files

        val_pos_images, val_pos_labels, val_pos_orders, val_pos_files = load_orders(val_positive_orders,
                                                                                    positive_image_paths, 1)
        val_neg_images, val_neg_labels, val_neg_orders, val_neg_files = load_orders(val_negative_orders,
                                                                                    negative_image_paths, 0)
        val_images = val_pos_images + val_neg_images
        val_labels = val_pos_labels + val_neg_labels
        val_orders = val_pos_orders + val_neg_orders
        val_files = val_pos_files + val_neg_files

        # now we want to make a dataset out of the images/labels
        train_dataset = CustomDataset(train_images, train_labels, data_transforms=train_transforms,
                                      metadata={
                                          'orders': train_orders,
                                          'filenames': train_files
                                      })
        val_dataset = CustomDataset(val_images, val_labels, data_transforms=val_transforms,
                                    metadata={
                                        'orders': val_orders,
                                        'filenames': val_files
                                    })
    # TODO: should we pin memory?
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
