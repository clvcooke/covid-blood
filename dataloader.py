import json
import torch
from torch.utils.data import DataLoader, random_split
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


class ImageCache:
    def __init__(self, cache_amnt=-1):
        self.cached = {}

    def get(self, path):
        if path not in self.cached:
            self.cached[path] = Image.open(path).copy()
        else:
            return self.cached[path]
        return self.cached[path]


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, bags, data_transforms=None, metadata=None, cache_images=True):
        """

        :param bags: list of dicts containing paths/labels/orders
        :param data_transforms: image transforms applied to images
        :param metadata: any extra data the should be stored with the dataset for convenience
        """
        self.bags = bags
        self.metadata = metadata
        self.data_transforms = data_transforms
        self.cache_images = cache_images
        if self.cache_images:
            self.image_cache = ImageCache()

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_paths = self.bags[index]['files']
        label = self.bags[index]['label']
        if self.cache_images:
            images = [self.image_cache.get(image_path) for image_path in image_paths]
        else:
            images = [Image.open(image_path) for image_path in image_paths]
        transformed_images = torch.stack([self.data_transforms(image) for image in images])
        return transformed_images, label


class SingleCellDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_labels, data_transforms=None, metadata=None, extract_filenames=False,
                 cache_images=True):
        """

        :param data_x: input data to network
        :param data_y: labels
        :param data_transforms: image transforms to be applied to images
        :param metadata: any extra data that should be stored with the images for convenience
        """
        assert len(image_paths) == len(image_labels), 'length mismatch between x and y'
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.metadata = metadata
        self.data_transforms = data_transforms
        self.extract_filenames = extract_filenames
        self.cache_images = cache_images
        if cache_images:
            self.image_cache = ImageCache()

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_path = self.image_paths[index]
        if self.cache_images:
            image = self.image_cache.get(image_path)
        else:
            image = Image.open(image_path)
        label = self.image_labels[index]
        if self.data_transforms is not None:
            image = self.data_transforms(image)
        if self.extract_filenames:
            return (image, image_path), label
        else:
            return image, label


def get_strat_fold(x, y, fold_seed=0, fold_index=0, fold_count=6):
    assert fold_index < fold_count
    folder = model_selection.StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=fold_seed)
    train_split, test_split = list(folder.split(x, y))[fold_index]
    x = np.array(x)
    y = np.array(y)
    train_x, train_y = x[train_split], y[train_split]
    test_x, test_y = x[test_split], y[test_split]
    return train_x, train_y, test_x, test_y


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
    split = list(folder.split(X=data))[fold_index]
    data = np.array(data)
    return data[split[0]], data[split[1]]


def get_patient_orders(exclude_orders=None):
    base_path = '/hddraid5/data/colin/covid-data/'
    label_files = glob.glob(os.path.join(base_path, '*Covid*.xlsx'))
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
    for order, test_result in tqdm(zip(orders, test_results), desc='reading excel files', total=len(orders)):
        try:
            np.int(order)
            if test_result.lower() == 'positive':
                label = True
            elif test_result.lower() == 'negative':
                label = False
            else:
                continue
        except (TypeError, AttributeError, ValueError):
            continue
        all_image_paths = glob.glob(os.path.join(base_path, 'COVID Research Images', '**', str(order), '**', '*.jpg'),
                                    recursive=True)
        image_paths = [image_path for image_path in all_image_paths if
                       (os.path.getsize(image_path) < IMAGE_SIZE_CUTOFF_UPPER and os.path.getsize(
                           image_path) > IMAGE_SIZE_CUTOFF_LOWER)]
        if len(image_paths) == 0:
            continue
        if label:
            positive_images[str(order)] = image_paths
        else:
            negative_images[str(order)] = image_paths
    # sort by order number, python 3.7 has dictionaries ordered by default
    negative_images = dict(sorted(negative_images.items()))
    positive_images = dict(sorted(positive_images.items()))
    all_images = dict(negative_images, **positive_images)
    return negative_images, positive_images, all_images


def load_orders_into_bags(orders, image_paths, labels, exclusion=None):
    bags = []
    for order, label in tqdm(zip(orders, labels)):
        if exclusion is None:
            files = image_paths[order]
        else:
            files = [image_path for image_path in image_paths[order] if os.path.basename(image_path) not in exclusion]
        if len(files) == 0:
            print(f"Order {order} has zero files")
            continue
        bag = {
            "order": order,
            "label": label,
            "files": files
        }
        bags.append(bag)
    return bags


def load_orders(orders, image_paths, labels, exclusion=None):
    all_labels = []
    all_orders = []
    all_files = []
    for order, label in tqdm(zip(orders, labels)):
        labels = []
        orders = []
        files = []
        for image_path in image_paths[order]:
            # exclude all files in the exclusion set
            if exclusion is not None and os.path.basename(image_path) in exclusion:
                continue
            labels.append(label)
            orders.append(order)
            files.append(image_path)
        all_labels += labels
        all_orders += orders
        all_files += files
    return all_labels, all_orders, all_files


def load_pbc_data(train_transforms=None, val_transforms=None, batch_size=8):
    """

    :param train_transforms:
    :param val_transforms:
    :param batch_size:
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
    train_loader, val_loader = [torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for
                                x in ['train', 'val']]
    return train_loader, val_loader


def load_all_patients(train_transforms=None, test_transforms=None, group_by_patient=False, batch_size=8, fold_number=0,
                      fold_seed=0,
                      fold_count=6,
                      extract_filenames=False,
                      exclusion=None):
    """
    Loads all the data from the Duke COVID +/- dataset
    :param group_by_patient: return one entry per patient (bag style), default false
    :param batch_size: images to load at a time
    :param fold_number: which fold to use for validation, default 0
    :param fold_seed: seed for fold generation, default 0
    :param fold_count: number of folds to use, validation split = 1/fold_count
    :return: iterable dataset objects for training and validation
    """
    negative_image_paths, positive_image_paths, all_image_paths = get_patient_orders()
    negative_orders = list(negative_image_paths.keys())
    positive_orders = list(positive_image_paths.keys())
    orders = negative_orders + positive_orders
    labels = [0] * len(negative_orders) + [1] * len(positive_orders)
    # split into train/val
    # TODO: add test data splitting
    train_orders, train_labels, test_orders, test_labels = get_strat_fold(orders, labels, fold_index=fold_number,
                                                                          fold_seed=fold_seed, fold_count=fold_count)
    if exclusion is not None:
        with open(exclusion) as fp:
            # set to go fast
            exclusion_set = set(json.load(fp))
    else:
        # empty set as default
        exclusion_set = set()
    if group_by_patient:
        train_bags = load_orders_into_bags(train_orders, all_image_paths, train_labels, exclusion=exclusion_set)
        test_bags = load_orders_into_bags(test_orders, all_image_paths, test_labels, exclusion=exclusion_set)
        training_dataset = BagDataset(train_bags, data_transforms=train_transforms)
        test_dataset = BagDataset(test_bags, data_transforms=test_transforms)
    else:
        train_labels, train_orders, train_files = load_orders(train_orders, all_image_paths, train_labels,
                                                              exclusion=exclusion_set)
        test_labels, test_orders, test_files = load_orders(test_orders, all_image_paths, test_labels,
                                                           exclusion=exclusion_set)
        # now we want to make a dataset out of the images/labels
        training_dataset = SingleCellDataset(train_files, train_labels, data_transforms=train_transforms,
                                             metadata={
                                                 'orders': train_orders,
                                             },
                                             extract_filenames=extract_filenames)

        test_dataset = SingleCellDataset(test_files, test_labels, data_transforms=test_transforms,
                                         metadata={
                                             'orders': test_orders,
                                         }, extract_filenames=extract_filenames)
    val_split = 0.8
    training_len = len(training_dataset)
    train_len = int(training_len * val_split)
    train_dataset, validation_dataset = random_split(training_dataset, [train_len, training_len - train_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader, val_loader, test_loader
