import json
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
    for order, test_result in tqdm(zip(orders, test_results), desc='reading excel files', total=len(orders)):
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
    negative_images = dict(sorted(negative_images.items()))
    positive_images = dict(sorted(positive_images.items()))
    return negative_images, positive_images


def load_orders_into_bags(orders, image_paths, label, exclusion=None):
    bags = []
    for order in tqdm(orders):

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


def load_orders(orders, image_paths, label, exclusion=None):
    all_labels = []
    all_orders = []
    all_files = []
    for order in tqdm(orders):
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


def load_all_patients(train_transforms=None, val_transforms=None, group_by_patient=False, batch_size=8, fold_number=0,
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
    negative_image_paths, positive_image_paths = get_patient_orders()
    negative_orders = list(negative_image_paths.keys())
    positive_orders = list(positive_image_paths.keys())
    # split into train/val
    # TODO: add test data splitting
    train_positive_orders, val_positive_orders = get_fold(positive_orders, fold_index=fold_number, fold_seed=fold_seed,
                                                          fold_count=fold_count)
    train_negative_orders, val_negative_orders = get_fold(negative_orders, fold_index=fold_number, fold_seed=fold_seed,
                                                          fold_count=fold_count)
    if exclusion is not None:
        with open(exclusion) as fp:
            # set to go fast
            exclusion_set = set(json.load(fp))
    else:
        # empty set as default
        exclusion_set = set()
    if group_by_patient:
        train_pos_bags = load_orders_into_bags(train_positive_orders,
                                               positive_image_paths, 1,
                                               exclusion=exclusion_set)
        train_neg_bags = load_orders_into_bags(train_negative_orders,
                                               negative_image_paths, 0,
                                               exclusion=exclusion_set)
        train_bags = train_pos_bags + train_neg_bags

        val_pos_bags = load_orders_into_bags(val_positive_orders,
                                             positive_image_paths, 1,
                                             exclusion=exclusion_set)
        val_neg_bags = load_orders_into_bags(val_negative_orders,
                                             negative_image_paths, 0,
                                             exclusion=exclusion_set)
        val_bags = val_pos_bags + val_neg_bags
        train_dataset = BagDataset(train_bags, data_transforms=train_transforms)
        val_dataset = BagDataset(val_bags, data_transforms=val_transforms)
    else:
        # first we load the data into memory from disk
        train_pos_labels, train_pos_orders, train_pos_files = load_orders(train_positive_orders,
                                                                          positive_image_paths, 1,
                                                                          exclusion=exclusion_set)
        train_neg_labels, train_neg_orders, train_neg_files = load_orders(train_negative_orders,
                                                                          negative_image_paths, 0,
                                                                          exclusion=exclusion_set)
        train_labels = train_pos_labels + train_neg_labels
        train_orders = train_pos_orders + train_neg_orders
        train_files = train_pos_files + train_neg_files

        val_pos_labels, val_pos_orders, val_pos_files = load_orders(val_positive_orders,
                                                                    positive_image_paths, 1,
                                                                    exclusion=exclusion_set)
        val_neg_labels, val_neg_orders, val_neg_files = load_orders(val_negative_orders,
                                                                    negative_image_paths, 0,
                                                                    exclusion=exclusion_set)
        val_labels = val_pos_labels + val_neg_labels
        val_orders = val_pos_orders + val_neg_orders
        val_files = val_pos_files + val_neg_files

        # now we want to make a dataset out of the images/labels
        train_dataset = SingleCellDataset(train_files, train_labels, data_transforms=train_transforms,
                                          metadata={
                                              'orders': train_orders,
                                          },
                                          extract_filenames=extract_filenames)
        val_dataset = SingleCellDataset(val_files, val_labels, data_transforms=val_transforms,
                                        metadata={
                                            'orders': val_orders,
                                        }, extract_filenames=extract_filenames)
    # TODO: should we pin memory?
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
