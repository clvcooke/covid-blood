import json
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn import model_selection
from pandas import read_excel
import os
import glob
from PIL import Image
from tqdm import tqdm
import cv2 as cv

# used to filter out large images (not of single cells)
IMAGE_SIZE_CUTOFF_UPPER = 800000
# size in bytes
IMAGE_SIZE_CUTOFF_LOWER = 100

CONTROL_FRACTION_IDS = [
    "10050819999",
    "10050936144",
    "10051030838",
    "10051031452",
    "10051045136",
    "10051055184",
    "10051065979",
    "10051182007",
    "10051195311",
    "10051195937"]


def filter_wbc(path, lower_bound=(0, 100, 0), upper_bound=(204, 255, 127), invert=False):
    image = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    image_g = image.copy()
    image_s = cv.cvtColor(image_g, cv.COLOR_RGB2LAB)
    mask = cv.inRange(image_s, lower_bound, upper_bound)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=np.ones((9, 9), np.uint8), iterations=1)
    mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    if invert:
        mask_rgb = (mask_rgb == 0).astype(mask_rgb.dtype)
    image_g = image_g & mask_rgb
    return Image.fromarray(image_g)


class ImageCache:
    def __init__(self, cache_amnt=-1, cell_mask=None):
        self.cached = {}
        self.cell_mask = cell_mask

    def get(self, path):
        if path not in self.cached:
            if self.cell_mask is not None:
                im = filter_wbc(path, invert=self.cell_mask == 'nuc')
            else:
                im = cv.imread(path)
                # im = Image.open(path).copy()
            self.cached[path] = im
        else:
            return self.cached[path]
        return self.cached[path]


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, bags, data_transforms=None, metadata=None, cache_images=True, cell_mask=None):
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
            self.image_cache = ImageCache(cell_mask=cell_mask)

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
                 cache_images=True, cell_mask=None):
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
            self.image_cache = ImageCache(cell_mask=cell_mask)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_path = self.image_paths[index]
        if self.cache_images:
            image = self.image_cache.get(image_path)
        else:
            image = cv.imread(image_path)
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
    val_folder = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    train_split, val_split = list(val_folder.split(train_x, train_y))[0]
    val_x, val_y = train_x[val_split], train_y[val_split]
    train_x, train_y = train_x[train_split], train_y[train_split]
    test_x, test_y = x[test_split], y[test_split]
    return train_x, train_y, val_x, val_y, test_x, test_y


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


def get_control_sample():
    base_path = '/hddraid5/data/colin/covid-data/july_22'
    if not os.path.exists(base_path):
        base_path = '/home/col/covid-data/july_22'
    control_ids = CONTROL_FRACTION_IDS
    all_image_paths = glob.glob(os.path.join(base_path, 'COVID Research Images', '**', '*.jpg'), recursive=True)
    control_images = {}
    for order in tqdm(control_ids):
        try:
            np.int(order)
            label = False
        except:
            continue
        order_paths = [ip for ip in all_image_paths if str(order) in ip]
        image_paths = [image_path for image_path in order_paths if (
                    os.path.getsize(image_path) < IMAGE_SIZE_CUTOFF_UPPER and os.path.getsize(
                image_path) > IMAGE_SIZE_CUTOFF_LOWER)]
        if len(image_paths) == 0:
            raise RuntimeError()
        control_images[str(order)] = image_paths
    total_imgs = sum([len(value) for value in control_images.values()])
    print(f"Retrieved {len(control_images)} control patients, totaling {total_imgs} images")
    return control_images


def get_patient_orders(exclude_orders=None):
    base_path = '/hddraid5/data/colin/covid-data/july_22'
    if not os.path.exists(base_path):
        base_path = '/home/col/covid-data/july_22'
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
    all_image_paths = glob.glob(os.path.join(base_path, 'COVID Research Images', '**', '*.jpg'),
                                recursive=True)
    all_image_paths = [image_path for image_path in all_image_paths if
                       (os.path.getsize(image_path) < IMAGE_SIZE_CUTOFF_UPPER and os.path.getsize(
                           image_path) > IMAGE_SIZE_CUTOFF_LOWER)]
    for path in all_image_paths:
        if 'Not WBC' in path:
            raise RuntimeError()
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
        order_paths = [ip for ip in all_image_paths if str(order) in ip]
        image_paths = [image_path for image_path in order_paths if
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
    neg_pat_count = len(negative_images)
    pos_pat_count = len(positive_images)
    neg_cell_count = sum([len(values) for values in negative_images.values()])
    pos_cell_count = sum([len(values) for values in positive_images.values()])

    print("Data Stats:")
    print(
        f"          - {neg_pat_count} negative patients, {pos_pat_count} positive_patients -- {pos_pat_count / (pos_pat_count + neg_pat_count)} positive pat. fraction")
    print(
        f"          - {neg_cell_count} negative cells, {pos_cell_count} positive_cells -- {pos_cell_count / (pos_cell_count + neg_cell_count)} positive cell fraction")

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
                      exclusion=None,
                      cell_mask=None, weighted_sample=True,
                      control_weighting=1.0,
                      include_control=False):
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
    train_orders, train_labels, val_orders, val_labels, test_orders, test_labels = get_strat_fold(orders, labels,
                                                                                                  fold_index=fold_number,
                                                                                                  fold_seed=fold_seed,
                                                                                                  fold_count=fold_count)

    if exclusion is not None:
        with open(exclusion) as fp:
            # set to go fast
            exclusion_set = set(json.load(fp))
    else:
        # empty set as default
        exclusion_set = set()
    if group_by_patient:
        train_bags = load_orders_into_bags(train_orders, all_image_paths, train_labels, exclusion=exclusion_set)
        val_bags = load_orders_into_bags(val_orders, all_image_paths, val_labels, exclusion=exclusion_set)
        test_bags = load_orders_into_bags(test_orders, all_image_paths, test_labels, exclusion=exclusion_set)
        training_dataset = BagDataset(train_bags, data_transforms=train_transforms)
        val_dataset = BagDataset(val_bags, data_transforms=test_transforms)
        test_dataset = BagDataset(test_bags, data_transforms=test_transforms)
    else:
        train_labels, train_orders, train_files = load_orders(train_orders, all_image_paths, train_labels,
                                                              exclusion=exclusion_set)
        if include_control:
            control_data = get_control_sample()
            all_image_paths.update(control_data)
            control_orders = list(control_data.keys())
            control_labels = [0] * len(control_orders)
            control_labels, control_orders, control_files = load_orders(control_orders, all_image_paths, control_labels,
                                                                        exclusion=exclusion_set)
            train_labels = train_labels + control_labels
            train_orders = train_orders + control_orders
            train_files = train_files + control_files
        val_labels, val_orders, val_files = load_orders(val_orders, all_image_paths, val_labels,
                                                        exclusion=exclusion_set)
        test_labels, test_orders, test_files = load_orders(test_orders, all_image_paths, test_labels,
                                                           exclusion=exclusion_set)
        # now we want to make a dataset out of the images/labels
        training_dataset = SingleCellDataset(train_files, train_labels, data_transforms=train_transforms,
                                             metadata={
                                                 'orders': train_orders,
                                             },
                                             extract_filenames=extract_filenames,
                                             cell_mask=cell_mask)
        val_dataset = SingleCellDataset(val_files, val_labels, data_transforms=test_transforms,
                                        metadata={
                                            'orders': val_orders
                                        }, extract_filenames=extract_filenames,
                                        cell_mask=cell_mask)
        test_dataset = SingleCellDataset(test_files, test_labels, data_transforms=test_transforms,
                                         metadata={
                                             'orders': test_orders,
                                         }, extract_filenames=extract_filenames,
                                         cell_mask=cell_mask)
    # swapping data transforms
    if weighted_sample:
        target = train_labels
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        if include_control:
            control_weights = np.ones_like(samples_weight)
            control_weights[-len(control_labels):] *= control_weighting
            samples_weight = control_weights*samples_weight
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(training_dataset, batch_size=batch_size, pin_memory=True, sampler=sampler)
    else:
        train_loader = DataLoader(training_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader, val_loader, test_loader


def load_control(transforms, batch_size=8, extract_filenames=False):
    if os.path.exists(('/hddraid5')):
        control_dir = "/hddraid5/data/colin/covid-data/control_data/negative/COVID Imaging Negative Controls"
    else:
        control_dir = '/home/col/covid-data/new_data/COVID Research Images/COVID Imaging Negative Controls'
    negative_control_orders = os.listdir(control_dir)
    negative_control_orders = [no for no in negative_control_orders if no.isdigit()]
    negative_control_orders = [no for no in negative_control_orders if no not in CONTROL_FRACTION_IDS]
    positive_image_paths = {}
    negative_image_paths = {}
    for order in negative_control_orders:
        order_images = glob.glob(f"{control_dir}/{order}/*.jpg")
        order_images = [image_path for image_path in order_images if
                        (os.path.getsize(image_path) < IMAGE_SIZE_CUTOFF_UPPER and os.path.getsize(
                            image_path) > IMAGE_SIZE_CUTOFF_LOWER)]
        negative_image_paths[order] = order_images
    all_image_paths = dict(negative_image_paths, **positive_image_paths)
    negative_orders = list(negative_image_paths.keys())
    orders = negative_orders
    labels = [0] * len(negative_orders)
    control_labels, control_orders, control_files = load_orders(orders, all_image_paths, labels)
    control_dataset = SingleCellDataset(control_files, control_labels, data_transforms=transforms,
                                        metadata={
                                            'orders': control_orders,
                                        }, extract_filenames=extract_filenames)
    control_loader = DataLoader(control_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return control_loader
