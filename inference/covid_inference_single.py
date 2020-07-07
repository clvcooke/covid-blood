import json
import os
# external lib
import torch
from tqdm import tqdm
# internal lib
from dataloader import load_all_patients
from models.imagenet import get_model
from utils import setup_torch, load_model, get_user, get_covid_transforms


def predict(arch_name, model_file, output_file, fold_number):
    assert fold_number is not None and 0 <= fold_number < 6, "bad fold"
    # since we can only trust a single fold this is pretty straightforward
    transforms = get_covid_transforms()
    batch_size = 8
    train_loader, val_loader, test_loader = load_all_patients(train_transforms=transforms['train'],
                                                              test_transforms=transforms['train'],
                                                              batch_size=batch_size, extract_filenames=True,
                                                              fold_number=fold_number)
    num_classes = 2
    model = get_model(model_name=arch_name, num_outputs=num_classes)
    load_model(model, model_file)
    model.cuda()
    # now predict for the test set
    inference_results = {}
    loaders, loader_names = [train_loader, val_loader, test_loader], ['train_', 'val_', '']
    with torch.no_grad():
        for loader, loader_name in zip(loaders, loader_names):
            for (images, filenames), labels in tqdm(loader):
                images = images.cuda()
                results = model(images)
                preds = torch.nn.functional.softmax(results, dim=-1)[:, 1]
                preds = preds.tolist()
                for filename, pred, label in zip(filenames, preds, labels):
                    order = os.path.basename(filename).split('_')[0]
                    try:
                        int(order)
                    except:
                        print("bad order:", order)
                        order = os.path.basename(os.path.dirname(filename))
                        int(order)
                        print("new order", order)
                    if order not in inference_results:
                        inference_results[order] = {}
                        inference_results[order][loader_name + 'predictions'] = []
                        inference_results[order][loader_name + 'label'] = int(label)
                        inference_results[order][loader_name + 'files'] = []
                    inference_results[order][loader_name + 'predictions'].append(pred)
                    inference_results[order][loader_name + 'files'].append(os.path.basename(filename))
    with open(output_file, 'w') as fp:
        json.dump(inference_results, fp)
    print("Success")


def run_single(model_id, fold, gpu_number):
    username = get_user()
    setup_torch(0, 1, gpu_number)
    model_path = f"/hddraid5/data/{username}/models/{model_id}.pth"
    if not os.path.exists(model_path):
        model_path = f"/home/{username}/models/{model_id}.pth"
    arch = "densenet"
    output_path = f"/home/{username}/results_cov/covid_class_v4_{model_id}_fold_{fold}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"Model ID {model_id} has already been run")
        return
    assert os.path.splitext(output_path)[1] == '.json'
    predict(arch, model_path, output_path, fold)
