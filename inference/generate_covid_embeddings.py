import json
import os
# external lib
import torch
from tqdm import tqdm
# internal lib
from dataloader import load_all_patients
from models.imagenet import get_model
from utils import setup_torch, load_model, get_user, get_covid_transforms
import torch.nn.functional as F


def gen_embeddings(arch_name, model_file, output_file, fold_number):
    transforms = get_covid_transforms()
    batch_size = 8
    test_transform = transforms['val']
    train_loader, val_loader, test_loader = load_all_patients(train_transforms=test_transform,
                                                              test_transforms=test_transform,
                                                              batch_size=batch_size, extract_filenames=True,
                                                              fold_number=fold_number)
    num_classes = 2
    model = get_model(model_name=arch_name, num_outputs=num_classes)
    load_model(model, model_file)
    model.cuda()
    inference_results = {}

    loaders, loader_names = [test_loader, val_loader, train_loader], ['test_', 'val_', 'train_']
    with torch.no_grad():
        for loader, loader_name in zip(loaders, loader_names):
            for (images, filenames), labels in tqdm(loader):
                images = images.cuda()
                features = model.features(images)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                results = torch.flatten(out, 1)
                print(results.shape)
                preds = results.tolist()
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
                        inference_results[order][loader_name + 'features'] = []
                        inference_results[order][loader_name + 'label'] = int(label)
                        inference_results[order][loader_name + 'files'] = []
                    inference_results[order][loader_name + 'features'].append(pred)
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
    output_path = f"/home/{username}/features_cov/covid_feat_{model_id}_fold_{fold}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"Model ID {model_id} has already been run")
        return
    assert os.path.splitext(output_path)[1] == '.json'
    gen_embeddings(arch, model_path, output_path, fold)


if __name__ == "__main__":
    run_single(model_id='3bcnioqn', fold=0, gpu_number=0)