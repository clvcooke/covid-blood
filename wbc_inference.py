import json
import os
import pwd
from torchvision import transforms
from utils import setup_torch, load_model
from dataloader import load_all_patients
from models.imagenet import get_model
import torch
from tqdm import tqdm


def infer_loader(model, loader):
    # how to get filenames
    inference_results = {}
    for (images, filenames), _ in tqdm(loader):
        # assume cuda
        images = images.cuda()
        results = model(images)
        _, preds = torch.max(results, 1)
        preds = preds.tolist()
        for filename, pred in zip(filenames, preds):
            inference_results[filename] = pred
    return inference_results


def infer_loader_cont(model, loader):
    # how to get filenames
    inference_results = {}
    for (images, filenames), _ in tqdm(loader):
        # assume cuda
        images = images.cuda()
        results = model(images)
        preds = results
        preds = preds.tolist()
        for filename, pred in zip(filenames, preds):
            inference_results[filename] = pred
    return inference_results


def predict(model_name, model_path, output_file, continous=False):
    """
    Process predictions for wbc classification
    :return:
    """
    if os.path.exists(output_file):
        raise RuntimeError("output file already exists, please choose another: " + output_file)
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    setup_torch(0, 1, 3)
    image_size = 224
    batch_size = 8
    # first
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader, val_loader, test_loader = load_all_patients(train_transforms=transform,
                                                              test_transforms=transform,
                                                              batch_size=batch_size,
                                                              extract_filenames=True)
    num_classes = 9
    model = get_model(model_name=model_name, num_outputs=num_classes)
    load_model(model, model_path)
    model.cuda()
    # now run prediction step
    if continous:
        infer = infer_loader_cont
    else:
        infer = infer_loader
    train_results = infer(model, train_loader)
    val_results = infer(model, val_loader)
    test_results = infer(model, test_loader)
    results = train_results
    results.update(val_results)
    results.update(test_results)
    with open(output_file, 'w') as fp:
        json.dump(results, fp)
    print("Success")


if __name__ == "__main__":
    username = pwd.getpwuid(os.getuid()).pw_name
    model_id = '1dr34rc5'
    model_path = f"/hddraid5/data/{username}/models/{model_id}.pth"
    model_name = 'densenet'
    continous = True
    if continous:
        output_file = f'/home/colin/testing/wbc_class_{model_id}_v2_cont.json'
    else:
        output_file = f'/home/colin/testing/wbc_class_{model_id}_v2.json'
    assert os.path.splitext(output_file)[1] == '.json'
    predict(model_name=model_name, model_path=model_path, output_file=output_file, continous=continous)
