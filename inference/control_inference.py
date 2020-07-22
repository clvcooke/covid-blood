import torch
import os
import json
from tqdm import tqdm

from utils import setup_torch, load_model, get_user, get_covid_transforms
from dataloader import load_control
from models.imagenet import get_model


if __name__ == "__main__":
    # hardcoding for now
    model_ids = ['3bcnioqn',
                 '3dzierwk',
                 '1629vjqj',
                 '29tpkvfa',
                 '29psm43i',
                 '3cqyqz7j']
    TTA_rounds = 0
    gpu_number = 0
    setup_torch(0, 1, gpu_number)
    batch_size = 8
    transforms = get_covid_transforms()
    control_loader = load_control(transforms['train'], batch_size=batch_size, extract_filenames=True)

    for m, model_id in enumerate(model_ids):
        for TTA_round in range(TTA_rounds):
            print(f"{m+1}/{len(model_ids)}")
            model_file = f"/hddraid5/data/{get_user()}/models/{model_id}.pth"
            output_path = f"/home/{get_user()}/results_cov/covid_class_v6_{model_id}_control_TTA_{TTA_round}.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            arch_name = 'densenet'
            # load model
            num_classes = 2
            model = get_model(arch_name, num_classes)
            load_model(model, model_file)
            model.cuda()
            model.eval()
            inference_results = {}
            with torch.no_grad():
                for (images, filenames), labels in tqdm(control_loader):
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
                            inference_results[order]['control_predictions'] = []
                            inference_results[order]['control_label'] = int(label)
                            inference_results[order]['control_files'] = []
                        inference_results[order]['control_predictions'].append(pred)
                        inference_results[order]['control_files'].append(os.path.basename(filename))
            with open(output_path, 'w') as fp:
                json.dump(inference_results, fp)
            print("Success")
