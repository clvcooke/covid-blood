from config import get_config
from torchvision import transforms
from utils import setup_torch
import wandb
from dataloader import load_all_patients, load_pbc_data
from models.imagenet import get_model
from trainer import ClassificationTrainer
from torch import optim

wandb.init("covid")


def main():
    """
    Train a classifier to detect covid positive vs. negative by fine tuning a pre-trained CNN
    :return:
    """
    config, unparsed = get_config()
    setup_torch(config.random_seed, config.use_gpu, config.gpu_number)
    wandb.config.update(config)
    image_size = 224
    # TODO: generate this in a function?
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
    if config.task == 'covid-class':
        train_loader, val_loader = load_all_patients(train_transforms=data_transforms['train'],
                                                     val_transforms=data_transforms['val'],
                                                     batch_size=config.batch_size,
                                                     fold_number=config.fold_number)
        num_classes = 2
    elif config.task == 'wbc-class':
        train_loader, val_loader = load_pbc_data(train_transforms=data_transforms['train'],
                                                 val_transforms=data_transforms['val'],
                                                 batch_size=config.batch_size)
        num_classes = 9
    else:
        raise RuntimeError("Task not supported")
    model = get_model(model_name=config.model_name, num_classes=num_classes, use_pretrained=config.pretrained_model)
    if config.use_gpu:
        model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ClassificationTrainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                                    batch_size=config.batch_size, epochs=config.epochs, patience=7)
    trainer.train()


if __name__ == "__main__":
    main()
