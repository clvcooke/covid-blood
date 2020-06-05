from config import get_config
from torchvision import transforms
from utils import setup_torch, get_covid_transforms
import wandb
from dataloader import load_all_patients, load_pbc_data
from models.imagenet import get_model
from models.multi_instance import AttentionModel, GatedAttentionModel
from trainer import ClassificationTrainer
from torch import optim
import warnings

wandb.init("covid")


def main():
    """
    Train a multi-instance classifier to detect covid positive vs. negative
    :return:
    """
    config, unparsed = get_config()
    # enforce batch_size of 1
    if config.batch_size != 1:
        warnings.warn("Batch size must be one for multi-instance learning, changing batch_size to 1")
        config.batch_size = 1
    setup_torch(config.random_seed, config.use_gpu, config.gpu_number)
    wandb.config.update(config)
    data_transforms = get_covid_transforms(image_size=224)
    if config.task != 'covid-class':
        raise RuntimeError("Task not supported")
    train_loader, val_loader, test_loader = load_all_patients(train_transforms=data_transforms['train'],
                                                 test_transforms=data_transforms['val'],
                                                 batch_size=config.batch_size,
                                                 fold_number=config.fold_number,
                                                 exclusion=config.exclusion,
                                                 group_by_patient=True)
    model = GatedAttentionModel(
        backbone_name=config.model_name,
        num_classes=2
    )
    if config.use_gpu:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    trainer = ClassificationTrainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                                    batch_size=config.batch_size, epochs=config.epochs, patience=7)
    trainer.train()


if __name__ == "__main__":
    main()
