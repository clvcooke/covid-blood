from config import get_config
from torchvision import transforms
from utils import setup_torch, get_covid_transforms, load_model
import wandb
from dataloader import load_all_patients, load_pbc_data
from models.imagenet import get_model
from models.multi_instance import AttentionModel, GatedAttentionModel, SimpleMIL
from mil_trainer import ClassificationTrainer
from torch import optim
import warnings




def main(config):
    """
    Train a multi-instance classifier to detect covid positive vs. negative
    :return:
    """
    wandb.init("covid")
    # enforce batch_size of 1
    if config.batch_size != 1:
        warnings.warn("Batch size must be one for multi-instance learning, changing batch_size to 1")
        config.batch_size = 1
    setup_torch(config.random_seed, config.use_gpu, config.gpu_number)
    wandb.config.update(config)
    data_transforms = get_covid_transforms(image_size=224, center_crop_amount=224)
    if config.task != 'covid-class':
        raise RuntimeError("Task not supported")
    train_loader, val_loader, test_loader = load_all_patients(train_transforms=data_transforms['train'],
                                                              test_transforms=data_transforms['val'],
                                                              batch_size=1,
                                                              fold_number=config.fold_number,
                                                              exclusion=config.exclusion,
                                                              group_by_patient=True,
                                                              weighted_sample=True)
    model = SimpleMIL(
        backbone_name=config.model_name,
        num_classes=2,
        pretrained_backbone=False,
        instance_hidden_size=1024,
        hidden_size=1024
    )

    if config.use_gpu:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.init_lr)
    trainer = ClassificationTrainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                                    test_loader=test_loader, test_interval=config.test_interval,
                                    batch_size=config.batch_size, epochs=config.epochs, patience=15, scheduler=None)
    trainer.train()


if __name__ == "__main__":
    conf, unparsed = get_config()

    main(conf)
