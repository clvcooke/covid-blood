from config import get_config
from torchvision import transforms
from utils import setup_torch, get_covid_transforms, load_model
import wandb
from dataloader import load_all_patients, load_pbc_data, load_control
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
    run = wandb.init("covid")
    # enforce batch_size of 1
    if config.batch_size != 1:
        warnings.warn("Batch size must be one for multi-instance learning, changing batch_size to 1")
        config.batch_size = 1
    print(config.random_seed)
    setup_torch(config.random_seed, config.use_gpu, config.gpu_number)
    wandb.config.update(config)
    data_transforms = get_covid_transforms(image_size=224, center_crop_amount=224, zoom=config.zoom)
    if config.task != 'covid-class':
        raise RuntimeError("Task not supported")
    train_loader, val_loader, test_loader = load_all_patients(train_transforms=data_transforms['train'],
                                                              test_transforms=data_transforms['val'],
                                                              batch_size=1,
                                                              fold_number=config.fold_number,
                                                              exclusion=config.exclusion,
                                                              group_by_patient=True,
                                                              weighted_sample=True,
                                                              mil_size=config.mil_size,
                                                              include_control=True)
    model = SimpleMIL(
        backbone_name=config.model_name,
        num_classes=2,
        pretrained_backbone=False,
        instance_hidden_size=1024,
        hidden_size=1024
    )

    if config.use_gpu:
        model.cuda()
   

    if config.lr_schedule == 'plateau':
        #optimizer = optim.SGD(model.parameters(), lr=config.init_lr, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=config.init_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, mode='max', factor=0.316,
                                                         verbose=True)
    elif config.lr_schedule == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=config.init_lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0000001, max_lr=0.001, mode='triangular',
                                                step_size_up=2000)
        # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=3e-1, mode='triangular',
        #                                        step_size_up=2000)
    else:
        scheduler = None
        if config.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=config.init_lr)
        elif config.optimizer.lower() == 'madgrad':
            from madgrad import MADGRAD
            optimizer = MADGRAD(model.parameters(), lr=config.init_lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.init_lr)
 
    trainer = ClassificationTrainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
             test_loader=test_loader, test_interval=config.test_interval,
                                    batch_size=config.batch_size, epochs=config.epochs, patience=15, scheduler=scheduler, schedule_type=config.lr_schedule)
    trainer.train()
    run.finish()


if __name__ == "__main__":
    conf, unparsed = get_config()

    main(conf)
