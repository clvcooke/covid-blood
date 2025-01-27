from config import get_config
from torchvision import transforms
from utils import setup_torch, get_covid_transforms
import wandb
from dataloader import load_all_patients, load_pbc_data, load_control
from models.imagenet import get_model
from trainer import ClassificationTrainer
from torch import optim
import subprocess


def main(config):
    wandb.init("covid")
    """
    Train a classifier to detect covid positive vs. negative by fine tuning a pre-trained CNN
    :return:
    """
    setup_torch(config.random_seed, config.use_gpu, config.gpu_number)
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()

    wandb.config.update(config)
    wandb.config['git_hash'] = git_head_hash
    print("NUC SEG: ", config.nucseg)
    data_transforms = get_covid_transforms(image_size=224,
                                           center_crop_amount=224,
                                           center_mask=config.center_mask,
                                           resize=config.resize,
                                           zoom=config.zoom,
                                           outer_mask=config.outer_mask,
                                           nucseg=config.nucseg,
                                           shear=config.shear,
                                           saturation=config.saturation,
                                           hue=config.hue,
                                           speckle=config.speckle)
    cell_mask = config.cell_mask
    include_control = config.control_weight is not None
    control_weight = config.control_weight
    if config.task == 'covid-class':
        train_loader, val_loader, test_loader = load_all_patients(
            train_transforms=data_transforms['train'],
            test_transforms=data_transforms['val'],
            batch_size=config.batch_size,
            fold_number=config.fold_number,
            exclusion=config.exclusion,
            cell_mask=cell_mask,
            extract_filenames=True,
            include_control=include_control,
            control_weighting=control_weight,
            weighted_sample=True)
        num_classes = 2
        negative_control_loader = load_control(data_transforms['val'], extract_filenames=True, )

    elif config.task == 'wbc-class':
        train_loader, val_loader = load_pbc_data(train_transforms=data_transforms['train'],
                                                 val_transforms=data_transforms['val'],
                                                 batch_size=config.batch_size)
        num_classes = 9
        test_loader = None
        negative_control_loader = None
    else:
        raise RuntimeError("Task not supported")
    model = get_model(model_name=config.model_name, num_outputs=num_classes, use_pretrained=config.pretrained_model)
    if config.use_gpu:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=config.init_lr, momentum=0.9)
    if config.lr_schedule == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, mode='max', factor=0.316,
                                                         verbose=True)
    elif config.lr_schedule == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=1, mode='triangular',
                                                step_size_up=4000)
    else:
        scheduler = None
    trainer = ClassificationTrainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                                    test_loader=test_loader, test_interval=config.test_interval,
                                    batch_size=config.batch_size, epochs=config.epochs,
                                    negative_control=negative_control_loader, lq_loss=config.lq_loss,
                                    scheduler=scheduler, schedule_type=config.lr_schedule)
    trainer.train()


if __name__ == "__main__":
    arg_config, unparsed = get_config()
    main(arg_config)
