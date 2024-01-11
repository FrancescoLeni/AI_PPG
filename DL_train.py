import argparse
import torch
import torch.nn as nn

from models.DL import ModelClass, check_load_model
from models.DL.common import Dummy, ConvNeXt, ConvNeXtSAM, ResNet1
from utils.dataloaders import Crops
from utils.DL.callbacks import Callbacks, EarlyStopping, Saver
from utils.DL.loaders import CropsDataset
from utils.DL.optimizers import get_optimizer, scheduler
from utils.DL.collates import padding_x
from utils.DL.metrics import Metrics
from utils.DL.logger import Loggers
from utils import random_state, increment_path, json_from_parser

# setting all random states
random_state(36)


def main(args):

    # unpacking
    folder = args.folder
    name = args.name
    save_path = increment_path(folder, name)
    epochs = args.epochs
    batch_size = args.batch_size
    device = args.device
    mode = args.mode
    if mode == "binary":
        num_classes = 2
    else:
        num_classes = 3

    # saving inputs
    json_from_parser(args, save_path)

    # checking whether you parsed weights or model name
    if "." not in args.model:
        if args.model == "CNN":
            model = Dummy()
            # loading model = bla bla bla
        elif args.model == "ConvNeXt":
            model = ConvNeXt(num_classes)
        elif args.model == "ConvNeXtSAM":
            model = ConvNeXtSAM(num_classes)
        elif args.model == "ResNet1":
            model = ResNet1(num_classes)
        else:
            raise TypeError("Model name not recognised")
    else:
        model = args.model

    # double-checking whether you parsed weights or model
    mod = check_load_model(model)

    # initializing callbacks
    stopper = EarlyStopping(patience=args.patience, monitor="val_loss", mode="min")
    saver = Saver(model=mod, save_best=True, save_path=save_path, monitor="val_loss",
                  mode='min')
    callbacks = Callbacks([stopper, saver])

    # initializing metrics
    metrics = Metrics(num_classes=num_classes, device=device, top_k=1, thresh=0.5)

    # dataset
    if args.crops:  # crops dataset
        crops_data = Crops()
        crops_data.split(test_size=0.15)
        test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, normalization=args.data_norm)
        val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, normalization=args.data_norm)
        train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, normalization=args.data_norm)
    elif args.sequences:  # sequences dataset
        ...
    elif args.crops_raw:
        crops_data = Crops(parent='dataset/crops_raw')
        crops_data.split(test_size=0.15)
        test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, raw=True, normalization=args.data_norm)
        val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, raw=True, normalization=args.data_norm)
        train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, raw=True, normalization=args.data_norm)
    else:
        raise ValueError("data format not recognised")

    # initializing loss and optimizer (to be modified)
    loss_fn = nn.CrossEntropyLoss(weight=None, reduction="mean", label_smoothing=args.lab_smooth)
    opt = get_optimizer(mod, args.opt, args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)

    # initializing loggers
    logger = Loggers(metrics=metrics, save_path=save_path, opt=opt, device=device)

    # lr scheduler
    sched = scheduler(opt, args.sched, args.lrf, epochs)

    # building loaders
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=padding_x)  # to pad
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=padding_x)  # to pad
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padding_x)  # to pad

    # building model
    model = ModelClass(mod, (train_loader, val_loader, test_loader), loss_fn=loss_fn, device=device, AMP=args.AMP,
                       optimizer=opt, metrics=metrics, loggers=logger, callbacks=callbacks, sched=sched)

    # training loop
    model.train_loop(epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('--model', type=str, required=True, choices=["CNN", "ConvNeXt", "ConvNeXtSAM", "ResNet1"], help='name of model or path to model weights')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--mode', type=str, required=True, choices=["binary", "all"], help="whether to use binary or full annotation")
    parser.add_argument('--data_norm', type=str, default=None, choices=["min_max", "RobustScaler", "Z-score"], help="type of scaler for data")
    parser.add_argument('--folder', type=str, default="runs", help='name of folder to which saving results')
    parser.add_argument('--name', type=str, default="exp", help='name of experiment folder inside folder')
    parser.add_argument('--opt', type=str, default="AdamW", choices=["SGD", "Adam", "AdamW"], help='name of optimizer to use')
    parser.add_argument('--sched', type=str, default=None, choices=["linear", "cos_lr"], help="name of the lr scheduler")
    parser.add_argument('--lr0', type=float, default=0.0004, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.001, help='final learning rate (multiplicative factor)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value (SGD) beta1 (Adam, AdamW)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay value')
    parser.add_argument('--lab_smooth', type=float, default=0, help='label smoothing value')
    parser.add_argument('--patience', type=int, default=30, help='number of epoch to wait for early stopping')
    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"], help='device to which loading the model')
    parser.add_argument('--AMP', action="store_true", help='whether to use AMP')
    # not yet implemented lul
    parser.add_argument('--weighten_loss', type=tuple, default=None, help='whether to weighten the loss and wheight for classes NOTE that len==num_class')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--crops', action="store_true", help='whether to use Crops dataset')
    group.add_argument('--crops_raw', action="store_true", help='whether to use Crops_raw dataset (extracted from raw signal)')
    group.add_argument('--sequences', action="store_true", help='whether to use Sequences dataset')

    args = parser.parse_args()

    main(args)


