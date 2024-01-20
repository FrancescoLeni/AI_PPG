import argparse
import torch
import torch.nn as nn

from models.DL import ModelClass, check_load_model
from models.DL.common import Dummy, ConvNeXt, ConvNeXtSAM, ResNet1, ResNet2, ResNetTransform, ResNetTransform2, \
                             ResNetTransformerAtt, TransformerEncDec, ResUnet, ResUnetAtt, DarkNetCSP, ResUnetAtt2, \
                             DarkNetCSPBoth, LearnableInitBiLSTM, LearnableInitBiLSTM2
from utils.dataloaders import Crops, CroppedSeq, Sequences
from utils.DL.callbacks import Callbacks, EarlyStopping, Saver
from utils.DL.loaders import CropsDataset, CroppedSeqDataset, WindowedSeq
from utils.DL.optimizers import get_optimizer, scheduler
from utils.DL.collates import padding_x, keep_unchanged
from utils.DL.metrics import Metrics
from utils.DL.logger import Loggers, LoggersBoth
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

    bi_head_f = False

    # saving inputs
    json_from_parser(args, save_path)

    # dataset
    if args.crops:  # crops dataset
        crops_data = Crops()
        crops_data.split(test_size=0.15)
        test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, normalization=args.data_norm,
                                sig_mode='single', bi_head=bi_head_f)
        val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, normalization=args.data_norm,
                               sig_mode='single', bi_head=bi_head_f)
        train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, normalization=args.data_norm,
                                 sig_mode='single', bi_head=bi_head_f)
        n_in = 1
    elif args.crops_raw:
        crops_data = Crops(parent='dataset/crops_raw')
        crops_data.split(test_size=0.15)
        test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, raw=True, normalization=args.data_norm,
                                sig_mode='single', bi_head=bi_head_f)
        val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, raw=True, normalization=args.data_norm,
                               sig_mode='single', bi_head=bi_head_f)
        train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, raw=True, normalization=args.data_norm,
                                 sig_mode='single', bi_head=bi_head_f)
        n_in = 1
    elif args.fixed_crops_raw:
        crops_data = Crops(parent='dataset/fixed_crops_raw')
        crops_data.split(test_size=0.15)
        test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, raw=True, normalization=args.data_norm,
                                sig_mode='single', bi_head=bi_head_f)
        val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, raw=True, normalization=args.data_norm,
                               sig_mode='single', bi_head=bi_head_f)
        train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, raw=True, normalization=args.data_norm,
                                 sig_mode='single', bi_head=bi_head_f)
        n_in = 1
    elif args.fixed_crops:
        crops_data = Crops(parent='dataset/fixed_crops')
        crops_data.split(test_size=0.15)
        test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, raw=False, normalization=args.data_norm,
                                sig_mode='single', bi_head=bi_head_f)
        val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, raw=False, normalization=args.data_norm,
                               sig_mode='single', bi_head=bi_head_f)
        train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, raw=False, normalization=args.data_norm,
                                 sig_mode='single', bi_head=bi_head_f)
        n_in = 1
    elif args.crops_der:
        crops_data = Crops()  # filtered
        crops_data.split(test_size=0.15, everything=True)
        test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, raw=True, normalization=args.data_norm,
                                sig_mode='derivatives', bi_head=bi_head_f)
        val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, raw=True, normalization=args.data_norm,
                               sig_mode='derivatives', bi_head=bi_head_f)
        train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, raw=True, normalization=args.data_norm,
                                 sig_mode='derivatives', bi_head=bi_head_f)
        n_in = 4
    elif args.crops_all:
        crops_data = Crops()  # filtered
        crops_data.split(test_size=0.15, everything=True)
        test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, raw=True, normalization=args.data_norm,
                                sig_mode='all', bi_head=bi_head_f)
        val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, raw=True, normalization=args.data_norm,
                               sig_mode='all', bi_head=bi_head_f)
        train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, raw=True, normalization=args.data_norm,
                                 sig_mode='all', bi_head=bi_head_f)
        n_in = 5
    elif args.cropped_seq:
        data = CroppedSeq(parent='dataset/crops/patients')  # loaded and split
        test_set = CroppedSeqDataset(data.test, mini_batch=256, mode=mode, normalization=args.data_norm)
        val_set = CroppedSeqDataset(data.val, mini_batch=256, mode=mode, normalization=args.data_norm)
        train_set = CroppedSeqDataset(data.train, mini_batch=256, mode=mode, normalization=args.data_norm)
        n_in = 1
    elif args.cropped_seq_raw:
        data = CroppedSeq()  # loaded and split
        test_set = CroppedSeqDataset(data.test, mini_batch=256, mode=mode, raw=True, normalization=args.data_norm)
        val_set = CroppedSeqDataset(data.val, mini_batch=256, mode=mode, raw=True, normalization=args.data_norm)
        train_set = CroppedSeqDataset(data.train, mini_batch=256, mode=mode, raw=True, normalization=args.data_norm)
        n_in = 1
    elif args.windowed_seq:
        data = Sequences()
        test_set = WindowedSeq(data.test, mode=mode, raw=False, normalization=args.data_norm)
        val_set = WindowedSeq(data.val, mode=mode, raw=False, normalization=args.data_norm)
        train_set = WindowedSeq(data.train, mode=mode, raw=False, normalization=args.data_norm)
        n_in = 1
    elif args.windowed_seq_raw:
        data = Sequences(raw=True)
        test_set = WindowedSeq(data.test, mode=mode, raw=True, normalization=args.data_norm)
        val_set = WindowedSeq(data.val, mode=mode, raw=True, normalization=args.data_norm)
        train_set = WindowedSeq(data.train, mode=mode, raw=True, normalization=args.data_norm)
        n_in = 1
    else:
        raise ValueError("data format not recognised")    # checking whether you parsed weights or model name
    if "." not in args.model:
        if args.model == "Dummy":
            model = Dummy(num_classes)
            # loading model = bla bla bla
        elif args.model == "ConvNeXt":
            model = ConvNeXt(num_classes)
        elif args.model == "ConvNeXtSAM":
            model = ConvNeXtSAM(num_classes)
        elif args.model == "ResNet1":
            model = ResNet1(num_classes, n_in)
        elif args.model == "ResNet2":
            model = ResNet2(num_classes)
        elif args.model == 'ResNetTransform':
            model = ResNetTransform(num_classes)
        elif args.model == 'ResNetTransform2':
            model = ResNetTransform2(num_classes)
        elif args.model == 'ResNetTransformerAtt':
            model = ResNetTransformerAtt(num_classes)
        elif args.model == 'TransformerEncDec':
            model = TransformerEncDec(num_classes)
        elif args.model == 'ResUnet':
            model = ResUnet(num_classes)
        elif args.model == 'ResUnetAtt':
            model = ResUnetAtt(num_classes)
        elif args.model == 'DarkNetCSP':
            model = DarkNetCSP(num_classes, n_in)
        elif args.model == "ResUnetAtt2":
            model = ResUnetAtt2(num_classes)
        elif args.model == 'DarkNetCSPBoth':
            model = DarkNetCSPBoth(in_dim=n_in, bi=False, tri=False, both=True)
        elif args.model == 'LearnableInitBiLSTM':
            model = LearnableInitBiLSTM(num_classes, n_in)
        elif args.model == 'LearnableInitBiLSTM2':
            model = LearnableInitBiLSTM2(num_classes, n_in)
        else:
            raise TypeError("Model name not recognised")
    else:
        model = args.model

    # double-checking whether you parsed weights or model and accounting for transfer learning
    mod = check_load_model(model, args.backbone)

    if args.freeze:
        freeze_list = ['stem', 'S', 'DownSample', 'sppf', 'classifier']
    else:
        freeze_list = None


    # initializing callbacks
    stopper = EarlyStopping(patience=args.patience, monitor="val_loss", mode="min")
    saver = Saver(model=mod, save_best=True, save_path=save_path, monitor="val_loss",
                  mode='min')
    callbacks = Callbacks([stopper, saver])

    # initializing metrics
    if not bi_head_f:
        metrics = Metrics(num_classes=num_classes, device=device, top_k=1, thresh=0.5)
    else:
        metrics = (Metrics(num_classes=2, device=device, top_k=1, thresh=0.5), Metrics(num_classes=3, device=device, top_k=1, thresh=0.5))

    if args.windowed_seq_raw or args.windowed_seq:
        seq_flag = True
    else:
        seq_flag = False

    if args.weighted_loss:
        if args.mode == 'binary':
            if args.cropped_seq or args.cropped_seq_raw:
                weights = torch.tensor([0.62963445, 2.42849968], dtype=torch.float32)  # only m9
                # weights = torch.tensor([0.54701633, 5.8173012], dtype=torch.float32)  # all split
            elif args.crops or args.crops_raw:
                # sill not computed (not userful actually as data are stratified and balanced)
                weights = None
            elif args.windowed_seq or args.windowed_seq_raw:
                # weights = torch.tensor([0.63646569, 2.33196228], dtype=torch.float32)  # w=1200, s=600
                # weights = torch.tensor([0.63420781, 2.36278286], dtype=torch.float32)  # w=1200, s=600 (removing borders)
                weights = torch.tensor([1., 2.], dtype=torch.float32)  # invented
                # weights = torch.tensor([0.64751537, 2.1947386], dtype=torch.float32)  # w=1000, s=500
                # weights = torch.tensor([0.64552263, 2.21794587], dtype=torch.float32)  # w=1000, s=500 (removing borders)
        else:
            # not yet implemented
            weights = None
    else:
        weights = None

    # initializing loss and optimizer
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="mean", label_smoothing=args.lab_smooth)
    opt = get_optimizer(mod, args.opt, args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)

    # initializing loggers
    if not bi_head_f:
        logger = Loggers(metrics=metrics, save_path=save_path, opt=opt, device=device)
    else:
        #  metrics1 -> binary; metrics2 -> all
        logger = LoggersBoth(metrics1=metrics[0], metrics2=metrics[1], save_path=save_path, opt=opt, device=device)

    # lr scheduler
    sched = scheduler(opt, args.sched, args.lrf, epochs)

    # building loaders
    if args.crops or args.crops_raw or args.crops_all or args.crops_der:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=padding_x)  # to pad
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=padding_x)  # to pad
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padding_x)  # to pad
    elif args.cropped_seq or args.cropped_seq_raw:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=keep_unchanged)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=keep_unchanged)  # shuffling has no effect
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=keep_unchanged)
    elif args.windowed_seq or args.windowed_seq_raw or args.fixed_crops or args.fixed_crops_raw:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)


    # building model
    model = ModelClass(mod, (train_loader, val_loader, test_loader), loss_fn=loss_fn, device=device, AMP=args.AMP,
                       optimizer=opt, metrics=metrics, loggers=logger, callbacks=callbacks, sched=sched,
                       freeze=freeze_list, sequences=seq_flag, bi_head=bi_head_f)

    # training loop
    model.train_loop(epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('--model', type=str, required=True, choices=["CNN", "ConvNeXt", "ConvNeXtSAM", "ResNet1", "ResNet2",
                                                                     "ResNetTransform", "ResNetTransform2", "ResNetTransformerAtt",
                                                                     "TransformerEncDec", "ResUnet", "ResUnetAtt", "DarkNetCSP",
                                                                     "ResUnetAtt2", "DarkNetCSPBoth", "LearnableInitBiLSTM",
                                                                     "Dummy", "LearnableInitBiLSTM2"],
                                                                      help='name of model to train or path to weights to train')
    parser.add_argument('--backbone', type=str, default=None, help='path to backbone weights for transformer architechtures')
    parser.add_argument('--freeze', action="store_true", help='whether to freeze backbone')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--mode', type=str, required=True, choices=["binary", "all", "both"], help="whether to use binary or full annotation")
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
    parser.add_argument('--weighted_loss', action="store_true", help='whether to weighten the loss and wheight for classes')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--crops', action="store_true", help='whether to use Crops dataset')
    group.add_argument('--crops_raw', action="store_true", help='whether to use Crops_raw dataset (extracted from raw signal)')
    group.add_argument('--crops_der', action="store_true", help='whether to use Crops + derivatives')
    group.add_argument('--crops_all', action="store_true", help='whether to use Crops + raw + derivatives')
    group.add_argument('--cropped_seq_raw', action="store_true", help='whether to use CroopedSeq dataset on raw crops')
    group.add_argument('--cropped_seq', action="store_true", help='whether to use CroopedSeq dataset')
    group.add_argument('--windowed_seq_raw', action="store_true", help="wheter to use raw windowed raw sequences")
    group.add_argument('--windowed_seq', action="store_true", help="wheter to use raw windowed sequences")
    group.add_argument('--fixed_crops', action="store_true", help='whether to use fixed Crops dataset')
    group.add_argument('--fixed_crops_raw', action="store_true", help='whether to use fixed Crops_raw dataset (extracted from raw signal)')

    args = parser.parse_args()

    main(args)


