
from utils.DL.callbacks import BaseCallback

import torch
import torchvision
import torchmetrics
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import pandas


class BaseMetric(BaseCallback):
    def __init__(self, num_classes=2, device="gpu"):
        """
        :param
            --num_classes: number of classes
            --device: device "cpu" or "gpu"
        """
        super().__init__()
        if device == "gpu" and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = "cpu"

        self.num_classes = num_classes

        self.t_value = 0.0
        self.v_value = 0.0
        self.t_value_mean = 0.0
        self.v_value_mean = 0.0

    def on_train_batch_end(self, output=None, target=None, batch=None):
        if self.device == "cpu":
            output = output.to("cpu")
            target = target.to("cpu")
        batch_value = self.metric(output, target)
        self.t_value = self.metric.compute()  # value computed along every batch
        self.t_value_mean = self.t_value.mean()

    def on_val_start(self):
        self.metric.reset()

    def on_val_batch_end(self, output=None, target=None, batch=None):
        if self.device == "cpu":
            output = output.to("cpu")
            target = target.to("cpu")
        batch_value = self.metric(output, target)
        self.v_value = self.metric.compute()  # value computed along every batch
        self.v_value_mean = self.v_value.mean()

    def on_epoch_end(self, epoch=None):
        self.t_value = 0.0
        self.v_value = 0.0
        self.t_value_mean = 0.0
        self.v_value_mean = 0.0
        self.metric.reset()


class Accuracy(BaseMetric):
    """
    :param
        --num_classes: number of classes
        --device: device "cpu" or "gpu"
        --top_k: Number of highest probability or logit score predictions considered to find the correct label.
        --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
    """
    def __init__(self, num_classes=2, device="gpu", top_k=1, thresh=0.5):
        super().__init__(num_classes, device)

        self.metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes,
                                                           top_k=top_k, average=None).to(self.device)


class Precision(BaseMetric):
    """
    :param
        --num_classes: number of classes
        --device: device "cpu" or "gpu"
        --top_k: Number of highest probability or logit score predictions considered to find the correct label.
        --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
    """
    def __init__(self, num_classes=2, device="gpu", top_k=1, thresh=0.5):
        super().__init__(num_classes, device)

        self.metric = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes,
                                                            top_k=top_k, average=None).to(self.device)



class Recall(BaseMetric):
    """
    :param
        --num_classes: number of classes
        --device: device "cpu" or "gpu"
        --top_k: Number of highest probability or logit score predictions considered to find the correct label.
        --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
    """
    def __init__(self, num_classes=2, device="gpu", top_k=1, thresh=0.5):
        super().__init__(num_classes, device)

        self.metric = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes,
                                                             top_k=top_k, average=None).to(self.device)



class AUC(BaseMetric):
    """
    :param
        --num_classes: number of classes
        --device: device "cpu" or "gpu"
        --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
    """
    def __init__(self, num_classes=2, device="gpu", thresh=None):
        super().__init__(num_classes, device)

        self.metric = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes,
                                                        average=None, thresholds=thresh).to(self.device)



class ROC(BaseMetric):
    """
       :param
           --num_classes: number of classes
           --device: device "cpu" or "gpu"
           --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
       """

    def __init__(self, num_classes=2, device="cpu", thresh=None):
        super().__init__(num_classes, device)

        self.run = False

        self.metric = torchmetrics.classification.ROC(task="multiclass", num_classes=num_classes,
                                                      thresholds=thresh).to(self.device)

        self.fpr = None
        self.tpr = None
        self.thresh = None

    def on_train_batch_end(self, output=None, target=None, batch=None):
        pass

    def on_val_start(self):
        pass

    def on_val_batch_end(self, output=None, target=None, batch=None):
        if self.run:
            if self.device == "cpu":
                output = output.to("cpu")
                target = target.to("cpu")
            if not hasattr(self, "preds"):
                setattr(self, "preds", output)
                setattr(self, "labs", target)
            else:
                self.preds = torch.cat([self.preds, output], dim=0)
                self.labs = torch.cat([self.labs, target], dim=0)
        else:
            pass

    def on_val_end(self, metrics=None, epoch=None):
        if self.run:
            self.fpr, self.tpr, self.thresh = self.metric(self.preds, self.labs)
            self.metric.reset()
            self.preds = 0
            self.labs = 0
        else:
            pass

    def on_epoch_end(self, epoch=None):
        pass

    def on_epoch_start(self, epoch=None, max_epoch=None):
        if epoch == max_epoch-1:
            self.run = True
        else:
            pass


class PRC(BaseMetric):
    """
       :param
           --num_classes: number of classes
           --device: device "cpu" or "gpu"
           --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
       """

    def __init__(self, num_classes=2, device="cpu", thresh=None):
        super().__init__(num_classes, device)

        self.run = False

        self.metric = torchmetrics.classification.PrecisionRecallCurve(task="multiclass", num_classes=num_classes,
                                                                       thresholds=thresh).to(self.device)

        self.metric_best_P = torchmetrics.classification.PrecisionAtFixedRecall(task='multiclass', min_recall=0.5,
                                                                                num_classes=num_classes)
        self.metric_best_R = torchmetrics.classification.RecallAtFixedPrecision(task='multiclass', min_precision=0.5,
                                                                                num_classes=num_classes)


        self.P = None
        self.R = None
        self.thresh = None  # REMEMBER TO SWAP ORDER [::-1]
        self.best_P = None
        self.best_R = None
        self.best_P_th = None
        self.best_R_th = None


    def on_train_batch_end(self, output=None, target=None, batch=None):
        pass

    def on_val_start(self):
        pass

    def on_val_batch_end(self, output=None, target=None, batch=None):
        if self.run:
            if self.device == "cpu":
                output = output.to("cpu")
                target = target.to("cpu")
            if not hasattr(self, "preds"):
                setattr(self, "preds", output)
                setattr(self, "labs", target)
            else:
                self.preds = torch.cat([self.preds, output], dim=0)
                self.labs = torch.cat([self.labs, target], dim=0)
        else:
            pass

    def on_val_end(self, metrics=None, epoch=None):
        if self.run:
            self.P, self.R, self.thresh = self.metric(self.preds, self.labs)
            self.metric.reset()
            self.best_P, self.best_P_th = self.metric_best_P(self.preds, self.labs)
            self.best_R, self.best_R_th = self.metric_best_R(self.preds, self.labs)
            self.metric_best_P.reset()
            self.metric_best_R.reset()
            self.preds = 0
            self.labs = 0
        else:
            pass

    def on_epoch_end(self, epoch=None):
        pass

    def on_epoch_start(self, epoch=None, max_epoch=None):
        if epoch == max_epoch-1:
            self.run = True
        else:
            pass


class Metrics(BaseCallback):
    def __init__(self, num_classes=2, device="gpu", top_k=1, thresh=0.5):
        self.A = Accuracy(num_classes=num_classes, device=device, top_k=top_k, thresh=thresh)
        self.P = Precision(num_classes=num_classes, device=device, top_k=top_k, thresh=thresh)
        self.R = Recall(num_classes=num_classes, device=device, top_k=top_k, thresh=thresh)
        self.AuC = AUC(num_classes=num_classes, device=device, thresh=None)
        self.metrics = [self.A, self.P, self.R, self.AuC]
        self.build_metrics_dict()

    def on_train_batch_end(self, output=None, target=None, batch=None):
        for obj in self.metrics:
            obj.on_train_batch_end(output, target, batch)

    def on_train_end(self, metrics=None):
        self.dict["train_loss"] = [np.float16(metrics)]
        for obj in self.metrics:
            name = "train_" + obj.__class__.__name__
            self.dict[name] = [[x] for x in obj.t_value.to("cpu").numpy().astype(np.float16)]

    def on_val_start(self):
        for obj in self.metrics:
            obj.on_val_start()

    def on_val_end(self, metrics=None, epoch=None):
        self.dict["val_loss"] = [np.float16(metrics)]
        for obj in self.metrics:
            name = "val_" + obj.__class__.__name__
            self.dict[name] = [[x] for x in obj.v_value.to("cpu").numpy().astype(np.float16)]



    def on_val_batch_end(self, output=None, target=None, batch=None):
        for obj in self.metrics:
            obj.on_val_batch_end(output, target, batch)

    def on_epoch_end(self, epoch=None):
        for obj in self.metrics:
            obj.on_epoch_end(epoch)


    def build_metrics_dict(self):

        names = [obj.__class__.__name__ for obj in self.metrics]
        names.append("loss")
        keys = ["train_"+name for name in names]
        keys.extend(["val_"+name for name in names])

        setattr(self, "dict", {key: None for key in keys})





#
# def plot_roc_curve(fpr, tpr, thresholds):
#     plt.figure(figsize=(8, 8))
#     plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
#
#     # Annotate points for the thresholds used in the ROC calculation
#     for threshold in thresholds:
#         idx = (thresholds >= threshold).nonzero()[0].max()
#         plt.annotate(f'{threshold:.2f}', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(0, 10),
#                      ha='center', fontsize=8, color='red')
#
#     plt.xlabel('False Positive Rate (FPR)')
#     plt.ylabel('True Positive Rate (TPR)')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# def plot_pr_curve(precision, recall, thresholds):
#     plt.figure(figsize=(8, 8))
#     plt.plot(recall, precision, color='green', lw=2, label='PR curve')
#
#     # Annotate points for every threshold value
#     for threshold in thresholds:
#         idx = (thresholds >= threshold).nonzero()[0].max()
#         plt.annotate(f'Threshold = {threshold:.2f}', (recall[idx], precision[idx]),
#                      textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='red')
#
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
