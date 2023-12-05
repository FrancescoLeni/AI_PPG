import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import pandas


# ----------------------------------------------------------------------------------------------------------------------
# BASE CALLBACK CLASS
# ----------------------------------------------------------------------------------------------------------------------

class BaseCallback:

    def on_train_start(self):
        pass

    def on_train_end(self, metrics = None):
        pass

    def on_val_start(self):
        pass

    def on_val_end(self, metrics=None, epoch=None):
        pass

    def on_train_batch_start(self, batch=None):
        pass

    def on_train_batch_end(self, output=None, target=None, batch=None):
        pass

    def on_val_batch_start(self, batch=None):
        pass

    def on_val_batch_end(self, output=None, target=None, batch=None):
        pass

    def on_epoch_start(self, epoch=None, max_epoch=None):
        pass

    def on_epoch_end(self, epoch=None):
        pass

# ----------------------------------------------------------------------------------------------------------------------
# CALLBACK SUBCLASSES
# ----------------------------------------------------------------------------------------------------------------------


class EarlyStopping(BaseCallback):

    def __init__(self, patience=30, monitor="vloss", mode='min'):
        super().__init__()
        self.mode = mode
        self.monitor = monitor
        self.best_fitness = 0.0  # validation
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch
        self.stop = False

    def on_val_end(self, metrics=None, epoch=None):
        fitness = metrics[self.monitor]
        if self.mode == "min":
            if fitness <= self.best_fitness:
                self.best_epoch = epoch
                self.best_fitness = fitness
        elif self.mode == "max":
            if fitness >= self.best_fitness:
                self.best_epoch = epoch
                self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        self.stop = delta >= self.patience  # stop training if patience exceeded

    def on_epoch_end(self, epoch=None):
        if self.stop:
            raise StopIteration


