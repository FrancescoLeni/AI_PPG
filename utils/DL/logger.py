import pandas as pd
import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import pandas

from utils import increment_path
from utils.DL.callbacks import BaseCallback


class BaseLogger(BaseCallback):
    """
        :param
            --metrics = metrics object
    """
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics
        self.build_metrics_dict()


    def build_metrics_dict(self):
        setattr(self, "dict", {key: [] for key in self.metrics.dict})

    def update(self, epoch=None):
        for key in self.dict:
            if len(self.dict[key]) == 0:
                self.dict[key].extend(self.metrics.dict[key])
            elif not isinstance(self.metrics.dict[key][0], list):
                self.dict[key].append(self.metrics.dict[key][0])
            else:
                for i in range(len(self.dict[key])):
                    self.dict[key][i].append(self.metrics.dict[key][i][0])


class SaveCSV(BaseLogger):
    def __init__(self, metrics, folder="runs", name="exp"):
        super().__init__(metrics)

        self.save_path = increment_path(Path(folder)/name)

        # riempire il dict con tutti i valori medi di tutto




