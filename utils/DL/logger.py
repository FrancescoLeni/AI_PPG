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


class LogsHolder(BaseCallback):
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
        dummy = self.dict.copy()
        for key in dummy:
            if "loss" not in key:
                for i in range(self.metrics.num_classes):
                    self.dict[key+f"_{i}"] = []

    def on_epoch_end(self, epoch=None):
        for key in self.metrics.dict:
            if "loss" in key:
                self.dict[key].append(self.metrics.dict[key][0])
            else:
                for i in range(len(self.metrics.dict[key])):
                    self.dict[key+f"_{i}"].append(self.metrics.dict[key][i][0])
                self.dict[key].append(np.float16(sum(self.metrics.dict[key][i])/len(self.metrics.dict[key])))  # mean value


class SaveCSV(BaseCallback):
    """
        :param
            --logs = LogsHolder object
    """
    def __init__(self, logs, folder="runs", name="exp", exist_ok = True):
        super().__init__()
        # RISOLVERE CASINO CON CREAZIONE PATH
        self.save_path = increment_path(Path(folder)/name, exist_ok=exist_ok)
        self.logs = logs
        self.file_name = "results.csv"
        self.build_dict()

    def on_epoch_start(self, epoch=None, max_epoch=None):
        for key in self.dict:
            self.dict[key] = []

    def on_epoch_end(self, epoch=None):
        self.update_dict()
        self.write_csv()

    def build_dict(self):
        # dictionary with only last metrics
        setattr(self, "dict", {key: [] for key in self.logs.dict})

    def update_dict(self):
        for key in self.dict:
            self.dict[key].append(self.logs.dict[key][-1])

    def write_csv(self):
        df = pd.DataFrame(self.dict)
        csv_path = self.save_path / self.file_name
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)


class Loggers(BaseCallback):
    def __init__(self, metrics, folder="runs", name="exp", exist_ok=True):
        super().__init__()
        self.logs = LogsHolder(metrics)
        self.csv = SaveCSV(self.logs, folder, name, exist_ok)
        self.build_list()

    def on_epoch_end(self, epoch=None):
        for obj in self.list:
            obj.on_epoch_end(epoch)

    def on_epoch_start(self, epoch=None, max_epoch=None):
        for obj in self.list:
            obj.on_epoch_start(epoch, max_epoch)

    def build_list(self):
        setattr(self, "list", [value for _, value in vars(self).items()])





