import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

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
                flat = [item[0] for item in self.metrics.dict[key]]
                self.dict[key].append(np.float16(sum(flat)/len(flat)))  # mean value


class SaveCSV(BaseCallback):
    """
        :param
            --logs = LogsHolder object
    """
    def __init__(self, logs, save_path):
        super().__init__()
        self.save_path = save_path
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


class SaveFigures(BaseCallback):
    def __init__(self, logs, save_path):
        self.save_path = save_path
        self.logs = logs

    def on_end(self):
        self.save_metrics()

    def save_metrics(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize= (20, 11))

        self.plot_metric("val_loss", axes[0,0])
        self.plot_metric("Accuracy", axes[0, 1])
        self.plot_metric("Precision", axes[0, 2])
        self.plot_metric("train_loss", axes[1, 0])
        self.plot_metric("AUC", axes[1, 1])
        self.plot_metric("Recall", axes[1, 2])

        fig.tight_layout()
        plt.savefig(self.save_path / "metrics.png", dpi=96)
        plt.close()

    def plot_metric(self, metric, ax):
        if "loss" not in metric:
            for key in self.logs.dict:
                if metric in key and "train_" not in key:
                    if key[-1].isdigit():
                        lab = key[-1]
                    else:
                        lab = "mean"
                    ax.plot(range(len(self.logs.dict[key])), self.logs.dict[key], label=lab)
            ax.legend()
        else:
            ax.plot(range(len(self.logs.dict[metric])), self.logs.dict[metric])
        ax.set_title(metric)


class LrLogger(BaseCallback):
    def __init__(self, opt, save_path):
        super().__init__()
        self.opt = opt
        self.save_path = save_path
        self.log = []
        self.last_epoch = 0

    def on_epoch_end(self, epoch=None):
        self.log.append(self.opt.param_groups[0]["lr"])
        self.last_epoch = len(self.log)

    def on_end(self):
        self.save()

    def save(self, name="lr.png"):
        plt.figure(figsize=(20, 11))
        plt.plot(range(self.last_epoch), self.log, marker='*')
        plt.xlabel('Epoch')
        plt.title('lr Scheduler')
        plt.savefig(self.save_path / name, dpi=96)
        plt.close()


class Loggers(BaseCallback):
    def __init__(self, metrics, opt, save_path):
        super().__init__()
        self.logs = LogsHolder(metrics)
        self.csv = SaveCSV(self.logs, save_path)
        self.lr = LrLogger(opt, save_path)
        self.figure_saver = SaveFigures(self.logs, save_path)
        self.build_list()

    def on_epoch_end(self, epoch=None):
        for obj in self.list:
            obj.on_epoch_end(epoch)

    def on_epoch_start(self, epoch=None, max_epoch=None):
        for obj in self.list:
            obj.on_epoch_start(epoch, max_epoch)

    def on_end(self):
        for obj in self.list:
            obj.on_end()

    def build_list(self):
        setattr(self, "list", [value for _, value in vars(self).items()])





