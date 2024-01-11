import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path


# SETTING GLOBAL VARIABLES
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


# ----------------------------------------------------------------------------------------------------------------------
# GENERAL MODEL CLASS FOR HANDLING TRAINING, VALIDATION AND INFERENCE
# ----------------------------------------------------------------------------------------------------------------------


class ModelClass(nn.Module):
    def __init__(self, model, loaders, device='cpu', callbacks=None, loss_fn=None, optimizer=None, sched=None,
                 metrics=None, loggers=None, AMP=True):
        super().__init__()
        """
        :param
            --model: complete Torch model to train/test
            --loaders: tuple with the Torch data_loaders like (train,val,test)
            --device: str for gpu or cpu
            --metrics: metrics instance for computing metrics callbacks
        """

        if isinstance(model, nn.Module):
            self.model = model
        else:
            raise TypeError("model not recognised")

        self.train_loader, self.val_loader, self.test_loader = loaders
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_properties = torch.cuda.get_device_properties(self.device)
                self.gpu_mem = gpu_properties.total_memory / (1024 ** 3)
            else:
                print('no gpu found')
                self.gpu_mem = 0
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_mem = 0

        print(f"loading model to device={self.device}")
        self.model.to(self.device)

        self.callbacks = callbacks
        self.metrics = metrics
        self.loggers = loggers

        self.loss_fun = loss_fn.to(self.device)
        self.opt = optimizer
        self.sched = sched

        if AMP and "cuda" in self.device:
            print("eneabling Automatic Mixed Precision (AMP)")
            self.AMP = True
            self.scaler = GradScaler()
        else:
            self.AMP = False


    def train_one_epoch(self,epoch_index,tot_epochs):
        running_loss = 0.
        last_loss = 0.

        # initializing progress bar
        description = 'Training'
        pbar_loader = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # train over batches
        for batch, data in pbar_loader:
            torch.cuda.empty_cache()  # Clear GPU memory
            gpu_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            inputs, labs = data

            inputs = inputs.to(self.device)
            labs = labs.to(self.device)

            self.opt.zero_grad()

            if self.AMP:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fun(outputs, labs)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fun(outputs, labs)
                loss.backward()
                self.opt.step()

            del inputs

            running_loss += loss.item()
            last_loss = running_loss #/ self.train_loader.batch_size  # loss per batch
            running_loss = 0.

            with torch.no_grad():
                # computing training metrics
                self.metrics.on_train_batch_end(outputs.float(), labs, batch)
                # calling callbacks
                self.callbacks.on_train_batch_end(outputs.float(), labs, batch)

            #updating pbar
            if self.metrics.num_classes != 2:
                A = self.metrics.A.t_value_mean
                P = self.metrics.P.t_value_mean
                R = self.metrics.R.t_value_mean
                AUC = self.metrics.AuC.t_value_mean
            else:
                A = self.metrics.A.t_value_mean
                P = self.metrics.P.t_value[1]
                R = self.metrics.R.t_value[1]
                AUC = self.metrics.AuC.t_value[1]

            pbar_loader.set_description(f'Epoch {epoch_index}/{tot_epochs-1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                                        f'train_loss: {last_loss:.4f}, A: {A :.2f}, P: {P :.2f}, R: {R :.2f}, AUC: {AUC :.2f}')
            if self.device != "cpu":
                torch.cuda.synchronize()

        # updating dictionary
        self.metrics.on_train_end(last_loss)

    def val_loop(self, epoch):
        running_loss = 0.0
        last_loss = 0.0

        #resetting metrics for validation
        self.metrics.on_val_start()
        # calling callbacks
        self.callbacks.on_val_start()

        # initializing progress bar
        description = f'Validation'
        pbar_loader = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # Disable gradient computation and reduce memory consumption, and model to evaluation mode.
        self.model.eval()
        with torch.no_grad():
            for batch, data in pbar_loader:
                torch.cuda.empty_cache()  # Clear GPU memory
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fun(outputs, labels)

                running_loss += loss.item()
                last_loss = running_loss #/ self.val_loader.batch_size  # loss per batch
                running_loss = 0.0

                if self.device != "cpu":
                    torch.cuda.synchronize()

                # computing metrics on batch
                self.metrics.on_val_batch_end(outputs, labels, batch)
                # updating roc and prc
                self.loggers.on_val_batch_end(outputs, labels, batch)
                # calling callbacks
                self.callbacks.on_val_batch_end(outputs, labels, batch)

                # updating pbar
                if self.metrics.num_classes != 2:
                    A = self.metrics.A.v_value_mean
                    P = self.metrics.P.v_value_mean
                    R = self.metrics.R.v_value_mean
                    AUC = self.metrics.AuC.v_value_mean
                else:
                    A = self.metrics.A.v_value_mean
                    P = self.metrics.P.v_value[1]
                    R = self.metrics.R.v_value[1]
                    AUC = self.metrics.AuC.v_value[1]
                description = f'Validation: val_loss: {last_loss:.4f}, val_A: {A :.2f}, ' \
                              f'val_P: {P :.2f}, val_R: {R :.2f}, val_AUC: {AUC :.2f}'
                pbar_loader.set_description(description)

        # updating metrics dict
        self.metrics.on_val_end(last_loss)
        # updating loggers (roc, prc)
        self.loggers.on_val_end()
        # calling callbacks
        self.callbacks.on_val_end(self.metrics.dict, epoch)



    def train_loop(self, num_epochs):

        for epoch in range(num_epochs):

            torch.cuda.empty_cache()

            self.metrics.on_epoch_start()
            self.loggers.on_epoch_start(epoch=epoch, max_epoch=num_epochs)

            self.model.train(True)

            # 1 epoch train
            self.train_one_epoch(epoch, num_epochs)
            # reshuffle for subsampling
            self.train_loader.dataset.build()

            # validation
            self.val_loop(epoch)

            # reshuffle for subsampling
            self.val_loader.dataset.build()

            # logging results
            self.loggers.on_epoch_end()
            # updating lr scheduler
            if self.sched:
                self.sched.step()
            #resetting metrics
            self.metrics.on_epoch_end()
            # calling callbacks
            try:
                self.callbacks.on_epoch_end(epoch)
            except StopIteration:  # (early stopping)
                print(f"early stopping at epoch {epoch}")
                break

        # logging metrics images
        self.loggers.on_end()
        # calling callbacks (saving last model)
        self.callbacks.on_end()

    def inference(self, return_preds=False):
        self.model.train(False)
        outputs = []

        model = nn.Sequential(self.model,
                              nn.Softmax(1))

        #resetting metrics for validation
        self.metrics.on_val_start()

        # initilialize progress bar
        description = f'Test'
        pbar_loader = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)
        with torch.no_grad():
            for batch, data in pbar_loader:
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = model(inputs)
                pred = np.uint8(np.argmax(output.to('cpu')))
                outputs.append(pred)

                if self.device != "cpu":
                    torch.cuda.synchronize()

                # computing metrics on batch
                self.metrics.on_val_batch_end(output, labels, batch)

                # updating pbar
                if self.metrics.num_classes != 2:
                    A = self.metrics.A.v_value_mean
                    P = self.metrics.P.v_value_mean
                    R = self.metrics.R.v_value_mean
                    AUC = self.metrics.AuC.v_value_mean
                else:
                    A = self.metrics.A.v_value_mean
                    P = self.metrics.P.v_value[1]
                    R = self.metrics.R.v_value[1]
                    AUC = self.metrics.AuC.v_value[1]
                description = f'item: {batch}/{len(self.test_loader)}, A: {A :.2f}, ' \
                              f'P: {P :.2f}, R: {R :.2f}, AUC: {AUC :.2f}'
                pbar_loader.set_description(description)

        # print TEST RESULTS
        # resetting metrics
        self.loggers.on_val_end()
        self.metrics.on_epoch_end()

        if return_preds:
            return outputs


def check_load_model(model):
    if isinstance(model, nn.Module):
        return model
    elif isinstance(model, str) and Path(model).suffix == ".pt" or ".pth":
        return torch.load(model)
    else:
        raise TypeError("model not recognised")



