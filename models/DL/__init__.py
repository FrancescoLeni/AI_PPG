import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchmetrics
from tqdm import tqdm
import numpy as np


import time


# SETTING GLOBAL VARIABLES
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


# ----------------------------------------------------------------------------------------------------------------------
# GENERAL MODEL CLASS FOR HANDLING TRAINING, VALIDATION AND INFERENCE
# ----------------------------------------------------------------------------------------------------------------------


class ModelClass(nn.Module):
    def __init__(self, model, loaders, device='gpu', callbacks=None, loss_fn=None, optimizer=None, metrics=None):
        super().__init__()
        """
        :param
            --model: complete Torch model to train/test
            --loaders: tuple with the Torch data_loaders like (train,val,test)
            --device: str for gpu or cpu
            --metrics: metrics instance for computing metrics callbacks
        """
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = loaders
        if device=='gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_properties = torch.cuda.get_device_properties(self.device)
                self.gpu_mem = gpu_properties.total_memory / (1024 ** 3)
            else:
                print('no gpu found')
                self.gpu_mem = 0
        else:
            self.device = 'cpu'

        print(f"loading model to device={self.device}")
        self.model.to(self.device)

        self.callbacks = callbacks
        self.metrics = metrics

        self.loss_fun = loss_fn
        self.opt = optimizer


    def train_one_epoch(self,epoch_index,tot_epochs):
        running_loss = 0.
        last_loss = 0.

        # initializing progress bar
        gpu_used = torch.cuda.memory_allocated() / (1024 ** 3)
        description = 'Training'
        pbar_loader = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # train over batches
        for batch, data in pbar_loader:
            torch.cuda.empty_cache()  # Clear GPU memory
            gpu_used = torch.cuda.memory_allocated() / (1024 ** 3)
            inputs, labs = data

            inputs = inputs.to(self.device)
            labs = labs.to(self.device)

            self.opt.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_fun(outputs, labs)
            loss.backward()
            self.opt.step()

            running_loss += loss.item()
            last_loss = running_loss #/ self.train_loader.batch_size  # loss per batch
            running_loss = 0.

            # computing training metrics
            self.metrics.on_train_batch_end(outputs, labs, batch)

            # updating pbar
            pbar_loader.set_description(f'Epoch {epoch_index}/{tot_epochs-1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                                        f'train_loss: {last_loss:.4f}, A: {self.metrics.A.t_value_mean :.2f}, ' 
                                        f'P: {self.metrics.P.t_value_mean :.2f}, R: {self.metrics.R.t_value_mean :.2f}, ' 
                                        f'AUC: {self.metrics.AuC.t_value_mean :.2f} ')

            torch.cuda.synchronize()

    def val_loop(self):
        running_loss = 0.0
        last_loss = 0.0

        #resetting metrics for validation
        self.metrics.on_val_start()

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

                torch.cuda.synchronize()

                # computing metrics on batch
                self.metrics.on_val_batch_end(outputs, labels, batch)

                # updating pbar
                description = f'Validation: val_loss: {last_loss:.4f}, val_A: {self.metrics.A.v_value_mean :.2f}, ' \
                              f'val_P: {self.metrics.P.v_value_mean :.2f}, val_R: {self.metrics.R.v_value_mean :.2f}, ' \
                              f'val_AUC: {self.metrics.AuC.v_value_mean :.2f}'
                pbar_loader.set_description(description)



    def train_loop(self, num_epochs):

        for epoch in range(num_epochs):

            self.metrics.on_epoch_start()

            self.model.train(True)

            # 1 epoch train
            self.train_one_epoch(epoch, num_epochs)
            # reshuffle for subsampling
            self.train_loader.dataset.build()

            # validation
            self.val_loop()
            # reshuffle for subsampling
            self.val_loader.dataset.build()

            #resetting metrics
            self.metrics.on_epoch_end()

    def inference(self, return_preds=False):
        self.model.train(False)
        outputs = []

        model = nn.ModuleList([self.model,
                               nn.Softmax(1)])

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

                output = model(inputs)
                pred = np.uint8(np.argmax(output.to('cpu')))
                outputs.append(pred)

                torch.cuda.synchronize()

                # computing metrics on batch
                self.metrics.on_val_batch_end(outputs, labels, batch)

                # updating pbar
                description = f'item: {i}/{len(self.test_loader)}, A: {self.metrics.A.v_value_mean :.2f}, ' \
                              f'P: {self.metrics.P.v_value_mean :.2f}, R: {self.metrics.R.v_value_mean :.2f}, ' \
                              f'AUC: {self.metrics.AuC.v_value_mean :.2f}'
                pbar_loader.set_description(description)

        # print TEST RESULTS
        # resetting metrics
        self.metrics.on_epoch_end()

        if return_preds:
            return outputs





