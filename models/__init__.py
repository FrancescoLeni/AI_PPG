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
        description = f'Epoch {epoch_index}/{tot_epochs-1} GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f} ' \
                      f'train_loss: {last_loss:.4f} (metriche)'
        pbar_loader = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # train over batches
        for i, data in pbar_loader:
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

            # updating pbar
            # train_desc.format(epoch_index, tot_epochs, gpu_used, self.gpu_mem, last_loss)
            pbar_loader.set_description(f'Epoch {epoch_index}/{tot_epochs-1} GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f} '
                                        f'train_loss: {last_loss:.4f} (metriche)')

            torch.cuda.synchronize()






    def val_loop(self):
        running_loss = 0.0
        last_loss = 0.0

        # initializing progress bar
        description = f'Validation: val_loss: {last_loss:.4f} (metriche)'
        pbar_loader = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # Disable gradient computation and reduce memory consumption, and model to evaluation mode.
        self.model.eval()
        with torch.no_grad():
            for i, data in pbar_loader:
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
                # updating pbar
                description = f'Validation: val_loss: {last_loss:.4f} (metriche)'
                pbar_loader.set_description(description)




    def train_loop(self, num_epochs):

        for epoch in range(num_epochs):
            self.model.train(True)

            # 1 epoch train
            self.train_one_epoch(epoch, num_epochs)
            # reshuffle for subsampling
            self.train_loader.dataset.build()

            # validation
            self.val_loop()
            # reshuffle for subsampling
            self.val_loader.dataset.build()

    def inference(self, return_preds=False):
        self.model.train(False)
        outputs = []

        model = nn.ModuleList([self.model,
                               nn.Softmax(1)])

        # initilialize progress bar
        description = f'item {0}/{len(self.test_loader)} predicted {"N/A"} true {"N/A"}'
        pbar_loader = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)
        with torch.no_grad():
            for i, data in pbar_loader:
                inputs, labels = data

                inputs = inputs.to(self.device)

                output = model(inputs)
                pred = np.uint8(np.argmax(output.to('cpu')))
                outputs.append(pred)

                torch.cuda.synchronize()
                # updating pbar
                description = f'item {i}/{len(self.test_loader)} predicted {pred} true {labels}'
                pbar_loader.set_description(description)

        # print TEST RESULTS

        if return_preds:
            return outputs





