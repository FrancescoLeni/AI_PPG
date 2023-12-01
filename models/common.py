import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchmetrics














# ----------------------------------------------------------------------------------------------------------------------
# GENERAL MODEL CLASS FOR HANDLING TRAINING, VALIDATION AND INFERENCE
# ----------------------------------------------------------------------------------------------------------------------


class ModelClass(nn.Module):
    def __init__(self, model, loaders, device='gpu', tb_writer=None, callbacks=None, loss_fn=None, optimizer=None, metrics=None):
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
                self.device = torch.cuda.get_device_name()
            else:
                print('no gpu found')
        else:
            self.device = 'cpu'

        print(f"loading model to device={self.device}")
        self.model.to(self.device)

        self.writer = tb_writer
        self.callbacks = callbacks
        self.metrics = metrics

        self.loss_fun = loss_fn
        self.opt = optimizer

        self.t_loss = 0.0
        self.v_loss = 0.0

    def train_one_epoch(self,epoch_index):
        running_loss = 0.
        metric = self.metrics
        last_loss = 0.
        # train over batches
        for i, data in enumerate(self.train_loader):
            inputs, labs = data

            inputs = inputs.to(self.device)
            labs = labs.to(self.device)

            self.opt.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_fun(outputs, labs)
            loss.backward()
            self.opt.step()

            running_loss += loss.item()

            last_loss = running_loss / self.train_loader.batch_size  # loss per batch
            print('  batch {} loss: {} parameters {}'.format(i + 1, last_loss,sum([p.sum() for p in self.model.parameters()])))
            tb_x = epoch_index * len(self.train_loader) + i + 1
            self.writerwriter.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        self.t_loss = last_loss # update loss after


    def val_loop(self):
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(self.val_loader):
                vinputs, vlabels = vdata

                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)

                voutputs = self.model(vinputs)
                vloss = self.loss_fun(voutputs, vlabels)
                running_vloss += vloss


        self.v_loss = running_vloss / (i + 1)


    # method to be called outside
    def train_loop(self, num_epochs):

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch + 1))
            self.model.train(True)

            # 1 epoch train
            self.train_one_epoch(epoch)
            # reshuffle subsampling
            self.train_loader.dataset.build()

            # validation
            self.val_loop()
            # reshuffle subsampling
            self.val_loader.dataset.build()

            print('LOSS train {} valid {}'.format(self.t_loss, self.v_loss))
            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars('Training vs. Validation Loss',
                               {'Training': self.t_loss, 'Validation': self.v_loss},
                               epoch + 1)
            self.writer.flush()

