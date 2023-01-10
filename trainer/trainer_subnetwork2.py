import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from utils.util import compute_Ii_from_Si, compute_Si_from_Ii


class DefaultTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None, **extra_args):
        super(DefaultTrainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                             train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, DoP_pred, DoP, AoP_pred, AoP, S0_pred, S0):
        acc_metrics = np.zeros(len(self.metrics) * 3)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i * 3] += metric(DoP_pred, DoP)
            self.writer.add_scalar('{}_DoP'.format(metric.__name__), acc_metrics[i * 3])
            acc_metrics[i * 3 + 1] += metric(AoP_pred, AoP)
            self.writer.add_scalar('{}_AoP'.format(metric.__name__), acc_metrics[i * 3 + 1])
            acc_metrics[i * 3 + 2] += metric(S0_pred, S0)
            self.writer.add_scalar('{}_S0'.format(metric.__name__), acc_metrics[i * 3 + 2])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set the model to train mode
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics) * 3)

        # start training
        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            # (1, 1, 1) GPU tensor
            amp = sample['amp']. to(self.device)

            # (N, 3, H, W) GPU tensor
            I1_in = sample['L1'].to(self.device) * amp
            I2_in = sample['L2'].to(self.device) * amp
            I3_in = sample['L3'].to(self.device) * amp
            I4_in = sample['L4'].to(self.device) * amp
            S0_in, S1_in, S2_in, DoP_in, AoP_in = compute_Si_from_Ii(I1_in, I2_in, I3_in, I4_in)
            # normalize to [0, 1]
            S0_in = S0_in / 2
            S1_in = (S1_in + 1) / 2
            S2_in = (S2_in + 1) / 2

            # (N, 3, H, W) GPU tensor
            I1 = sample['I1'].to(self.device)
            I2 = sample['I2'].to(self.device)
            I3 = sample['I3'].to(self.device)
            I4 = sample['I4'].to(self.device)
            S0, S1, S2, DoP, AoP = compute_Si_from_Ii(I1, I2, I3, I4)
            # normalize to [0, 1]
            S0 = S0 / 2
            S1 = (S1 + 1) / 2
            S2 = (S2 + 1) / 2

            # get network output
            # (N, 3, H, W) GPU tensor
            S0_out, S1_out, S2_out = self.model(S0_in, S1_in, S2_in)
            I1_out, I2_out, I3_out, I4_out, DoP_out, AoP_out = compute_Ii_from_Si(S0_out * 2, S1_out * 2 - 1,
                                                                                  S2_out * 2 - 1)

            # visualization
            with torch.no_grad():
                if batch_idx % 100 == 0:
                    # save images to tensorboardX
                    self.writer.add_image('S0_in', make_grid(S0_in))
                    self.writer.add_image('S1_in', make_grid(S1_in))
                    self.writer.add_image('S2_in', make_grid(S2_in))
                    self.writer.add_image('DoP_in', make_grid(DoP_in))
                    self.writer.add_image('AoP_in', make_grid(AoP_in / np.pi))

                    self.writer.add_image('S0_out', make_grid(S0_out))
                    self.writer.add_image('S1_out', make_grid(S1_out))
                    self.writer.add_image('S2_out', make_grid(S2_out))
                    self.writer.add_image('DoP_out', make_grid(DoP_out))
                    self.writer.add_image('AoP_out', make_grid(AoP_out / np.pi))

                    self.writer.add_image('S0', make_grid(S0))
                    self.writer.add_image('S1', make_grid(S1))
                    self.writer.add_image('S2', make_grid(S2))
                    self.writer.add_image('DoP', make_grid(DoP))
                    self.writer.add_image('AoP', make_grid(AoP / np.pi))

            # train model
            self.optimizer.zero_grad()
            model_loss = self.loss(S0_out, S1_out, S2_out, S0, S1, S2)
            model_loss.backward()
            self.optimizer.step()

            # calculate total loss/metrics and add scalar to tensorboard
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            total_metrics += self._eval_metrics(DoP_out, DoP, AoP_out / np.pi, AoP / np.pi, S0_out, S0)

            # show current training step info
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item(),  # it's a tensor, so we call .item() method
                    )
                )

        # turn the learning rate
        self.lr_scheduler.step()

        # get batch average loss/metrics as log and do validation
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # set the model to validation mode
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics) * 3)

        # start validating
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                # get data and send them to GPU
                # (1, 1, 1) GPU tensor
                amp = sample['amp'].to(self.device)

                # (N, 3, H, W) GPU tensor
                I1_in = sample['L1'].to(self.device) * amp
                I2_in = sample['L2'].to(self.device) * amp
                I3_in = sample['L3'].to(self.device) * amp
                I4_in = sample['L4'].to(self.device) * amp
                S0_in, S1_in, S2_in, DoP_in, AoP_in = compute_Si_from_Ii(I1_in, I2_in, I3_in, I4_in)
                # normalize to [0, 1]
                S0_in = S0_in / 2
                S1_in = (S1_in + 1) / 2
                S2_in = (S2_in + 1) / 2

                # (N, 3, H, W) GPU tensor
                I1 = sample['I1'].to(self.device)
                I2 = sample['I2'].to(self.device)
                I3 = sample['I3'].to(self.device)
                I4 = sample['I4'].to(self.device)
                S0, S1, S2, DoP, AoP = compute_Si_from_Ii(I1, I2, I3, I4)
                # normalize to [0, 1]
                S0 = S0 / 2
                S1 = (S1 + 1) / 2
                S2 = (S2 + 1) / 2

                # get network output
                # (N, 3, H, W) GPU tensor
                S0_out, S1_out, S2_out = self.model(S0_in, S1_in, S2_in)

                I1_out, I2_out, I3_out, I4_out, DoP_out, AoP_out = compute_Ii_from_Si(S0_out * 2, S1_out * 2 - 1,
                                                                                      S2_out * 2 - 1)

                loss = self.loss(S0_out, S1_out, S2_out, S0, S1, S2)

                # calculate total loss/metrics and add scalar to tensorboardX
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(DoP_out, DoP, AoP_out / np.pi, AoP / np.pi, S0_out, S0)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

