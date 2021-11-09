from __future__ import absolute_import, division, print_function

import os
import sys
import time
import torch
import random
import numpy as np
import pathlib
from PIL import Image

import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import albumentations as alb
import albumentations.augmentations.transforms as alb_tr
from barbar import Bar
from natsort import natsorted

# Imports from current project
import utils
import losses
from models import UNet
from dataloader import MonoDatasetWithMask


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SegmentationTrainer:
    def __init__(self, options=None):
        self.opt = options
        self.device = torch.device('cuda:0')

        """
        Callbacks
        """
        self.fold = "fold_" + str(self.opt.fold)
        # If save flag is set, then no timestamp is added, because then same exp name can be used for all folds
        suffix = "_{}".format(str(utils.getTimeStamp())) if not self.opt.save else ""
        self.folder_name = "{}_{}{}".format(self.opt.model_name, self.fold, suffix)

        self.log_path = os.path.join(self.opt.log_dir, self.folder_name)
        if os.path.exists(self.log_path):
            print(self.log_path)
            raise FileExistsError
        self.writer = SummaryWriter(self.log_path)  # init tensorboard summary writer

        """
        Model setup
        """
        self.kernel = torch.ones(3, 3).to(self.device)
        self.model = UNet(n_channels=3, n_classes=1, kernel=self.kernel, bilinear=True).to(self.device)

        #  init optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                       patience=10,
                                                                       min_lr=1e-10,
                                                                       factor=0.1)

        if self.opt.pretrained:  # Load pre-trained unet model
            self.exp_name = pathlib.Path(self.opt.exp_dir).name[:-7]  # remove fold
            self.checkpoint_path = os.path.join(self.opt.exp_dir,
                                               "model_weights", "weights_{}".format(str(self.opt.load_epoch)))
            self.checkpoint = torch.load(os.path.join(self.checkpoint_path, self.exp_name + ".pt"))

            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.model.train()
            print("Loaded pre-trained Unet from experiment: {}".format(self.exp_name))

        else:
            self.model = utils.init_net(self.model, type="kaiming", mode="fan_in",
                                        activation_mode="relu",
                                        distribution="normal")

        """
        Loss functions
        """
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()
        self.dice_loss = losses.dice_coeff_loss
        self.metric_fn = losses.dice_coeff

        """
        Data setup
        """
        self.image_ext = '.png'
        split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, self.fold, "{}_files.txt")
        self.train_filenames = utils.read_lines_from_text_file(split_file_path.format("train"))

        self.image_aug = alb.Compose([alb.Resize(height=self.opt.height, width=self.opt.width),
                                      alb_tr.ColorJitter(brightness=0.2,
                                                         contrast=(0.3, 1.5),
                                                         saturation=(0.5, 2),
                                                         hue=0.1,
                                                         p=0.5)])

        self.image_mask_aug = alb.Compose([alb.Rotate(limit=(-60, 60),
                                                      p=self.opt.aug_prob),
                                           alb.IAAAffine(translate_percent=10, shear=0.1,
                                                         p=self.opt.aug_prob),
                                           alb.HorizontalFlip(p=0.5),
                                           alb.VerticalFlip(p=0.5)])

        self.train_dataset = MonoDatasetWithMask(data_root_folder=self.opt.dataroot,
                                                 filenames=self.train_filenames,
                                                 height=self.opt.height,
                                                 width=self.opt.width,
                                                 camera="left",
                                                 aug=self.image_mask_aug,
                                                 image_aug=self.image_aug,
                                                 aug_prob=self.opt.aug_prob,
                                                 image_ext=self.image_ext)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size,
                                           shuffle=True, num_workers=self.opt.num_workers,
                                           drop_last=True, worker_init_fn=seed_worker)  # put into dataloader

        #######################
        # Save image augmentations to config file
        self.aug_dict = {"image_aug": alb.to_dict(self.image_aug),
                         "image_mask_aug": alb.to_dict(self.image_mask_aug)}

    """
    Training and optimisation
    """
    def train(self):
        self.model = utils.init_net(self.model, type="kaiming", mode="fan_in",
                                    activation_mode="relu",
                                    distribution="normal")

        self.save_configs()  # save script config to json file
        self.append_configs(item=self.aug_dict)
        print("Running experiment named: {} on device:{}...".format(self.folder_name,
                                                                    self.opt.device_num))

        for epoch in range(self.opt.num_epochs):
            print("Epoch {}".format(epoch + 1))

            self.model.train()
            time_before_epoch_train = time.time()
            train_loss, train_metric, train_pred = self.compute_epoch(dataloader=self.train_dataloader, train=True)
            self.lr_scheduler.step(train_loss)

            epoch_train_duration = time.time() - time_before_epoch_train
            self.log_losses('train_loss', train_loss, epoch + 1)
            self.log_losses('train_metric', train_metric, epoch + 1)
            print('Epoch {} mean batch train loss: {:0.5f} | train metric: {:0.4f} | epoch train time: {:0.2f}s'.
                  format(epoch + 1, train_loss, train_metric, epoch_train_duration))
            # save model checkpoint every save_freq epochs
            if (epoch + 1) % self.opt.save_freq == 0: self.save_checkpoint(epoch=epoch + 1,
                                                                           loss=train_loss)

    def compute_epoch(self, dataloader, train=True):
        running_loss = 0
        running_metric = 0

        for i, batch in enumerate(Bar(dataloader), 0):
            image, mask, filename = batch
            image, mask = image.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()  # set the gradients to zero
            pred_mask, _ = self.model(image)
            loss = self.compute_losses(y_pred_stage_1=pred_mask, y_true_stage_1=mask,
                                       y_pred_stage_2=pred_mask, y_true_stage_2=mask)

            if train:
                loss.backward()  # backward pass
                self.optimizer.step()  # Update parameters

            metric = self.metric_fn(pred=pred_mask, target=mask)
            running_metric += metric.detach() * self.opt.batch_size
            running_loss += loss.detach() * self.opt.batch_size  # Mean of one batch times the batch size

        epoch_loss = running_loss.item() / len(dataloader.dataset)  # Sum of all samples over number of samples in dataset
        epoch_metric = (running_metric.item() * 100) / len(dataloader.dataset)
        return epoch_loss, epoch_metric, pred_mask[0]

    def log_losses(self, name, loss, epoch):
        """Write an event to the tensorboard events file"""
        self.writer.add_scalar(name, loss, epoch)

    def log_images(self, name, loss, epoch):
        """Write an image to the tensorboard events file"""
        self.writer.add_image(name, loss, epoch)

    def save_model(self, epoch):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "model_weights", "weights_{}".format(epoch))
        os.makedirs(save_folder)
        save_path = os.path.join(save_folder, "{}.pth".format(self.opt.model_name))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

    def save_checkpoint(self, epoch, loss):
        """ Save model weights and optim state to disk
        """
        save_folder = os.path.join(self.log_path, "model_weights", "weights_{}".format(epoch))
        os.makedirs(save_folder)
        save_path = os.path.join(save_folder, "{}.pt".format(self.opt.model_name))
        checkpoint = {'epoch': epoch,
                   'model_state_dict': self.model.state_dict(),
                   'optimizer_state_dict': self.optimizer.state_dict(),
                   'loss': loss}
        torch.save(checkpoint, save_path)

    def save_configs(self):
        utils.write_to_json_file(content=self.opt.__dict__,
                                    path=os.path.join(self.log_path, "config.json"))
        print("Saving script configs...")

    def append_configs(self, item):
        config_dict = utils.json_loader(os.path.join(self.log_path, "config.json"))
        config_dict.update(item)
        utils.write_to_json_file(content=config_dict,
                                 path=os.path.join(self.log_path, "config.json"))

    def compute_losses(self, y_true_stage_1, y_pred_stage_1,
                       y_true_stage_2, y_pred_stage_2,
                       stage_1_loss="mse_dice"):
        if stage_1_loss == "f_beta":
            stage_1_loss = 1 - losses.f_beta_loss(pred=y_pred_stage_1, gt=y_true_stage_1)
        elif stage_1_loss == "mse_dice":
            stage_1_loss = self.mse(input=y_pred_stage_1, target=y_true_stage_1) + \
                           1 - losses.dice_coeff(pred=y_pred_stage_1, target=y_true_stage_1)

        stage_2_loss = self.mse(input=y_pred_stage_2, target=y_true_stage_2) + \
                      1 - losses.dice_coeff(pred=y_pred_stage_2, target=y_true_stage_2)
        return 0.5 * stage_1_loss + 0.5 * stage_2_loss
