from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import torch
import random
import pathlib
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import json
from barbar import Bar

# Imports from current project
import utils
import losses
from models import UNet
from dataloader import MonoDataset, MonoDatasetWithMaskPrediction


def add_blurred_point(base_mask, x_mean, y_mean, spread, blur_func):
    """
    Blur the point at (x_mean, y_mean) with the blur function specified by "func"
    :param base_mask: numpy array of shape (h, w)
    :param x_mean: float value denoting center_x of func
    :param y_mean: float value denoting center_y of func
    :param func: Blur function, default is gaussian. Currently gaussian and tanh are supported
    :param spread: In case of gaussian, spread is sigma; In case of tanh, spread is alpha
    :return: numpy array with gaussian values added around x_mean, y_mean
    """
    y = np.linspace(0, base_mask.shape[0] - 1, base_mask.shape[0])
    x = np.linspace(0, base_mask.shape[1] - 1, base_mask.shape[1])
    x, y = np.meshgrid(x, y)

    if blur_func == "tanh":
        drawn_mask = base_mask + (
                    255 * (1+(np.tanh(-(np.pi * np.sqrt((x-x_mean)**2 + (y-y_mean)**2)) / spread))))
    elif blur_func == "gaussian":
        drawn_mask = base_mask + (
                    255 * np.exp(-((x-x_mean)**2 / (2 * spread**2) + (y-y_mean)**2 / (2 * spread**2))))
    else:
        raise ValueError("Please specify a valid blur function. The values currently supported are: "
                         "gaussian and tanh")
    # Euclidean distance function
    drawn_mask[drawn_mask > 255] = 255  # Round off extra values
    return drawn_mask


def create_mask_from_points(points, height, width, spread, blur_func, coordinates={"x": 0, "y": 1}, binary=False):
    """Creates mask from a list of points
    """
    base_mask = np.zeros((height, width))
    if points:  # If no points are found, do nothing, return empty mask
        if not binary:
            for point in points:
                base_mask = add_blurred_point(base_mask=base_mask,
                                                   x_mean=point[coordinates["x"]],
                                                   y_mean=point[coordinates["y"]],
                                             spread=spread,
                                             blur_func=blur_func)
        else:
            for point in points:
                # point[y,x] in i,j indexing
                base_mask[int(point[coordinates["y"]]), int(point[coordinates["x"]])] = 1
    # return Image.fromarray(np.uint8(mask)).convert('L')  # alb aug needs PIL image
    return np.uint8(base_mask)


def to_float(tuple_):
    return (float(tuple_[0]), float(tuple_[1]), float(tuple_[2]))


def to_numpy_tuple(tensor_tuple):
    return tuple(torch.cat(tensor_tuple).numpy()) + (0,)


def convert_points_to_numpy(point_tensor):
    return [to_float(to_numpy_tuple(point_elem)) for point_elem in point_tensor]


class SegTester:
    def __init__(self, options=None):
        self.opt = options
        self.device = torch.device('cuda:0')

        """
        Load configs
        """
        self.fold = "fold_" + str(self.opt.fold)
        self.exp_name = pathlib.Path(self.opt.exp_dir).name  # remove fold name
        self.weights_path = os.path.join(self.opt.exp_dir, "model_weights",
                                         "weights_{}".format(str(self.opt.load_epoch)))
        self.config_path = os.path.join(self.opt.exp_dir, "config.json")

        with open(os.path.join(self.config_path), 'r') as configfile:
            self.exp_opts = json.load(configfile)
            print("Loaded experiment configs...")

        """
        Load models
        """
        self.kernel = torch.ones(3, 3).to(self.device)
        self.unet = UNet(n_channels=3, n_classes=1, kernel=self.kernel, bilinear=True).to(self.device)

        checkpoint = torch.load(os.path.join(self.weights_path, self.exp_opts["model_name"] + ".pt"))
        self.unet.load_state_dict(checkpoint["model_state_dict"])
        self.unet.eval()
        print("Loaded pre-trained Unet for experiment: {}".format(self.exp_opts["model_name"]))

        """
        Load data
        """
        split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, self.fold, "val_files.txt")
        if self.opt.fake:
            split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, "fake_B/train_files.txt")

        self.test_filenames = utils.read_lines_from_text_file(split_file_path)
        self.test_dataset = MonoDatasetWithMaskPrediction(data_root_folder=self.opt.dataroot,
                                                          filenames=self.test_filenames,
                                                          height=self.exp_opts["height"],
                                                          width=self.exp_opts["width"])
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                          num_workers=self.opt.num_workers, drop_last=False)  # put into dataloader

        """
        Set up pred dirs
        """
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.save_path = os.path.join(self.opt.pred_dir, self.exp_name, "{}_{}").format(self.opt.data_split, self.fold)
        if self.opt.fake:
            self.save_path = os.path.join(self.opt.pred_dir, self.exp_name, "{}").format(self.opt.data_split)

        self.pred_folder_names = list(set([file.split()[0]
                                           for file in self.test_filenames]))  # Get all unique 'op/video' names
        self.pred_folder_paths = [os.path.join(self.save_path, folder) for folder in self.pred_folder_names]

        if self.opt.save_pred_points:
            # append op/video/pred_points_02,_03
            self.save_pred_points_paths = [os.path.join(folder, points_path)
                                          for folder in self.pred_folder_paths
                                          for points_path in ['pred_points_02', 'pred_points_03']]
            # In Python3, map function is lazy so you have to consume it with list() else it won't execute
            success_bool = list(map(utils.check_and_create_folder, self.save_pred_points_paths))

        if self.opt.save_pred_tp_points:
            # append op/video/pred_points_02,_03
            self.save_pred_tp_points_paths = [os.path.join(folder, points_path)
                                              for folder in self.pred_folder_paths
                                              for points_path in ['pred_tp_points_02', 'pred_tp_points_03']]
            # In Python3, map function is lazy so you have to consume it with list() else it won't execute
            success_bool = list(map(utils.check_and_create_folder, self.save_pred_tp_points_paths))

        if self.opt.save_pred_mask:
            self.save_pred_mask_paths = [os.path.join(folder, mask_path)
                                          for folder in self.pred_folder_paths
                                          for mask_path in ['pred_mask_02', 'pred_mask_03']]
            success_bool = list(map(utils.check_and_create_folder, self.save_pred_mask_paths))

        if self.opt.save_annotated:
            self.save_annotated_paths = [os.path.join(folder, annotated_path)
                                          for folder in self.pred_folder_paths
                                          for annotated_path in ['annotated_02', 'annotated_03']]
            success_bool = list(map(utils.check_and_create_folder, self.save_annotated_paths))

    def predict(self):
        print("Running prediction on test dataset...")
        metrics = []
        points = []

        for i, batch in enumerate(Bar(self.test_dataloader), 0):
            image, point, filename = batch
            image_input = image.to(self.device)

            pred_mask, _ = self.unet(image_input)

            image_np = image.detach().cpu().clone().numpy().transpose((0, 2, 3, 1))
            point = convert_points_to_numpy(point)
            pred_mask_np = pred_mask.detach().cpu().clone().numpy().transpose((0, 2, 3, 1))

            rel_folder, frame_number, side = self.test_dataset.get_split_filename(filename[0])

            if self.opt.save_pred_points:
                json_name = "{:06d}{}".format(int(frame_number), ".json")
                json_path = os.path.join(self.save_path, rel_folder,
                                         "pred_points_0{}".format(self.side_map[side]),
                                         json_name)
                utils.save_points(pred_mask_np[0, ...], json_path)

            if self.opt.save_pred_tp_points:
                json_name = "{:06d}{}".format(int(frame_number), ".json")
                json_path = os.path.join(self.save_path, rel_folder,
                                         "pred_tp_points_0{}".format(self.side_map[side]),
                                         json_name)
                utils.save_points(pred_mask_np[0, ...], json_path, tp_only=True, point=point)

            if self.opt.save_pred_mask:
                pred_mask_name = "{:06d}{}".format(int(frame_number), ".png")
                pred_mask_path = os.path.join(self.save_path, rel_folder,
                                              "pred_mask_0{}".format(self.side_map[side]),
                                              pred_mask_name)
                save_image(pred_mask, pred_mask_path)

            if self.opt.save_annotated:
                annotated_image_name = "{:06d}{}".format(int(frame_number), ".png")
                annotated_path = os.path.join(self.save_path, rel_folder,
                                              "annotated_0{}".format(self.side_map[side]),
                                              annotated_image_name)
                save_image(image, annotated_path)
                image_cv = cv2.imread(annotated_path)
                annotated_image = utils.get_annotated_from_points(image=image_cv,
                                                                  point=point,
                                                                  pred_mask=pred_mask_np[0, ..., 0])
                cv2.imwrite(annotated_path, annotated_image)

            # Compute metrics and save
            #metric = losses.dice_coeff(pred=pred_mask, target=mask[..., np.newaxis])
            #metrics.append(metric.item())

        print("Evaluation completed. Metric score: {0:.2f} %".format(np.mean(metrics) * 100))
        print("Saved predictions to: {}".format(os.path.join(self.opt.pred_dir, self.exp_name)))
        print('Successfully wrote annotated images to disk')


class SegTesterWithoutMask:
    def __init__(self, options=None):
        self.opt = options
        self.device = torch.device('cuda:0')

        """
        Load configs
        """
        self.fold = "fold_" + str(self.opt.fold)
        self.exp_name = pathlib.Path(self.opt.exp_dir).name  # remove fold name
        self.weights_path = os.path.join(self.opt.exp_dir, "model_weights",
                                         "weights_{}".format(str(self.opt.load_epoch)))
        self.config_path = os.path.join(self.opt.exp_dir, "config.json")

        with open(os.path.join(self.config_path), 'r') as configfile:
            self.exp_opts = json.load(configfile)
            print("Loaded experiment configs...")

        """
        Load models
        """
        self.kernel = torch.ones(3, 3).to(self.device)
        self.unet = UNet(n_channels=3, n_classes=1, kernel=self.kernel, bilinear=True).to(self.device)

        checkpoint = torch.load(os.path.join(self.weights_path, self.exp_opts["model_name"] + ".pt"))
        self.unet.load_state_dict(checkpoint["model_state_dict"])
        self.unet.eval()
        print("Loaded pre-trained Unet for experiment: {}".format(self.exp_opts["model_name"]))

        """
        Load data
        """
        split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, self.fold, "val_files.txt")
        if self.opt.fake:
            split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, "fake_B/train_files.txt")

        self.test_filenames = utils.read_lines_from_text_file(split_file_path)
        self.test_dataset = MonoDataset(data_root_folder=self.opt.dataroot,
                                        filenames=self.test_filenames,
                                        height=self.exp_opts["height"],
                                        width=self.exp_opts["width"])
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                          num_workers=self.opt.num_workers, drop_last=False)  # put into dataloader

        """
        Set up pred dirs
        """
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.save_path = os.path.join(self.opt.pred_dir, self.exp_name, "{}_{}").format(self.opt.data_split, self.fold)
        if self.opt.fake:
            self.save_path = os.path.join(self.opt.pred_dir, self.exp_name, "{}").format(self.opt.data_split)

        self.pred_folder_names = list(set([file.split()[0]
                                           for file in self.test_filenames]))  # Get all unique 'op/video' names
        self.pred_folder_paths = [os.path.join(self.save_path, folder) for folder in self.pred_folder_names]

        if self.opt.save_pred_points:
            # append op/video/pred_points_02,_03
            self.save_pred_points_paths = [os.path.join(folder, points_path)
                                          for folder in self.pred_folder_paths
                                          for points_path in ['pred_points_02', 'pred_points_03']]
            # In Python3, map function is lazy so you have to consume it with list() else it won't execute
            success_bool = list(map(utils.check_and_create_folder, self.save_pred_points_paths))

        if self.opt.save_pred_mask:
            self.save_pred_mask_paths = [os.path.join(folder, mask_path)
                                          for folder in self.pred_folder_paths
                                          for mask_path in ['pred_mask_02', 'pred_mask_03']]
            success_bool = list(map(utils.check_and_create_folder, self.save_pred_mask_paths))

        if self.opt.save_annotated:
            self.save_annotated_paths = [os.path.join(folder, annotated_path)
                                          for folder in self.pred_folder_paths
                                          for annotated_path in ['annotated_02', 'annotated_03']]
            success_bool = list(map(utils.check_and_create_folder, self.save_annotated_paths))

    def predict(self):
        print("Running prediction on test dataset...")
        metrics = []

        for i, batch in enumerate(Bar(self.test_dataloader), 0):
            image, filename = batch
            image_input = image.to(self.device)

            pred_mask, _ = self.unet(image_input)

            image_np = image.detach().cpu().clone().numpy().transpose((0, 2, 3, 1))
            pred_mask_np = pred_mask.detach().cpu().clone().numpy().transpose((0, 2, 3, 1))

            rel_folder, frame_number, side = self.test_dataset.get_split_filename(filename[0])

            if self.opt.save_pred_points:
                json_name = "{:06d}{}".format(int(frame_number), ".json")
                json_path = os.path.join(self.save_path, rel_folder,
                                              "pred_points_0{}".format(self.side_map[side]),
                                              json_name)
                utils.save_points(pred_mask_np[0, ...], json_path)

            if self.opt.save_pred_mask:
                pred_mask_name = "{:06d}{}".format(int(frame_number), ".png")
                pred_mask_path = os.path.join(self.save_path, rel_folder,
                                              "pred_mask_0{}".format(self.side_map[side]),
                                              pred_mask_name)
                save_image(pred_mask, pred_mask_path)

            if self.opt.save_annotated:
                annotated_image_name = "{:06d}{}".format(int(frame_number), ".png")
                annotated_path = os.path.join(self.save_path, rel_folder,
                                              "annotated_0{}".format(self.side_map[side]),
                                              annotated_image_name)
                save_image(image, annotated_path)
                image_cv = cv2.imread(annotated_path)
                annotated_image = utils.get_annotated(image=image_cv, #Image.fromarray(image_cv),
                                                      pred_mask=pred_mask_np[0, ..., 0])
                cv2.imwrite(annotated_path, annotated_image)

        print("Saved predictions to: {}".format(os.path.join(self.opt.pred_dir, self.exp_name)))
        print('Successfully wrote annotated images to disk')