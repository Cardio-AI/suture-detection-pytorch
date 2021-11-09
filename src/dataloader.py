# Imports
import os
import sys
import time
import torch
import random
import numpy as np
from torchvision.transforms import functional as func
from torchvision import transforms
from torch.utils.data import Dataset

import utils as ut


class MonoDataset(Dataset):
    """ This dataset works with the endoscopic dataset which is organized in the KITTI format
    Folder structures are Surgery--> Video --> image_02 or image_03 which contain images
    Left camera corresponds to the folder image_02 and right to image_03
    The images are named in the format {:06d}, example: 000012.png

    This mono class, works with split files which are text files that contain the
    relative path, frame number and optionally the camera (or the side left or right) to be used.
    The class reads the image file paths that are specified in the text file and loads the images.
    It applies the specified transforms to the image, else it just converts it into a tensor.
    :returns Pre-processed Image from either left or right camera (default is left)
    """

    def __init__(self, data_root_folder=None,
                 filenames=None,
                 height=448,
                 width=448,
                 aug=None,
                 color_aug=None,
                 aug_prob=0.5,
                 camera="left",
                 image_ext='.png'):
        super(MonoDataset).__init__()
        self.data_root_folder = data_root_folder
        self.filenames = filenames
        self.height = height
        self.width = width
        self.image_ext = image_ext
        self.camera = camera

        self.aug = aug
        self.color_aug = color_aug
        self.aug_prob = aug_prob
        self.resize = transforms.Resize((self.height, self.width))
        self.resize_bigger = transforms.Resize((int(self.height * 1.2),
                                                int(self.width * 1.2)))

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.cam_to_side = {"left": "l", "right": "r"}
        if self.image_ext == '.npy':
            self.image_loader = ut.numpy_loader
        else:
            self.image_loader = ut.pil_loader

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns the image with transforms applied to it"""
        filename, frame_number, side = self.get_split_filename(self.filenames[index])
        image = self.image_loader(self.make_image_path(self.data_root_folder, filename,
                                                       frame_number, side))  # get
        image = self.preprocess(image)
        return image, self.filenames[index]

    def get_split_filename(self, filename):
        """ Splits a filename string comprising of "relative_path <space> frame_number <space> side"
            If side is not given, then adds the camera (l or r) as the side
        :param filename A string comprising of "relative_path <space> frame_number <space> side"
        :return split_filename- "relative_path, frame_number, side"
        """
        split_filename = filename.split()
        if len(split_filename) == 2:
            return split_filename[0], split_filename[1], self.cam_to_side[self.camera]  # folder, frame_num, side
        else:
            return split_filename

    def make_image_path(self, data_root, rel_folder, frame_number, side):
        """Combines the relative path with the data root to get the complete path of image
           If left, then images have to be taken from the folder image_02 and if not, from image_03
           Image names are 6 decimal places long, example: 000012.png
        """
        image_name = "{:06d}{}".format(int(frame_number), self.image_ext)
        return os.path.join(data_root, rel_folder,
                            "image_0{}".format(self.side_map[side]), image_name)

    def preprocess(self, image):
        #image = self.resize(image)  # Output: Resized PIL Image
        if self.image_ext == '.npy':
            image = np.resize(image, (self.height, self.width, 3))  # Output: Resized PIL Image
        else:
            image = self.resize(image)  # Output: Resized PIL Image

        if self.color_aug and random.random() > self.aug_prob: image = self.color_aug(image)
        if self.aug: image = self.aug(image=np.asarray(image))["image"]  # alb needs np input
        return func.to_tensor(image)


class MonoDatasetWithMask(MonoDataset):
    """ Generate masks for the corresponding images and return them
    Masks are present in the folder called 'point_labels'
    """

    def __init__(self, mask_transform=None,
                 aug=None,
                 image_aug=None,
                 aug_prob=0.5,
                 in_memory=False,
                 **kwargs):
        super(MonoDatasetWithMask, self).__init__(**kwargs)

        self.aug = aug
        self.aug_prob = aug_prob
        self.image_aug = image_aug
        self.in_memory = in_memory

        if self.image_ext == '.npy':
            self.mask_loader = ut.numpy_loader
        else:
            self.mask_loader = ut.mask_loader

        if mask_transform is None:
            self.mask_transform = transforms.ToTensor()
        else:
            self.mask_transform = transforms.Compose(mask_transform)

        self.coordinates = {"x": 0, "y": 1}
        self.label_loader = ut.json_loader

        if self.in_memory:
            time_before_loading = time.time()
            self.train_images = []
            self.train_masks = []
            for file in self.filenames:
                filename, frame_number, side = self.get_split_filename(file)
                image = self.image_loader(self.make_image_path(self.data_root_folder, filename,
                                                               frame_number, side))  # pil loader
                mask = self.mask_loader(self.make_mask_path(self.data_root_folder, filename,
                                                            frame_number, side))  # pil loader
                self.train_images.append(image)
                self.train_masks.append(mask)
            time_after_loading = time.time()
            print("Loaded all data into memory...")
            print("Time for loading all data into memory: ", time_after_loading - time_before_loading)

    def __getitem__(self, index):
        if self.in_memory:
            image = self.train_images[index]
            mask = self.train_masks[index]
        else:
            # Get the relative path, frame number and side : need for both mask and image
            #time_before = time.time()
            filename, frame_number, side = self.get_split_filename(self.filenames[index])
            #time_after_split = time.time()
            image = self.image_loader(self.make_image_path(self.data_root_folder, filename,
                                                           frame_number, side))  # pil loader
            #time_after_image = time.time()
            mask = self.mask_loader(self.make_mask_path(self.data_root_folder, filename,
                                                        frame_number, side))  # pil loader
            #time_after_mask = time.time()

            #print("Time for splitting: ", time_after_split - time_before)
            #print("Time for loading image: ", time_after_image - time_after_split)
            #print("Time for loading mask: ", time_after_mask - time_after_image)

        if self.image_ext == '.npy':
            mask = mask[..., 0]
        image, mask = self.preprocess_image_mask(image=image, mask=mask)
        return image, mask, self.filenames[index]

    def make_mask_path(self, data_root, rel_folder, frame_number, side):
        """Combines the relative path with the data root to get the complete path of label file
           Label file names are 6 decimal places long, example: 000012.json
        """
        mask_folder_suffix = "_gaussian_3"
        #mask_folder_suffix = "_low_res"
        #mask_folder_suffix = ""
        mask_name = "{:06d}{}".format(int(frame_number), self.image_ext)
        return os.path.join(data_root, rel_folder,
                            "mask_0{}{}".format(self.side_map[side], mask_folder_suffix), mask_name)

    def preprocess_image_mask(self, image, mask):
        #image = self.resize(image)

        if self.image_ext == '.npy':
            image = np.resize(image, (self.height, self.width, 3))  # Output: Resized PIL Image
        else:
            image = self.resize(image)  # Output: Resized PIL Image

        if self.image_aug: image = self.image_aug(image=np.asarray(image))["image"]

        if self.aug:
            augmented = self.aug(image=np.asarray(image), mask=np.asarray(mask))
            # alb needs np input

            image = augmented["image"]
            mask = augmented["mask"]

        image = func.to_tensor(np.array(image))
        mask = func.to_tensor(np.array(mask))
        return image, mask


class MonoDatasetWithMaskPrediction(MonoDataset):
    """ Generate masks for the corresponding images and return them
    Masks are present in the folder called 'point_labels'
    """

    def __init__(self, mask_transform=None,
                 aug=None,
                 image_aug=None,
                 aug_prob=0.5,
                 in_memory=False,
                 corresponding_only=False,
                 **kwargs):
        super(MonoDatasetWithMaskPrediction, self).__init__(**kwargs)

        self.aug = aug
        self.aug_prob = aug_prob
        self.image_aug = image_aug
        self.in_memory = in_memory
        self.json_loader = ut.json_loader
        self.coordinates = {"x": 0, "y": 1}
        self.side_to_mask_list_index = {"l": 0, "r": 1}
        self.corresponding_only = corresponding_only

    def __getitem__(self, index):
        # Get the relative path, frame number and side : need for both mask and image
        filename, frame_number, side = self.get_split_filename(self.filenames[index])
        image = self.image_loader(self.make_image_path(self.data_root_folder, filename,
                                                       frame_number, side))  # pil loader
        labels = self.json_loader(self.make_json_path(self.data_root_folder, filename, frame_number))
        l_r_points = self.extract_points_from_json_kitti(labels)
        points = l_r_points[self.side_to_mask_list_index[side]]

        image = self.preprocess_image_mask(image=image)
        return image, points, self.filenames[index]

    @staticmethod
    def make_json_path(data_root, rel_folder, image_name, side=None):
        """Combines the relative path with the data root to get the complete path of mask
        """
        return os.path.join(data_root, rel_folder, "point_labels", image_name + ".json")

    @staticmethod
    def interpolate_point(h_original=None,
                          w_original=None,
                          h_resized=None,
                          w_resized=None,
                          x=0, y=0):
        """
        Interpolate a point from original size to target size
        :param h_original: Original height of image in which points were labeled
        :param w_original: Original width of mask in which points were labeled
        :param h_resized: Target height to be resized to
        :param w_resized: Target width to be resized to
        :param x: x coordinate of point to be resized
        :param y: y coordinate of point to be resized
        :return: Interpolated x,y
        """
        if not (h_resized or w_resized or h_original or w_original):
            raise ValueError("Please specify valid values for interpolation")
        if (h_original == h_resized) and (w_original == w_resized): return x,y

        factor_h = h_resized / h_original
        factor_w = w_resized / w_original
        interpolated_x = round(x * factor_w)
        interpolated_y = round(y * factor_h)

        # Handling rounding errors
        if interpolated_x > w_resized - 1: interpolated_x = w_resized - 1
        if interpolated_y > h_resized - 1: interpolated_y = h_resized - 1
        return interpolated_x, interpolated_y

    def extract_points_from_json_kitti(self, json_labels):
        """
        Read all the points from the json file
        If the shape is 'line', it means there are corresponding points, else
        only left point is present. They are appended to lists and returned.
        :param json_labels: json file which contain the labeled points
        :return: [Left_Points, Right_Points] interpolated to target size as a list
        """
        points_dict = json_labels['shapes']
        left_points_list, right_points_list = [], []

        h_original = json_labels["imageHeight"]//2
        w_original = json_labels["imageWidth"]

        for i in range(len(points_dict)):
            if points_dict[0]['label'] == 'None':
                pass
            else:
                if points_dict[i]['shape_type'] == 'None':
                    pass
                elif points_dict[i]['shape_type'] == 'line':
                    # Case 1: The first labelled point file_items["shapes"][n]['points'][0] --> [x,y]
                    # The y coordinate lies in the lower half of the image
                    # which means y coordinate>im_h/2; then first point belongs to the right image
                    # else first point belongs to left image
                    if points_dict[i]['points'][0][1] > json_labels["imageHeight"] // 2:
                        side = {"left": 1, "right": 0}
                    else:
                        side = {"left": 0, "right": 1}

                    left_x = points_dict[i]['points'][side["left"]][self.coordinates["x"]]
                    left_y = points_dict[i]['points'][side["left"]][self.coordinates["y"]]
                    right_x = points_dict[i]['points'][side["right"]][self.coordinates["x"]]
                    right_y = points_dict[i]['points'][side["right"]][self.coordinates["y"]] \
                              - json_labels["imageHeight"] // 2

                    left_points_list.append((left_x, left_y))
                    right_points_list.append((right_x, right_y))

                elif (not self.corresponding_only and
                      points_dict[i]['shape_type'] == 'point'):
                    # If correspondences_only flag is set to False, these single points are also included
                    x = points_dict[i]['points'][0][0]
                    y = points_dict[i]['points'][0][1]

                    # Check if the y-coordinate belongs to left or right image, append accordingly
                    left_points_list.append((x, y)) if (
                                points_dict[i]['points'][0][1] < json_labels["imageHeight"] // 2) \
                        else right_points_list.append((x, y - json_labels["imageHeight"] // 2))

        # Interpolate the points if there is a target height and width specified
        # Or if there are points found
        if self.width or self.height:
            if len(left_points_list) >= 1:
                left_points_list = [self.interpolate_point(h_original=h_original,
                                                           w_original=w_original,
                                                           h_resized=self.height,
                                                           w_resized=self.width,
                                                           x=point[self.coordinates["x"]],
                                                           y=point[self.coordinates["y"]]) for point in
                                    left_points_list]

            if len(right_points_list) >= 1:
                right_points_list = [self.interpolate_point(h_original=h_original,
                                                            w_original=w_original,
                                                            h_resized=self.height,
                                                            w_resized=self.width,
                                                            x=point[self.coordinates["x"]],
                                                            y=point[self.coordinates["y"]]) for point in
                                     right_points_list]
        return [left_points_list, right_points_list]

    def preprocess_image_mask(self, image):
        if self.image_ext == '.npy': image = np.resize(image, (self.height, self.width, 3))
        else: image = self.resize(image)  # Output: Resized PIL Image
        image = func.to_tensor(np.array(image))
        return image