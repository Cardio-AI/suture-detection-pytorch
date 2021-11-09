# Set up imports
import os
import sys
import warnings

import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted

# Project imports
import utils as ut


class GenSutureMasks:
    """
    This class contains functionalities to convert the labels into masks
    It provides options to create binary masks or alternatively apply a blurring function
    Currently supported blurring functions are gaussian and tanh
    """
    def __init__(self, target_height=None,
                 target_width=None,
                 binary="False",
                 blur_func="gaussian",
                 data_format="mono",
                 spread=1,
                 corresponding_only=False):
        super(GenSutureMasks).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.binary = binary
        self.blur_func = blur_func
        self.format = data_format
        self.spread = spread
        self.corresponding_only = corresponding_only
        self.coordinates = {"x": 0, "y": 1}

        if self.format == "kitti": self.get_points = self.extract_points_from_json_kitti
        else: self.get_points = self.extract_points_from_json_mono

    def add_blurred_point(self, base_mask, x_mean, y_mean):
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

        if self.blur_func == "tanh":
            drawn_mask = base_mask + (
                        255 * (1+(np.tanh(-(np.pi * np.sqrt((x-x_mean)**2 + (y-y_mean)**2)) / self.spread))))
        elif self.blur_func == "gaussian":
            drawn_mask = base_mask + (
                        255 * np.exp(-((x-x_mean)**2 / (2 * self.spread**2) + (y-y_mean)**2 / (2 * self.spread**2))))
        else:
            raise ValueError("Please specify a valid blur function. The values currently supported are: "
                             "gaussian and tanh")
        # Euclidean distance function
        drawn_mask[drawn_mask > 255] = 255  # Round off extra values
        return drawn_mask

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

    def extract_points_from_json_mono(self, json_labels):
        """
        Extract the points from the json files when the points are in mono or AdaptOR challenge format
        :param json_labels: json file which contain the labeled points
        :return: Points, interpolated to target size as a list
        """
        h_original, w_original = self.get_original_dims(json_labels)
        points_list = [(point["x"], point["y"]) for point in json_labels["points"]]

        # Interpolate the points if there is a target height and width specified
        # Or if there are points found
        if len(points_list) >= 1:
            points_list = [self.interpolate_point(h_original=h_original,
                                                  w_original=w_original,
                                                  h_resized=self.target_height,
                                                  w_resized=self.target_width,
                                                  x=point[self.coordinates["x"]],
                                                  y=point[self.coordinates["y"]]) for point in points_list]
        return [points_list]

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
        h_original, w_original = self.get_original_dims(json_labels)

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
                    if points_dict[i]['points'][0][1] > json_labels["imageHeight"] // 2: side = {"left": 1, "right": 0}
                    else: side = {"left": 0, "right": 1}

                    left_x = points_dict[i]['points'][side["left"]][self.coordinates["x"]]
                    left_y = points_dict[i]['points'][side["left"]][self.coordinates["y"]]
                    right_x = points_dict[i]['points'][side["right"]][self.coordinates["x"]]
                    right_y = points_dict[i]['points'][side["right"]][self.coordinates["y"]] \
                              - json_labels["imageHeight"] // 2

                    left_points_list.append((left_x, left_y))
                    right_points_list.append((right_x, right_y))

                elif (not self.corresponding_only and
                      points_dict[i]['shape_type'] == 'point'):
                    # If correspondences_only flag is set to False, these single points are skipped
                    x = points_dict[i]['points'][0][0]
                    y = points_dict[i]['points'][0][1]

                    # Check if the y-coordinate belongs to left or right image, append accordingly
                    left_points_list.append((x,y)) if (points_dict[i]['points'][0][1] < json_labels["imageHeight"]//2) \
                        else right_points_list.append((x, y - json_labels["imageHeight"] // 2))

        # Interpolate the points if there is a target height and width specified
        # Or if there are points found
        if self.target_width or self.target_height:
            if len(left_points_list) >= 1:
                left_points_list = [self.interpolate_point(h_original=h_original,
                                                           w_original=w_original,
                                                           h_resized=self.target_height,
                                                           w_resized=self.target_width,
                                                           x=point[self.coordinates["x"]],
                                                           y=point[self.coordinates["y"]]) for point in
                                    left_points_list]

            if len(right_points_list) >= 1:
                right_points_list = [self.interpolate_point(h_original=h_original,
                                                            w_original=w_original,
                                                            h_resized=self.target_height,
                                                            w_resized=self.target_width,
                                                            x=point[self.coordinates["x"]],
                                                            y=point[self.coordinates["y"]]) for point in
                                     right_points_list]
        return [left_points_list, right_points_list]

    def get_original_dims(self, json_labels):
        """Returns original dims os specified in the json file
        """
        original_height = json_labels["imageHeight"] if self.format == "mono" else json_labels["imageHeight"]//2
        original_width = json_labels["imageWidth"]
        return original_height, original_width

    def create_mask_from_points(self, points):
        """Creates mask from a list of points
        """
        base_mask = np.zeros((self.target_height, self.target_width))
        if points:  # If no points are found, do nothing, return empty mask
            if not self.binary:
                for point in points:
                    base_mask = self.add_blurred_point(base_mask=base_mask,
                                                       x_mean=point[self.coordinates["x"]],
                                                       y_mean=point[self.coordinates["y"]])
            else:
                for point in points:
                    # point[y,x] in i,j indexing
                    base_mask[point[self.coordinates["y"]], point[self.coordinates["x"]]] = 1
        # return Image.fromarray(np.uint8(mask)).convert('L')  # alb aug needs PIL image
        return np.uint8(base_mask)

    def labels_to_mask(self, json_labels):
        """Master function that takes in json labels as input and outputs a mask
        Can handle both the formats"""
        if not (self.target_height or self.target_width):
            # If target dims are not specified then set it to original dims
            self.target_height, self.target_width = self.get_original_dims(json_labels)
        # In case of mono a single list, in case of kitti format a list containing [left_points, right_points]
        list_of_point_lists = self.get_points(json_labels)
        return [self.create_mask_from_points(point_list) for point_list in list_of_point_lists]


class ConflictingArgumentsWarning(UserWarning):
    pass


parser = argparse.ArgumentParser('Generate masks from json files')
parser.add_argument("--dataroot", type=str, help="path where the datasets are located")
parser.add_argument("--mask_suffix", help="If True then the blur_func and spread value is appended to mask name",
                    action="store_true")

parser.add_argument("--binary", help="If set, generates a binary mask without a blur function",
                    action="store_true")
parser.add_argument("--blur_func", type=str,
                    help="If blur, then the blurring function that has to be applied to the point",
                    default="gaussian")
parser.add_argument("--data_format", type=str, help="The format in which the data is stored",
                    default="mono", choices=["mono", "kitti"])
parser.add_argument("--spread", type=int, help="Spread parameter of the blur function", default="2")
parser.add_argument("--corresponding_only",
                    help="Only return corresponding points (in KITTI format)", action="store_true")
parser.add_argument("--target_height", type=int, help="input image height", default=None)
parser.add_argument("--target_width", type=int, help="input image width", default=None)

if __name__ == '__main__':

    args = parser.parse_args()
    args.corresponding_only = False if not args.corresponding_only else True
    gen_suture_masks = GenSutureMasks(target_height=args.target_height,
                                      target_width=args.target_width,
                                      binary=args.binary,
                                      blur_func=args.blur_func,
                                      data_format=args.data_format,
                                      spread=args.spread,
                                      corresponding_only=args.corresponding_only)
    op_dirs = ut.get_sub_dirs(args.dataroot)
    mask_suffix = "_{}_{}".format(str(args.blur_func), str(args.spread)) if args.mask_suffix else ""
    mask_name_dict = {"mono": ["masks"],
                      "kitti": ["mask_02", "mask_03"]}

    if args.binary and args.blur_func is not None:
        warnings.warn("You have set the binary flag to True and also specified a blur function."
                      "In this case, the binary flag will override the blur function and you will get"
                      "binary masks as output", ConflictingArgumentsWarning)

    print("Extracting points from the json files and creating the masks...")
    for op in op_dirs:  # For a surgery folder path
        videos = ut.get_sub_dirs(op)  # Video folder path
        for video in videos:
            label_list = natsorted(glob.glob(os.path.join(video, "point_labels", "*.json")))  # get all label files
            mask_paths = [os.path.join(video, mask_name+mask_suffix)
                          for mask_name in mask_name_dict[args.data_format]]
            success = [ut.check_and_create_folder(path) for path in mask_paths]  # Create folders to write masks
            for label_file in tqdm(label_list):
                filename = os.path.splitext(Path(label_file).name)[0]  # Get the path to a json file
                labels = ut.json_loader(label_file)  # Get labels from the json file
                masks = gen_suture_masks.labels_to_mask(labels)
                write_paths = [os.path.join(mask_path, filename+".png") for mask_path in mask_paths]
                [cv2.imwrite(path, mask) for (path, mask) in zip(write_paths, masks)]

    print("Successfully wrote masks to disk. Exiting program...")
