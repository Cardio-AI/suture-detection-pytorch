# Set up imports
import os
import sys
import glob
import random
import utils
import argparse
from natsort import natsorted
import torchvision.transforms as transforms


def get_images_in_surgery_folder(surgery_path):
    indices = []
    videos = (lambda x: [item for item in os.listdir(x) if os.path.isdir(item)])(surgery_path)
    for video in videos:
        image_filepaths = natsorted(glob.glob(os.path.join(video, "images", "*.png")))

        # Prepare text file information
        rel_path_name = os.path.join(os.path.basename(surgery_path), os.path.basename(video))  # Folder name
        frame_indices = [os.path.basename(os.path.splitext(path)[0]) for path in image_filepaths]

        newline_batch = [' '.join((rel_path_name, frame_index)) for frame_index in frame_indices]
        indices += newline_batch
    return indices


parser = argparse.ArgumentParser('Generate masks from json files')
parser.add_argument("--dataroot", type=str, help="path where the data is located")
parser.add_argument("--splits_root", type=str, help="Where the split file will be saved", default="../splits/")
parser.add_argument("--splits_name", type=str, help="Name of the split", default="mkr_dataset")
parser.add_argument("--num_folds", type=int, help="Number of folds to split the data into", default="4")


if __name__ == '__main__':
    args = parser.parse_args()
    op_dirs = (lambda x: [item for item in os.listdir(x) if os.path.isdir(item)])(args.dataroot)

    for fold in range(args.num_folds):
        train_indices, val_indices = [], []
        success = utils.check_and_create_folder(
            os.path.join(args.splits_root, args.splits_name, "fold_" + str(fold + 1)))  # splits/mkr_dataset/fold_1
        for i, session in enumerate(op_dirs):
            # If fold then validation else put into training list
            if i % args.num_folds == fold:
                val_indices += get_images_in_surgery_folder(surgery_path=session)
            else:
                train_indices += get_images_in_surgery_folder(surgery_path=session)

        random.shuffle(train_indices)
        random.shuffle(val_indices)

        f_writepath = os.path.join(args.splits_root, args.splits_name, "fold_{}", "{}_files.txt")
        utils.write_list_to_text_file(save_path=f_writepath.format(fold + 1, "train"),
                                      text_list=train_indices,
                                      verbose=False)
        utils.write_list_to_text_file(save_path=f_writepath.format(fold + 1, "val"),
                                      text_list=val_indices,
                                      verbose=False)
        print("Fold {}: Extracted {} training files and {} validation files "
              "and wrote them to disk".format(str(fold + 1), len(train_indices), len(val_indices)))
