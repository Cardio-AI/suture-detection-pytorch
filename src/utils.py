import random
import datetime
import os
import skimage.draw
import skimage.measure
import skimage.morphology
from scipy import ndimage
from copy import deepcopy
import math
import json
from PIL import Image

import torch
import numpy as np
from ipyfilechooser import FileChooser


def set_cwd():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)


def numpy_loader(path):
    return np.load(path)


def json_loader(path):
    with open(path, 'rb') as file:
        return json.load(file)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        with Image.open(file) as image:
            return image.convert('RGB')


def check_and_create_folder(path):
    """Check if a folder in given path exists, if not then create it"""
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else: return False


def write_to_json_file(content, path):
    with open(path, 'w') as file:
        json.dump(content, file, indent=4)


def mask_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        with Image.open(file) as image:
            return image.convert('L')


def getTimeStamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def read_lines_from_text_file(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_hrs_min_sec(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = (total_seconds % 3600) % 60
    return int(hours), int(minutes), int(seconds)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def get_experiment(load_pretrained=True):
    pretrainpath = FileChooser('../')
    pretrainpath.default_path = '/home/lalith/codebase/detCycleGAN/experiments'
    pretrainpath.title = '<b>Choose the path to pre-trained weights file</b>'
    return load_pretrained, pretrainpath


def match_keypoints(kplist1, kplist2, threshold):
    """
    compare two lists of keypoints and match the keypoints.

    :param list kplist1: list containing keypoints
    :param list kplist2: list containing keypoints
    :param float threshold: maximal distance for matching
    :return: list of matching keypoints with length of kplist1
    """
    labellist_matched = []
    for elem1 in kplist1:
        match = 0
        labeldistance = threshold
        for elem2 in kplist2:
            if compute_distance(elem1, elem2) < labeldistance:
                labeldistance = compute_distance(elem1, elem2)
                # print(labeldistance, elem1, elem2)
                match = elem2
        if match:
            labellist_matched.append([elem1, match])
            # kplist1.remove(elem1)
            # kplist2.remove(match)
        # else:
        #     labellist_matched.append([elem1, [None, None]])
    # from scipy.spatial import distance
    # m2 = distance.cdist(pred_regions, mask_regions, 'euclidean') --> Matrix
    return remove_dupes(labellist_matched)


def remove_dupes(matchlist):
    """
    removes duplicated second elements in matchlist
    m = [[(1, 2), (3, 4)], [(4, 2), (3, 7)], [(2, 8), (3, 7)], [(5, 8), (3, 7)], [(10, 2), (3, 4)], [(6, 3), (2, 4)]]
    m_c = [[(1, 2), (3, 4)], [(2, 8), (3, 7)], [(6, 3), (2, 4)]]
    :return:
    """
    dupe_idx = []
    for ind1, row1 in enumerate(matchlist):
        for ind2, row2 in enumerate(matchlist):
            if (row1[1] == row2[1]) and (ind1 != ind2):
                if [ind2, ind1] not in dupe_idx:
                    dupe_idx.append([ind1, ind2])

    m_cleared = deepcopy(matchlist)
    for dupe in dupe_idx:
        pair1 = matchlist[dupe[0]]
        pair2 = matchlist[dupe[1]]
        d1 = compute_distance(pair1[0], pair1[1])
        d2 = compute_distance(pair2[0], pair2[1])
        if d1 < d2:
            if pair2 in m_cleared:
                m_cleared.remove(pair2)
        else:
            if pair1 in m_cleared:
                m_cleared.remove(pair1)
    return m_cleared


def compute_distance(p1, p2):
    if list(p2) == [None, None]:
        return None
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def centres_of_mass(mask, threshold):
    mask_bin = deepcopy(mask)
    mask_bin[mask >= threshold] = 1
    mask_bin[mask < threshold] = 0

    label_regions, num_regions = skimage.measure.label(mask_bin, background=0, return_num=True)
    indexlist = [item for item in range(1, num_regions + 1)]
    return ndimage.measurements.center_of_mass(mask_bin, label_regions, indexlist)


def generate_points(image_out, connectivity=2):
    # generate centre of mass
    label_img = skimage.measure.label(image_out, connectivity=2)
    props = skimage.measure.regionprops(label_img)
    # generate prediction points
    points = []
    areas = []
    bboxs = []
    for prop in props:
        point = {}
        point["y"] = prop.centroid[0]
        point["x"] = prop.centroid[1]
        bboxs.append(prop.bbox)
        points.append(point)
        areas.append(prop.area)
    return points, areas, bboxs


def apply_coordinates(image, coordlist, size, color=(57, 255, 20)):
    """
    paint a circle with a cross in it to coordinates in list

    :param image: (1080,1920,3),float
    :param coordlist: list of coord-Pairs (x,y)
    :param size: radius of circle and size of cross
    :param color: color of circle, default neon green
    :return:
    """
    img_masked = deepcopy(image)

    for x, y in coordlist:
        line1 = skimage.draw.line(int(x), int(y - size), int(x), int(y + size))
        line2 = skimage.draw.line(int(x - size), int(y), int(x + size), int(y))

        skimage.draw.set_color(img_masked, line1, (1, 1, 0), alpha=0.5)  # yellow cross
        skimage.draw.set_color(img_masked, line2, (1, 1, 0), alpha=0.5)  # yellow cross
        # skimage.draw.set_color(img_masked, line1, color, alpha=0.5)  # cross
        # skimage.draw.set_color(img_masked, line2, color, alpha=0.5)  # cross

    for x, y in coordlist:
        img_backup = deepcopy(img_masked)
        # circle = skimage.draw.circle_perimeter(int(x), int(y), size)
        # skimage.draw.set_color(img_masked, circle, color, alpha=1)  # circle
        circle = skimage.draw.circle(int(x), int(y), size + 3, shape=image.shape[:-1])
        img_masked[circle] = color
        # skimage.draw.set_color(img_masked, circle, color, alpha=1)  # circle
        circle = skimage.draw.circle(int(x), int(y), size, shape=image.shape[:-1])
        img_masked[circle] = img_backup[circle]
        # skimage.draw.set_color(img_masked, circle, color, alpha=1)  # circle

    return img_masked


def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal"):
    assert (torch.cuda.is_available())
    net = net.cuda()
    if type == "glorot":
        glorot_weight_zero_bias(net, distribution=distribution)
    else:
        kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net


def glorot_weight_zero_bias(model, distribution="uniform"):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    distribution: string
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    torch.nn.init.xavier_normal_(module.weight, gain=1)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    if activation_mode == "leaky_relu":
        print("Leaky relu is not supported yet")
        assert False

    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=activation_mode)
                else:
                    torch.nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=activation_mode)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


def get_f_beta_score(tpr, ppv, beta=1):
    assert beta in [0.5, 1, 2], "beta should be one of 0.5, 1, 2, but got beta={}".format(beta)
    f_beta = (((1 + (np.square(beta))) * ppv * tpr) / ((np.square(beta) * ppv) + tpr))
    return f_beta


def save_points(pred_mask, json_filepath="", threshold=0.5, tp_only=False, point=None):
    """
    Extract points from mask and save them to disk
    """
    pred_regions = centres_of_mass(pred_mask, threshold)

    if tp_only:
        swapped_point = (lambda x: [(point[1], point[0], point[2]) for point in x])(point)
        matched_regions = match_keypoints(pred_regions, swapped_point, threshold=6)
        true_positives = [elem[0] for elem in matched_regions]
        output_points = true_positives
    else:
        output_points = pred_regions

    with open(json_filepath, 'w') as outfile:
        # save the resulting object in a new json file
        json.dump(output_points, outfile, indent=4)


def get_annotated(image, pred_mask, gt_mask=None):
    pred_regions = centres_of_mass(pred_mask, threshold=0.5)

    if gt_mask is None:
        annotated_image = apply_coordinates(image, pred_regions, size=5, color=(128, 0, 0))
        return annotated_image

    else:
        gt_regions = centres_of_mass(gt_mask, threshold=0.5)
        matched_regions = match_keypoints(pred_regions, gt_regions, threshold=6)

        true_positives = [elem[0] for elem in matched_regions]
        true_positives_mask = [elem[1] for elem in matched_regions]
        false_negatives = [fn for fn in gt_regions if fn not in true_positives_mask]

        annotated_image = apply_coordinates(image, pred_regions, size=5, color=(34, 57, 191))  # Red, FP
        annotated_image = apply_coordinates(annotated_image, false_negatives, size=5, color=(0, 178, 255))  # Orange, FN
        annotated_image = apply_coordinates(annotated_image, true_positives, size=5)  # Default green, TP
        return annotated_image


def get_annotated_from_points(image, pred_mask, point=None):
    pred_regions = centres_of_mass(pred_mask, threshold=0.5)
    swapped_point = (lambda x: [(point[1], point[0], point[2]) for point in x])(point)

    matched_regions = match_keypoints(pred_regions, swapped_point, threshold=6)

    true_positives = [elem[0] for elem in matched_regions]
    true_positives_mask = [elem[1] for elem in matched_regions]
    false_negatives = [fn[:-1] for fn in swapped_point if fn not in true_positives_mask]

    annotated_image = apply_coordinates(image, pred_regions, size=5, color=(34, 57, 191))  # Red, FP
    annotated_image = apply_coordinates(annotated_image, false_negatives, size=5, color=(0, 178, 255))  # Orange, FN
    annotated_image = apply_coordinates(annotated_image, true_positives, size=5)  # Default green, TP
    return annotated_image


def evaluate(predictions, labels, radius):
    """
    Evaluate an array of predictions based on an array of labels and their given radius
    True positive gets calculated by matching predictions with labels from shortest distance to longest distance.
    False positive are all predictions without a label.
    False negative are all label without a prediction.
    :param predictions: an array of predictions with x and y coordinates
    :param labels: an array of labels with x and y coordinates
    :param radius: the radius around the labels within which a prediction is still correct
    :returns: the amount of true positive (TP) (label with prediction), false positive (FP) (prediction with no label)
    and false negative (FN) (label with no prediction) labels
    """
    # count all labels in radius of each prediction
    labels_in_radius_of_all_predictions = []

    # iterate all predictions
    for prediction_index, prediction in enumerate(predictions):
        labels_in_radius_of_prediction = []

        # for each label
        for label_index, label in enumerate(labels):

            # get the distance to all close labels for each prediction
            distance = abs(math.sqrt(
                (label[0] - prediction[0]) ** 2 + (label[1] - prediction[1]) ** 2))

            # save all close labels of the prediction
            if distance <= radius:
                labels_in_radius_of_prediction.append(
                    {"prediction_index": prediction_index, "label_index": label_index, "distance": distance})

        labels_in_radius_of_all_predictions.append(
            labels_in_radius_of_prediction)

    # all true positive predictions with labels and distance
    true_positive_predictions = []

    # check if any predictions have close labels
    # find all matching pairs of predictions and labels starting with the closest pair
    while max([len(_) for _ in labels_in_radius_of_all_predictions], default=0) >= 1:

        # the closest pair of any prediction and any label
        closest_prediction_label_pair = None

        # iterate the predictions
        for labels_in_radius_of_prediction in labels_in_radius_of_all_predictions:
            # choose the prediction and label with the shortest distance
            for close_label in labels_in_radius_of_prediction:
                if closest_prediction_label_pair == None or close_label["distance"] <= closest_prediction_label_pair[
                    "distance"]:
                    closest_prediction_label_pair = close_label

        # the best prediction is a true positive prediction
        true_positive_predictions.append(closest_prediction_label_pair)

        # make sure this prediction does not get picked again
        labels_in_radius_of_all_predictions[closest_prediction_label_pair["prediction_index"]] = [
        ]

        # make sure this label does not get picked again
        for index, labels_in_radius_of_prediction in enumerate(labels_in_radius_of_all_predictions):
            # remove the label of the best prediction from all other predictions
            labels_in_radius_of_all_predictions[index] = [
                close_label for close_label in labels_in_radius_of_prediction if
                close_label["label_index"] != closest_prediction_label_pair["label_index"]]

    # the amount of true positives is just the amount of found predictions and labels matches
    true_positive = len(true_positive_predictions)
    # the amount of false positives is the amount of predictions not found in the predictions and labels matches
    false_positive = len([prediction for index, prediction in enumerate(predictions) if len(
        [tp_prediction for tp_prediction in true_positive_predictions if
         tp_prediction["prediction_index"] == index]) == 0])
    # the amount of false negatives is the amount of labels not found in the predictions and labels matches
    false_negative = len([label for index, label in enumerate(labels) if len(
        [tp_prediction for tp_prediction in true_positive_predictions if tp_prediction["label_index"] == index]) == 0])

    return true_positive, false_positive, false_negative


def write_list_to_text_file(save_path, text_list, verbose=True):
    """
    Function to write a list to a text file.
    Each element of the list is written to a new line.
    Note: Existing text in the file will be overwritten!
    :param save_path: Path to save-should be complete with .txt extension)
    :param text_list: List of text-each elem of list written in new line)
    :param verbose: If true, prints success message to console
    :return: No return, writes file to disk and prints a success message
    """
    with open(save_path, 'w+') as write_file:
        for text in text_list:
            if isinstance(text, str):
                write_file.write(text + '\n')
            else:
                write_file.write(str(text) + '\n')
        write_file.close()
    if verbose: print("Text file successfully written to disk at {}".format(save_path))