import pandas as pd
import os, sys
import cv2
import numpy as np
from utils.imutils import jimshow as show
from utils.imutils import jimshow_channel as show_channel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import norm
from tqdm import notebook
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, preprocess_input)
from sklearn.neighbors import NearestNeighbors


def parser():
    """
    The user can specify whether to perform image search using color histograms or a pretrained CNN VGG16 model
    and K-Nearst Neighbour
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",
                        "-m",
                        required = True,
                        help = "Color histogram or VGG16?")
    args = parser.parse_args()
    return args


def read_image(filepath):
    return cv2.imread(filepath)


def calculate_histogram(image):
    return cv2.calcHist([image], [0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])


def normalize_histogram(hist):
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)


def update_distance_df(distance_df, filename, distance):
    distance_df.loc[len(distance_df.index)] = [filename, distance]


def compare_histograms(hist1, hist2):
    return round(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR), 3)


def save_dataframe_to_csv(distance_df, csv_outpath):
    distance_df.to_csv(csv_outpath)


def process_images(filepath_f1, filepath):

    """
    The process_images function systematically analyzes images by extracting and comparing normalized histograms.
    Initially, a normalized histogram encompassing all color channels is extracted from the target image (f1).

    Subsequently, a pandas dataframe is initialized to store image filenames and their corresponding distance. As
    the function iterates through all images in a sorted order, each image undergoes histogram calculation and
    normalization. The resulting normalized histogram will then be compared to that of the target image.

    The dataframe aims to hold the five most similar images to the target image. If the dataframe contains
    fewer than six entries, the function appends the current image's filename and distance. Otherwise, it
    finds the image with the highest distance (i.e., the least similar to the target) and assesses whether the
    current image has a smaller distance. If so, the dataframe is updated accordingly.
    """

    image_f1 = read_image(filepath_f1)
    hist_f1 = calculate_histogram(image_f1)
    norm_hist_f1 = normalize_histogram(hist_f1)


    distance_df = pd.DataFrame(columns=("Filename", "Distance"))

    for file in sorted(os.listdir(filepath)):
        if file != filepath_f1:
            individual_filepath = os.path.join(filepath, file)
            image = read_image(individual_filepath)
            image_name = file.split(".jpg")[0]

            hist = calculate_histogram(image)
            norm_hist = normalize_histogram(hist)

            dist = compare_histograms(norm_hist_f1, norm_hist)

            if len(distance_df) < 6:
                update_distance_df(distance_df, image_name, dist)
            else:
                max_dist_idx = distance_df['Distance'].idxmax()
                max_dist = distance_df.loc[max_dist_idx, 'Distance']

                if dist < max_dist:
                    distance_df.at[max_dist_idx, 'Filename'] = image_name
                    distance_df.at[max_dist_idx, 'Distance'] = dist

    return distance_df


def extract_features_input(filepath, model):
    """
    Extract features from image input using the pretrained VGG16 model
    """
    input_shape = (224, 224, 3)
    img = load_img(filepath, target_size = (input_shape[0], input_shape[1]))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose = False)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(features)
    return flattened_features


def pretrained_model():
    """
    Load pretrained VGG16 model
    """
    model = VGG16(weights = 'imagenet',
                include_top = False,          
                pooling = 'avg',
                input_shape = (224, 224, 3))
    return model

def load_classifier():
    """
    Load the K-Nearst Neighbour classification model
    """
    neighbors = NearestNeighbors(n_neighbors = 10,
                             algorithm = 'brute',
                             metric = 'cosine').fit(feature_list)
    return neighbors, feature_list


def extract_features_image(model, feature_list, filenames):
    feature_list = []
    for i in notebook.tqdm(range(len(filenames))):
        feature_list.append(extract_features(filenames[i], model))
    return feature_list


def calculate_nn(model, feature_list, target_image):
    """
    Calculate the nearest neighbours for target image
    """
    distances, indices = neighbors.kneighbors([feature_list[target_image]])
    return distances, indices


def save_indices(distances, indices):
    idx = []
    for i in range(1, 6):
        distances[0][i], indices[0][i]
        idx.append(indices[0][i])
    return idx


def plot_target_vs_closest(idx, filenames, target_image):
    
    plt.imshow(mpimg.imread(filenames[target_image]))
    f, axarr = plt.subplots(1,5)
    axarr[0].imshow(mpimg.imread(filenames[idxs[0]]))
    axarr[1].imshow(mpimg.imread(filenames[idxs[1]]))
    axarr[2].imshow(mpimg.imread(filenames[idxs[2]]))
    axarr[3].imshow(mpimg.imread(filenames[idxs[3]]))
    axarr[4].imshow(mpimg.imread(filenames[idxs[4]]))



def main():

    args = parser()

    if args.method.lower() == 'histogram':

        filepath_f1 = os.path.join("in", "image_0001.jpg")
        filepath = os.path.join("in")

        distance_df = process_images(filepath_f1, filepath)

        print(distance_df)

        csv_outpath = os.path.join("out", "output.csv")
        save_dataframe_to_csv(distance_df, csv_outpath)

    else:
        
        model = pretrained_model()

        target_image = os.path.join("..", "..", "..", "cds-vis-data", "flowers") # "in", "image_0001.jpg"
        features = extract_features_input(target_image, model)

        root_dir = os.path.join("..", "..", "..", "cds-vis-data", "flowers") # # "in", "image_0001.jpg
        filenames = [root_dir + "/" + name for name in sorted(os.listdir(root_dir))]

        feature_list = extract_features_image(model, feature_list, filenames)

        neighbors, feature_list = load_classifier()
        distances, indices = calculate_nn(neighbors, feature_list, target_image)

        idx = save_indices(distances, indices)
        plot_target_vs_closest(idx, filenames, target_image)

if __name__ == "__main__":
    main()