import os, sys
import cv2
import numpy as np
from utils.imutils import jimshow as show
from utils.imutils import jimshow_channel as show_channel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import norm
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, preprocess_input)
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import argparse

def parser():
    """
    The user can specify whether to perform image search using color histograms or a pretrained CNN VGG16 model
    and K-Nearst Neighbour.
    Additionally, the user must provide a target image, that will form the basis of the image search.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",
                        "-m",
                        required = True,
                        help = "Color histogram or pretrained models?")
    parser.add_argument("--target",
                        "-t",
                        required = False,
                        default = "image_0001.jpg",
                        help = "Enter a taget image")
    args = parser.parse_args()
    args.method = args.method.lower()
    args.target = args.target.lower()
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


def process_images(target_image, filepath):

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

    image_f1 = read_image(target_image)
    hist_f1 = calculate_histogram(image_f1)
    norm_hist_f1 = normalize_histogram(hist_f1)


    distance_df = pd.DataFrame(columns=("Filename", "Distance"))

    for file in sorted(os.listdir(filepath)):
        if file != target_image:
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



def pretrained_model():
    """
    Load pretrained VGG16 model with default weights
    """
    model = VGG16(weights = 'imagenet',
                include_top = False,          
                pooling = 'avg',
                input_shape = (224, 224, 3))
    return model


def extract_features_input(filepath, model):
    """
    Extract features from the preprocessed target image using the pretrained VGG16 model
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


def load_classifier(feature_list):
    """
    Load the K-Nearst Neighbour classification model
    """
    neighbors = NearestNeighbors(n_neighbors = 10,
                             algorithm = 'brute',
                             metric = 'cosine').fit(feature_list)
    return neighbors, feature_list


def extract_features_image(model, filenames):
    """
    will extract features for all images in the filepath...
    returns a list of numpy arrays
    """
    feature_list = []
    for i in range(len(filenames)):
        feature_list.append(extract_features_input(filenames[i], model))
    return feature_list


def find_target_index(target_image, filenames):
    for i, filename in enumerate(filenames):
        if filename == target_image:
            target_image_index = i
            break
    else:
        print("Target image filename not found in the list of filenames.")
        target_image_index = None
    
    return target_image_index



def calculate_nn(neighbors, feature_list, target_image_index):
    """
    Calculate the nearest neighbours for target image and returns
    their index and distances to the target image
    """
    distances, indices = neighbors.kneighbors([feature_list[target_image_index]])
    return distances, indices


def save_indices(distances, indices):
    """ 
    Saves the indices of similar images to the target and
    respectively their distance
    """
    distance_df = pd.DataFrame(columns=("Filename", "Distance"))
    idx = []
    for i in range(1, 6):
        #distances[0][i], indices[0][i]
        
        distance = distances[0][i]
        filename = indices[0][i]
        distance_df = distance_df.append({"Filename": filename, "Distance": distance}, ignore_index = True)
        
        idx.append(indices[0][i])
    return idx, distance_df

def save_indices(distances, indices, filenames):
    """
    Saves the indices of similar images to the target and their distances
    """
    distance_df = pd.DataFrame(columns=["Filename", "Distance"])
    idxs = []

    # Add the target image to the DataFrame with distance 0
    target_filename = os.path.basename(filenames[0])
    distance_df.loc[0] = [target_filename, 0]

    # Add the closest images to the DataFrame and idx list
    for i in range(1, 6):
        distance = distances[0][i]
        filename_index = indices[0][i]
        filename = os.path.basename(filenames[filename_index])
        distance_df.loc[i] = [filename, distance]
        idxs.append(filename_index)

    return distance_df, idxs
 
    

def plot_target_vs_closest(idxs, filenames, target_image, outpath):
    """
    Plot the target image and the 5 most similar images
    """
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    axes[0].imshow(mpimg.imread(target_image))
    axes[0].set_title('Target Image')

    for i in range(1, 6):
        axes[i].imshow(mpimg.imread(filenames[idxs[i-1]]))
        axes[i].set_title(f'Closest Image {i}')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(outpath)
    return print("The plot has been saved to the out folder")


def main():

    args = parser()

    if args.method == 'hist':

        target_image = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers", args.target) # "in", args.target
        filepath = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers") # "in"

        distance_df = process_images(target_image, filepath)

        filenames = [os.path.join(filepath, filename + ".jpg") for filename in distance_df['Filename'].tolist()]

        save_dataframe_to_csv(distance_df, "out/output_hist.csv")
        
        # Plot
        # Extract inxs??
        filenames = [os.path.join(filepath, filename + ".jpg") for filename in distance_df['Filename'].tolist()]

        plot_target_vs_closest(idx, filenames, target_image, "out/target_closest_hist.png")

    else:
        
        model = pretrained_model()

        target_image = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers", args.target) # "in", args.target
        features = extract_features_input(target_image, model)

        root_dir = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers") # # "in", args.target
        filenames = [root_dir + "/" + name for name in sorted(os.listdir(root_dir))]

        feature_list = extract_features_image(model, filenames)

        neighbors, feature_list = load_classifier(feature_list)
        
        target_image_index = find_target_index(target_image, filenames)
        distances, indices = calculate_nn(neighbors, feature_list, target_image_index)

        distance_df, idxs = save_indices(distances, indices, filenames)
        plot_target_vs_closest(idxs, filenames, target_image, "out/target_closest_pretrained.png")
        
        save_dataframe_to_csv(distance_df, "out/output_pretrained.csv")


if __name__ == "__main__":
    main()