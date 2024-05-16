import os, sys
import cv2
import numpy as np
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
    """
    Reads an image file from a specified filepath.
    The returned image will be a NumPy array.
    """
    return cv2.imread(filepath)


def calculate_histogram(image):
    """Calculates a histogram encompassing all color channel of an image"""
    return cv2.calcHist([image], [0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])


def normalize_histogram(hist):
    """
    Normalizes the full histogram using MinMax, which involces subtracting the minimum pixel value for all pixels
    in the image, then dividing that difference by the range of pixel values (max minus min).
    All pixel values will now be between 0 and 1. 
    """
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)


def update_distance_df(distance_df, filename, distance):
    """
    Updates the distance dataframe with a new entry containing the filename and its corresponding 
    distance from the target image.
    """
    distance_df.loc[len(distance_df.index)] = [filename, distance]


def compare_histograms(hist1, hist2):
    """ Compares two histograms using the chi-squared distance metric """
    return round(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR), 3)


def save_dataframe_to_csv(distance_df, csv_outpath):
    """ Saves the dataframe as a .csv file"""
    distance_df.to_csv(csv_outpath)
    return print("The results have been saved in the out folder")


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
    Extracts features for all images in the filepath using the pretrainged VGG16 model
    and returns a list of numpy arrays
    """
    feature_list = []
    for i in range(len(filenames)):
        feature_list.append(extract_features_input(filenames[i], model))
    return feature_list


def find_target_index(target_image, filenames):
    """
    Finds the index of the target image in the list of filenames
    """
    for i, filename in enumerate(filenames):
        if filename == target_image:
            target_image_index = i
            break
    else:
        print("Target image filename not found in the list of filenames")
        target_image_index = None
    
    return target_image_index



def calculate_nn(neighbors, feature_list, target_image_index):
    """
    Calculate the nearest neighbours for target image and returns
    their index and distances to the target image
    """
    distances, indices = neighbors.kneighbors([feature_list[target_image_index]])
    return distances, indices


def save_indices(distances, indices, filenames):
    """
    Saves the indices of similar images to the target and their distances.
    The distances are converted to strings with three decimals to ensure the format of the saved distances.
    """
    distance_df = pd.DataFrame(columns=["Filename", "Distance"])
    idxs = []

    target_filename = os.path.basename(filenames[0])
    distance_df.loc[0] = [target_filename, "0.00"]

    for i in range(1, 6):
        distance = round(distances[0][i], 3)
        filename_index = indices[0][i]
        filename = os.path.basename(filenames[filename_index])
        distance_str = "{:.3f}".format(distance)
        distance_df.loc[i] = [filename, distance_str]
        idxs.append(filename_index)

    return distance_df, idxs
    

def plot_target_vs_closest(idxs, filenames, target_image, outpath):
    """
    Plot the target image and the 5 most similar images.
    As the list of idxs of the two pipelines are not the same length, the function accommodates both cases.
    """
    fig, axes = plt.subplots(1, 6, figsize = (20, 4))
    axes[0].imshow(mpimg.imread(target_image))
    axes[0].set_title('Target Image')

    if len(idxs) == 6:  # For the histogram method
        for i in range(6):
            if i != 0:
                axes[i].imshow(mpimg.imread(filenames[idxs[i]]))
                axes[i].set_title(f'Closest Image {i}')
            else:
                continue

    elif len(idxs) == 5:  # For the pretrained method
        for i in range(5):
            axes[i+1].imshow(mpimg.imread(filenames[idxs[i]]))
            axes[i+1].set_title(f'Closest Image {i+1}')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
    return print("The plot has been saved to the out folder")


def main():

    args = parser()

    target_image = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers", args.target) # "in", args.target
    image_number = target_image.split("_")[1].replace(".jpg", "")
    filepath = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers") # "in"

    if args.method == 'hist':

        distance_df = process_images(target_image, filepath)
        
        distance_df = distance_df.sort_values(by = "Distance")
        save_dataframe_to_csv(distance_df, f"out/distances_{image_number}_hist.csv")

        filenames = [os.path.join(filepath, filename + ".jpg") for filename in distance_df['Filename'].tolist()]
        idxs = [filenames.index(filename) for filename in filenames]
        plot_target_vs_closest(idxs, filenames, target_image, f"out/target_closest_{image_number}_hist.png")

    else:
        
        model = pretrained_model()

        root_dir = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers") # # "in", args.target
        filenames = [root_dir + "/" + name for name in sorted(os.listdir(root_dir))]

        feature_list = extract_features_image(model, filenames)

        neighbors, feature_list = load_classifier(feature_list)
        
        target_image_index = find_target_index(target_image, filenames)

        distances, indices = calculate_nn(neighbors, feature_list, target_image_index)

        distance_df, idxs = save_indices(distances, indices, filenames)

        plot_target_vs_closest(idxs, filenames, target_image, f"out/target_closest_{image_number}_pretrained.png")
        
        save_dataframe_to_csv(distance_df, f"out/distances_{image_number}_pretrained.csv")


if __name__ == "__main__":
    main()