import pandas as pd
import os
import cv2
import numpy as np
from utils.imutils import jimshow as show
from utils.imutils import jimshow_channel as show_channel
import matplotlib.pyplot as plt


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


def main():

    filepath_f1 = os.path.join("in", "image_0001.jpg")
    filepath = os.path.join("in")

    distance_df = process_images(filepath_f1, filepath)

    print(distance_df)

    csv_outpath = os.path.join("out", "output.csv")
    save_dataframe_to_csv(distance_df, csv_outpath)

if __name__ == "__main__":
    main()