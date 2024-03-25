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
    The process_images function will ...

    Initially, one normalised histogram of all color channels will be extracted from one particular image.

    Afterwards, a pandas dataframe with specified column names will be initialized. 

    Loop through all images in sorted order.

    In the loop, the current image will gain a calculated histogram and normalize it.
    The norm hist of the current image will be compared to image 1. 

    As the dataframe will end up having five images which are most simlar to the target image, the function
    will first append the image name and distance if there is less than 6 filled rows in the dataframe.
    Otherwise, it will find the image with the highest distance (so the lest similar image to the target)
    in the dataframe.
 
        First, I want to append the first 5 images' names and dist values to the distance_df. When the table
        consitsts of five images, I want to compare the distance between the image with the biggest dist value
        in the distance_df ( = so the image least similar to the target image (image_f1)) and the dist value
        of the current image. If the current image has a smaller dist, I want to update the df.


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

    filepath_f1 = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers", "image_0001.jpg")
    filepath = os.path.join("..", "..", "..", "..", "cds-vis-data", "flowers")

    distance_df = process_images(filepath_f1, filepath)

    print(distance_df)

    csv_outpath = os.path.join("out", "output.csv")
    save_dataframe_to_csv(distance_df, csv_outpath)

if __name__ == "__main__":
    main()