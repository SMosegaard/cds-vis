# Assignment 1 - Building a Simple Image Search Algorithm
*By Sofie Mosegaard, 01-03-2024*

This repository aims to design a simple image search algorithm. The algorithm allows its users to retrieve similar images to a given target image and potentially unveil patterns and similarities within a dataset.

Specifically, the project will conduct image search on flower images. In order to extract the most similar images to a given target image, the project offers two methods:

1. Color histogram comparison:
    - Initializes a dataframe to store image filenames and corresponding distances
    - Extracts a normalized histogram encompassing all color channels for the target image using OpenCV
    - Iterates through all images, calculating and normalizing histograms
    - Compares the normalized histogram to that of the target image using the chi-squared distance metric
    - Updates the dataframe with the five most similar images to the target
    - Saves a dataframe containing the target image, the most similar images, and their distances
    - Visualizes the target image and its five most similar images

2. Pretrained CNN VGG16 model and K-Nearest Neighbours (K-NN):
    - Loads the pretrained VGG16 model and creates a database of extracted features from images
    - Utilizes a K-NN classifier to find the nearest neighbors for the target image
    - Saves a dataframe with the target image, the most similar images, and their distances
    - Visualizes the target image and its five most similar images

To gain a better understanding of the code, all functions in the script ```src/image_search.py``` have a short description.

## Data source

In this repository, image search will be done on images of flowers. The dataset is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford. All images are in color.

You can download the dataset [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) and place it in the ```in``` folder. Make sure to unzip the data within the folder before executing the code.

## Repository structure

The repository consists of the following elements:

- 2 bash scripts for setup of the virtual environments, installation of requirements, and execution of the code
- 1 .txt file specifying the required packages including versioned dependencies
- 1 README.md file
- 3 folders
    - in: contains data to be processed
    - src: consists of the Python script to be executed
    - out: stores the saved results, i.e., the dataframe in a .csv format and the visualization as .png.

## Reproducibility 

1.   Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-1"
```
2.  Navigate into the folder in your terminal.
```python
$ cd assignment-1
```
3.  Run the setup bash script to create a virtual environment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```

## Usage

Run the run bash script in your terminal with the required input information. This involves specifying the pipeline by writing ```--method / -m``` to either perform the image search using color histograms (enter ```hist```) or a pretrained CNN and K-NN (enter ```pretrained```). Additionally, you must provide a target image by writing ```--target / -t```, which will form the basis of the image search. If no target image is specified, the script will by default use image 'image_0001.jpg'.

```python
$ source run.sh --method {hist/pretrained} --target {image_xxxx.jpg}
```
The inputs will be converted to lowercase, so it makes no difference how it's spelled.

Then, the code will perform image search utilizing the specified method and retrieve similar images to the provided target image. Once the script has finished running, it will print that the results have been saved in the terminal output.

## Summary of results

The algorithm's primary objective is to retrieve visually similar images to a specified target image using two pipelines: color histogram comparison and a pretrained VGG16 model with K-NN classification.
 
The results present the most similar images to the target image based on their distances, indicating their level of similarity. A lower distance indicates a higher degree of resemblance, while a higher distance equals greater dissimilarity. The target image itself serves as the reference point, denoted by a distance of 0.0.

For example, using image 0001 of a daffodil from the flower dataset, the color histogram comparison pipeline yields the following results:

|Filename|Distance
|---|---|
|image_0001|0.0|
|image_0928|178.124|
|image_0876|188.548|
|image_0773|190.081|
|image_0142|190.209|
|image_1316|190.222|

![Visualisation of results for the pretrained pipeline](https://raw.githubusercontent.com/SMosegaard/cds-vis/main/assignments/assignment-1/out/target_closest_0001_hist.png)

In comparison, the pretrained pipeline provides the following results:

|Filename|Distance
|---|---|
|image_0001|0.0|
|image_0037|0.133|
|image_0016|0.139|
|image_0036|0.161|
|image_0017|0.162|
|image_0049|0.164|

![Visualisation of results for the pretrained pipeline](https://raw.githubusercontent.com/SMosegaard/cds-vis/main/assignments/assignment-1/out/target_closest_0001_pretrained.png)

The distances presented in the two tables are not directly comparable as they are calculated using different methods and metrics. However, a clear trend does emerge, as the pretrained pipeline yields significantly smaller distances between the target image and retrieved similar images. Additionally, the pretrained method showcases images of the same flower variety as the target, whereas the histogram method retrieves images with varying degrees of similarity, encompassing other yellow flowers as well.

It is a consistent pattern across multiple examples, that the pretrained pipeline outperforms the color histogram comparison method. All results can be found in the ```out``` folder. This indicates the effectiveness and robustness of the pretrained pipeline in identifying visually similar images within the dataset.

In summary, the results provide insights into image similarity and potential patterns within a dataset.

## Discussion

The comparison between the two methods utilized in this project, color histogram comparison and pretrained models (VGG16 and K-NN), sheds light on their respective strengths and limitations in image search.

The color histogram comparison method offers a computationally lightweight approach. By quantifying only the distribution of colors in images, it provides a simple yet effective way of assessing similarity. However, the method struggles to capture more nuanced visual patterns and variations.

In contrast, the pretrained pipeline presents a more sophisticated solution by using advanced deep learning techniques. By extracting high-level features from images and utilizing a classification algorithm, it can recognize similarities and identify subtle variations between complex images. Naturally, this method also has limitations, as it is more computationally heavy and time-consuming.

In conclusion, the first method is less computationally heavy, but struggles to accurately capture differences between objects in complex images, as it only extracts color distributions. The second method has enhanced performance and is capable of extracting complex featureas as edges, shapes, and textures, however at the cost of increased computational resources.

The image search algorithm can be applied to numerous image datasets. The choice between the two methods depends on the specific application. If the data is simple and one just wants a general trends within a dataset, method one might be sufficient. Contrary, method two probably is preferred if the data is more complex.