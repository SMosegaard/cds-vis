# Portfolio 1 - Building a simple image search algorithm
*By Sofie Mosegaard, 01-03-2024*

This repository aims at designing a simple image search algorithm. The algorithm allows its users to retrive similar images to a given target image and potentially unveiling patterns and similarities within a dataset.

Specifically, the project will image search on flower images. In order to extract the five most similar images to a given target image, the project offers two methods:

1. Color histogram comparison:
    - Initiates a dataframe to store image filenames and corresponding distances
    - Extracts a normalized histogram encompassing all color channels for the target image using OpenCV
    - Iterates through all images, calculating and normalizing histograms
    - Compares the normalized histogram to that of the target image
    - Updates the dataframe with the five most similar images to the target
    - Saves a dataframe containing the target image, the most similar images, and their distances
    - Visualises the target image and its five most similar images

2. Pretrained CNN VGG16 model and K-Nearest Neighbours:
    - Loads the VGG16 model and creates a database of extracted features from images
    - Utilizes a K-Nearest Neighbors classifier to find the nearest neighbors for the target image
    - Saves a dataframe with the target image, the most similar images, and their distances
    - Visualises the target image and its five most similar images

To gain a better understanding of the code, all functions in the script ```src.py``` will have a short descriptive text.

### Data source

The dataset is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford. All images are in color.

You can download the dataset [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) and place it in the ```in``` folder. Ensure to unzip the data within the folder before executing the code.

### Repository structure

The repository consists of the following elements:

- 2 bash scripts for setup of the virtual environments, installqtion of requirements, and execution of the code
- 1 .txt file specifying the required packages
- 1 README.md file
- 3 folders
    - in: contains data to be processed
    - src: consists of the Python code to be executed
    - out: stores the saved results in a .csv format

### Reproducibility 

-   Clone the repository: $ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-1"
-   Navigate into the folder in your terminal.
-   Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt: $ source setup.sh
-   Run the run bash script in the terminal to execude the code: $ source run.sh

1.   Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-1"
```
2.  Navigate into the folder in your terminal.
```python
$ cd assignment-1
```
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```

## Usage

Run the run bash script in the terminal with the required input information (```--method / -m``` and ```--target / -t```). 

You can specify the pipeline by writing ```--method / -m```. You can either execute the image search algorithm using color histograms (enter ```hist```) or a pretrained CNN and K-Nearst Neighbour (enter ```pretrained```).

Additionally, you must provide a target image, that will form the basis of the image search. If none specified, the code will by default use image 'image_0001.jpg'.

```python
$ source run.sh --method {hist/pretrained} --target {'image_xxxx.jpg'}
```
The code will then execude image search utilizing the given method and retrieve similar images to the provided input, target image.

## Summary of results

The image search algoritm will perform as follows on the flowers dataset:

|Filename|Distance
|---|---|
|target|0.0|
|image_0928|178.124|
|image_0773|190.081|
|image_0142|190.209|
|image_0876|188.548|
|image_1316|190.222|

In this example, the target image is a daffodils... 
The most similar images are also yelllow....


The algorithm's primary objective is to retrieve images that are visually similar to a given target image. By utilizing histogram comparison, the algorithm assesses the similarity between images based on their color distributions. The results illustrates the images that are the most similar to the target image in distance, which serves as a measure of similarity. A lower distance indicates a higher degree of resemblance, while a higher distance equals greater dissimilarity. The target image itself serves as the reference point, denoted by a distance of 0.0.

In summary, the results provide insights into the similarity between images and potential patterns within the dataset. The image search algorithm can be applied to numerous image datasets.


### Discussion

discussion of results
difference between two methods
limitations

...

The two methods employed in this project, color histogram comparison and pretrained models (VGG16 and K-NN), offer distinct approaches to image similarity search.

The first method 
- less computationally heavy
- might struggle to capture more complex patterns

Second method
- is better, but at the same time it requires substantial more computational resources --> therefore, it might not be "worth"

Limitations
- both have flaws --> method 1 may struggle a bit, but method 2 is more expensive...
