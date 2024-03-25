# Portfolio 1 - Building a simple image search algorithm
*By Sofie Mosegaard, 01-03-2024*

This repository is designed to design a simple image search algorithm on flower images. The project primarily uses ```OpenCV``` and ```Pandas```. The output will 

The assignment has the objective:
-   Work with larger datasets of images
-   Extract structured information from image data using ```OpenCV```
-   Quantaitively compare images based on these features, performing *distant viewing*

### Data source

The dataset is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford, and full details of the data can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).

Please download the data and place it in the folder called ```in```. Note, that you will need to unzip the data inside the folder before, you can execude the code as described below (see: Reproducibility). 

### Repository structure

The repository consists of 2 bash scripts, 1 README.md file, and 3 folders. The folders contains the following:
-   in: contains the data. 
-   src: consists of the scipt that will perform the repository objective.
-   out: holds the saved results in a .csv format.

### Reproducibility 

-   Clone the repository: $ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-1"
-   Navigate into the folder in your terminal.
-   Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt: $ source setup.sh
-   Run the run bash script in the terminal to execude the code: $ source run.sh

### Discussion

The image search algoritm will perform as follows on the flowers dataset:

|Filename|Distance
|---|---|
|target|0.0|
|image_0928|178.124|
|image_0773|190.081|
|image_0142|190.209|
|image_0876|188.548|
|image_1316|190.222|

The algorithm's primary objective is to retrieve images that are visually similar to a given target image. By utilizing techniques as histogram comparison, the algorithm assesses the similarity between images based on their color distributions. The results illustrates the images that are the most similar to the target image in distance, which serves as a measure of similarity. A lower distance indicates a higher degree of resemblance, while a higher distance equals greater dissimilarity. The target image itself serves as the reference point, denoted by a distance of 0.0.

In summary, the results provide insights into the similarity between images and potential patterns within the dataset. The image search algorithm can be applied to numerous image datasets.
