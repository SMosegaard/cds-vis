# Assignment 4 - Detecting faces in historical newspapers
*By Sofie Mosegaard, 26-04-2024*

This repository is designed to utilize a pretrained Convolutional Neural Network (CNN) for extracting meaningful information from image data. With the increase in image content in printed media and advancements in technology, exploring this trend through automated face detection in newspapers becomes interesting.

In recent years, face detection has emerged as a crucial technology for a wide range of applications. With the advancements in machine learning and computer vision, face detection algorithms have gained great accuracy and efficiency. The project utilizes the pretrained Multi-Task Cascaded Convolutional Network (MTCNN), which is a deep learning algorithm used for face detection. The model uses a cascading series of neural networks to detect, localise, and align facial features from images. More detailed information about the MTCNN model can be found [here](https://medium.com/@danushidk507/facenet-pytorch-pretrained-pytorch-face-detection-mtcnn-and-facial-recognition-b20af8771144).

This project will specifically investigate the prevelance of images of human faces in historic newspapers over the last 200 years, by exploring the following:
- Load the MTCNN model and initialize a dataframe with columns to store the results.
- Iterate through historic newspapers to detect human faces using the MTCNN model and update the dataframe accordingly.
- Calculate the total number of faces per decade and the percentage of issues with faces per dacade.
- Visualize the percentage of issues with faces per decade across newspapers.
- Save the dataframe and the plot in the out folder.

To get a better understanding of the code, all functions in the script ```main.py``` will have a short descriptive text.

## Data source

In this repository, the pretrained MTCNN will detect faces in newspapers. The dataset is a collection of three historic Swiss newspapers: Gazette de Lausanne (GDL), Impartial (IMP), and Journal de Genève (JDG). The newspapers spans over 4000 issues ranging from 1790s to 2010s. All newspapers are in black and white format and contains metadata in their titles. 

You can download the dataset [here](https://zenodo.org/records/3706863) and place it in the ```in``` folder. Ensure to unzip the data within the folder before executing the code.

## Repository structure

The repository consists of the following elements:
- 2 bash scripts for setup and execution of the code
- 1 README.md file
- 1 .txt file specifying the required packages
- 3 folders
    - in: contains data to be processed
    - src: consists of the Python code to be executed
    - out: stores the saved results, including a .csv file illustrating the number of faces per decade and a plot in .png format.

## Reproducibility

1.   Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-4"
```
2.  Navigate into the folder in your terminal.
```python
$ cd assignment-4
```
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```
4.  Run the run bash script in your terminal:
```python
$ source run.sh
```

## Summary of results

-   Finally, remember your repository should include a writtens summary and interpretation of what you think this analysis might being showing. You do not need to be an expert in the history of printed Swiss media - just describe what you see and what that might mean in this context. Make sure also to mention any possible limitations of your approach!

...
- er der betydeligt flere billeder efter 19th century?? 
- how has the prevelance of images of human faces changed in print media over the last roughly 200 years?

The project aims to highlight the increasement of images of human faces in historic newspapers over 200 years.
...


## Discussion
- To show understanding of how to interpret machine learning outputs

...

still purely predictions, so the model might detect something that is not a face as a face or visa versa.