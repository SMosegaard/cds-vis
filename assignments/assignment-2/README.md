# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
By Sofie Mosegaard, 15-03-2023

This assignment is designed to train two multiclass classification models on image data from the ```Cifar10``` dataset and assess their performances using ```scikit-learn```. More information about the ```Cifar10``` dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html)

The assignment has the objective:
-   To ensure that you can use ```scikit-learn``` to build simple benchmark classifiers on image classification data
-   To demonstrate that you can build reproducible pipelines for machine learning projects
-   To make sure that you can structure repos appropriately

## Installation and requirements
-   Clone the repository: $ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-2"
-   Select Python 3 kernel
-   Install the required packages (cv2, numpy, scikit-learn, matplot, tensorflow)

## Usage
When cloned, your repository 'assignment 2' will contain two folders. The folder ```src``` consists of two python scripts for Logistic Regression (LR_classifier.py) and Neural Network (NN_classifier.py) classification. Both scripts will preform preprocessing of the image data (including greyscaled, standardized, and reshaped), training of the classifiers, and evaluation of the models. The results (i.e., the classification reports for both models and the loss curve of the neural networks training) will be saved in the folder ```out```.
