# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
By Sofie Mosegaard, 15-03-2023

This repository is designed to train two multiclass classification models on image data from the ```Cifar10``` dataset and assess their performances using ```scikit-learn```. 

The project has the objective:
-   To build simple benchmark classifiers on image classification data using ```scikit-learn```
-   To demonstrate that you can build reproducible pipelines for machine learning projects
-   To make sure that you can structure repos appropriately

### Data source

The two classification models will be trained on image data from ```Cifar10```, which can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). The data will be loaded automatically throug the main functions in the two python scripts. Additionally, the data has already been spitted into train/test. 

### Repository structure

The repository consists of 2 bash scripts, 1 README.md file, and 2 folders. The folders contains the following:
-   src: consists of two python scripts for Logistic Regression (```LR_classifier.py```) and Neural Network ```NN_classifier.py```() classification. 
-   out: holds the saved results in .txt and .png format.

### Reproducibility 

-   Clone the repository: $ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-2"
-   Navigate into the folder in your terminal.
-   Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt: $ source setup.sh
-   Run the run bash script in the terminal to execude the code: $ source run.sh

### Discussion

The scripts will preform preprocessing of the image data (including greyscaling, standardization, and reshaping), training of the two classifiers, and evaluation of the models.

- mention gridsearch

Both classifiers perform quite well with average accuracy scores of 30% for the logistic regression classifier and 35% for the neural network classifier. Both multiclass classification tasks significantly surpasses chance level of 10%. Both models are best at classifiying trucks, ships, and automobiles, wheras they struggle on predicting cats. This could be because they vary greatly in appearances, colors, patterns, sizes, and orientations.

To demonstrate how the models extract information in its attempt to predict, SHAP (Shapley Additive exPlanations) methodology is included. SHAP provides each feature an importance value, indicating how much it contributes to the given prediction.

-  The red areas increase the probability of predicting a certain class, where the blue area decreases the probability


here are the results... what we can see from the classification report (insert) is that the model is perfoming very well on average. However it struggles on predicting dogs, which could be because X and X. This can be visualised (inset SHAP findings)


When cloned, your repository 'assignment 2' will contain two folders. The folder ```src``` consists of two python scripts for Logistic Regression (LR_classifier.py) and Neural Network (NN_classifier.py) classification. Both scripts will preform preprocessing of the image data (including greyscaled, standardized, and reshaped), training of the classifiers, and evaluation of the models. The results (i.e., the classification reports for both models and the loss curve of the neural networks training) will be saved in the folder ```out```.

