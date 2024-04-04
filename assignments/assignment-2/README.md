# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
By Sofie Mosegaard, 15-03-2023

This repository is designed to train two multiclass classification models on image data from the ```Cifar10``` dataset and assess their performances using ```scikit-learn```. 

The project has the objective:
1.  To build simple benchmark classifiers on image classification data using ```scikit-learn```
2.   To demonstrate that you can build reproducible pipelines for machine learning projects
3.   To make sure that you can structure repos appropriately

## Data source

The two classification models will be trained on image data from ```Cifar10```, which can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). The data will be loaded automatically throug the main functions in the two python scripts. Additionally, the data has already been spitted into train/test. 

## Repository structure

The repository consists of 2 bash scripts, 1 README.md file, and 2 folders. The folders contains the following:
-   src: consists of two python scripts for Logistic Regression (```LR_classifier.py```) and Neural Network (```NN_classifier.py```) classification. Additionally, there are two python scripts performing GridSearch (```LR_classifier.py```, ```NN_classifier.py```).
-   out: holds the saved results, consisting of classification reports in .txt format, plots of permutation testing, and loss curve of the NN classifier.

## Reproducibility 

1.   Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-2"
```
2.  Navigate into the folder in your terminal.
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```

## Usage

Run the run bash script in the terminal and specify, whether you want to perform GridSearch (--GridSearch / -gs):
```python
$ source run.sh --gs {yes/no}
```
The input will be converted to lowercase, so it makes no difference how it's spelled. Also, 
script to run both classifiers sequentially

Based on the input (i.e., yes/no), the script will perform GridSearch or simply use default parameters. The GridSearch will be performed in other scripts (LR_gridsearch.py, NN_gridsearch.py). The parameters will be tuned through k-fold cross-validation with 5 folds to improve robustness of the model and the tested parameters. In your terminal, it will print the grid results and used parameters like so:
```python
Best Accuracy for 0.30062 using the parameters {'max_iter': 100, 'tol': 0.1}
 mean=0.2886, std=0.005596 using {'max_iter': 100, 'tol': 0.01}
 mean=0.3006, std=0.006379 using {'max_iter': 100, 'tol': 0.1}
 mean=0.2812, std=0.008556 using {'max_iter': 100, 'tol': 1}
 mean=0.2886, std=0.005596 using {'max_iter': 200, 'tol': 0.01}
 mean=0.3006, std=0.006379 using {'max_iter': 200, 'tol': 0.1}
 mean=0.2812, std=0.008556 using {'max_iter': 200, 'tol': 1}
 mean=0.2886, std=0.005596 using {'max_iter': 300, 'tol': 0.01}
 mean=0.3006, std=0.006379 using {'max_iter': 300, 'tol': 0.1}
 mean=0.2812, std=0.008556 using {'max_iter': 300, 'tol': 1}
```
*(This is for the logistic regression tuning)*

Be aware that permutation tuning using GridSearch is very computationally heavy and will take some time to perform.

The best parameters will then be imported to the main script and used to fit the classifier.

If you choose to use the default parameters, the models will use the parameters:
-   LR: {'max_iter': 100, 'tol': 0.1}
-   NN: {'activation': 'logistic', 'learning_rate_init' = 0.001, 'solver': 'adam', 'hidden_layer_sizes': 20}

## Discussion

The 'XX_classifier' scripts will preform preprocessing of the image data (including greyscaling, standardization, and reshaping), training of the two classifiers, and evaluation of the models.

Both classifiers perform quite well with average accuracy scores of 30% for the logistic regression classifier and 35% for the neural network classifier. Both multiclass classification tasks significantly surpasses chance level of 10%. Both models are best at classifiying trucks, ships, and automobiles, wheras they struggle on predicting cats. This could be because they vary greatly in appearances, colors, patterns, sizes, and orientations.

The models were permutation tested to examine whether the obtained accuracy scores are statistically significant. The tests demonstrated, that both models are statistically independed and that the classification accuracies are significantly better than what could be expected by chance. 

