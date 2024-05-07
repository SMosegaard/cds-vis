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

The repository consists of the following elements:

- 2 bash scripts for setup of the virtual environments, installation of requirements, and execution of the code
- 1 .txt file specifying the required packages including versioned dependencies
- 1 README.md file
- 3 folders
    - in: contains data to be processed
    - src: consists of the Python code to be executed. Specifically, two scripts performing classification (LR_classifier.py, NN_classifier.py) and GridSearch (LR_gridsearch.py, NN_gridsearch.py).
    - out: stores the saved results, i.e., classification reports in .txt format, plots of permutation testing, and loss curve of the NN classifier.

## Reproducibility 

1.   Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-2"
```
2.  Navigate into the folder in your terminal.
```python
$ cd assignment-2
```
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```

## Usage

Run the run bash script in the terminal and specify, whether you want to perform GridSearch (--GridSearch / -gs):
```python
$ source run.sh --gs {yes/no}
```
The input will be converted to lowercase, so it makes no difference how it's spelled.

Based on the input (i.e., yes/no), the script will perform GridSearch or simply use default parameters. The GridSearch will be performed in other scripts. The parameters will be tuned through k-fold cross-validation with 5 folds to improve robustness of the model and the tested parameters. In your terminal, it will print the grid results and used parameters like so:

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

The best parameters will then be imported to the main script and used to fit the classifier.

Be aware that permutation tuning using GridSearch is very computationally heavy and will take some time to perform.

If you choose to use the default parameters, the models will use the parameters:
-   LR: {'max_iter': 100, 'tol': 0.1}
-   NN: {'activation': 'logistic', 'learning_rate_init' = 0.001, 'solver': 'adam', 'hidden_layer_sizes': 20}

Please note, that the script will run both classifiers sequentially. If you wish to run a specific model, you can uncomment the corresponding scripts within the run.sh file.

## Summary of results

it is possible to adjust the parameters to do grid search on in the gridsearch.py scripts

results from the best paramters found from the grid search, which is also the default parameters


|LR|NN |
|---|---|---|---|---|---|---|---|---|
||precision|recall|f1-score||precision|recall|f1-score|support|
|---|---|---|---|---|---|---|---|---|
|airplane|0.29|0.35|0.32||0.41|0.46|0.43|1000|
|automobile|0.36|0.39||0.37|0.48|0.51|0.49|1000|
|bird|0.25|0.20|0.22||0.30|0.39|0.34|1000|
|cat|0.23|0.16|0.19||0.29|0.19|0.23|1000|
|deer|0.25|0.21|0.23||0.30|0.31|0.31|1000|
|dog|0.31|0.30|0.31||0.39|0.33|0.36|1000|
|frog|0.28|0.28|0.28||0.38|0.40|0.39|1000|
|horse|0.30|0.31|0.31||0.42|0.50|0.46|1000|
|ship|0.33|0.41|0.37||0.51|0.47|0.49|1000|
|truck|0.38|0.43|0.41||0.49|0.41|0.44|1000|
||||||||||
|accuracy|||0.30||||0.40|10000|
|macro avg|0.30|0.30|0.30||0.40|0.40|0.39|10000|
|weighted avg|0.30|0.30|0.30||0.40|0.40|0.39|10000|


NN:

||precision|recall|f1-score|support|
|---|---|---|---|---|
|airplane|0.40|0.29|0.33|1000|
|automobile|0.46|0.49|0.43|1000|
|bird|0.28|0.30|0.29|1000|
|cat|0.27|0.12|0.17|1000|
|deer|0.31|0.22|0.26|1000|
|dog|0.40|0.30|0.34|1000|
|frog|0.28|0.49|0.36|1000|
|horse|0.30|0.54|0.39|1000|
|ship|0.50|0.42|0.45|1000|
|truck|0.43|0.46|0.44|1000|
||||||1000|
|accuracy|||0.35|10000|
|macro avg|0.36|0.35|0.35|10000|
|weighted avg|0.36|0.35|0.35|10000|

![Loss curve NN](https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-2/out/NN_loss_curve.png)

![Permutation test LR](https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-2/out/LG_permutation.png) ![Permutation test NN](https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-2/out/NN_permutation.png) 


 |  


## Discussion

The 'XX_classifier' scripts will preform preprocessing of the image data (including greyscaling, standardization, and reshaping), training of the two classifiers, and evaluation of the models.

Both classifiers perform quite well with average accuracy scores of 30% for the logistic regression classifier and 35% for the neural network classifier. Both multiclass classification tasks significantly surpasses chance level of 10%. Both models are best at classifiying trucks, ships, and automobiles, wheras they struggle on predicting cats. This could be because they vary greatly in appearances, colors, patterns, sizes, and orientations.

The models were permutation tested to examine whether the obtained accuracy scores are statistically significant. The tests demonstrated, that both models are statistically independed and that the classification accuracies are significantly better than what could be expected by chance. 

