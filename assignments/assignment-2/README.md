# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
*By Sofie Mosegaard, 15-03-2023*

This repository hosts two pipelines for multiclass image classification: Logistic Regression (LR) and Neural Network (NN). The pipelines utilize supervised machine learning techniques, where the models learn patterns from labeled data to make predictions on unseen data. 

LR and NN are both popular benchmarks machine learning models within the field. In this repository, the models will be compared and offer insights into their respective strengths and weaknesses for multiclass image classification tasks. LR offers a simple yet effective linear classification algorithm, whereas NN, in this case the Multi-Layer Perceptron (MLP) classifier, is a complex feedforward artificial neural network with fully connected neurons. Both models will be implemented using standard scikit-learn pipelines.

Specifically, the project will conduct image classification utilizing the two benchmark machine learning models on image data from the CIFAR-10 dataset, by doing the following:
- Preprocesses the data which includes greyscaling, scaling, and reshaping
- Optionally, conducts GridSearch tuning to tune hyperparameters and enhance classification accuracy
- Defines respectively the LR and NN classifier on tuned or default parameters depending on the user
- Fits the classifier to the training data
- Evaluates the trained classifier on new, unseen test data
- Generates a classification report and saves it for further analysis
- Plots the loss curve during training of the NN classifier to visualize model performance
- Conduct permutation tests to assess classifier significance

To gain a better understanding of the code, all functions in the 'XX_classifier.py' scripts will have short descriptions.

## Data source

In this repository, the two classification models will be trained on image data from CIFAR-10, which is one of the most used dataset for machine learning. The CIFAR-10 dataset consists of 60,000 32x32 colour images in 10 classes. The classes represent airplaines, automobiles, birds, cats, deers, dogs, frogs, horses, ships, and truckes.

You can download the dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html) and place it in the in folder. Ensure to unzip the data within the folder before executing the code.

The data has already been split into training (50,000 images) and test data (1,000 randomly seletec images per class, meaning 10,000 images in total).

## Repository structure

The repository consists of the following elements:

- 2 bash scripts for setup of the virtual environments, installation of requirements, and execution of the code
- 1 .txt file specifying the required packages including versioned dependencies
- 1 README.md file
- 3 folders
    - in: contains data to be processed
    - src: consists of the Python code to be executed. Specifically, two scripts performing classification ('LR_classifier.py', 'NN_classifier.py') and GridSearch ('LR_gridsearch.py', 'NN_gridsearch.py').
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

Based on the input (i.e., yes/no), the script will perform GridSearch or simply use default parameters. The GridSearch will be performed in other scripts. The parameters will be tuned through k-fold cross-validation with 5 folds to improve robustness of the model and the tested parameters. Natrually, the grid of parameters to be tuned can be adjusted in the 'XX_gridsearch.py' sctipts.

-
- *skriv kommentar om hvad, der bliver tunet. Hvorfor er de parametre, udvalgt
-

In your terminal, it will print the grid results and used parameters like so:

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
*(This is the output of the LR tuning)*

The best parameters will then be imported to the main script and used to fit the classifier.

If you choose to use the default parameters, the models will use the parameters:
-   LR: {'max_iter': 100, 'tol': 0.1}
-   NN: {'activation': 'logistic', 'learning_rate_init' = 0.001, 'solver': 'adam', 'hidden_layer_sizes': 100}

*** nævn at NN validates + early stopping
As the classification task is multiclass, the validation split is random. Therefore, it could in theory
only improve the accuracy and robustness of the classes, that are being validated. However, I cant 
if any classes are not included in the validation split
-
validation improved the results by 2% acc and made the classes a bit more balanced.
Also it improved the loss score (from 1.6 to 1.4) and made the model run faster, due to implementaiton of early stopping
***

The bash script for execution of the code will run both classifiers sequentially. If you wish to run a specific model, you can uncomment the corresponding scripts within the run.sh file.

*Be aware that permutation tuning using GridSearch is very computationally heavy and will take some time to perform.*

When the code has finished running, it will print that the results have been saved in the terminal.

## Summary of results

The results present the performance metrics for both the logistic regression and neural network models in a multiclass image classification task when utilizing default parameters:

<div align="center">

||*LR*|precision|recall|f1-score||*NN*|precision|recall|f1-score||support|
|---|---|---|---|---|---|---|---|---|---|---|---|
|airplane||0.29|0.35|0.32|||0.41|0.46|0.43||1000|
|automobile||0.36|0.39|0.37|||0.48|0.51|0.49||1000|
|bird||0.25|0.20|0.22|||0.30|0.39|0.34||1000|
|cat||0.23|0.16|0.19|||0.29|0.19|0.23||1000|
|deer||0.25|0.21|0.23|||0.30|0.31|0.31||1000|
|dog||0.31|0.30|0.31|||0.39|0.33|0.36||1000|
|frog||0.28|0.28|0.28|||0.38|0.40|0.39||1000|
|horse||0.30|0.31|0.31|||0.42|0.50|0.46||1000|
|ship||0.33|0.41|0.37|||0.51|0.47|0.49||1000|
|truck||0.38|0.43|0.41|||0.49|0.41|0.44||1000|
|||||||||||||
|accuracy||||0.30|||||0.40||10000|
|macro avg||0.30|0.30|0.30|||0.40|0.40|0.39||10000|
|weighted avg||0.30|0.30|0.30|||0.40|0.40|0.39||10000|

</div>

The logistic regression model achieved an overall accuracy of 30% on the CIFAR-10 dataset. The model demonstrated strong performance classifying categories such as "truck" and "automobile" with precision scores of 0.38 and 0.36, however it struggled with the categories "cat" and "bird" with notably lower precision scores.

The neural network model produced even better accuracies with an overall accuracy of 40%. Similarly, it struggles on predicting animals especially the category "cat". This could be because they vary greatly in appearances, colors, patterns, sizes, and orientations.

Both multiclass classification tasks significantly surpasses chance level of 10%. To examine whether the obtained accuracy scores are statistically significant, both models were permutation tested. The tests demonstrated, that both models are statistically independed and that the classification accuracies are significantly better than what could be expected by chance. 

<div align="center">

<p float="left">
    <img src="https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-2/out/LR_permutation.png" width="400">
    <img src="https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-2/out/NN_permutation.png" width="400">
</p>

</div>

Finally, the loss curve during training of the NN classifier was visualized to assess the models training process and performance:

<div align="center">

<img src="https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-2/out/NN_loss_curve.png" width="400"/>

</div>

The loss curve indicates a consistent decrease in the loss function as the model undergoes training. This suggests that the model is effectively learning from the training data and gradually improving its ability to make accurate predictions.

... kommenter på validation

## Discussion

The neural network model outperforms logistic regression in the multiclass classification task by achieving higher accuracy and demonstrating more balanced performance across different categories. This might be as a result of the NN model's complex architecture and superiority at learning relationships and patterns within complex data.

However, the NN model is also significantly more computationally heavy to run compared to the simple LR classifier. This raises the question of whether the improvement in accuracy justifies the increased computational resources required for training.

Additionally, the importance of hyperparameter tuning should be considered. While the project includes code for tuning hyperparameters, it is important to conisder whether the potential performance advantages justifies the  the additional costs.
