import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, GridSearchCV, learning_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import cifar10
import shap

# Convert images to greyscale
def greyscale(X):
    greyscaled_images  = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(X.shape[0]):
        greyscaled_images [i] = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
    return greyscaled_images

# Scale image features
def scale(X):
    scaled_images  = X  / 255.0
    return scaled_images 

# Reshape images to 2D
def reshape(X):
    reshaped_images = X.reshape(-1, 1024)
    return reshaped_images

def preprocess_data(X_train, X_test):
    """
    Preprocesses the data - greyscale, scale, and reshape.

    Parameters:
    X_train : numpy array
        Array of training images in RGB format.
    X_test : numpy array
        Array of testing images in RGB format.

    Returns:
    X_train_processed : numpy array
        Preprocessed training images.
    X_test_processed : numpy array
        Preprocessed testing images.
    """
    X_train_greyed = greyscale(X_train)
    X_test_greyed = greyscale(X_test)

    X_train_scaled = scale(X_train_greyed)
    X_test_scaled = scale(X_test_greyed)

    X_train_scaled_reshape = reshape(X_train_scaled)
    X_test_scaled_reshape = reshape(X_test_scaled)

    return X_train_scaled_reshape, X_test_scaled_reshape


def define_and_fit_classifier(X_train, y_train):
    
    """
    Function that defines and fits the neural netork classifier to the data 
    """

    classifier = LogisticRegression(solver = 'saga',
                                    multi_class = 'multinomial',
                                    random_state = 123,
                                    verbose = True)

    #classifier = classifier.fit(X_train, y_train)

    return classifier


def grid_search(classifier, X_train, y_train):

    param_grid = {'tol': [0.01, 0.1, 1], 'max_iter': [100, 200, 300]}
    
    grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, cv = 5, n_jobs = -1)
    grid_result = grid_search.fit(X_train, y_train)

    print(f'Best Accuracy for {grid_result.best_score_} using the parameters {grid_result.best_params_}')

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4} using {param}')

    return grid_result.best_estimator_


def evaluate_classifier(classifier, X_train_scaled_reshape, y_train, X_test_scaled_reshape,  y_test):

    """
    Function that evaluates the trained classifier on new, unseen data. This includes plotting a confusion
    matrix and calculating a classification report, which will be saved.
    """
    y_pred = classifier.predict(X_test_scaled_reshape) 
            
    metrics.ConfusionMatrixDisplay.from_estimator(classifier,
                                                X_train_scaled_reshape,
                                                y_train,
                                                cmap = plt.cm.Blues)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names = labels)
    print(classifier_metrics)

    filepath_report = "../out/LR_classification_report.txt"
    with open(filepath_report, 'w') as file:
        file.write(classifier_metrics)



# Main function that executes all the functions above in a structered manner on the CIFAR-10 dataset
def main():

    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocess data
    X_train_scaled_reshape, X_test_scaled_reshape = preprocess_data(X_train, X_test)

    # Define and fit classifier
    LR_classifier = define_and_fit_classifier(X_train_scaled_reshape, y_train)

    # Grid search
    best_LR_classifier = grid_search(LR_classifier, X_train_scaled_reshape, y_train)

    # Evaluate classifier
    evaluate_classifier(best_LR_classifier, X_train_scaled_reshape, y_train, X_test_scaled_reshape, y_test)

if __name__ == "__main__":
    main()
