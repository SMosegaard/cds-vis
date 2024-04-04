import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import cifar10
import argparse
from NN_gridsearch import main as grid_search_main 

def parser():
    """
    The user can specify to perform GridSearch or not
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--GridSearch",
                        "-gs",
                        required = True,
                        help = "Perform GridSearch (yes or no)")
    args = parser.parse_args()
    return args

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

# Preprocesses the data - greyscale, scale, and reshape 
def preprocess_data(X_train, X_test):

    X_train_greyed = greyscale(X_train)
    X_test_greyed = greyscale(X_test)

    X_train_scaled = scale(X_train_greyed)
    X_test_scaled = scale(X_test_greyed)

    X_train_scaled_reshape = reshape(X_train_scaled)
    X_test_scaled_reshape = reshape(X_test_scaled)

    return X_train_scaled_reshape, X_test_scaled_reshape


# Function that defines and fits the neural netork classifier to the data
def define_classifier():

    classifier = MLPClassifier(max_iter = 1000,
                                random_state = 123,
                                verbose = True)

    return classifier


def fit_classifier(classifier, X_train, y_train):
    classifier = classifier.fit(X_train, y_train)
    return classifier


# Function that evaluates the trained classifier on new, unseen data
def evaluate_classifier(classifier, X_train, y_train, X_test,  y_test):

    y_pred = NN_classifier.predict(X_test)     # Generate predictions

    # Plot confusion matrix
    metrics.ConfusionMatrixDisplay.from_estimator(NN_classifier,
                                                X_train,
                                                y_train,
                                                cmap = plt.cm.Blues)

    # Change labels from numbers to object names
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Calculate classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names = labels)
    print(classifier_metrics)

    # Save classification report
    filepath_report = "../out/NN_classification_report.txt"
    with open(filepath_report, 'w') as file:
        file.write(classifier_metrics)

    # Plot loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(NN_classifier.loss_curve_)
    plt.title("Loss curve during training for the neural network classifier")
    plt.ylabel('Loss score')
    plt.savefig("../out/NN_loss_curve.png")
    plt.close()


def permutation_test(classifier, X_test, y_test):

    score, permutation_scores, pvalue = permutation_test_score(classifier, X_test, y_test, cv = 5, 
                                                                n_permutations = 50, n_jobs = 1,
                                                                random_state = 123, verbose = True,
                                                                scoring = None)

    n_classes = 10

    plt.figure(figsize = (8, 6))
    plt.hist(permutation_scores, 20, label = 'Permutation scores', edgecolor = 'black')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth = 3,label = 'Classification Score'' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth = 3, label = 'Chance level')
    plt.title("Permutation test neural network classifier")
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.show()
    plt.savefig("out/NN_permutation.png")
    plt.close()


# Function that executes all the functions above in a structered manner on the CIFAR-10 dataset
def main():

    args = parser()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train_scaled_reshape, X_test_scaled_reshape = preprocess_data(X_train, X_test)
    
    if args.GridSearch.lower() == 'yes':
        best_NN_classifier = grid_search_main()
    else:
        best_NN_classifier = define_classifier()

    # Fit classifier
    best_NN_classifier = fit_classifier(best_NN_classifier, X_train_scaled_reshape, y_train)

    # Evaluate classifier
    evaluate_classifier(best_NN_classifier, X_train_scaled_reshape, y_train, X_test_scaled_reshape, y_test)

    # Permutation test for significance
    permutation_test(best_NN_classifier, X_test_scaled_reshape, y_test)

if __name__ == "__main__":
    main()
