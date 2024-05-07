import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import permutation_test_score
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import cifar10
import argparse
from LR_gridsearch import main as grid_search_main 


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

def greyscale(X):
    """ Convert images to greyscale """
    greyscaled_images  = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(X.shape[0]):
        greyscaled_images [i] = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
    return greyscaled_images


def scale(X):
    """ Scale image features """
    scaled_images  = X  / 255.0
    return scaled_images 


def reshape(X):
    """ Reshape images to 2D """
    reshaped_images = X.reshape(-1, 1024)
    return reshaped_images

def preprocess_data(X_train, X_test):
    """
    Preprocesses the data, which includes greyscaling, scaling, and reshaping.
    """
    X_train_greyed = greyscale(X_train)
    X_test_greyed = greyscale(X_test)

    X_train_scaled = scale(X_train_greyed)
    X_test_scaled = scale(X_test_greyed)

    X_train_scaled_reshape = reshape(X_train_scaled)
    X_test_scaled_reshape = reshape(X_test_scaled)

    return X_train_scaled_reshape, X_test_scaled_reshape


def define_classifier():
    """
    Function that defines LR classifier
    """
    classifier = LogisticRegression(tol = 0.1,
                                    max_iter = 100,
                                    solver = 'saga',
                                    multi_class = 'multinomial',
                                    random_state = 123,
                                    verbose = True)
    return classifier


def fit_classifier(classifier, X_train, y_train):
    """
    Function that fits the LR classifier to the data
    """
    classifier = classifier.fit(X_train, y_train)
    return classifier


def evaluate_classifier(classifier, X_train, y_train, X_test,  y_test, outpath):
    """
    Function that evaluates the trained classifier on new, unseen data. This includes plotting a confusion
    matrix and calculating a classification report, which will be saved.
    """
    y_pred = classifier.predict(X_test) 
            
    metrics.ConfusionMatrixDisplay.from_estimator(classifier,
                                                X_train,
                                                y_train,
                                                cmap = plt.cm.Blues)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names = labels)
    print(classifier_metrics)

    with open(outpath, 'w') as file:
        file.write(classifier_metrics)
    return print("The classification report has been saved to the out folder")


def permutation_test(classifier, X_test, y_test, outpath):

    score, permutation_scores, pvalue = permutation_test_score(classifier, X_test, y_test, cv = 5, 
                                                                n_permutations = 100, n_jobs = -1,
                                                                random_state = 123, verbose = True,
                                                                scoring = None)

    n_classes = 10

    plt.figure(figsize = (8, 6))
    plt.hist(permutation_scores, 20, label = 'Permutation scores', edgecolor = 'black')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth = 3,label = 'Classification Score'' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth = 3, label = 'Chance level')
    plt.title("Permutation test logistic regression classifier")
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.savefig(outpath)
    plt.show()
    return print("The permutation test has been saved to the out folder")


def main():
    
    args = parser()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train_scaled_reshape, X_test_scaled_reshape = preprocess_data(X_train, X_test)

    if args.GridSearch.lower() == 'yes':
        best_LR_classifier = grid_search_main()
    else:
        best_LR_classifier = define_classifier()
    
    best_LR_classifier = fit_classifier(best_LR_classifier, X_train_scaled_reshape, y_train)

    evaluate_classifier(best_LR_classifier, X_train_scaled_reshape, y_train, X_test_scaled_reshape,
                        y_test, "out/LR_classification_report.txt")

    permutation_test(best_LR_classifier, X_test_scaled_reshape, y_test, "out/LR_permutation.png")

if __name__ == "__main__":
    main()