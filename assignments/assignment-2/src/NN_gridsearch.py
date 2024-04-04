import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow
from tensorflow.keras.datasets import cifar10


def greyscale(X):
    greyscaled_images  = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(X.shape[0]):
        greyscaled_images [i] = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
    return greyscaled_images

def scale(X):
    scaled_images  = X  / 255.0
    return scaled_images 

def reshape(X):
    reshaped_images = X.reshape(-1, 1024)
    return reshaped_images

def preprocess_data(X_train, X_test):

    X_train_greyed = greyscale(X_train)
    X_test_greyed = greyscale(X_test)

    X_train_scaled = scale(X_train_greyed)
    X_test_scaled = scale(X_test_greyed)

    X_train_scaled_reshape = reshape(X_train_scaled)
    X_test_scaled_reshape = reshape(X_test_scaled)

    return X_train_scaled_reshape, X_test_scaled_reshape


def define_classifier():
    classifier = MLPClassifier(max_iter = 1000,
                                random_state = 123,
                                verbose = True)
    return classifier


def grid_search():

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train_scaled_reshape, X_test_scaled_reshape = preprocess_data(X_train, X_test)
    print(X_train_scaled_reshape)

    NN_classifier = define_classifier()

    param_grid = {'activation': ('logistic', 'relu'),
                'solver': ('adam', 'sgd'),
                'learning_rate_init': [0.01, 0.001],
                'hidden_layer_sizes': [20, 50, 100]}
    
    grid_search = GridSearchCV(estimator = NN_classifier, param_grid = param_grid, cv = 5, n_jobs = -1)
    grid_result = grid_search.fit(X_train_scaled_reshape, y_train)

    print(f'Best Accuracy for {grid_result.best_score_} using the parameters {grid_result.best_params_}')

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4} using {param}')

    best_estimator = grid_result.best_estimator_
    return best_estimator


def main():
    best_estimator = grid_search()
    return best_estimator

if __name__ == "__main__":
    best_estimator = grid_search()
