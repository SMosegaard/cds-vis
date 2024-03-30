import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import cifar10

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
def define_and_fit_classifier(X_train, y_train):

    classifier = MLPClassifier(max_iter = 1000,
                                random_state = 123,
                                verbose = True)

    #classifier = classifier.fit(X_train, y_train)

    return classifier

# GridsearchCV
def grid_search(classifier, X_train, y_train):

    param_grid = {'activation': ('logistic', 'relu'),
                'solver': ('adam', 'sgd'),
                'learning_rate_init': [0.01, 0.001],
                'hidden_layer_sizes': [20, 50]}
    
    grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, cv = 5, n_jobs = -1)
    grid_result = grid_search.fit(X_train, y_train)

    print(f'Best Accuracy for {grid_result.best_score_} using the parameters {grid_result.best_params_}')

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4} using {param}')

    return grid_result.best_estimator_

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


def shap_values(classifier, X_train, X_test):
    #explainer = shap.Explainer(classifier.predict_proba, X_test)
    explainer = shap.GradientExplainer(classifier, X_train[:5])
    shap_values = explainer.shap_values(X_test)
    shap.image_plot([shap_values[i] for i in range(10)], X_test[:5])
    
    plt.savefig("../out/LR_SHAP.png")

# Function that executes all the functions above in a structered manner on the CIFAR-10 dataset
def main():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocess data
    X_train_scaled_reshape, X_test_scaled_reshape = preprocess_data(X_train, X_test)
    
    # Define and fit classifier
    NN_classifier = define_and_fit_classifier(X_train_scaled_reshape, y_train)

    # GridSearch
    best_NN_classifier = grid_search(NN_classifier, X_train_scaled_reshape, y_train)

    # Evaluate classifier
    evaluate_classifier(NN_classifier, X_train_scaled_reshape, y_train, X_test_scaled_reshape, y_test)


if __name__ == "__main__":
    main()
