import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import cifar10


# Function that preprocesses the data
def preprocess_data(X_train, X_test):
    
    # Convert training images to grayscale
    X_train_greyed = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    for i in range(X_train.shape[0]):
        X_train_greyed[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)
        
    # Convert test images to grayscale
    X_test_greyed = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    for i in range(X_test.shape[0]):
        X_test_greyed[i] = cv2.cvtColor(X_test[i], cv2.COLOR_RGB2GRAY)

    # Scale features
    X_train_scaled = X_train_greyed / 255.0
    X_test_scaled = X_test_greyed / 255.0

    # Reshape images to 2D (number of images, number of pixels for each flattened image)
    X_train_scaled_reshape = X_train_scaled.reshape(-1, 1024)   # (50000, 1024)
    X_test_scaled_reshape = X_test_scaled.reshape(-1, 1024)     # (10000, 1024)

    return X_train_scaled_reshape, X_test_scaled_reshape


# Function that defines and fits the neural netork classifier to the data
def define_and_fit_classifier(X_train_scaled_reshape, y_train):
    classifier = MLPClassifier(activation = "logistic",
                                solver = "adam",
                                hidden_layer_sizes = (20,),
                                max_iter = 1000,
                                random_state = 123,
                                verbose = True)

    NN_classifier = classifier.fit(X_train_scaled_reshape, y_train)

    return NN_classifier


# Function that evaluates the trained classifier on new, unseen data
def evaluate_classifier(NN_classifier, X_train_scaled_reshape, y_train, X_test_scaled_reshape, y_test):
    y_pred = NN_classifier.predict(X_test_scaled_reshape)     # Generate predictions

    # Plot confusion matrix
    metrics.ConfusionMatrixDisplay.from_estimator(NN_classifier,
                                                X_train_scaled_reshape,
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

    # Evaluate classifier
    evaluate_classifier(NN_classifier, X_train_scaled_reshape, y_train, X_test_scaled_reshape, y_test)

if __name__ == "__main__":
    main()
