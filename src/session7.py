from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

# Function that preprocesses the data
def preprocess_data(data, labels):
    
    # Normalise data
    data = data.astype("float")/255.0

    # train:test split data
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size = 0.2)

    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    return X_train, X_test, y_train, y_test


# Define neural network
def define_model():
    model = Sequential()
    model.add(Dense(256, input_shape = (784,), activation = "relu"))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))

    return model

def compile_and_fit_classifier(model, X_train, y_train):
    sgd = SGD(0.01) # Learning rate (0.01 = default)
    model.compile(loss = "categorical_crossentropy", 
                optimizer = sgd, 
                metrics = ["accuracy"]) # Optimize based on accuracy
    
    classifier = model.fit(X_train, y_train, 
                    validation_split = 0.1,
                    epochs = 10, 
                    batch_size = 32)

    return classifier


# Evaluate
def evaluate_classifier(model, X_test, y_test, lb):
    predictions = model.predict(X_test, batch_size = 32)

    classifier_metrics = classification_report(y_test.argmax(axis=1), 
                            predictions.argmax(axis=1), 
                            target_names=[str(x) for x in lb.classes_])
    print(classifier_metrics)

    filepath_report = "../output/session7_classification_report.txt"
    with open(filepath_report, 'w') as file:
        file.write(classifier_metrics)


# Function that executes all the functions above in a structered manner on the CIFAR-10 dataset
def main():
    # Load the dataset
    data, labels = fetch_openml('mnist_784', version = 1, return_X_y = True)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data, labels)

    # Define model
    model = define_model()

    # Compile and fit classifier
    classifier = compile_and_fit_classifier(model, X_train, y_train)
    
    # Evaluate classifier
    lb = LabelBinarizer()
    lb.fit(labels)
    evaluate_classifier(model, X_test, y_test, lb)


if __name__ == "__main__":
    main()