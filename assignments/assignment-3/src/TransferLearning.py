import os
import argparse
import tensorflow as tf
from PIL import UnidentifiedImageError
from tensorflow.keras.preprocessing.image import (load_img, img_to_array, ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input, decode_predictions, VGG16)
from tensorflow.keras.layers import (Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def parser():
    """
    The user can specify which optimizer to use
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer",
                        "-o",
                        required = True,
                        help = "Choose sgd or adam optimizer")
    args = parser.parse_args()
    return args


def load_images(folder_path):

    list_of_images = [] 
    list_of_labels = []
    
    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path  = os.path.join(folder_path, subfolder)
        
        for file in os.listdir(subfolder_path):
            individual_filepath = os.path.join(subfolder_path, file)
            
            try:
                image = load_img(individual_filepath, target_size = (224, 224))
                image = img_to_array(image)
                list_of_images.append(image)

                label = subfolder_path.split("/")[-1]
                list_of_labels.append(label)

            except (UnidentifiedImageError):
                print(f"Skipping {individual_filepath}")
        
    array_of_images = np.array(list_of_images)
    X = preprocess_input(array_of_images)
    y = list_of_labels
    
    return X, y


def data_split(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 123)
    X_train = X_train.astype("float32") / 255.
    X_test = X_test.astype("float32") / 255.
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test) 

    return X_train, X_test, y_train, y_test


def define_model():

    model = VGG16(include_top = False, pooling = 'avg', input_shape = (224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu')(flat1)
    output = Dense(10, activation = 'softmax')(class1)

    model = Model(inputs = model.inputs, outputs = output)

    return model



def compile_model(model, optimizer):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01,
                                                                 decay_steps = 10000,
                                                                 decay_rate = 0.9)
    
    sgd = SGD(learning_rate = lr_schedule)
    adam = Adam(learning_rate = lr_schedule)

    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


def fit_model(model, X_train, y_train):

    H = model.fit(X_train, y_train, 
                  validation_split = 0.1,
                  batch_size = 128,
                  epochs = 10,
                  verbose = 1)

    return H


def plot_history(H, epochs, outpath):
    plt.figure(figsize = (12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label = "train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label = "val_loss", linestyle = ":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label = "train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label = "val_acc", linestyle = ":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig(outpath)


def evaluate(X_test, y_test, model, H, optimizer):
    
    label_names = ["ADVE", "Email", "Form", "Letter", "Memo", "News", "Note", "Report", "Resume", "Scientific"]

    predictions = model.predict(X_test, batch_size = 32)

    classifier_metrics = classification_report(y_test.argmax(axis = 1),
                                               predictions.argmax(axis = 1),
                                               target_names = label_names)

    filepath_metrics = open(f'out/VGG16_metrics_{optimizer}.txt', 'w')
    filepath_metrics.write(classifier_metrics)
    filepath_metrics.close()

    plot_history(H, 10, f"out/VGG16_losscurve_{optimizer}.png")

    return print("Results have been saved to the out folder")


def main():
    
    args = parser()

    folder_path = os.path.join("../../../../cds-vis-data/Tobacco3482") # ("in/Tobacco3482")

    X, y = load_images(folder_path)
    X_train, X_test, y_train, y_test = data_split(X, y)
    
    model = define_model()

    model = compile_model(model, args.optimizer)

    H = fit_model(model, datagen, X_train, y_train)

    evaluate(X_test, y_test, model, H, args.optimizer)

if __name__ == "__main__":
    main()