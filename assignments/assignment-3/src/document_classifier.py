import os
import tensorflow as tf
from PIL import UnidentifiedImageError
from tensorflow.keras.preprocessing.image import (load_img, img_to_array, ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input, decode_predictions, VGG16)
from tensorflow.keras.layers import (Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
import keras
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier

def parser():
    """
    The user can specify which optimizer to use, whether to perform GridSearch to tune the hyperparameters, and
    to implement batch normalization and/or data augmentation.
    The function will then parse command-line arguments and make them lower case.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer",
                        "-o",
                        required = True,
                        choices = ["adam", "sgd"],
                        help = "Choose optimizer") 
    parser.add_argument("--GridSearch",
                        "-gs",
                        required = True,
                        choices = ["yes", "no"],
                        help = "Perform GridSearch (yes or no)")
    parser.add_argument("--BatchNorm",
                        "-bn",
                        required = True,
                        choices = ["yes", "no"],
                        help = "Perform batch normalization (yes or no)")   
    parser.add_argument("--DatAug",
                        "-da",
                        required = True,
                        choices = ["yes", "no"],
                        help = "Perform data augmentation (yes or no)")                 
    args = parser.parse_args()
    args.optimizer = args.optimizer.lower()
    args.GridSearch = args.GridSearch.lower()
    args.BatchNorm = args.BatchNorm.lower()
    args.DatAug = args.DatAug.lower()
    return args


def load_images(folder_path):
    """
    Loads the data from the specified folder path, generates labels for each image, 
    and preprocesses them for model input.
    The dataset contains certain files, i.e., Thumbs.db, could not be loaded and
    returned the error 'UnidentifiedImageError'. These will simplt be ignored.
    """
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
    """
    Splits the data into training and testing sets (80:20 split) by stratifing y. Normalizes X
    by simply dividing by the maximum possible value and performs label binarization on y.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 123)
    X_train = X_train.astype("float32") / 255.
    X_test = X_test.astype("float32") / 255.
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test) 

    return X_train, X_test, y_train, y_test


def define_model(BatchNorm, Optimizer):
    """
    Defines the model architecture. First, the VGG16 model is loaded from TensorFlow without the classification
    layers. The convolutional layers are marked as not trainable to retain their pretrained weights. The
    user specifies whether the model should be defined with or without batch normalization. Subsequently, a
    new fully connected layer with ReLU activation is added followed by an output layer with softmax
    activation for multi-class classification. Finally, the model will be compiled with the specified
    optimizer and respective learning rate.
    """
    model = VGG16(include_top = False, pooling = 'avg', input_shape = (224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    if BatchNorm == "no":
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(128, activation = 'relu')(flat1)
        output = Dense(10, activation = 'softmax')(class1)

    elif BatchNorm == "yes":
        flat1 = Flatten()(model.layers[-1].output)
        bn = BatchNormalization()(flat1)
        class1 = Dense(128, activation='relu')(bn)
        output = Dense(10, activation='softmax')(class1)
    
    model = Model(inputs = model.inputs, outputs = output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, 
                                                                decay_steps = 10000, decay_rate = 0.9)
    if Optimizer == "adam":
        optimizer = Adam(learning_rate = lr_schedule)
    if Optimizer == "sgd":
        optimizer = SGD(learning_rate = lr_schedule)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model


def sklearn_object(model):
    """
    Convert the model from a KerasClassifier to an object, that can be used in a scikit-learn pipeline.
    """
    model = KerasClassifier(model = model, verbose = 0)
    return model


def data_generator():
    """
    The function creates an image data generator with ImageDataGenerator from TensorFlow.
    The implemented data augmentation involves settings as horizontal flipping and rotation.
    A validation split is set to 10%.
    """
    datagen = ImageDataGenerator(horizontal_flip = True, 
                                rotation_range = 90,
                                validation_split = 0.1)
    return datagen


def fit_model(model, X_train, y_train, DatAug, batchsize = 32, epochs = 10):
    """
    The function fits the defined and compiled model to the training data.

    If the parameters have not been gridsearch to obtain the best parameters, the model will use a batch
    size of 32 and 10 epochs as default. 

    The user specifies whether to implement data augmentation...
    Fits the compiled model to the training data with or without data augmentation and returns the training history.

    early stopping implemented to minimise loss and avoid overfitting. Monitors val_loss
    After 3 epochs with no improvement, the training will be stopped
    
    When the model is fitted the function returns the fitted model, H.
    """

    callback = [keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 3)]

    if DatAug == "no":
        H = model.fit(X_train, y_train, 
                    validation_split = 0.1,
                    batch_size = batchsize,
                    epochs = epochs,
                    callbacks = callbacks)
    
    elif DatAug == "yes":
        DatAug = data_generator()
        datagen.fit(X_train)
        H = model.fit(datagen.flow(X_train, y_train, batch_size = batchsize),
                                    validation_data = datagen.flow(X_train, y_train, 
                                                                    batch_size = batchsize,
                                                                    subset = "validation"),
                                                                    epochs = epochs,
                                                                    callbacks = callbacks) 
    return H


def plot_history(H, epochs, outpath):
    """
    Plots the training and validation loss and accuracy curves and saves the plot to a specified outpath.
    """
    plt.figure(figsize = (12,6))

    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label = "training loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label = "validation loss", linestyle = ":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label = "training accuracy")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label = "validation accuracy", linestyle = ":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(outpath)
    plt.show()


def evaluate(model, X_test, y_test, H, BatchSize, epochs, BatchNorm, DatAug, Optimizer):
    """
    Evaluates the model on the test data, generates classification reports, and saves the results.
    """
    label_names = ["ADVE", "Email", "Form", "Letter", "Memo", "News", "Note", "Report", "Resume", "Scientific"]
    predictions = model.predict(X_test, batch_size = BatchSize)
    #y_preds = model.predict(X_test)
    classifier_metrics = classification_report(y_test.argmax(axis = 1),
                                               predictions.argmax(axis = 1),
                                               target_names = label_names)

    if BatchNorm == "yes":
        if DatAug == "yes": 
            model_param = "BatchNorm_DatAug"
        elif DatAug == "no":
            model_param = "BatchNorm"
    elif BatchNorm == "no": 
        if DatAug == "yes":
            model_param = "DatAug"
        elif DatAug == "no": 
            model_param = "baseline"

    filepath_metrics = open(f'out/{model_param}_metrics_{Optimizer}.txt', 'w')
    filepath_metrics.write(classifier_metrics)
    filepath_metrics.close()

    plot_history(H, epochs, f"out/{model_param}_losscurve_{Optimizer}.png")

    return print("Results have been saved to the out folder")




def main():
    
    args = parser()

    folder_path = os.path.join("../../../../cds-vis-data/Tobacco3482") # ("in/Tobacco3482")

    X, y = load_images(folder_path)
    X_train, X_test, y_train, y_test = data_split(X, y)
    
    # define model - BatchNorm yes/no
    model = define_model(args.BatchNorm, args.Optimizer)

    # grid search
    # fit
    if args.GridSearch == 'yes':
        model = sklearn_object(model)
        model, batchsize, epochs = gridsearch(model, X_train, y_train, batchsize = batchsize, epochs = epoch) #DatAug
        H = fit_model(model, X_train, y_train, args.DatAug, batchsize = batchsize, epochs = epochs)
    else:
        H = fit_model(odel, X_train, y_train, args.DatAug, batchsize = batchsize, epochs = epochs)

  
    # evaluate
    evaluate(model, X_test, y_test, H, BatchSize, epochs, args.BatchNorm, args.DatAug, args.Optimizer)

if __name__ == "__main__":
    main()