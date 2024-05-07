import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import tensorflow
from tensorflow.keras.datasets import cifar10


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


def define_model_BatchNorm():
    model = VGG16(include_top = False, pooling = 'avg', input_shape = (224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(128, activation='relu')(bn)
    output = Dense(10, activation='softmax')(class1)
    model = Model(inputs = model.inputs, outputs = output)
    return model


def define_model_baseline():
    model = VGG16(include_top = False, pooling = 'avg', input_shape = (224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu')(flat1)
    output = Dense(10, activation = 'softmax')(class1)
    model = Model(inputs = model.inputs, outputs = output)
    return model



def compile_model(model):
    """
    Compiles the model with the specified optimizer.
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01,
                                                                 decay_steps = 10000,
                                                                 decay_rate = 0.9)
    
    adam = Adam(learning_rate = lr_schedule) 0.001,

    model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model



def grid_search():

    folder_path = os.path.join("../../../../cds-vis-data/Tobacco3482") # ("in/Tobacco3482")
    X, y = load_images(folder_path)
    X_train, X_test, y_train, y_test = data_split(X, y)
    
    # define model
    if args.BatchNorm == "yes":
        model = define_model_BatchNorm()
    else:
        model = define_model_baseline()

    model = compile_model()
    model = KerasClassifier(model = model, verbose = 1)

    param_grid = {'optimizer' = ['sgd', 'adam'], 'learning_rate' = [0.1, 0.01, 0.001, 0.0001]}
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, n_jobs = -1,
                                scoring = 'accuracy', verbose = 3)

    grid_result = grid_search.fit(X_train, y_train)
    grid_result = grid_search.fit_model(X_train, y_train)


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
