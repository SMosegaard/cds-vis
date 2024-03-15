


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

def model:
    Sequential()
    model.add(Dense(256, input_shape = (784,), activation = "relu"))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))

    return model

def define_and_fit_classifier(X_train, y_train):
    
    sgd = SGD(0.01) # Learning rate (0.01 = default)
    model.compile(loss = "categorical_crossentropy", 
                optimizer = sgd, 
                metrics = ["accuracy"]) # Optimize based on accuracy
    
    history = model.fit(X_train, y_train, 
                    validation_split = 0.1,
                    epochs = 10, 
                    batch_size = 32)

    return history, model


# Evaluate
def evaluate():
    
    predictions = model.predict(X_test, batch_size = 32)

# Function that executes all the functions above in a structered manner on the CIFAR-10 dataset
def main():
    # Load the dataset
    data, labels = fetch_openml('mnist_784', version = 1, return_X_y = True)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data, labels)

    # Define and fit classifier

    # Evaluate classifier


if __name__ == "__main__":
    main()