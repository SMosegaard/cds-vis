# Portfolio 3 - Document Classification using Pretrained Image Embeddings
*By Sofie Mosegaard, 12-04-2024*

This repository is designed to perform document classification using Transfer Learning (TL) with a pretrained Convolutional Neural Network (CNN). The project aims to investigate whether document types can be predicted based on appearance rather than content. Leveraging the differences in appearance for example between scientific papers and emails, the repository seeks to exploit these variations for accurate document classification.

The project utilizes the VGG16 model, which is a state-of-the-art CNN architecture. VGG16 has a deep and complex architecture with 16 layers, totaling 134 million trainable parameters. The model expects input tensors of size (224, 224) with 3 RGB channels. Training deep CNNs are very time-consuming and computationally heavy. Luckily, VGG16 is well-suited for transfer learning. TL is the process of applying pretrained models to new classification tasks and data.

Despite CNN's remarkable performances, they are prone to overfitting, particularly when dealing with limited datasets. To enhance model robustness and generalizability, the user will be given the opportunity to implement batch normalization and data augmentation. The batch normalization technique makes the training process more stable through normalization of the layers' inputs by recentering and rescaling, while data augmentation avoids overfitting by increasing the amout of data by augmenting already existing data.

Specifically, the project will conduct document classification using pretrained image embeddings, by doing the following:
1. Data preparation:
    - Load the data and generate labels for all images
    - Preprocess the images using TensorFlow's preprocess_input function
2. Data splitting:
    - Split the data into 80% training and 20% testing sets by stratifying the labels (y). Later, 10% of the training data will used for validation.
    - Scale the input data (X) and perform label binarization on the labels (y)
3. Model definition and compilation:
    - Define and compile the model architecture
    - Optionally, implement batch normalization to mitigate overfitting
4. Hyperparameter tuning:
    - Optionally, conduct GridSearch to tune hyperparameters through k-fold cross-validation with 5 folds to enhance classification accuracy and robustness
    - The number of epochs and batch size will be tuned, as they are anticipated to significantly impact model performance
5.  Model training:
    - Fit the compiled classifier on the training data
    - Optionally, implement data augmentation by generating new data through horizontal flipping and rotation of the original data, which then will be used for model fitting
6. Model evaluation:
    - Evaluate the trained classifier on new, unseen test data
6. Generate results:
    - Generate a classification report and save it for further analysis
    - Plot and save training and validation loss and accuracy curves to visualize model performance
7. Explorative analysis

To gain a better understanding of the code, all functions in the script ```document_classifier.py``` have a short description.

For the purpose of comparing different model adjustments, three parameters were modified: the optimizer, batch normalization, and data augmentation. This resulted in six distinct model architectures, including two baseline models with the adam and SGD optimizers, two implementing batch normalization, and two with data augmentation. The results will be summarized and discussed below. 

## Data source

In this repository, the pretrained VGG16 CNN is transferred and finetuned to classify a new dataset comprising image data of documents from the ```Tobacco3482``` dataset. The dataset comprises 3,842 images in black and white across 10 classes. The different document types represent ADVE, email, form, letter, memo, news, note, report, resume, and scientific paper.

You can download the dataset [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download) and place it in the in folder. Ensure to unzip the data within the folder before executing the code.

## Repository structure

The repository consists of the following elements:

- 2 bash scripts for setup of the virtual environments, installation of requirements, and execution of the code
- 1 .txt file specifying the required packages including versioned dependencies
- 1 README.md file
- 3 folders
    - in: contains data to be processed
    - src: consists of the Python code to be executed
    - out: stores the saved results, i.e., classification reports in .txt format loss curves in .png format
- (Additionally, there is 1 bash script, 1 .txt file, and one python script for the explorative analysis. This analysis will be descriped in the last section)

## Reproducibility

1.   Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-3"
```
2.  Navigate into the folder in your terminal.
```python
$ cd assignment-3
```
3.  Run the setup bash script to create a virtual environment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```

## Usage

Run the run bash script in the terminal and specify, which optimizer you want to use (--optimizer / -o) and whether you want to perform GridSearch (--GridSearch / -gs), batch normalization (--BatchNorm / -bn), and/or data augmentation (--DatAug / -da):
```python
$ source run.sh -o {adam/sgd} -gs {yes/no} -bn {yes/no} -da {yes/no}
```
*Be aware that hyperparameter tuning using GridSearch are computationally heavy and will take some time to perform.*

The inputs will be converted to lowercase, so it makes no difference how it's spelled.

Based on the user input, the script will perform GridSearch or simply use default parameters. The results from the GridSearch will be printed in the terminal output. The best parameters will then be used to fit the classifier. If you choose to use default parameters, the models will use 10 epochs, and a batch size of 32. The selection of number of epochs and batch size was made considering the small size of the dataset and through iterative testing.

Once the script has finished running, it will print that the results have been saved in the terminal output.

## Summary of results

The reported results are based on training the models for 10 epochs with a batch size of 32.

<div align="center">

|classifier|optimizer|accuracy|macro accuracy|weighted accuracy|
|---|---|---|---|---|
baseline|adam|0.69|0.66|0.69|
baseline|sgd|0.46|0.28|0.38|
BatchNorm|adam|0.71|0.66|0.69|
BatchNorm|sgd|0.68|0.63|0.67|
BatchNorm + DatAug|adam|0.63|0.59|0.61|
BatchNorm + DatAug|sgd|0.63|0.59|0.62|

</div>

The training loss and validation accuracy curves were visualized to assess the models training process and performance: 

<div align = "center">

<p float = "left">
    <img src = "https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-3/out/VGG16_losscurve_adam.png" width = "400">
    <img src = "https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-3/out/VGG16_losscurve_BatchNorm_adam.png" width = "400">
    <img src = "https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-3/out/VGG16_losscurve_DatAug_adam.png" width = "400">
</p>

</div>

<div align = "center">

<p float = "left">
    <img src = "https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-3/out/VGG16_losscurve_sgd.png" width = "400">
    <img src = "https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-3/out/VGG16_losscurve_BatchNorm_sgd1.png" width = "400">
    <img src = "https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-3/out/VGG16_losscurve_DatAug_sgd.png" width = "400">
</p>

</div>

In the baseline model trained with the SGD optimizer, the overall accuracy stands at 46%. Certain classes like Email and ADVE exhibit relatively high precision and recall, while others such as Resume and Scientific show significantly low performance metrics. The learning curve demonstrated a good fit for the model. The curve showed a steady decrease in training loss and an increase in training accuracy over the epochs, suggesting effective learning and model optimization. When changing the optimizer to adam, there's a notable improvement in the model's performance, with an average classification accuracy at 69%. However, its learning curve do suggest some overfitting. The model becomes too in learning the training data, so it is not able to generalize to unseen data.

Greater average classification accuracy is observed with the inclusion of batch normalization in the model architecture. This modification leads to a higher accuracy of 68% with SGD optimizer and 71% with adam optimizer. Both reports demonstrate balanced performance across all classes, which suggests a well-optimized and effective model configuration. However, the learning curves of both models suggest some overfitting, especially the model with the adam optimizer.

Finally, models incorporating data augmentation show mixed results. While they both optimizers achieve an average accuracy of 63%, their learning curves do not reflect an optimal training nor fit.

It can be concluded that inclusion of batch normalization and SGD optimizer yields the best results while not overfitting too much.

## Discussion

Several limitations and factors have been identified that may influence the model's effectiveness:

Firstly, the utilization of the data augmentation technique, including horizontal flipping and rotation of 90 degrees, have not yielded significant improvements in performance. It could be that the generated images were too different from the real data, which leads to a steep learning curve and a model struggling to navigate. Besides  implementing batch normalization and data augmentation, where the latter is very dominant, it would have been interesting to implement early stopping when fitting the model. Early stopping also migrates oversitting and minimises loss, however it does not improve robustness and generalizability as the other methods.

The dataset itself possesses challenges as its classes are imbalanced, which is possibly the reason why some classes are extremely difficult for the model to classify. A solution could be to balance all classes using upsampling or downsampling. It is also relevant to consider that the pretrained VGG16 model is trained on specific sets of images. If reports and resumes were not included, it could potentially result in poorer performance even after finetuning.

Additionally, the choice of optimizer plays a crucial role in model training. The models were tested with both SGD and adam optimizers to assess their performance under different optimization strategies. Generally, the adam optimizer performs very well on complex data such as images and is robust to noisy data, which likely contributed to its effectiveness in optimizing the models on the Tobacco3482 dataset.

It is also worth noting that the models were trained for a relatively small number of epochs (10 epochs) and a batch size of 32 due to computational constraints. While the models may not have fully converged within this setup, the results still provide valuable insights into their performance and capabilities. It would be relevant to further explore the potential of the models with longer training.

Which is also why, the project offers the option for hyperparameter tuning on batch size and number of epochs, which has the potential to enhance classification accuracy. However, tuning of hyperparameters is very computationally heavy and should be considered, as potential performance advantages does not necessarily rationalize additional costs. Given the inherent complexity of neural networks, it is very difficult if not impossible to estimate how a CNN trains. Therefore, one could argue for conducting GridSearch in order to have evidence for one's selection of parameters. Considering the nature of the data, the parameter grid for batch size was set to [16, 32, 64] and [10, 15, 20] for the number of epochs, however, these are primarily educated guesses.

Likewise, the method employed for hyperparameter tuning can also be discussed. GridSearch tests all predefined parameters and combinations slavishly, which leads to a long execution time and requires user interference in setting up and estimating relevant paramter values. It would have been relevant to implement Bayesian optimization or Optuna tuning, as they provide a more systematic approach and require less of the user, as the parameter grid only needs to be a range of values. Therefore, Bayesian optimization or Optuna tuning might also be less computationally heavy, as they do not have to exhaustively search through all possible combinations. 

## Exploratory - what does the model see?

While neural networks yield remarkable outcomes, they often operate as black boxes. This makes them difficult to understand what actually happens and thus validate the results. To address this, activation heatmaps can be employed. Heatmaps are a smart tool that unveils the regions in an image that significantly influence the model's predictions.

Heatmaps offer a visual representation of the model's attention by highlighting the influential areas in an image that contribute the most to its classification decision. By superimposing these heatmaps onto the original image, we can interpret the specific features that drive the predictions.

To explain individual predictions, the python script ```src/explainer.py``` has been developed. The script can be executed using the following bash command and specify which image to investigate by writing ```--image / -i``` . If no image is specified, the script will by default use the first image within the first folder, i.e., 'ADVE/0000136188.jpg'.
```python
$ source explainer.sh -i {folder name/xxxx.jpg}
```

Heatmaps will be generated both from the baseline model and the model with the implementation of batch normalization:

<div align = "center">

<img src = "https://github.com/SMosegaard/cds-vis/blob/main/assignments/assignment-3/out/heatmap_Scientific10064589_10064594.jpg.png" width = "800">

</div>

The red-yellow representations highlight the most informative features within the input images, which is what the models have used to form the predictions. In the example, we can see, that when the model classifies an image as a scientific report, the heatmap highlights its characteristic layout and the presence of graphs/equations. This indicates, that these features played an important role in the classification. It can also be seen, that the two heatmaps also looks different, as they represent different architectures. 

In the example, we can see, that when the model classifies an image as a scientific report, the heatmap highlights its characteristic layout and the presence of graphs/equations. This indicates, that these features played an important role in the classification. 

For example, it can be seen in the scientific image that the model focuses on its specific layout with abstract, columns and graphs, where the model bases its prediction about the email based on the very white.

In summary, by employing heatmaps, we can visually explore and explain the behavior of the model. By visualizing the important regions in a given image, the heatmaps offer a glimpse into the model's perception and decision-making process. The methodology becomes more interpretable, as we can see, how the CNNs operates in practise.