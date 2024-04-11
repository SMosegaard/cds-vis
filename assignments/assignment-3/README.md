# Assignment 3 - Document classification using pretrained image embeddings
## Transfer learning w/ pretrained CNNs
*By Sofie Mosegaard, 12-04-2024*

This repository is designed to perform document classification using transfer learning with pretrained Convolutional Neural Networks (CNNs). The project aims to investigate whether document types can be predicted based on appearance rather than contents. Leveraging the differences in appearance for example between scientific papers and poems, the repository seeks to exploit these variations for accurate document classification.

#### Pretrained CNN

The project utilizes the VGG16 model, which is a state-of-the-art CNN architecture. VGG16 has a deep and complex architecture with 16 layers, totaling 134 million trainable parameters. The model expects input tensors of size (224, 224) with 3 RGB channels.

#### Transfer Learning

Training deep CNNs are very time-consuming and computationally heavy. However, VGG16 is well-suited for transfer learning. Transfer learning is the process of applying pretrained models to new classification tasks and data.

In this repository, the pretrained VGG16 CNN is transferred and finetuned to classify a new dataset comprising image data of documents from the ```Tobacco3482``` dataset.

#### Mitigating Overfitting

CNNs are prone to overfitting, particularly when dealing with limited datasets. To enhance model robustness and generalizability, batch normalization and data augmentation techniques are implemented. All results will be summarized and discussed below. 

## Data source

The model will be finetuned and trained on image data from the ```Tobacco3482``` dataset, which can be found [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). The data comprises 3842 images in black and white across 10 different document types.

To proceed, download the dataset and place it in the 'in' folder. Note that you'll need to unzip the data within the folder before executing the code (see: Reproducibility)..

## Repository structure

The repository consists of 2 bash scripts, 1 README.md file, 1 txt file, and 3 folders. The folders contains the following:

-   in: for storing input data
-   src: consists of three scipt to achieve the repository's objectives: one baseline model (TransferLearning.py), one implementing batch normalization (TransferLearning_BatchNorm.py), and one incorporating data augmentation (TransferLearning_DatAug.py)
-   out: holds the saved results, including classification reports in .txt format and loss curves in .png format for all models.

## Reproducibility

1.   Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-vis/tree/main/assignments/assignment-3"
```
2.  Navigate into the folder in your terminal.
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```

## Usage

Run the run bash script in the terminal and specify, which optimizer you want to use(--optimizer / -o):
```python
$ source run.sh -o {sgd/adam}
```
The model will then be compiled with the given optimizer (i.e., sgd/adam).

Please note that the bash script will run all three models (baseline, with batch normalization, and with data augmentation) sequentially. If you wish to run only specific models, you can uncomment the corresponding scripts within the run.sh file.

## Discussion

-   Your repository should also include a short description of what the classification report and learning curve show.
-   To show understanding of how to interpret machine learning outputs

The best models were ... based on acc. score and loss curve during training..

The DatAug models didnt perform very well, which might be bacuse, the generated data was too different from the real data. The DatAug method is used to improve robustness, but as they were to different, the learning curve was too steep for the models to learn the patterns.

To increate the robustness and generalizability of the model, batch normalization and augmentation were implemented. 
- difference between techiniques?

additionally, the models will be tested on two optimizers: sgd and adam.
- difference between optimizers?

- the saved results were obtained by training the models on 10 epochs and a batch size of 64. Natrually, the models havent fully learned on so few epochs, but as the models are computationally heavy to run, they still provide an indication of their performance. 