# Assignment 3 - Document classification using pretrained image embeddings
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

## Summary of results

The reported results are based on training the models for 10 epochs with a batch size of 32.

In the baseline model trained with the SGD optimizer, the overall accuracy stands at 48%. Certain classes like Email and Memo exhibit relatively high precision and recall, while others such as Report and Resume show significantly low performance metrics. When changing the optimizer to Adam, there's a notable improvement in the model's performance, with an average classification accuracy at 69%. Both learning curves demonstrated a good fit for the model. The curves showed a steady decrease in training loss and an increase in training accuracy over the epochs, suggesting effective learning and model optimization

Greater average classification accuracy is observed with the inclusion of batch normalization in the model architecture. This modification leads to a higher accuracy of 71% with SGD optimizer and 76% with Adam optimizer. Both reports demonstrates balanced performance across all classes, which suggests a well-optimized and effective model configuration. However, the learning curve of the batch normalized model with Adam optimizer do suggest overfitting. It can be seen, that the model becomes to specialized in learning the training data, so it is not able to generalize to new data.

Finally, models incorporating data augmentation show mixed results. While they both optimizers achieve an average accuracy of 62% and exhibit some improvements over the baseline, their learning curves do not reflect the same good fit.

It can be concluded, that inclusion of batch normalization and SGD optimizer yields the best reulsts while not overfitting.

## Discussion

Several factors have been identified that may influence the model's effectiveness:

Firstly, the utilization of the data augmentation technique, including horizontal flipping and rotation of 90 degrees, have not yielded significant improvements in performance. It could be that the generated images were too different from the real data, which leads to a steep learning curve and a model struggeling to navigate.

The dataset itself posses challenges as its classes are imbalanced, which is possibly the reason why some classes are extremely difficult for the model to classify. A solution could be to balance all classes using upsampling or downsampling. It is also relevant to consider, that the pretrained VGG16 model is trained on specific sets of images. If reports and resumes were not included, it could potentially result in poorer performance even after finetuning.

Additionally, the choice of optimizer plays a crucial role in model training. The models were tested with both SGD and Adam optimizers to assess their performance under different optimization strategies. Generally, the Adam optimizer performs very well on complex data such as images and is robust to noisy data, which likely likely contributed to its effectiveness in optimizing the models on the Tobacco3482 dataset.

Finally, it is worth noting that the models were trained for a relatively small number of epochs (10 epochs) and a batch size of 64 due to computational constraints. While the models may not have fully converged within this setup, the results still provide valuable insights into their performance and capabilities. It would be relevant to further explore the potential of the models with longer training.
