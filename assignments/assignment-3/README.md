# Assignment 3 - Document classification using pretrained image embeddings
## Transfer learning w/ pretrained CNNs
*By Sofie Mosegaard, 12-04-2024*

This repository is designed to perform transfer learning with pretrained CNN's on document classification using.

The project will utilize the VGG16 model, which ...

16 layers; 134 million trainable parameters
Deep, complex

Multiple stacked pooling and convolution layers

Massive – around 500MB

Final output layer of 1000 nodes


I will test, whether we can predict types of document purely based only on its appearance rather than its contents? As a scientific paper appears differently to a poem, this repository tries to leverage this knowledge to try to predict what type of document we have, based on its visual appearance.

The project has the objective:
- To demonstrate that you can use ```tensorflow``` to train Convolutional Neural Networks
- To synthesize existing skills to create pipelines for transfer learning with pretrained CNNs
- To show understanding of how to interpret machine learning outputs

## Data source

The model will be trained on image data from the ```Tobacco3482``` dataset, which can be found [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). The data comprises 3842 images in black and white across 10 different document types.


- images in the categories adds, emails, forms, letters, memos, news, notes, reports, resumes, and scientific articles.

Please download the data and place it in the folder called in. Note, that you will need to unzip the data inside the folder before, you can execude the code as described below (see: Reproducibility).

## Repository structure

The repository consists of 2 bash scripts, 1 README.md file, 1 txt file, and 3 folders. The folders contains the following:

-   in: where you should locate your data
-   src: consists of three scipt for transfer learning
that will perform the repository objective. The script 

src: consists of two python scripts for Logistic Regression (LR_classifier.py) and Neural Network (NN_classifier.py) classification. Additionally, there are two python scripts performing GridSearch (LR_classifier.py, NN_classifier.py).


-   out: holds the saved results, which consists of classification reports in .txt format and loss curves in .png format for all models.

## Purpose

- To demonstrate that you can use ```tensorflow``` to train Convolutional Neural Networks
- To synthesize existing skills to create pipelines for transfer learning with pretrained CNNs
- To show understanding of how to interpret machine learning outputs


## Summary 

- the data is arranged into folders which have the label name for those images
when the data are loaded in, the images will be added to a numpy array
additionally, the labels will be extracted and added to an array as well

To increate the robustness and generalizability of the model, batch normalization and augmentation were implemented. 

- the saved results were obtained by training the models on 10 epochs and a batch size of 64. Natrually, the models havent fully learned on so few epochs, but as the models are computationally heavy to run, they still provide an indication of their performance. 



## Discussion

-   Your repository should also include a short description of what the classification report and learning curve show.
-   To show understanding of how to interpret machine learning outputs

The best models were ... based on acc. score and loss curve during training..

The DatAug models didnt perform very well, which might be bacuse, the generated data was too different from the real data. The DatAug method is used to improve robustness, but as they were to different, the learning curve was too steep for the models to learn the patterns.