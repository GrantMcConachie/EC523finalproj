# EC523 Final Project: Counterfactual Image Generation of Alzheimer's Disease MRI Images

## Overview
This project is dedicated to developing a framework capable of predicting Alzheimer's Disease (AD) severity and generating counterfactual samples using low-resolution MRI images. Utilizing Convolutional Neural Network (CNN) architectures, the project validates model efficacy on the MNIST dataset before applying the same architecture for brain image classification. The generation of counterfactual instances is achieved using the Contrastive Explanation Method (CEM), pushing images across different classes for further analysis.

### Objectives
- Predict disease states from low-resolution MRI images (2D structural MRI images, 176 x 208).
- Identify brain areas responsible for AD.
- Validate and compare the model with existing methods and accuracy metrics.

## Repository Contents
- `CAE_MNIST_model.ipynb`: Jupyter notebook implementing a Convolutional Autoencoder for the MNIST dataset.
- `CAE_MNIST_model.h5`: Trained model file.
- `CEM_brain.py`, `CEM_MNIST.py`: Python scripts for implementing the Contrastive Explanation Method.
- `DenseNetModel.ipynb`: Jupyter notebook providing DenseNet architecture for both MNIST and brain image datasets.
- `Inceptionv3_model.py`: Implementation of Inception V3 architecture.
- `cf_MRI_kaggle.ipynb`: Jupyter notebook for applying counterfactual methods to the MRI dataset using a new CNN architecture and the omnixai library.
- `cf_mnist.ipynb`: Jupyter notebook for applying counterfactual methods to the MNIST dataset using the alibi library.
- `metrics.py`: Python script for the implementation and calculation of metrics used in the final report.

## Installation Instructions
To install the necessary packages, run the following commands:
```bash
# Install Alibi library
pip install tensorflow[alibi]

# Install omnixai
pip install omnixai
```

## Acknowledgements
This project is part of the EC523 coursework at Boston University. We extend our gratitude to our professors and teaching assistants for their invaluable guidance and support.
