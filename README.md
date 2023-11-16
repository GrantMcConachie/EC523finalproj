# EC523finalproj

## Counterfactual Image Generation of Alzheimer's Disease MRI Images

This project aims to develop an accurate framework to both predict the severity of AD and generate counterfactual samples using low-resolution MRI images. The model will consist of a convolutional Autoencoder (CAE) classifier as well as a diffusion model. The model will be able to predict disease states from low-resolution images and give insight into areas of the brain that are responsible for AD. We will use 2D structural MRI images (176 x 208), as shown in Figure 1, to train the model. The model will be evaluated by comparing it to previous work and with accuracy metrics. In addition, the project aims to train dense net and CAE on the MNIST model to validate its accuracy and then also train on MRI images. Diffusion model on MNIST and MRI images will produce counterfactual examples.

CAE_model.py contains a Convolution AutoEncoder that maps MNIST data images to latent space and generates images of the MNIST dataset. 

Densenet classifies MNIST data with high accuracy.

CS253_ISTA_FISTA.pdf provides a brief review of the FISTA algorithm by referencing recent research. The purpose of the review is to see its feasibility and ease of use with the diffusion models.


