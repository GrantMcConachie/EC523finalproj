# EC523finalproj

## Counterfactual Image Generation of Alzheimer's Disease MRI Images

This project aims to develop an accurate framework to both predict the severity of AD and generate counterfactual samples using low-resolution MRI images. The model will consist of a convolutional neural network (CNN) classifier as well as a diffusion model. The model will be able to predict disease states from low-resolution images and give insight into areas of the brain that are responsible for AD. We will use 2D structural MRI images (176 x 208), as shown in Figure 1, to train the model. The model will be evaluated by comparing it to previous work and with accuracy metrics. 

CAE_model.py contains a Convolution AutoEncoder that maps MNIST data images to latent space and generates images of the MNIST dataset. 

Densenet classifiest MNIST data
