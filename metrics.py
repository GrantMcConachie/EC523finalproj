"""
Functions that calculate metrics for our model
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models, optimizers
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm

def class_switch_percent(model_path, original_im, counterfac_im, pred_axis=1):
    """
    Args:
        model_path `str` - path to pretrained DenseNet model (e.g. "./path/to/densenet.h5")
        original_im `numpy.ndarray` - sequence of original images (batch x img_x x img_y x channel)
        counterfac_im `numpy.ndarray` - sequence of counterfactual images (batch x img_x x img_y x channel)
    """
    densenet = load_model(model_path, compile=False)
    correct_class = np.argmax(densenet.predict(original_im), axis=pred_axis)
    switch_class = np.argmax(densenet.predict(counterfac_im), axis=pred_axis)
    p_switch = np.sum(correct_class == switch_class) / original_im.shape[0]

    return f"Percentage of images successfully swithced classes {100 * p_switch}%"

def fid(original_im, counterfac_im):
    """
    calculates the Frechet Inception Distance score which is the distance between
    vectors calculated for real and generated images. A lower FID indicates better
    quality images and a higher FID indicates lower quality images. Takes a while 
    to run.

    Args:
        original_im `numpy.ndarray` - sequence of original images (batch x img_x x img_y x channel)
        counterfac_im `numpy.ndarray` - sequence of counterfactual images (batch x img_x x img_y x channel)

    Retuns:
        fid `float` - fid score between the real images and the counterfactual ones
    """
    # upscale images if less than 75 pixels
    new_or_img = []
    new_c_img = []
    img_shape = original_im[0].shape[0]

    if img_shape < 75:
        pad_num = round((75-img_shape) / 2)
        for o_img, c_img in zip(original_im, counterfac_im):
            new_or_img.append(np.pad(o_img, ((pad_num, pad_num), (pad_num, pad_num), (0, 0)), mode='constant', constant_values=0))
            new_c_img.append(np.pad(c_img, ((pad_num, pad_num), (pad_num, pad_num), (0, 0)), mode='constant', constant_values=0))

        original_im = np.array(new_or_img)
        counterfac_im = np.array(new_c_img)

    # Load in the inception v3 network
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False, 
        pooling='avg', 
        input_shape=original_im[0].shape
    )

    # Calculating img imbeddings
    o_img_embed = inceptionv3.predict(original_im)
    c_img_embed = inceptionv3.predict(counterfac_im)

    # calculate mean and covariance statistics
    mu1, sigma1 = o_img_embed.mean(axis=0), np.cov(o_img_embed, rowvar=False)
    mu2, sigma2 = c_img_embed.mean(axis=0), np.cov(c_img_embed, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid

def knn(original_im, counterfac_im):
    """
    Calculated the k-th nearest neighbors of each perturbed image

    Args:
        original_im `numpy.ndarray` - sequence of original images (batch x img_x x img_y x channel)
        counterfac_im `numpy.ndarray` - sequence of counterfactual images (batch x img_x x img_y x channel)
    """
    # Load a pretrained resnet and remove last layer
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=original_im[0].shape
    )
    resnet.layers.pop()
    resnet.layers.pop()
    resnet.outputs = [resnet.layers[-1].output]

    # Generating codes for all the counterfac and original images
    or_im_code = np.squeeze(resnet.predict(original_im))
    count_im_code = np.squeeze(resnet.predict(counterfac_im))

    # Fitting KNN
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(or_im_code)
    nearest_neighbors = neigh.kneighbors(count_im_code, return_distance=False)

    # percentage of nearest neighbors being the counterfactual example
    correct = 0
    incorrect = 0
    for i, n in enumerate(nearest_neighbors):
        if n[0] == i:
            correct += 1
        else:
            incorrect += 1

    return f"Percentage of images neareast their counterfactual image: {100 * correct / (incorrect + correct)}%"
    
def convert_2d_to_3d(images, num_channels=3):
    """
    Converts a batch of 2d images to 3d images with reeating channels

    Args:
        images `numpy.ndarray` - set of images to convert (batch x img_x x img_y)

    Returns:
        new_images `numpy.ndarray` - converted images (batch x img_x x img_y x num_channels)
    """
    new_images = np.array([
        np.repeat(np.pad(img, ((2, 2), (2, 2)), mode='constant', constant_values=0)[:, :, np.newaxis], num_channels, axis=2)
        for img in images
    ])

    return new_images

if __name__ == '__main__':
    # Load in MNIST, make 3 channeled + pad
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = convert_2d_to_3d(x_train)
    x_test = convert_2d_to_3d(x_test)

    # test funcs
    # print(class_switch_percent("./test_models/DenseNet_MNIST_tfmodel.h5", x_test, x_test))
    # print(knn(x_train[:10000], x_test))
    # print(knn(x_test, x_test))
    print("fid:", fid(x_train[:10000], x_test))