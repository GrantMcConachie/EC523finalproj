"""
Functions that calculate metrics for our model. All metrics in paper can be 
recovered by changing variables `PN_or_PP` and `MNIST_or_BRAIN`. PN_or_PP variable
does not affect counterfactual instance results. Class switch % varies for 
counterfactual instances as the models used to calculate them vary.

NOTE: Have to unzip the data files and may have to change the relative directories to
get the script to run.

@author: Grant
"""

PN_or_PP = "PN" # 'PN' or 'PP'
MNIST_or_BRAIN = "BRAIN" # 'BRAIN' or 'MNIST

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models, optimizers
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm
import os
from PIL import Image

def class_switch_percent(model_path, original_im, counterfac_im, pred_axis=1, MNIST_or_BRAIN="BRAIN"):
    """
    Args:
        model_path `str` - path to pretrained DenseNet model (e.g. "./path/to/densenet.h5")
        original_im `numpy.ndarray` - sequence of original images (batch x img_x x img_y x channel)
        counterfac_im `numpy.ndarray` - sequence of counterfactual images (batch x img_x x img_y x channel)
    """
    if MNIST_or_BRAIN == "MNIST":
        densenet = load_model(model_path, compile=False)
        densenet.compile(optimizer='adam')
    elif MNIST_or_BRAIN == "BRAIN":
        densenet = load_model(model_path, compile=False)
        densenet.compile(optimizer='adam',
                                loss=tf.losses.CategoricalCrossentropy())
    correct_class = np.argmax(densenet.predict(original_im), axis=pred_axis)
    switch_class = np.argmax(densenet.predict(counterfac_im), axis=pred_axis)
    p_switch = np.sum(correct_class == switch_class) / original_im.shape[0]

    return f"Percentage of images successfully swithced classes {100 * (1-p_switch)}%"

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
    or_im_code = resnet.predict(original_im)[:, 0, 0, :]
    count_im_code = resnet.predict(counterfac_im)[:, 0, 0, :]

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
    
def convert_2d_to_3d_and_pad(images, num_channels=3):
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

def convert_2d_to_3d(images, num_channels=3):
    """
    Converts a batch of 2d images to 3d images with reeating channels

    Args:
        images `numpy.ndarray` - set of images to convert (batch x img_x x img_y)

    Returns:
        new_images `numpy.ndarray` - converted images (batch x img_x x img_y x num_channels)
    """
    new_images = np.array([
        np.repeat(img[:, :, np.newaxis], num_channels, axis=2)
        for img in images
    ])

    return new_images

if __name__ == '__main__':

    print("\nCEM:")
    #########################
    ########## CEM ##########
    #########################

    # Load in the data
    if MNIST_or_BRAIN == "BRAIN":
        data_dir = "git_folder\\CEM_data\\npy_kaggle\\" 
    elif MNIST_or_BRAIN == "MNIST" and PN_or_PP == "PP":
        data_dir = "git_folder\\CEM_data\\saved_PP\\"
    elif MNIST_or_BRAIN == "MNIST" and PN_or_PP == "PN":
        data_dir = "git_folder\\CEM_data\\saved_PN\\" 

    MNIST_reg = []
    MNIST_pp = []
    brain_reg = []
    brain_pp = []

    for images in reversed(os.listdir(data_dir)):
        # check if the image ends with png
        if "brain" in images:
            if PN_or_PP in images:
                try:
                    img = np.squeeze(np.load(data_dir + images))
                    brain_pp.append(img)
                except:
                    print(images)
            elif "reg" in images:
                img = np.squeeze(np.load(data_dir + images))
                brain_reg.append(img)
        elif PN_or_PP in images:
            img = np.squeeze(np.load(data_dir + images, allow_pickle=True))
            MNIST_pp.append(img)
        elif "reg" in images:
            img = np.squeeze(np.load(data_dir + images, allow_pickle=True)) 
            MNIST_reg.append(img)

    # convert to arrays
    if MNIST_or_BRAIN == "MNIST":
        MNIST_reg = np.array(MNIST_reg)
        MNIST_pp = np.array(MNIST_pp)
        if PN_or_PP == "PP":
            MNIST_pp = MNIST_reg - MNIST_pp

    elif MNIST_or_BRAIN == "BRAIN":
        brain_reg = np.array(brain_reg)
        brain_pp = np.array(brain_pp)
        if PN_or_PP == "PP":
            brain_pp = brain_reg - brain_pp

    # funcs
    if MNIST_or_BRAIN == "MNIST":
        print(class_switch_percent("git_folder\\test_models\\mnist_cnn.h5", MNIST_reg, MNIST_pp, MNIST_or_BRAIN=MNIST_or_BRAIN))
    elif MNIST_or_BRAIN == "BRAIN":
        print(class_switch_percent("git_folder\\test_models\\kaggle_model.h5", brain_reg, brain_pp, MNIST_or_BRAIN=MNIST_or_BRAIN))

    # adding 3 channels for FID and KNN
    if MNIST_or_BRAIN == "MNIST":
        MNIST_pp_3d = convert_2d_to_3d_and_pad(MNIST_pp)
        MNIST_reg_3d = convert_2d_to_3d_and_pad(MNIST_reg)
    
    if MNIST_or_BRAIN == "MNIST":
        print(knn(MNIST_reg_3d, MNIST_pp_3d))
        print(fid(MNIST_reg_3d, MNIST_pp_3d))
    elif MNIST_or_BRAIN == "BRAIN":
        print(knn(brain_reg, brain_pp))
        print(fid(brain_reg, brain_pp))

    print("\nCounterfactual instances:")
    #########################
    ###### Counterfact ######
    #########################

    # Load in the data
    data_dir_MNIST_count = "git_folder\\CEM_data\\counterfactual\\MNIST\\counterfactual_images\\"
    data_dir_MNIST_org = "git_folder\\CEM_data\\counterfactual\\MNIST\\mnist_images\\"
    data_dir_brain_count = "git_folder\\CEM_data\\sudan_cf\\counterfactual_images\\"
    data_dir_brain_org = "git_folder\\CEM_data\\sudan_cf\\original_images\\"
    MNIST_count = []
    MNIST_org = []
    brain_count = []
    brain_org = []
    
    for images in os.listdir(data_dir_MNIST_count):
        MNIST_count.append(np.asarray(Image.open(data_dir_MNIST_count + images)) / 255)

    for images in os.listdir(data_dir_MNIST_org):
        MNIST_org.append(np.asarray(Image.open(data_dir_MNIST_org + images)) / 255)

    for images in os.listdir(data_dir_brain_count):
        brain_count.append(np.asarray(Image.open(data_dir_brain_count + images)) / 255)

    for images in os.listdir(data_dir_brain_org):
        brain_org.append(np.asarray(Image.open(data_dir_brain_org + images)) / 255)

    MNIST_count = np.squeeze(np.array(MNIST_count)[:, :, :, 0])
    MNIST_org = np.squeeze(np.array(MNIST_org)[:, :, :, 0])
    brain_count = convert_2d_to_3d(np.array(brain_count))
    brain_org = convert_2d_to_3d(np.array(brain_org))

    # % correct
    if MNIST_or_BRAIN == "MNIST":
        print(class_switch_percent("git_folder\\test_models\\mnist_cnn.h5", MNIST_org, MNIST_count))
    elif MNIST_or_BRAIN == "BRAIN":
        print(class_switch_percent("git_folder\\test_models\\kaggle_model.h5", brain_org, brain_count))

    # adding 3 channels for FID and KNN
    MNIST_count_3d = convert_2d_to_3d_and_pad(MNIST_count)
    MNIST_org_3d = convert_2d_to_3d_and_pad(MNIST_org)

    if MNIST_or_BRAIN == "MNIST":
        print(knn(MNIST_org_3d, MNIST_count_3d))
        print(fid(MNIST_org_3d, MNIST_count_3d))
    elif MNIST_or_BRAIN == "BRAIN":
        print(knn(brain_org, brain_count))
        print(fid(brain_org, brain_count))

    print("complete")