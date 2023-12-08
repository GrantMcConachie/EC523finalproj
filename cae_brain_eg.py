import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model
import os
from PIL import Image
import matplotlib.pyplot as plt

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

def ae_model():
    x_in = Input(shape=(256, 256, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation=None, padding='same')(x)

    autoencoder = Model(x_in, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

if __name__ == '__main__':
    # Load data
    data_path = "./data/brain/allimages_train/"
    data_path_train = "data\\archive\\Alzheimer_s Dataset\\reshape\\allimages_train\\"
    data_path_test = "data\\archive\\Alzheimer_s Dataset\\reshape\\allimages_test\\"

    # Train
    all_img_train = []
    for images in os.listdir(data_path_train):
        img = np.asarray(Image.open(data_path_train + images))
        all_img_train.append(img)

    all_img_train = np.array(all_img_train)
    all_img_train = convert_2d_to_3d(all_img_train)

    # Test
    all_img_test = []
    for images in os.listdir(data_path_test):
        img = np.asarray(Image.open(data_path_test + images))
        all_img_test.append(img)

    all_img_test = np.array(all_img_test)
    all_img_test = convert_2d_to_3d(all_img_test)

    ae = ae_model()
    print(ae.summary())

    ae.fit(all_img_train, all_img_train, batch_size=128, epochs=4, validation_data=(all_img_test, all_img_test), verbose=1)
    decode_im = ae.predict(all_img_test)

    plt.figure()
    plt.title("original image")
    plt.imshow(all_img_test[0][:,:,0])

    plt.figure()
    plt.title("reconstructed image")
    plt.imshow(decode_im[0][:,:,0])

    print("done")