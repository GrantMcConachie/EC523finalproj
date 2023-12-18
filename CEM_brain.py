import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
#tf.get_logger().setLevel(40) # suppress deprecation messages
#tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
#from omnixai.explainers.vision import ContrastiveExplainer

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False

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
    
## Load and prepare brain data

x_train = np.load("/projectnb/ec523kb/students/gdmac/project/alibi/data/kaggle/train_data_kaggle.npy")
x_test = np.load("/projectnb/ec523kb/students/gdmac/project/alibi/data/kaggle/test_data_kaggle.npy")
y_train = np.load("/projectnb/ec523kb/students/gdmac/project/alibi/data/kaggle/train_labels_kaggle.npy")
y_test = np.load("/projectnb/ec523kb/students/gdmac/project/alibi/data/kaggle/test_labels_kaggle.npy")

# converting to 3d
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)

# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# x_train = np.reshape(x_train, x_train.shape + (1,))
# x_test = np.reshape(x_test, x_test.shape + (1,))
# print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
# print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)

# xmin, xmax = -.5, .5
# x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
# x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin

## Define and train the CNN

def cnn_model():
    x_in = Input(shape=(208, 176, 1))
    x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_out = Dense(4, activation='softmax')(x)
    
    cnn = Model(inputs=x_in, outputs=x_out)
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return cnn
    
#cnn = cnn_model()
#cnn.summary()
#cnn.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1)
#cnn.save('./models/brain_cnn.h5', save_format='h5')

cnn = load_model('./models/kaggle_model.h5')
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])

## Define and train the Autoencoder

def ae_model():
    x_in = Input(shape=(152, 152, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    encoded = Conv2D(3, (3, 3), activation=None, padding='same')(x)

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
    
ae = ae_model()
ae.summary()
ae.fit(x_train, x_train, batch_size=128, epochs=100, validation_data=(x_test, x_test), verbose=1)
ae.save('./models/brain_ae_kaggle.h5', save_format='h5')

ae = load_model('./models/brain_ae_kaggle.h5')

assert False
decoded_imgs = ae.predict(x_test)
n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i][:,:,0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i][:,:,0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

## generate constrastive explinatation with pertinant negatives

idx = 15
X = x_test[idx].reshape((1,) + x_test[idx].shape)

print("prediction:", cnn.predict(X).argmax(), "confidence:", cnn.predict(X).max())

preprocess = lambda x: x

explainer = ContrastiveExplainer(
    model=cnn,
    preprocess_function=preprocess,
    ae_model=ae
)

explanation = explainer.explain(X)

assert False

mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
shape = (1,) + x_train.shape[1:]  # instance shape
kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
            # class predicted by the original instance and the max probability on the other classes 
            # in order for the first loss term to be minimized
beta = .1  # weight of the L1 loss term
gamma = 100  # weight of the optional auto-encoder loss term
c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or 
              # the same class (PP) for the perturbed instance compared to the original instance to be explained
c_steps = 10  # nb of updates for c
max_iterations = 1000  # nb of iterations per value of c
feature_range = (x_train.min(),x_train.max())  # feature range for the perturbed instance
clip = (-1000.,1000.)  # gradient clipping
lr = 1e-2  # initial learning rate
no_info_val = -1. # a value, float or feature-wise, which can be seen as containing no info to make a prediction
                  # perturbations towards this value means removing features, and away means adding features
                  # for our MNIST images, the background (-0.5) is the least informative, 
                  # so positive/negative perturbations imply adding/removing features


# initialize CEM explainer and explain instance
cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
          gamma=gamma, ae_model=ae, max_iterations=max_iterations, 
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

explanation = cem.explain(X)

print(f'Pertinent negative prediction: {explanation.PN_pred}')
plt.figure()
plt.imshow(explanation.PN.reshape(150, 150))
plt.show()

assert False

mode = 'PP'

# initialize CEM explainer and explain instance
cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
          gamma=gamma, ae_model=ae, max_iterations=max_iterations, 
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

for idx in range(50):
    X = x_test[idx].reshape((1,) + x_test[idx].shape)
    explanation = cem.explain(X)
    np.save(f"./saved_PP/brain_non/PP_{idx}_brain.npy", explanation.PP)
    np.save(f"./saved_PP/brain_non/reg_{idx}_brain.npy", X)

print(f'Pertinent positive prediction: {explanation.PP_pred}')
plt.figure()
plt.imshow(explanation.PP.reshape(208, 176))
plt.show()
